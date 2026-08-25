"""AWS Bedrock passthrough handlers for HeadroomProxy.

A client pointed at a Bedrock endpoint does not speak the Anthropic Messages
protocol. Claude Code with ``CLAUDE_CODE_USE_BEDROCK=1`` POSTs to
``/model/{modelId}/invoke`` and ``/model/{modelId}/invoke-with-response-stream``;
the Vercel AI SDK provider that OpenCode uses POSTs to
``/model/{modelId}/converse`` and ``/model/{modelId}/converse-stream``. Both
shapes fell through Headroom's catch-all and were forwarded verbatim, so those
clients saved nothing.

These handlers intercept both shapes, compress the request body with the same
``anthropic_pipeline`` used for ``/v1/messages``, and forward to
``config.bedrock_api_url``.

Credentials. A SigV4 signature covers a hash of the body, and the caller signed
for Headroom's own host, so an inbound signature is void twice over by the time
we forward. When the configured upstream is an AWS-operated endpoint we therefore
re-sign the outbound request ourselves, taking the region from the endpoint
hostname (the endpoint we are calling is the only authority on which region it
belongs to). A Bedrock API key needs none of that: a bearer token commits to
neither body nor host, so one sent by the caller is forwarded as-is, and one in
``AWS_BEARER_TOKEN_BEDROCK`` is used for callers that hold no AWS credentials.
When the upstream is anything else — LiteLLM, LocalStack, a corporate gateway —
the caller's headers pass through untouched and that gateway owns signing,
exactly as before.

Body shapes. An InvokeModel body for an Anthropic model already IS the Anthropic
Messages shape (the model travels in the URL), so the pipeline applies with no
translation. A Converse body is not: its content blocks are typeless unions, and
every transform dispatches on ``block["type"]``. ``tag_converse_text`` and
``untag_converse_text`` add and remove that discriminator around the pipeline
call.

The Bedrock knowledge these handlers lean on — the endpoint vocabulary, the
signer, the Converse shims — lives in :mod:`headroom.proxy.bedrock`.

Responses are forwarded byte-faithfully — JSON for the unary calls, AWS
event-stream binary framing for the streaming ones. Nothing response-side is
parsed or mutated, since all compression happens request-side.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING
from urllib.parse import quote

from headroom.proxy.bedrock import (
    BEDROCK_ACTIONS,
    INBOUND_SIGNATURE_HEADERS,
    aws_bedrock_signing_target,
    bearer_token,
    control_plane_base,
    has_bearer_auth,
    sigv4_headers,
    tag_converse_text,
    untag_converse_text,
)

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger("headroom.proxy")

LOG_TAG = "bedrock_invoke"


class BedrockHandlerMixin:
    """Mixin providing the Bedrock passthrough handlers."""

    def _bedrock_upstream_base(self) -> str | None:
        """Resolved Bedrock upstream, or ``None`` when unconfigured.

        Returns the normalized ``config.bedrock_api_url`` (trailing slash
        stripped). ``None`` means the feature is off — the routes are not even
        registered in that case, so a ``None`` here is a defensive guard only.
        """
        base = getattr(self.config, "bedrock_api_url", None)  # type: ignore[attr-defined]
        return base.rstrip("/") if base else None

    def _unconfigured_response(self) -> Response:
        """503 for the unreachable case where a route ran without an upstream."""
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "configuration_error",
                    "message": "Bedrock passthrough requested but --bedrock-api-url is unset.",
                }
            },
        )

    async def handle_bedrock_invoke(
        self,
        request: Request,
        model_id: str,
        *,
        action: str,
    ) -> Response | StreamingResponse:
        """Compress and forward a Bedrock model-invocation request.

        Args:
            request: The inbound FastAPI request.
            model_id: The Bedrock model / inference-profile id captured from the
                URL path (may contain ``.``, ``:`` and ``/``).
            action: The URL suffix being served, a key of :data:`BEDROCK_ACTIONS`.
        """
        from fastapi.responses import Response

        from headroom.proxy.auth_mode import classify_client
        from headroom.proxy.helpers import (
            COMPRESSION_TIMEOUT_SECONDS,
            MAX_MESSAGE_ARRAY_LENGTH,
            _headroom_bypass_enabled,
            _strip_internal_headers,
            extract_tags,
            read_request_json_with_bytes,
        )
        from headroom.proxy.modes import is_cache_mode
        from headroom.utils import extract_user_query

        start_time = time.time()
        request_id = await self._next_request_id()  # type: ignore[attr-defined]
        stream, converse = BEDROCK_ACTIONS[action]

        base = self._bedrock_upstream_base()
        if base is None:
            # Routes only register when configured, so this is unreachable in
            # practice; fail loud rather than silently forwarding nowhere.
            return self._unconfigured_response()

        url = f"{base}/model/{quote(model_id, safe='')}/{action}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        # Outbound headers (case-insensitive drops). Two header sets:
        #   - verbatim: forwards the original bytes, so the inbound
        #     content-length / content-encoding still describe the body.
        #   - rewritten: the body we forward is decompressed JSON (possibly
        #     compressed by the pipeline), so content-length must be recomputed
        #     by httpx and the stale content-encoding dropped. Keeping the
        #     inbound content-length here is the classic "Too little data for
        #     declared Content-Length" footgun once the body shrinks.
        # Auth headers are left alone here; _forward_bedrock replaces them when
        # the upstream turns out to be AWS itself.
        in_headers = _strip_internal_headers(dict(request.headers.items()))
        client = classify_client(dict(request.headers.items()))
        tags = extract_tags(dict(request.headers.items()))
        verbatim_drop = {"host", "accept-encoding"}
        rewritten_drop = verbatim_drop | {"content-length", "content-encoding"}
        verbatim_headers = {k: v for k, v in in_headers.items() if k.lower() not in verbatim_drop}
        out_headers = {k: v for k, v in in_headers.items() if k.lower() not in rewritten_drop}

        # Read the body up front so we can fail open to a verbatim forward on any
        # parse error (a malformed body is the upstream's problem, not ours).
        try:
            body, raw = await read_request_json_with_bytes(request)
        except Exception as err:
            from starlette.requests import ClientDisconnect

            if isinstance(err, ClientDisconnect):
                logger.debug("[%s] %s client disconnected during body read", request_id, LOG_TAG)
                return Response(status_code=204)
            logger.warning(
                "[%s] %s could not parse body; forwarding verbatim: %s",
                request_id,
                LOG_TAG,
                err,
            )
            raw_only = await request.body()
            return await self._forward_bedrock(
                url=url,
                headers=verbatim_headers,
                content=raw_only,
                stream=stream,
                request_id=request_id,
            )

        messages = body.get("messages")
        bypass = (
            _headroom_bypass_enabled(request.headers)
            or not getattr(self.config, "optimize", True)  # type: ignore[attr-defined]
            or is_cache_mode(getattr(self.config, "mode", "token"))  # type: ignore[attr-defined]
            or not isinstance(messages, list)
            or not messages
            or len(messages) > MAX_MESSAGE_ARRAY_LENGTH
        )

        outbound = raw
        original_tokens = 0
        optimized_tokens = 0
        tokens_saved = 0
        transforms_applied: tuple[str, ...] = ()
        pipeline_timing: dict[str, float] | None = None

        if not bypass:
            try:
                if converse:
                    tag_converse_text(messages)
                context_limit = self.anthropic_provider.get_context_limit(model_id)  # type: ignore[attr-defined]
                result = await self._run_compression_in_executor(  # type: ignore[attr-defined]
                    lambda: self.anthropic_pipeline.apply(  # type: ignore[attr-defined]
                        messages=messages,
                        model=model_id,
                        model_limit=context_limit,
                        context=extract_user_query(messages),
                        request_id=request_id,
                    ),
                    timeout=COMPRESSION_TIMEOUT_SECONDS,
                )
                if result.messages != messages:
                    rewritten = (
                        untag_converse_text(result.messages) if converse else result.messages
                    )
                    body["messages"] = rewritten
                    outbound = json.dumps(body).encode("utf-8")
                    original_tokens = result.tokens_before
                    optimized_tokens = result.tokens_after
                    tokens_saved = max(0, result.tokens_before - result.tokens_after)
                    transforms_applied = tuple(result.transforms_applied)
                    pipeline_timing = result.timing
                    logger.info(
                        "[%s] %s compressed %d→%d tokens (%d saved) action=%s model=%s",
                        request_id,
                        LOG_TAG,
                        result.tokens_before,
                        result.tokens_after,
                        tokens_saved,
                        action,
                        model_id,
                    )
            except Exception as err:
                # Fail open: never break a request because compression failed.
                # `raw` is the untouched inbound bytes, so a half-tagged body
                # can never reach the wire.
                logger.warning(
                    "[%s] %s compression failed; forwarding verbatim: %s",
                    request_id,
                    LOG_TAG,
                    err,
                )
                outbound = raw

        out_headers["content-type"] = "application/json"
        response = await self._forward_bedrock(
            url=url,
            headers=out_headers,
            content=outbound,
            stream=stream,
            request_id=request_id,
        )

        # Best-effort metrics. Output tokens are left at 0 (the RequestOutcome
        # contract treats 0 as "not measured") — Bedrock responses are forwarded
        # byte-faithfully and never parsed. The valuable figure, request-side
        # compression, is recorded in full.
        try:
            from headroom.proxy.outcome import RequestOutcome

            await self._record_request_outcome(  # type: ignore[attr-defined]
                RequestOutcome(
                    request_id=request_id,
                    provider="bedrock",
                    model=model_id,
                    original_tokens=original_tokens,
                    optimized_tokens=optimized_tokens,
                    output_tokens=0,
                    tokens_saved=tokens_saved,
                    attempted_input_tokens=original_tokens,
                    total_latency_ms=(time.time() - start_time) * 1000,
                    transforms_applied=transforms_applied,
                    pipeline_timing=pipeline_timing,
                    tags=tags,
                    client=client,
                )
            )
        except Exception:
            logger.debug("[%s] %s outcome recording failed", request_id, LOG_TAG, exc_info=True)

        return response

    async def handle_bedrock_control(
        self,
        request: Request,
        path: str,
    ) -> Response | StreamingResponse:
        """Forward a Bedrock control-plane GET, e.g. ``/inference-profiles``.

        There is nothing to compress here. The route exists because a client
        pointed at Headroom sends its control-plane calls to Headroom too, and
        the catch-all would forward them to an inference host with a signature
        AWS rejects. Claude Code lists inference profiles at startup, so this is
        the difference between "no models available" and a working session.
        """
        from headroom.proxy.helpers import _strip_internal_headers

        request_id = await self._next_request_id()  # type: ignore[attr-defined]
        base = self._bedrock_upstream_base()
        if base is None:
            return self._unconfigured_response()

        url = f"{control_plane_base(base)}/{path.lstrip('/')}"
        if request.url.query:
            url = f"{url}?{request.url.query}"
        headers = {
            k: v
            for k, v in _strip_internal_headers(dict(request.headers.items())).items()
            if k.lower() not in ("host", "accept-encoding", "content-length", "content-encoding")
        }
        return await self._forward_bedrock(
            url=url,
            headers=headers,
            content=b"",
            stream=False,
            request_id=request_id,
            method="GET",
        )

    def _authorize_bedrock_request(
        self,
        *,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
        region: str,
        request_id: str,
        service: str = "bedrock",
    ) -> dict[str, str]:
        """Give the outbound request a credential AWS will actually accept.

        Three cases, in the order AWS's own SDKs resolve them:

        1. The caller sent a Bedrock API key. A bearer token commits to neither
           the body nor the host, so it is still valid after compression and is
           forwarded untouched.
        2. Headroom's environment holds a Bedrock API key. The caller's SigV4
           headers are void, so they are dropped and the key replaces them. This
           is the path for a caller with no AWS credentials at all.
        3. Otherwise re-sign with SigV4, since the caller signed a different body
           for a different host (Headroom's own) and its signature cannot satisfy
           AWS either way.

        If signing fails we forward the headers unchanged and let AWS answer: a
        real ``InvalidSignatureException`` names the problem better than a
        synthetic 502 from us, and the failure is logged here regardless.
        """
        if has_bearer_auth(headers):
            return headers

        token = bearer_token()
        if token:
            out = {k: v for k, v in headers.items() if k.lower() not in INBOUND_SIGNATURE_HEADERS}
            out["authorization"] = f"Bearer {token}"
            return out

        profile = getattr(self.config, "bedrock_profile", None)  # type: ignore[attr-defined]
        try:
            signed = sigv4_headers(
                method=method,
                url=url,
                body=body,
                region=region,
                profile=profile,
                service=service,
            )
        except Exception as err:
            logger.warning(
                "[%s] %s SigV4 signing failed; forwarding unsigned: %s", request_id, LOG_TAG, err
            )
            return headers
        out = {k: v for k, v in headers.items() if k.lower() not in INBOUND_SIGNATURE_HEADERS}
        # httpx owns framing headers; host is derived from the URL.
        out.update({k: v for k, v in signed.items() if k.lower() not in ("host", "content-length")})
        return out

    def _authorize_bedrock_target_request(
        self,
        *,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
        request_id: str,
    ) -> dict[str, str]:
        signing_target = aws_bedrock_signing_target(url)
        if signing_target is None:
            return headers
        region, service = signing_target
        return self._authorize_bedrock_request(
            method=method,
            url=url,
            body=body,
            headers=headers,
            region=region,
            request_id=request_id,
            service=service,
        )

    async def _forward_bedrock(
        self,
        *,
        url: str,
        headers: dict[str, str],
        content: bytes,
        stream: bool,
        request_id: str,
        method: str = "POST",
    ) -> Response | StreamingResponse:
        """Send a request to the Bedrock upstream and stream the reply back.

        Every forward funnels through here, which is why authorizing lives here
        too: the bytes and URL are final at this point, and no caller can forget.

        Uses the canonical httpx-as-reverse-proxy pattern: open the upstream with
        ``stream=True`` so status + headers are available immediately, then hand
        the raw byte iterator to ``StreamingResponse`` and close the upstream
        connection via a background task. Works for the JSON replies and for the
        event-stream ones alike — neither is buffered or mutated.
        """
        import httpx
        from fastapi.responses import JSONResponse, StreamingResponse
        from starlette.background import BackgroundTask

        headers = self._authorize_bedrock_target_request(
            method=method,
            url=url,
            body=content,
            headers=headers,
            request_id=request_id,
        )

        assert self.http_client is not None  # type: ignore[attr-defined]
        upstream_request = self.http_client.build_request(  # type: ignore[attr-defined]
            method,
            url,
            headers=headers,
            content=content,
        )
        try:
            upstream = await self.http_client.send(upstream_request, stream=True)  # type: ignore[attr-defined]
        except (httpx.ConnectError, httpx.TimeoutException) as err:
            logger.warning("[%s] %s upstream connect failed: %s", request_id, LOG_TAG, err)
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "type": "connection_error",
                        "message": f"Failed to connect to Bedrock upstream: {err}",
                    }
                },
            )

        # Forward raw (still-encoded) bytes, so strip hop-by-hop headers that
        # would conflict with StreamingResponse's own framing. content-encoding
        # and content-type are preserved.
        resp_headers = {
            k: v
            for k, v in upstream.headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "connection")
        }
        media_type = upstream.headers.get("content-type")
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=resp_headers,
            media_type=media_type,
            background=BackgroundTask(upstream.aclose),
        )
