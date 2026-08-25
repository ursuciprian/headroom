"""Tests for the AWS Bedrock passthrough handlers.

Covers the routes registered by ``register_provider_routes`` when
``--bedrock-api-url`` is set, and the handler behavior:

1. Routes register ONLY when ``bedrock_api_url`` is configured.
2. A large request body is compressed via ``anthropic_pipeline.apply`` and the
   compressed messages are what gets forwarded upstream.
3. Inference-profile model ids (dots/colons) are captured whole and re-encoded
   into the upstream URL.
4. The streaming route forwards upstream bytes byte-faithfully and closes the
   upstream connection.
5. Fail-open: a malformed JSON body is forwarded verbatim, never a 500.
6. Bypass (``optimize=False``) forwards verbatim — no compression.
7. The request outcome is recorded with ``provider="bedrock"``.
8. ``BEDROCK_TARGET_API_URL`` feeds the env config path.
9. An AWS upstream is re-signed with SigV4 and a gateway upstream is not;
   signing failure forwards unsigned instead of erroring.
9a. A Bedrock API key is honoured ahead of SigV4, whether it comes from the
   caller or from ``AWS_BEARER_TOKEN_BEDROCK``, and never leaks to a gateway.
10. Converse bodies compress and come back out in Converse block shape.
11. The control-plane listing goes to the control-plane host as a GET.

All forwarding is mocked at ``proxy.http_client`` so no real upstream is needed.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

UPSTREAM = "http://127.0.0.1:4000"
AWS_UPSTREAM = "https://bedrock-runtime.us-east-1.amazonaws.com"
SONNET_BEDROCK = "anthropic.claude-3-5-sonnet-20241022-v2:0"
INVOKE = f"/model/{SONNET_BEDROCK}/invoke"
INVOKE_STREAM = f"/model/{SONNET_BEDROCK}/invoke-with-response-stream"
CONVERSE = f"/model/{SONNET_BEDROCK}/converse"


class _FakeUpstream:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(
        self,
        status_code: int = 200,
        headers: dict | None = None,
        chunks: tuple[bytes, ...] = (b'{"ok":true}',),
    ) -> None:
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {"content-type": "application/json"})
        self._chunks = list(chunks)
        self.closed = False

    async def aiter_raw(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


class _FakeResult:
    """Stand-in for TransformPipeline.apply's TransformResult."""

    def __init__(
        self,
        messages: list[dict],
        tokens_before: int,
        tokens_after: int,
        transforms: tuple[str, ...] = ("smartcrush",),
    ) -> None:
        self.messages = messages
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.transforms_applied = list(transforms)
        self.timing = {"total": 1.0}


def _make_config(**overrides) -> ProxyConfig:
    base = {
        "bedrock_api_url": UPSTREAM,
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _install_fake_client(proxy, upstream: _FakeUpstream) -> MagicMock:
    """Replace proxy.http_client so forwarding never touches the network."""
    client = MagicMock()
    client.build_request = MagicMock(return_value=MagicMock(name="upstream_request"))
    client.send = AsyncMock(return_value=upstream)
    client.aclose = AsyncMock()  # awaited by proxy.shutdown() on lifespan exit
    proxy.http_client = client
    return client


def _forwarded(client: MagicMock) -> tuple[str, dict]:
    """Return (url, parsed_json_body_or_raw) handed to build_request."""
    call = client.build_request.call_args
    url = call.args[1] if len(call.args) > 1 else call.kwargs["url"]
    content = call.kwargs["content"]
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        parsed = content
    return url, parsed


# ── route gating ──────────────────────────────────────────────────────


def _paths(cfg: ProxyConfig) -> set[str]:
    app = create_app(cfg)
    return {r.path for r in app.routes if hasattr(r, "path")}


def test_routes_absent_when_bedrock_api_url_unset():
    paths = _paths(ProxyConfig())
    assert "/model/{model_id:path}/invoke" not in paths


def test_routes_present_when_bedrock_api_url_set():
    paths = _paths(_make_config())
    assert "/model/{model_id:path}/invoke" in paths
    assert "/model/{model_id:path}/invoke-with-response-stream" in paths


# ── compression ───────────────────────────────────────────────────────


def test_invoke_forwards_compressed_messages():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        compressed = [{"role": "user", "content": "short"}]
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult(compressed, tokens_before=5000, tokens_after=200)
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "x" * 5000}],
            "max_tokens": 100,
        }
        resp = client.post(INVOKE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == compressed


def test_compressed_body_drops_stale_content_length():
    """Regression: a shrunk body must not carry the inbound content-length, or
    httpx raises 'Too little data for declared Content-Length' (caught in
    live testing). content-encoding is dropped on the same path so a stale
    gzip claim can't mislabel the re-serialized JSON."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "tiny"}], 5000, 100)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "x" * 8000}], "max_tokens": 8},
        )

    assert resp.status_code == 200
    sent_headers = http.build_request.call_args.kwargs["headers"]
    lower = {k.lower() for k in sent_headers}
    assert "content-length" not in lower
    assert "content-encoding" not in lower


def test_invoke_preserves_non_message_body_fields():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult(
                [{"role": "user", "content": "c"}], tokens_before=900, tokens_after=100
            )
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": "y" * 3000}],
            "max_tokens": 256,
        }
        client.post(INVOKE, json=body)

    _, forwarded = _forwarded(http)
    assert forwarded["anthropic_version"] == "bedrock-2023-05-31"
    assert forwarded["max_tokens"] == 256


# ── model id encoding ─────────────────────────────────────────────────


def test_inference_profile_model_id_is_captured_and_reencoded():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 900, 100)
        )
        profile = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
        client.post(
            f"/model/{profile}/invoke",
            json={"messages": [{"role": "user", "content": "z" * 3000}], "max_tokens": 8},
        )

    url, _ = _forwarded(http)
    # Colon is percent-encoded; the whole profile id survives in the path.
    assert url == f"{UPSTREAM}/model/us.anthropic.claude-sonnet-4-5-20250929-v1%3A0/invoke"


# ── streaming ─────────────────────────────────────────────────────────


def test_invoke_with_response_stream_is_byte_faithful():
    app = create_app(_make_config())
    upstream = _FakeUpstream(chunks=(b"event-stream-chunk-1", b"event-stream-chunk-2"))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_fake_client(proxy, upstream)
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 900, 100)
        )
        resp = client.post(
            INVOKE_STREAM,
            json={"messages": [{"role": "user", "content": "w" * 3000}], "max_tokens": 8},
        )

    assert resp.status_code == 200
    assert resp.content == b"event-stream-chunk-1event-stream-chunk-2"
    assert upstream.closed is True


# ── fail-open + bypass ────────────────────────────────────────────────


def test_malformed_body_is_forwarded_verbatim():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        # Pipeline must NOT be invoked on an unparseable body.
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        resp = client.post(
            INVOKE,
            content=b"not-json-at-all",
            headers={"content-type": "application/json"},
        )

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded == b"not-json-at-all"


def test_optimize_disabled_forwards_verbatim():
    app = create_app(_make_config(optimize=False))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8}
        resp = client.post(INVOKE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == body["messages"]


def test_compression_failure_forwards_verbatim():
    """Fail-open: if the pipeline raises, forward the ORIGINAL body untouched
    rather than 500ing the request."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=RuntimeError("boom"))
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8}
        resp = client.post(INVOKE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == body["messages"]


def test_bypass_header_skips_compression():
    """`x-headroom-bypass: true` forwards verbatim — the pipeline never runs."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        body = {"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8}
        resp = client.post(INVOKE, json=body, headers={"x-headroom-bypass": "true"})

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded["messages"] == body["messages"]


def test_upstream_connect_error_returns_502():
    """A transport failure to the gateway surfaces as a clean 502, not a crash."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        client_mock = _install_fake_client(proxy, _FakeUpstream())
        client_mock.send = AsyncMock(side_effect=httpx.ConnectError("no route"))
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 900, 100)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "z" * 3000}], "max_tokens": 8},
        )

    assert resp.status_code == 502


# ── metrics ───────────────────────────────────────────────────────────


def test_outcome_recorded_with_bedrock_provider():
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_fake_client(proxy, _FakeUpstream())
        proxy._record_request_outcome = AsyncMock()
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 1000, 250)
        )
        client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "q" * 3000}], "max_tokens": 8},
        )

    assert proxy._record_request_outcome.await_count == 1
    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.provider == "bedrock"
    assert outcome.tokens_saved == 750


# ── env config path ───────────────────────────────────────────────────


def test_env_var_feeds_config(monkeypatch):
    from headroom.proxy.server import _proxy_config_from_env

    monkeypatch.delenv("HEADROOM_PROXY_CONFIG_JSON", raising=False)
    monkeypatch.setenv("BEDROCK_TARGET_API_URL", UPSTREAM)
    cfg = _proxy_config_from_env()
    assert cfg.bedrock_api_url == UPSTREAM


# ── endpoint classification ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://bedrock-runtime.us-east-1.amazonaws.com", "us-east-1"),
        ("https://bedrock-runtime.eu-west-1.amazonaws.com/model/x/invoke", "eu-west-1"),
        ("https://bedrock.us-east-1.amazonaws.com/inference-profiles", "us-east-1"),
        ("https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com", "us-gov-west-1"),
        ("https://bedrock-runtime.cn-north-1.amazonaws.com.cn", "cn-north-1"),
        # Gateways: everything below must NOT be signed for.
        ("http://127.0.0.1:4000", None),
        ("https://litellm.internal.example.com", None),
        ("https://bedrock-runtime.us-east-1.amazonaws.com.evil.example", None),
        ("https://not-bedrock.us-east-1.amazonaws.com", None),
        ("https://s3.us-east-1.amazonaws.com", None),
    ],
)
def test_aws_bedrock_region_discriminates_aws_from_gateway(url, expected):
    from headroom.proxy.bedrock import aws_bedrock_region

    assert aws_bedrock_region(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://bedrock-runtime.us-east-1.amazonaws.com/",
            "https://bedrock.us-east-1.amazonaws.com",
        ),
        (
            "https://bedrock-runtime-fips.us-east-1.amazonaws.com",
            "https://bedrock-fips.us-east-1.amazonaws.com",
        ),
        # Already control plane, and non-AWS gateways, are left alone.
        ("https://bedrock.us-east-1.amazonaws.com", "https://bedrock.us-east-1.amazonaws.com"),
        ("http://127.0.0.1:4000/", "http://127.0.0.1:4000"),
    ],
)
def test_control_plane_base(url, expected):
    from headroom.proxy.bedrock import control_plane_base

    assert control_plane_base(url) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    (
        (
            "https://bedrock-mantle.eu-west-1.api.aws/openai/v1/responses",
            ("eu-west-1", "bedrock-mantle"),
        ),
        (
            "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1/responses",
            ("eu-west-1", "bedrock"),
        ),
        ("https://litellm.internal.example.com/openai/v1/responses", None),
    ),
)
def test_aws_bedrock_signing_target(url, expected):
    from headroom.proxy.bedrock import aws_bedrock_signing_target

    assert aws_bedrock_signing_target(url) == expected


# ── credentials ───────────────────────────────────────────────────────


def _no_api_key(monkeypatch):
    """Drop any real Bedrock API key from the environment.

    A key outranks SigV4, so an operator who has one exported would otherwise
    flip every signing assertion below.
    """
    from headroom.proxy.bedrock import BEARER_TOKEN_ENV

    monkeypatch.delenv(BEARER_TOKEN_ENV, raising=False)


def _fake_credentials(monkeypatch):
    """Point the signer at fixed credentials so signatures are reproducible."""
    pytest.importorskip("botocore")
    from botocore.credentials import Credentials

    _no_api_key(monkeypatch)
    session = MagicMock()
    session.get_credentials.return_value = Credentials(
        "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "session-token"
    )
    monkeypatch.setattr(
        "headroom.proxy.bedrock.signing.aws_session", MagicMock(return_value=session)
    )


def test_sigv4_headers_accepts_mantle_service(monkeypatch):
    _fake_credentials(monkeypatch)
    from headroom.proxy.bedrock import sigv4_headers

    headers = sigv4_headers(
        method="POST",
        url="https://bedrock-mantle.eu-west-1.api.aws/openai/v1/responses",
        body=b"{}",
        region="eu-west-1",
        profile="corp",
        service="bedrock-mantle",
    )

    assert "/bedrock-mantle/aws4_request" in headers["Authorization"]


def _sent_headers(client: MagicMock) -> dict[str, str]:
    return {k.lower(): v for k, v in client.build_request.call_args.kwargs["headers"].items()}


def _post_to_aws(monkeypatch, path=INVOKE, headers=None, **cfg):
    """POST a compressible body at an AWS-hosted upstream; return (resp, http_mock)."""
    app = create_app(_make_config(bedrock_api_url=AWS_UPSTREAM, **cfg))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 5000, 200)
        )
        resp = client.post(
            path,
            json={"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8},
            headers=headers
            or {
                "authorization": "AWS4-HMAC-SHA256 Credential=stale/20200101/us-east-1/bedrock/aws4_request",
                "x-amz-date": "20200101T000000Z",
                "x-amz-security-token": "stale-token",
            },
        )
    return resp, http


def test_aws_upstream_is_resigned(monkeypatch):
    """The caller signed a different body for Headroom's host, so AWS can only
    accept a signature we compute over what we actually send."""
    _fake_credentials(monkeypatch)
    resp, http = _post_to_aws(monkeypatch)

    assert resp.status_code == 200
    sent = _sent_headers(http)
    assert sent["authorization"].startswith("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/")
    assert "/us-east-1/bedrock/aws4_request" in sent["authorization"]
    assert sent["x-amz-date"] != "20200101T000000Z"
    assert sent["x-amz-security-token"] == "session-token"


def test_gateway_upstream_keeps_caller_signature(monkeypatch):
    """A non-AWS upstream owns signing; we must not touch its auth headers."""
    monkeypatch.setattr(
        "headroom.proxy.bedrock.signing.aws_session",
        MagicMock(side_effect=AssertionError("must not sign for a gateway")),
    )
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 5000, 200)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8},
            headers={"authorization": "Bearer gateway-key"},
        )

    assert resp.status_code == 200
    assert _sent_headers(http)["authorization"] == "Bearer gateway-key"


def test_signing_failure_forwards_unsigned(monkeypatch):
    """No credentials must not become a Headroom error: forward and let AWS
    answer, so the operator sees the real AWS message."""
    _no_api_key(monkeypatch)
    monkeypatch.setattr(
        "headroom.proxy.bedrock.signing.aws_session",
        MagicMock(side_effect=RuntimeError("no SSO cache")),
    )
    resp, http = _post_to_aws(monkeypatch, headers={"authorization": "AWS4-HMAC-SHA256 stale"})

    assert resp.status_code == 200
    assert _sent_headers(http)["authorization"] == "AWS4-HMAC-SHA256 stale"


def test_verbatim_forward_to_aws_is_still_signed(monkeypatch):
    """Even an uncompressed body needs a fresh signature: the caller signed for
    Headroom's host, not for the AWS one we forward to."""
    _fake_credentials(monkeypatch)
    resp, http = _post_to_aws(monkeypatch, optimize=False)

    assert resp.status_code == 200
    assert "Credential=AKIAIOSFODNN7EXAMPLE/" in _sent_headers(http)["authorization"]


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Authorization": "Bearer ABSKkey"}, True),
        ({"authorization": "bearer abskey"}, True),
        ({"authorization": "  Bearer padded"}, True),
        ({"authorization": "AWS4-HMAC-SHA256 Credential=x/20200101/us-east-1/bedrock/x"}, False),
        ({"content-type": "application/json"}, False),
        ({"authorization": ""}, False),
    ],
)
def test_has_bearer_auth(headers, expected):
    from headroom.proxy.bedrock import has_bearer_auth

    assert has_bearer_auth(headers) is expected


def test_caller_api_key_reaches_aws_untouched(monkeypatch):
    """A Bedrock API key commits to neither body nor host, so compression cannot
    invalidate it and re-signing would only throw away the caller's identity."""
    _no_api_key(monkeypatch)
    monkeypatch.setattr(
        "headroom.proxy.bedrock.signing.aws_session",
        MagicMock(side_effect=AssertionError("must not sign when a key was sent")),
    )
    resp, http = _post_to_aws(monkeypatch, headers={"authorization": "Bearer ABSKcaller-key"})

    assert resp.status_code == 200
    assert _sent_headers(http)["authorization"] == "Bearer ABSKcaller-key"


def test_env_api_key_replaces_the_void_signature(monkeypatch):
    """A caller with no AWS credentials still sends a signature (Claude Code signs
    with whatever it has). Headroom's own key replaces it, and every leftover
    ``x-amz-`` header has to go with it or AWS rejects the mix."""
    from headroom.proxy.bedrock import BEARER_TOKEN_ENV

    monkeypatch.setenv(BEARER_TOKEN_ENV, "ABSKheadroom-key")
    monkeypatch.setattr(
        "headroom.proxy.bedrock.signing.aws_session",
        MagicMock(side_effect=AssertionError("must not sign when a key is configured")),
    )
    app = create_app(_make_config(bedrock_api_url=AWS_UPSTREAM))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 5000, 200)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8},
            headers={
                "authorization": "AWS4-HMAC-SHA256 Credential=stale/20200101/us-east-1/bedrock/x",
                "x-amz-date": "20200101T000000Z",
                "x-amz-security-token": "stale-token",
            },
        )

    assert resp.status_code == 200
    sent = _sent_headers(http)
    assert sent["authorization"] == "Bearer ABSKheadroom-key"
    assert "x-amz-date" not in sent
    assert "x-amz-security-token" not in sent


def test_env_api_key_is_not_sent_to_a_gateway(monkeypatch):
    """Headroom's key belongs to AWS. A gateway gets the caller's headers and
    nothing of ours, or a key leaks to whatever host was configured."""
    from headroom.proxy.bedrock import BEARER_TOKEN_ENV

    monkeypatch.setenv(BEARER_TOKEN_ENV, "ABSKheadroom-key")
    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(
            return_value=_FakeResult([{"role": "user", "content": "c"}], 5000, 200)
        )
        resp = client.post(
            INVOKE,
            json={"messages": [{"role": "user", "content": "x" * 5000}], "max_tokens": 8},
            headers={"authorization": "Bearer gateway-key"},
        )

    assert resp.status_code == 200
    assert _sent_headers(http)["authorization"] == "Bearer gateway-key"


# ── Converse ──────────────────────────────────────────────────────────


def test_converse_routes_present_when_configured():
    paths = _paths(_make_config())
    assert "/model/{model_id:path}/converse" in paths
    assert "/model/{model_id:path}/converse-stream" in paths
    assert "/inference-profiles" in paths


def test_converse_text_blocks_are_tagged_for_the_pipeline():
    """A Converse content block is a typeless union, so the transforms — which
    all dispatch on block["type"] — see nothing without the discriminator."""
    import copy

    seen: list = []

    def _capture(**kwargs):
        seen.append(copy.deepcopy(kwargs["messages"]))
        return _FakeResult(
            [{"role": "user", "content": [{"type": "text", "text": "small"}]}], 900, 50
        )

    app = create_app(_make_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=_capture)
        resp = client.post(
            CONVERSE,
            json={
                "messages": [{"role": "user", "content": [{"text": "x" * 5000}]}],
                "system": [{"text": "system prompt"}],
                "inferenceConfig": {"maxTokens": 100},
            },
        )

    assert resp.status_code == 200
    assert seen[0][0]["content"][0]["type"] == "text"
    _, forwarded = _forwarded(http)
    # Back out in Converse shape: no discriminator, other top-level fields kept.
    assert forwarded["messages"] == [{"role": "user", "content": [{"text": "small"}]}]
    assert forwarded["system"] == [{"text": "system prompt"}]
    assert forwarded["inferenceConfig"] == {"maxTokens": 100}


def test_converse_untag_strips_what_converse_rejects():
    from headroom.proxy.bedrock import untag_converse_text

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                {"cachePoint": {"type": "default"}},
                {"toolUse": {"toolUseId": "t1", "name": "read", "input": {}}},
            ],
        },
        {"role": "assistant", "content": "collapsed by a transform"},
    ]
    assert untag_converse_text(messages) == [
        {
            "role": "user",
            "content": [
                {"text": "hi"},
                {"cachePoint": {"type": "default"}},
                {"toolUse": {"toolUseId": "t1", "name": "read", "input": {}}},
            ],
        },
        {"role": "assistant", "content": [{"text": "collapsed by a transform"}]},
    ]


def test_converse_bypass_forwards_original_shape():
    """A bypassed Converse request must ship its original bytes — never a body
    left half-tagged by the adapter."""
    app = create_app(_make_config(optimize=False))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        proxy.anthropic_pipeline.apply = MagicMock(side_effect=AssertionError("should not run"))
        body = {"messages": [{"role": "user", "content": [{"text": "x" * 5000}]}]}
        resp = client.post(CONVERSE, json=body)

    assert resp.status_code == 200
    _, forwarded = _forwarded(http)
    assert forwarded == body


# ── control plane ─────────────────────────────────────────────────────


def test_inference_profiles_uses_control_plane_host(monkeypatch):
    """Model invocation lives on bedrock-runtime.*, the profile listing on
    bedrock.* — sending it to the runtime host is a 404 either way."""
    _fake_credentials(monkeypatch)
    app = create_app(_make_config(bedrock_api_url=AWS_UPSTREAM))
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http = _install_fake_client(proxy, _FakeUpstream())
        resp = client.get("/inference-profiles?maxResults=100")

    assert resp.status_code == 200
    method = http.build_request.call_args.args[0]
    url = http.build_request.call_args.args[1]
    assert method == "GET"
    assert url == "https://bedrock.us-east-1.amazonaws.com/inference-profiles?maxResults=100"
    assert "Credential=AKIAIOSFODNN7EXAMPLE/" in _sent_headers(http)["authorization"]
