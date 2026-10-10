"""Optional MLflow tracing for Headroom (``pip install headroom-ai[mlflow]``).

Off unless ``HEADROOM_MLFLOW_ENABLED`` is truthy. Traces go to
``MLFLOW_TRACKING_URI`` / ``MLFLOW_EXPERIMENT_NAME`` (defaults below). Full
request and response bodies are recorded, same exposure as ``--log-messages``.
"""

from __future__ import annotations

import atexit
import functools
import importlib.util
import json
import logging
import os
from typing import Any

from .metrics import _parse_bool

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://localhost:5050"
DEFAULT_EXPERIMENT_NAME = "my-experiment"

# SDKs Headroom itself calls (HeadroomClient wraps openai/anthropic clients,
# the litellm backend and memory embedders). The proxy's default forwarder is
# raw httpx, which instrument_proxy() covers with manual spans.
_AUTOLOG_FLAVORS = ("openai", "anthropic", "litellm")


def configure_mlflow_tracing() -> bool:
    """Point MLflow at the tracking server and enable SDK autologging.

    Returns True when tracing is active. Call before any LLM client is created.
    """
    if not _parse_bool(os.environ.get("HEADROOM_MLFLOW_ENABLED"), default=False):
        return False
    # Unified OTel provider: Headroom's existing OpenTelemetry compression
    # spans (headroom.compression.*) land in the same MLflow trace.
    os.environ.setdefault("MLFLOW_USE_DEFAULT_TRACER_PROVIDER", "false")
    try:
        import mlflow
    except ImportError:
        logger.warning(
            "HEADROOM_MLFLOW_ENABLED is set but mlflow is not installed. "
            "Install headroom-ai[mlflow] to enable MLflow tracing."
        )
        return False

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME)
    for flavor in _AUTOLOG_FLAVORS:
        if importlib.util.find_spec(flavor) is not None:
            getattr(mlflow, flavor).autolog()
    atexit.register(mlflow.flush_trace_async_logging)
    logger.info("MLflow tracing enabled (tracking_uri=%s)", mlflow.get_tracking_uri())
    return True


def _body_of(response: Any) -> Any:
    # ponytail: streaming responses record no output; wrap body_iterator if needed.
    if hasattr(response, "body_iterator"):
        return {"streaming": True}
    body = getattr(response, "body", b"")
    try:
        return json.loads(body)
    except ValueError:
        return body.decode(errors="replace")


def _trace_request_handler(handler: Any, name: str) -> Any:
    import mlflow
    from mlflow.entities import SpanType

    @functools.wraps(handler)
    async def wrapper(request: Any, *args: Any, **kwargs: Any) -> Any:
        with mlflow.start_span(name=name, span_type=SpanType.CHAIN) as span:
            try:
                span.set_inputs(await request.json())  # Starlette caches the body
            except Exception:
                span.set_inputs({"path": request.url.path})
            response = await handler(request, *args, **kwargs)
            span.set_attribute("http.status_code", getattr(response, "status_code", None))
            span.set_outputs(_body_of(response))
            return response

    return wrapper


def _trace_upstream(retry_request: Any) -> Any:
    import mlflow
    from mlflow.entities import SpanType

    @functools.wraps(retry_request)
    async def wrapper(
        method: str, url: str, headers: dict, body: dict, stream: bool = False, **kw: Any
    ) -> Any:
        # Headers carry provider credentials: never record them.
        with mlflow.start_span(name="upstream", span_type=SpanType.LLM) as span:
            span.set_inputs(body)
            span.set_attribute("url", url)
            response = await retry_request(method, url, headers, body, stream, **kw)
            span.set_attribute("http.status_code", response.status_code)
            if not stream:
                try:
                    span.set_outputs(response.json())
                except Exception:
                    span.set_outputs(response.text)
            return response

    return wrapper


def instrument_proxy(proxy: Any) -> None:
    """Wrap the proxy's LLM endpoints and upstream call in MLflow spans.

    Routes call ``proxy.handle_*`` at request time, so instance attributes win.
    """
    for attr, name in (
        ("handle_openai_chat", "openai.chat.completions"),
        ("handle_openai_responses", "openai.responses"),
        ("handle_anthropic_messages", "anthropic.messages"),
    ):
        setattr(proxy, attr, _trace_request_handler(getattr(proxy, attr), name))
    proxy._retry_request = _trace_upstream(proxy._retry_request)
