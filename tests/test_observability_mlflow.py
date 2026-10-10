"""Tests for the optional MLflow tracing hooks."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")

from headroom.observability import configure_mlflow_tracing, instrument_proxy  # noqa: E402


def test_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_MLFLOW_ENABLED", raising=False)
    assert configure_mlflow_tracing() is False


def test_instrument_proxy_passes_calls_through(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # sqlite store writes ./mlruns
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    calls = []

    class Request:
        url = SimpleNamespace(path="/v1/chat/completions")

        async def json(self):
            return {"messages": [{"role": "user", "content": "hi"}]}

    class Proxy:
        async def handle_openai_chat(self, request):
            upstream = await self._retry_request("POST", "u", {"authorization": "x"}, {"a": 1})
            return SimpleNamespace(status_code=200, body=upstream.content)

        handle_openai_responses = handle_anthropic_messages = handle_openai_chat

        async def _retry_request(self, method, url, headers, body, stream=False, **kw):
            calls.append((method, url, body, stream, kw))
            return SimpleNamespace(status_code=200, content=b'{"ok": true}', json=lambda: {"ok": 1})

    proxy = Proxy()
    instrument_proxy(proxy)
    response = asyncio.run(proxy.handle_anthropic_messages(Request()))
    mlflow.flush_trace_async_logging()
    spans = mlflow.get_trace(mlflow.get_last_active_trace_id()).data.spans
    asyncio.run(proxy._retry_request("POST", "u", {}, {}, True, request_id="r"))

    assert response.body == b'{"ok": true}'
    assert [s.name for s in spans] == ["anthropic.messages", "upstream"]
    assert spans[0].outputs == {"ok": True}
    assert "authorization" not in str(spans[1].to_dict())
    assert calls == [
        ("POST", "u", {"a": 1}, False, {}),
        ("POST", "u", {}, True, {"request_id": "r"}),
    ]
