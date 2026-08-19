"""Bedrock + Claude Code `wrap claude` wiring.

Mirrors test_azure_foundry_claude_compression.py: when CLAUDE_CODE_USE_BEDROCK=1
is set, `wrap claude` must derive a real Bedrock upstream and redirect Claude
Code's ANTHROPIC_BEDROCK_BASE_URL through the proxy, the same way it already
does for Vertex and Foundry.

No real AWS endpoint is contacted — helpers are unit-tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.cli import wrap as wrap_cli
from headroom.providers.claude import proxy_base_url as _claude_proxy_base_url

# --------------------------------------------------------------------------
# Default upstream derivation from region
# --------------------------------------------------------------------------


def test_default_bedrock_upstream_url_uses_given_region() -> None:
    assert (
        wrap_cli._default_bedrock_upstream_url("eu-west-1")
        == "https://bedrock-runtime.eu-west-1.amazonaws.com"
    )


def test_default_bedrock_upstream_url_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("AWS_REGION", "ap-southeast-2")
    assert (
        wrap_cli._default_bedrock_upstream_url(None)
        == "https://bedrock-runtime.ap-southeast-2.amazonaws.com"
    )


def test_default_bedrock_upstream_url_defaults_to_us_west_2(monkeypatch) -> None:
    monkeypatch.delenv("HEADROOM_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    assert (
        wrap_cli._default_bedrock_upstream_url(None)
        == "https://bedrock-runtime.us-west-2.amazonaws.com"
    )


# --------------------------------------------------------------------------
# Explicit ANTHROPIC_BEDROCK_BASE_URL / BEDROCK_TARGET_API_URL handling
# --------------------------------------------------------------------------


def test_bedrock_target_url_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_BEDROCK_BASE_URL", raising=False)
    monkeypatch.delenv("BEDROCK_TARGET_API_URL", raising=False)
    proxy_url = _claude_proxy_base_url(8787)
    assert wrap_cli._bedrock_target_url_from_claude_env(proxy_url) is None


def test_bedrock_target_url_none_when_already_pointed_at_proxy(monkeypatch) -> None:
    proxy_url = _claude_proxy_base_url(8787)
    monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", proxy_url)
    monkeypatch.delenv("BEDROCK_TARGET_API_URL", raising=False)
    assert wrap_cli._bedrock_target_url_from_claude_env(proxy_url) is None


def test_bedrock_target_url_returns_real_upstream(monkeypatch) -> None:
    proxy_url = _claude_proxy_base_url(8787)
    monkeypatch.setenv(
        "ANTHROPIC_BEDROCK_BASE_URL", "https://bedrock-runtime.us-east-1.amazonaws.com"
    )
    monkeypatch.delenv("BEDROCK_TARGET_API_URL", raising=False)
    assert (
        wrap_cli._bedrock_target_url_from_claude_env(proxy_url)
        == "https://bedrock-runtime.us-east-1.amazonaws.com"
    )


def test_bedrock_target_url_prefers_explicit_target_api_url(monkeypatch) -> None:
    proxy_url = _claude_proxy_base_url(8787)
    monkeypatch.setenv("ANTHROPIC_BEDROCK_BASE_URL", proxy_url)
    monkeypatch.setenv("BEDROCK_TARGET_API_URL", "https://bedrock-runtime.us-east-1.amazonaws.com")
    assert (
        wrap_cli._bedrock_target_url_from_claude_env(proxy_url)
        == "https://bedrock-runtime.us-east-1.amazonaws.com"
    )


# --------------------------------------------------------------------------
# settings.json written with ANTHROPIC_BEDROCK_BASE_URL in Bedrock mode
# --------------------------------------------------------------------------


def _settings(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def test_write_bedrock_mode_sets_bedrock_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    proxy_url = _claude_proxy_base_url(8787)
    wrap_cli._write_claude_wrap_base_url(proxy_url, bedrock_mode=True, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BEDROCK_BASE_URL"] == "http://127.0.0.1:8787"
    assert "ANTHROPIC_BASE_URL" not in payload["env"]


def test_write_non_bedrock_mode_does_not_set_bedrock_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    proxy_url = _claude_proxy_base_url(8787)
    wrap_cli._write_claude_wrap_base_url(proxy_url, settings_path=path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert "ANTHROPIC_BEDROCK_BASE_URL" not in payload["env"]


def test_restore_bedrock_mode_removes_bedrock_key(tmp_path: Path) -> None:
    path = _settings(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"env": {"ANTHROPIC_BEDROCK_BASE_URL": "http://127.0.0.1:8787"}}),
        encoding="utf-8",
    )
    wrap_cli._restore_claude_wrap_base_url(None, bedrock_mode=True, settings_path=path)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "ANTHROPIC_BEDROCK_BASE_URL" not in payload.get("env", {})
    # else: file deleted — key is gone, which is also correct


# --------------------------------------------------------------------------
# Env-key selection is mutually exclusive with Vertex/Foundry
# --------------------------------------------------------------------------


def test_claude_wrap_base_url_env_key_bedrock_mode() -> None:
    assert wrap_cli._claude_wrap_base_url_env_key(bedrock_mode=True) == "ANTHROPIC_BEDROCK_BASE_URL"


def test_claude_wrap_base_url_env_key_vertex_beats_bedrock() -> None:
    # Mutually exclusive in practice (wrap sets at most one True), but the
    # lookup order must still be deterministic if it's ever asked for both.
    assert (
        wrap_cli._claude_wrap_base_url_env_key(vertex_mode=True, bedrock_mode=True)
        == "ANTHROPIC_VERTEX_BASE_URL"
    )
