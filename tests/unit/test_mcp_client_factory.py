import pytest

from ai_agent.config import Settings
from ai_agent.integrations.mcp import build_mcp_client_for_sid


def _make_settings(**overrides) -> Settings:
    base = dict(
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
    )
    base.update(overrides)
    return Settings(**base)


def test_client_is_built_with_sid_cookie_header():
    settings = _make_settings()
    client = build_mcp_client_for_sid(settings, sid="abc123")

    # Exactly one server registered — catches regressions where both old and
    # new factory names end up registered together.
    assert list(client.connections.keys()) == ["frappe"]
    server_cfg = client.connections["frappe"]
    assert server_cfg.get("url") == "http://mcp:8080/mcp"
    assert server_cfg.get("transport") == "streamable_http"
    assert server_cfg.get("headers", {}).get("Cookie") == "sid=abc123"


def test_two_sids_produce_two_distinct_clients():
    settings = _make_settings()
    a = build_mcp_client_for_sid(settings, sid="aaa")
    b = build_mcp_client_for_sid(settings, sid="bbb")
    assert a is not b
    assert a.connections["frappe"]["headers"]["Cookie"] == "sid=aaa"
    assert b.connections["frappe"]["headers"]["Cookie"] == "sid=bbb"


def test_empty_sid_raises():
    settings = _make_settings()
    with pytest.raises(ValueError):
        build_mcp_client_for_sid(settings, sid="")


def test_whitespace_sid_raises():
    settings = _make_settings()
    with pytest.raises(ValueError):
        build_mcp_client_for_sid(settings, sid="   ")
