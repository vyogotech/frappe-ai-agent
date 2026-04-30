from typing import cast

import pytest
from langchain_mcp_adapters.sessions import StreamableHttpConnection

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
    return Settings(_env_file=None, **base)  # pyright: ignore[reportCallIssue]


def _http(client_conn) -> StreamableHttpConnection:
    """Narrow the Connection union to StreamableHttpConnection.

    The factory always builds StreamableHttpConnection, but client.connections
    is typed as the wider Connection union — a cast lets the tests inspect
    `headers` / `url` / `transport` without per-line pyright noise.
    """
    return cast(StreamableHttpConnection, client_conn)


def test_client_is_built_with_sid_cookie_header():
    settings = _make_settings()
    client = build_mcp_client_for_sid(settings, sid="abc123")

    # Exactly one server registered — catches regressions where both old and
    # new factory names end up registered together.
    assert list(client.connections.keys()) == ["frappe"]
    server_cfg = _http(client.connections["frappe"])
    assert server_cfg.get("url") == "http://mcp:8080/mcp"
    assert server_cfg.get("transport") == "streamable_http"
    headers = server_cfg.get("headers") or {}
    assert headers.get("Cookie") == "sid=abc123"


def test_two_sids_produce_two_distinct_clients():
    settings = _make_settings()
    a = build_mcp_client_for_sid(settings, sid="aaa")
    b = build_mcp_client_for_sid(settings, sid="bbb")
    assert a is not b
    a_headers = _http(a.connections["frappe"]).get("headers") or {}
    b_headers = _http(b.connections["frappe"]).get("headers") or {}
    assert a_headers["Cookie"] == "sid=aaa"
    assert b_headers["Cookie"] == "sid=bbb"


def test_empty_sid_raises():
    settings = _make_settings()
    with pytest.raises(ValueError):
        build_mcp_client_for_sid(settings, sid="")


def test_whitespace_sid_raises():
    settings = _make_settings()
    with pytest.raises(ValueError):
        build_mcp_client_for_sid(settings, sid="   ")
