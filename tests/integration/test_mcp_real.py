"""Real MCP handshake against a live frappe-mcp-server.

`build_mcp_client_for_sid` opens a Streamable HTTP connection and forwards
the caller's Frappe sid as a Cookie header. If MCP + Frappe are both up and
the sid is valid, `get_tools()` returns a non-empty list of tools.
"""

from __future__ import annotations

import os

import pytest

from ai_agent.config import Settings
from ai_agent.integrations.mcp import build_mcp_client_for_sid

pytestmark = pytest.mark.integration

_MCP_URL = os.getenv("AI_AGENT_INTEGRATION_MCP_URL")


@pytest.mark.skipif(
    not _MCP_URL,
    reason="set AI_AGENT_INTEGRATION_MCP_URL (and Frappe fixtures) to run",
)
async def test_mcp_lists_tools(frappe_sid: str):
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        mcp_server_url=_MCP_URL,
    )
    client = build_mcp_client_for_sid(settings, frappe_sid)
    tools = await client.get_tools()
    assert tools, "frappe-mcp-server returned zero tools — check Frappe auth"
