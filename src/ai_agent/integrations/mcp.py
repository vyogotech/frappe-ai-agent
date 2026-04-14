"""MCP server connection via langchain-mcp-adapters (Streamable HTTP)."""

from __future__ import annotations

from datetime import timedelta

from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_agent.config import Settings


def create_mcp_client(settings: Settings) -> MultiServerMCPClient:
    """Create MCP client configured for Streamable HTTP transport."""
    return MultiServerMCPClient(
        {
            "erpnext": {
                "transport": "http",
                "url": settings.mcp_server_url,
                "timeout": timedelta(seconds=30),
            },
        }
    )


def build_mcp_client_for_sid(settings: Settings, sid: str) -> MultiServerMCPClient:
    """Return a new MCP client configured to forward the caller's Frappe sid.

    Every call to this function returns a NEW client. Sharing clients across
    requests would leak one user's sid into another user's tool calls.

    Raises:
        ValueError: if sid is empty or whitespace-only.
    """
    if not sid or not sid.strip():
        raise ValueError("build_mcp_client_for_sid requires a non-empty sid")
    return MultiServerMCPClient(
        {
            "frappe": {
                "url": settings.mcp_server_url,
                "transport": "streamable_http",
                "headers": {"Cookie": f"sid={sid}"},
            }
        }
    )
