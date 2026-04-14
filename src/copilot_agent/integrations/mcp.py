"""MCP server connection via langchain-mcp-adapters (Streamable HTTP)."""

from __future__ import annotations

from datetime import timedelta

from langchain_mcp_adapters.client import MultiServerMCPClient

from copilot_agent.config import Settings


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
