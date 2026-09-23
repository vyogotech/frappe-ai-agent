"""MCP server connection via langchain-mcp-adapters (Streamable HTTP)."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from langchain_mcp_adapters.client import MultiServerMCPClient

from ai_agent.config import Settings

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

# Kept by frappe-mcp-server for older clients, hidden from the LLM here: they fail on
# sites without the project doctypes; aggregate_documents and run_report replace them.
DEPRECATED_TOOLS = frozenset(
    {
        "get_project_status",
        "analyze_project_timeline",
        "get_resource_allocation",
        "generate_project_report",
        "resource_utilization_analysis",
        "budget_variance_analysis",
    }
)


def build_mcp_client_for_sid(
    settings: Settings, sid: str, confirmation_token: str | None = None
) -> MultiServerMCPClient:
    """A new client per call, forwarding `sid`: shared, it would carry one user's sid to another.

    `confirmation_token` is the one-time grant the MCP server redeems before a write. It travels
    as a header, so it is in no tool argument, no saved row and nothing the model ever reads.

    Raises:
        ValueError: `sid` is empty or whitespace-only.
    """
    if not sid or not sid.strip():
        raise ValueError("build_mcp_client_for_sid requires a non-empty sid")
    headers = {"Cookie": f"sid={sid}"}
    if confirmation_token:
        headers["X-Frappe-Confirmation"] = confirmation_token
    return MultiServerMCPClient(
        {
            "frappe": {
                "url": settings.mcp_server_url,
                "transport": "streamable_http",
                "headers": headers,
                "timeout": timedelta(seconds=settings.mcp_tool_timeout_s),
                # what a stalled tool call waits on; the adapter's own default is 5 minutes
                "sse_read_timeout": timedelta(seconds=settings.mcp_tool_timeout_s),
            }
        }
    )


def filter_deprecated(tools: list[BaseTool]) -> list[BaseTool]:
    return [t for t in tools if t.name not in DEPRECATED_TOOLS]
