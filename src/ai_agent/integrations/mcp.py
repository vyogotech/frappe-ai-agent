"""MCP server connection via langchain-mcp-adapters (Streamable HTTP)."""

from __future__ import annotations

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


def filter_deprecated(tools: list[BaseTool]) -> list[BaseTool]:
    """Drop tools whose name is in `DEPRECATED_TOOLS`.

    Returned list preserves input order; the original list is not mutated.
    """
    return [t for t in tools if t.name not in DEPRECATED_TOOLS]
