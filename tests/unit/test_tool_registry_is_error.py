"""An MCP isError result reaches the model as a failed call, whatever the adapter does with it."""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool, ToolException

from ai_agent.agent.tool_registry import ToolRegistry


async def _refused(doctype: str) -> str:
    raise ToolException("Permission denied (HTTP 403). PermissionError")


def _mcp_tool() -> StructuredTool:
    # langchain-mcp-adapters 0.3 raises ToolException for an isError result, then
    # returns its text as the tool's output through handle_tool_error
    return StructuredTool.from_function(
        coroutine=_refused,
        name="get_document",
        description="Get a document.",
        handle_tool_error=lambda e: str(e),
    )


@pytest.mark.asyncio
async def test_an_error_result_is_a_failed_call():
    result = await ToolRegistry([_mcp_tool()]).ainvoke("get_document", {"doctype": "Sales Invoice"})
    assert result == (
        "Access denied: permission error — Permission denied (HTTP 403). PermissionError"
    )
