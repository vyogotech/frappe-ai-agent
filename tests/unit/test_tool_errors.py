# tests/unit/test_tool_errors.py
import pytest
from langchain_core.tools import StructuredTool, ToolException
from pydantic import BaseModel

from ai_agent.agent.tool_errors import (
    PermissionDeniedError,
    install_tool_error_handler,
    is_permission_error,
    to_tool_result_message,
)


def test_permission_denied_error_carries_tool_and_doctype():
    err = PermissionDeniedError(tool="list_invoices", doctype="Sales Invoice")
    assert err.tool == "list_invoices"
    assert err.doctype == "Sales Invoice"
    assert "Sales Invoice" in str(err)
    assert "list_invoices" in str(err)


def test_to_tool_result_message_handles_permission_denied():
    err = PermissionDeniedError(tool="list_invoices", doctype="Sales Invoice")
    msg = to_tool_result_message(err)
    assert "Access denied" in msg
    assert "Sales Invoice" in msg
    # The message must be actionable for the LLM — it should hint that the
    # user lacks permission, not expose a stack trace.
    assert "permission" in msg.lower()


def test_to_tool_result_message_handles_generic_exception():
    err = RuntimeError("upstream timeout")
    msg = to_tool_result_message(err)
    assert "upstream timeout" in msg
    # Generic errors still surface the message, prefixed consistently.
    assert "failed" in msg.lower() or "error" in msg.lower()


@pytest.mark.parametrize(
    "status_code,text,expected",
    [
        (403, "Forbidden", True),
        (401, "Not Permitted", True),
        (200, "OK", False),
        (500, "Internal Server Error", False),
        (None, "User does not have permission to read Sales Invoice", True),
        (None, "not permitted", True),
        (None, "some other error", False),
    ],
)
def test_is_permission_error_classifies_correctly(status_code, text, expected):
    class _FakeErr(Exception):
        def __init__(self, status_code, text):
            self.status_code = status_code
            super().__init__(text)

    err = _FakeErr(status_code, text)
    assert is_permission_error(err) is expected


# --- install_tool_error_handler regression tests ---------------------------- #
#
# Before `install_tool_error_handler` existed, the service set
# `tool.handle_tool_error = to_tool_result_message` directly. That hook only
# fires for `ToolException`, so any other exception from an MCP-adapter tool
# (which is the common case — MCP raises `McpError`, Frappe raises validation
# errors, httpx raises ConnectError) escaped the ToolNode and aborted the graph.
# These tests lock the two-part fix: wrap the coroutine to re-raise as
# ToolException AND set the handler.


class _EmptyArgs(BaseModel):
    pass


def _build_mcp_like_tool(coroutine):
    """Mimic how langchain_mcp_adapters builds a StructuredTool — coroutine-only."""
    return StructuredTool(
        name="fake_mcp_tool",
        description="",
        args_schema=_EmptyArgs,
        coroutine=coroutine,
    )


async def test_install_tool_error_handler_catches_generic_exception():
    """The whole point of install_tool_error_handler: a raw Exception from
    the tool becomes an LLM-visible string observation instead of escaping."""

    async def _boom(**_):
        raise RuntimeError("frappe: Unknown column 'address' on Customer")

    tool = _build_mcp_like_tool(_boom)
    install_tool_error_handler(tool)

    # arun invokes langchain's error-handling machinery; after our fix, the
    # runtime exception has been converted to ToolException and the hook
    # has returned the LLM-facing string.
    result = await tool.arun({})
    assert isinstance(result, str)
    assert "Tool call failed" in result
    assert "Unknown column" in result


async def test_install_tool_error_handler_preserves_existing_tool_exception():
    """If a tool raises ToolException directly, our wrapper must not mask it —
    langchain's native path should still convert it via the handler."""

    async def _raise_tool_exc(**_):
        raise ToolException("explicit tool exception")

    tool = _build_mcp_like_tool(_raise_tool_exc)
    install_tool_error_handler(tool)

    result = await tool.arun({})
    assert "explicit tool exception" in result


async def test_install_tool_error_handler_routes_permission_errors():
    """PermissionDeniedError → dedicated 'Access denied' wording, not the
    generic 'Tool call failed' prefix."""

    async def _forbidden(**_):
        raise PermissionDeniedError(tool="list_invoices", doctype="Sales Invoice")

    tool = _build_mcp_like_tool(_forbidden)
    install_tool_error_handler(tool)

    result = await tool.arun({})
    assert "Access denied" in result
    assert "Sales Invoice" in result


async def test_install_tool_error_handler_skips_when_no_coroutine():
    """Sync-only tools (no `coroutine` attr) are left alone — we're only
    aware of async MCP tools in practice, but defensive."""
    tool = StructuredTool(
        name="sync_tool",
        description="",
        args_schema=_EmptyArgs,
        func=lambda: "ok",
    )
    tool.coroutine = None

    install_tool_error_handler(tool)
    # No crash; coroutine attribute left alone since there's nothing to wrap.
    assert tool.coroutine is None
