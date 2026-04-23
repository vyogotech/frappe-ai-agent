"""Tool error translation — turn Frappe/MCP errors into LLM-friendly messages.

Tool errors must NOT raise past the ToolNode in the LangGraph ReAct loop. If
they did, the whole graph run would abort and the user would see an SSE error.
Instead, we wrap each MCP tool so any exception becomes a string return value
that flows back into the LLM as a normal tool observation. The LLM then writes
a human-readable response explaining what failed.

Two-step wiring (see `install_tool_error_handler`):

1. Wrap the tool's coroutine so any non-`ToolException` is re-raised as a
   `ToolException`. LangChain's `BaseTool.handle_tool_error` hook only fires
   for `ToolException`; without this re-raise, raw `McpError` / `httpx` /
   Frappe validation errors escape straight through the ToolNode and abort
   the graph run.
2. Set `handle_tool_error = to_tool_result_message` so LangChain folds the
   now-`ToolException` into a string the LLM sees as the tool observation.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, ToolException


class PermissionDeniedError(Exception):
    """Raised when a Frappe tool call is refused because the user lacks permission.

    Carries both the tool that was invoked and the underlying doctype so the
    LLM's response can be specific ("cannot read Sales Invoice") rather than
    a generic auth failure.
    """

    def __init__(self, tool: str, doctype: str):
        self.tool = tool
        self.doctype = doctype
        super().__init__(f"Permission denied calling {tool} for doctype {doctype}")


def is_permission_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like a Frappe permission denial?

    Checks (in order):
      1. The exception is a PermissionDeniedError
      2. It has a status_code attribute equal to 401 or 403
      3. Its string representation contains common permission wording
    """
    if isinstance(exc, PermissionDeniedError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return True
    text = str(exc).lower()
    permission_phrases = (
        "permission denied",
        "not permitted",
        "forbidden",
        "does not have permission",
        "not authorized",
        "unauthorized",
    )
    return any(phrase in text for phrase in permission_phrases)


def to_tool_result_message(exc: Exception) -> str:
    """Convert any exception into a string suitable for a ToolMessage content.

    The result is fed back to the LLM as the tool's output. It must be
    actionable and free of stack traces or internal details.

    NOTE: signature must be annotated with `Exception` (not `BaseException`)
    because LangGraph's ToolNode uses `_infer_handled_types` on this callable
    to decide which exception classes to catch. BaseException is rejected as
    "too broad" at registration time. See `langgraph/prebuilt/tool_node.py`.
    """
    if isinstance(exc, PermissionDeniedError):
        return f"Access denied: you do not have permission to read {exc.doctype} via {exc.tool}."
    if is_permission_error(exc):
        return f"Access denied: permission error — {exc}"
    return f"Tool call failed: {exc}"


def install_tool_error_handler(tool: BaseTool) -> None:
    """Route any exception from `tool` through `to_tool_result_message`.

    LangChain's built-in `handle_tool_error` path only covers `ToolException`,
    so MCP / Frappe / httpx errors (which are not ToolException subclasses)
    would escape the ToolNode and abort the graph. We wrap the tool's
    async entry point so non-`ToolException` errors are re-raised as
    `ToolException`, then let the built-in hook turn them into a string
    observation the LLM can reason about.
    """
    # `coroutine` is declared on StructuredTool (what the MCP adapter uses)
    # but not on BaseTool. Read via getattr for nullability; write via direct
    # assignment (ignoring the BaseTool-scope type warning) because the
    # runtime object is StructuredTool in every case we care about.
    original = getattr(tool, "coroutine", None)
    if original is None:
        # MCP adapter always sets `coroutine`; defensive for other tool kinds.
        return

    async def _wrapped(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except ToolException:
            raise
        except Exception as exc:
            # `from exc` preserves the original traceback in the log stream
            # without putting it in the LLM-facing message.
            raise ToolException(str(exc)) from exc

    tool.coroutine = _wrapped  # type: ignore[attr-defined]
    tool.handle_tool_error = to_tool_result_message
