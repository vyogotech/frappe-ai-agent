"""Tool error translation — turn Frappe/MCP errors into LLM-friendly messages.

Tool errors must NOT raise past the ToolNode in the LangGraph ReAct loop. If
they did, the whole graph run would abort and the user would see an SSE error.
Instead, we wrap each MCP tool so any exception becomes a string return value
that flows back into the LLM as a normal tool observation. The LLM then writes
a human-readable response explaining what failed.
"""

from __future__ import annotations


class PermissionDeniedError(Exception):
    """Raised when a Frappe tool call is refused because the user lacks permission.

    Carries both the tool that was invoked and the underlying doctype so the
    LLM's response can be specific ("cannot read Sales Invoice") rather than
    a generic auth failure.
    """

    def __init__(self, tool: str, doctype: str):
        self.tool = tool
        self.doctype = doctype
        super().__init__(
            f"Permission denied calling {tool} for doctype {doctype}"
        )


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
        return (
            f"Access denied: you do not have permission to read "
            f"{exc.doctype} via {exc.tool}."
        )
    if is_permission_error(exc):
        return f"Access denied: permission error — {exc}"
    return f"Tool call failed: {exc}"
