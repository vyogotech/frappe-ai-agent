"""A narrowed handler still turns its known failures into its fallback, and lets a bug out."""

import pytest
import respx
import structlog
from httpx import Response
from langchain_core.tools import StructuredTool, ToolException

from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.integrations.frappe_history import FrappeHistoryClient

FRAPPE = "http://frappe.test"


class _Schema:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def model_json_schema(self) -> dict:
        raise self._exc


async def _refused(doctype: str) -> str:
    raise ToolException("Permission denied (HTTP 403)")


def _tool(schema_error: Exception | None = None) -> StructuredTool:
    t = StructuredTool.from_function(coroutine=_refused, name="get_doc", description="Get one.")
    if schema_error is not None:
        t.args_schema = _Schema(schema_error)  # pyright: ignore[reportAttributeAccessIssue]
    return t


def test_a_schema_bug_fails_the_turn_instead_of_hiding_the_args():
    with pytest.raises(RuntimeError, match="bug"):
        ToolRegistry([_tool(RuntimeError("bug"))]).schemas()


def test_a_schema_pydantic_cannot_render_still_lists_no_args():
    assert "args: {}" in ToolRegistry([_tool(TypeError("unrenderable"))]).schemas()


async def test_a_failed_tool_call_is_logged_with_its_type():
    with structlog.testing.capture_logs() as logs:
        await ToolRegistry([_tool()]).ainvoke("get_doc", {"doctype": "Customer"})
    failed = [r for r in logs if r["event"] == "tool_call_failed"]
    assert [(r["tool"], r["error_type"]) for r in failed] == [("get_doc", "ToolException")]


async def test_reading_history_from_a_closed_client_raises_as_a_write_does():
    client = FrappeHistoryClient(base_url=FRAPPE)
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.list_messages(sid="s", session="c")


@respx.mock
async def test_a_bug_in_the_token_fetch_fails_the_session_write_without_raising():
    respx.get(f"{FRAPPE}/app").mock(side_effect=RuntimeError("bug"))
    post = respx.post(f"{FRAPPE}/api/resource/AI Chat Session").mock(
        return_value=Response(200, json={"data": {"name": "chat-1"}})
    )
    with structlog.testing.capture_logs() as logs:
        name = await FrappeHistoryClient(base_url=FRAPPE).create_session(
            sid="s", title="t", context_json="{}"
        )
    assert name is None
    assert not post.called
    failed = [r for r in logs if r["event"] == "frappe_history_write_failed"]
    assert [r["error_type"] for r in failed] == ["RuntimeError"]
