"""A failed tool call is a result to the model, and an outcome in the trace and audit row."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool
from opentelemetry.trace import StatusCode
from test_agent_loop import _fake_llm_with_yields

from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService

CALL = {
    "type": "tool_call",
    "payload": {"name": "list_documents", "arguments": {"doctype": "Customer"}},
}
ANSWER = {"type": "text", "payload": {"content": "Here is what I found."}}


def _tool(name: str, *, boom: bool) -> StructuredTool:
    async def _run(**_kwargs: Any) -> str:
        if boom:
            raise RuntimeError("Frappe returned HTTP 500")
        return "one row"

    return StructuredTool(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}, "additionalProperties": True},
        coroutine=_run,
        metadata={"readOnlyHint": True},
    )


async def _turn(*, boom: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one turn whose single tool call succeeds or fails; return its events and saved rows."""
    saved: list[dict[str, Any]] = []
    service = ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp"),  # pyright: ignore[reportCallIssue]
        llm=_fake_llm_with_yields([[{"blocks": [CALL]}], [{"blocks": [ANSWER]}]]),
        system_prompt_builder=lambda _ctx: "",
        history=MagicMock(
            save_message=AsyncMock(
                side_effect=lambda **kw: (saved.append(kw), f"m-{len(saved)}")[1]
            ),
            ensure_session=AsyncMock(return_value="s-1"),
            list_messages=AsyncMock(return_value=[]),
        ),
    )
    mcp = MagicMock(get_tools=AsyncMock(return_value=[_tool("list_documents", boom=boom)]))
    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp):
        events = [
            e
            async for e in service.handle_message(
                message="list the customers",
                session_id="s-1",
                context={},
                user_context=UserContext(sid="abc"),
            )
        ]
    return events, saved


def _trail(saved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assistant = next(row for row in saved if row["role"] == "assistant")
    return json.loads(assistant["tool_args_json"])


async def test_a_turn_whose_tools_all_failed_is_not_high_quality():
    events, saved = await _turn(boom=True)

    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"
    trail = _trail(saved)
    assert len(trail) == 1
    assert trail[0].pop("duration_ms") >= 0
    assert trail[0] == {
        "name": "list_documents",
        "args": {"doctype": "Customer"},
        "ok": False,
        "error_type": "RuntimeError",
    }


async def test_a_turn_whose_tools_worked_is_unchanged():
    events, saved = await _turn(boom=False)

    assert events[-1]["data_quality"] == "high"
    row = _trail(saved)[0]
    assert row["ok"] is True
    assert "error_type" not in row


async def test_each_call_gets_its_own_span(otel_spans):
    await _turn(boom=True)

    spans = [s for s in otel_spans.get_finished_spans() if s.name.startswith("execute_tool")]
    assert [s.name for s in spans] == ["execute_tool list_documents"]
    attributes = dict(spans[0].attributes or {})
    assert attributes["gen_ai.operation.name"] == "execute_tool"
    assert attributes["gen_ai.tool.name"] == "list_documents"
    assert attributes["error.type"] == "RuntimeError"
    assert spans[0].status.status_code is StatusCode.ERROR


async def test_the_chat_the_agent_filled_in_stays_out_of_the_audit_trail():
    registry = ToolRegistry([_tool("search_knowledge_base", boom=False)])
    await registry.ainvoke("search_knowledge_base", {"query": "invoices", "session": "s-1"})

    assert registry.invocations[0]["args"] == {"query": "invoices"}


async def test_a_tool_name_no_server_offers_is_a_failed_call_too():
    """A small model can write a name the schema's enum does not hold; nothing ran, so the
    turn's answer rests on no more than a failed call would have left it."""
    registry = ToolRegistry([_tool("list_documents", boom=False)])
    result = await registry.ainvoke("summon_documents", {"doctype": "Customer"})

    assert result.startswith("error: unknown tool")
    assert registry.invocations[0]["ok"] is False
    assert registry.invocations[0]["error_type"] == "UnknownTool"
