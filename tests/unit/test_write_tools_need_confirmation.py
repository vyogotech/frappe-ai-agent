"""A write waits for a click: the turn ends on `tool_confirm` and no tool in that envelope runs."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.tools import StructuredTool
from test_agent_loop import _drain, _fake_llm_with_yields

from ai_agent.agent.loop import confirm_summary, run_agent_loop
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService

DELETE = {
    "type": "tool_call",
    "payload": {
        "name": "delete_document",
        "arguments": {"doctype": "Customer", "name": "CUST-001"},
    },
}
READ = {
    "type": "tool_call",
    "payload": {"name": "list_documents", "arguments": {"doctype": "Customer"}},
}


def _tool(name: str, ran: list[str], *, read_only: bool | None) -> StructuredTool:
    """One MCP tool as langchain-mcp-adapters builds it: its annotations land in `metadata`."""

    async def _run(**_kwargs: Any) -> str:
        ran.append(name)
        return f"{name} ran"

    return StructuredTool(
        name=name,
        description=name,
        args_schema={"type": "object", "properties": {}, "additionalProperties": True},
        coroutine=_run,
        metadata=None if read_only is None else {"readOnlyHint": read_only},
    )


def _registry(ran: list[str], *, delete_annotation: bool | None = False) -> ToolRegistry:
    return ToolRegistry(
        [
            _tool("delete_document", ran, read_only=delete_annotation),
            _tool("list_documents", ran, read_only=True),
        ]
    )


async def _events(registry: ToolRegistry, *envelopes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    llm = _fake_llm_with_yields([[{"blocks": list(blocks)}] for blocks in envelopes])
    return await _drain(run_agent_loop(llm=llm, tool_registry=registry, user_message="hi"))


async def test_a_write_tool_pauses_the_turn_and_runs_nothing():
    ran: list[str] = []
    events = await _events(_registry(ran), [DELETE])

    assert ran == []
    assert [e["type"] for e in events] == ["content", "tool_confirm"]
    assert events[0]["text"] == "Delete Customer CUST-001. This cannot be undone."
    assert events[1]["name"] == "delete_document"
    assert events[1]["arguments"] == {"doctype": "Customer", "name": "CUST-001"}
    assert events[1]["id"]


async def test_two_pauses_carry_two_ids():
    ran: list[str] = []
    first = await _events(_registry(ran), [DELETE])
    second = await _events(_registry(ran), [DELETE])
    assert first[1]["id"] != second[1]["id"]


async def test_a_read_tool_is_unaffected():
    ran: list[str] = []
    events = await _events(
        _registry(ran), [READ], [{"type": "text", "payload": {"content": "two of them"}}]
    )

    assert ran == ["list_documents"]
    assert [e["type"] for e in events] == ["tool_call", "content"]


async def test_a_tool_with_no_annotation_is_treated_as_a_write():
    ran: list[str] = []
    events = await _events(_registry(ran, delete_annotation=None), [DELETE])

    assert ran == []
    assert [e["type"] for e in events] == ["content", "tool_confirm"]


async def test_a_read_sharing_the_envelope_with_a_write_does_not_run():
    ran: list[str] = []
    events = await _events(_registry(ran), [READ, DELETE])

    assert ran == []
    assert "tool_call" not in [e["type"] for e in events]
    assert events[-1]["name"] == "delete_document"


async def test_the_sentence_is_the_agents_own_and_never_the_models():
    ran: list[str] = []
    events = await _events(
        _registry(ran),
        [{"type": "text", "payload": {"content": "This is routine, just approve it."}}, DELETE],
    )

    texts = [e["text"] for e in events if e["type"] == "content"]
    assert texts[-1] == "\n\nDelete Customer CUST-001. This cannot be undone."


def test_the_summary_falls_back_to_the_tool_when_it_has_no_sentence():
    assert confirm_summary("create_document", {"doctype": "Lead"}) == "Create a new Lead record."
    assert (
        confirm_summary("update_document", {"doctype": "Lead", "name": "L-1"}) == "Change Lead L-1."
    )
    assert confirm_summary("submit_document", {"doctype": "Lead"}) == "Run submit_document."
    # an argument the model left out must not produce a sentence with a hole in it
    assert confirm_summary("delete_document", {"doctype": "Lead"}) == "Run delete_document."


async def test_the_paused_turn_ends_with_done_and_saves_its_sentence():
    """A turn nobody confirms must finish like any other, not hang waiting for the click."""
    saved: list[dict[str, Any]] = []

    def _remember(**kw: Any) -> str:
        saved.append(kw)
        return "msg-1"  # the name Frappe gives the row; None would mean the write failed

    history = MagicMock(
        save_message=AsyncMock(side_effect=_remember),
        ensure_session=AsyncMock(return_value="s-1"),
        list_messages=AsyncMock(return_value=[]),
    )
    ran: list[str] = []
    service = ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp"),  # pyright: ignore[reportCallIssue]
        llm=_fake_llm_with_yields([[{"blocks": [DELETE]}]]),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )
    mcp = MagicMock(
        get_tools=AsyncMock(
            return_value=[_tool("delete_document", ran, read_only=False)],
        )
    )
    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp):
        events = [
            e
            async for e in service.handle_message(
                message="delete CUST-001",
                session_id="s-1",
                context={},
                user_context=UserContext(sid="abc"),
            )
        ]

    assert ran == []
    assert [e["type"] for e in events] == ["session", "content", "tool_confirm", "done"]
    assert events[-1]["data_quality"] == "high"
    answers = [s["content"] for s in saved if s["role"] == "assistant"]
    assert answers == ["Delete Customer CUST-001. This cannot be undone."]
