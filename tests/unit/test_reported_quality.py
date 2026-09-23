"""`done.data_quality` follows the turn's real outcome, not just whether an exception escaped."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService


async def _refused(doctype: str) -> str:
    raise RuntimeError("ERPNext said no")


def _history() -> MagicMock:
    history = MagicMock(spec=FrappeHistoryClient)
    history.create_session.return_value = "chat-1"
    history.ensure_session.side_effect = lambda *, name, **_: name
    history.save_message.return_value = "msg-1"
    history.list_messages.return_value = []
    return history


def _service(history: MagicMock) -> ChatService:
    return ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp.test/mcp"),  # pyright: ignore[reportCallIssue]
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )


def _answers(**_kwargs):
    async def _gen():
        yield {"type": "content", "text": "There are 42 open sales orders."}

    return _gen()


def _calls_the_tool(**kwargs):
    """Stand-in loop that puts one tool call through the real registry, as the loop does."""

    async def _gen():
        yield {"type": "tool_call", "name": "get_doc", "arguments": {"doctype": "Customer"}}
        await kwargs["tool_registry"].ainvoke("get_doc", {"doctype": "Customer"})
        yield {"type": "content", "text": "There are 42 open sales orders."}

    return _gen()


async def _turn(service: ChatService, tools: list[Any], loop) -> list[dict[str, Any]]:
    mcp = MagicMock(get_tools=AsyncMock(return_value=tools))
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp),
        patch("ai_agent.services.chat.run_agent_loop", loop),
    ):
        return [
            e
            async for e in service.handle_message(
                message="how many open sales orders?",
                session_id="chat-1",
                context={},
                user_context=UserContext(sid="abc123"),
            )
        ]


async def test_a_turn_that_worked_still_reports_high():
    events = await _turn(_service(_history()), [], _answers)

    assert events[-1]["data_quality"] == "high"


async def test_a_failed_tool_call_is_not_a_high_quality_answer():
    tool = StructuredTool.from_function(coroutine=_refused, name="get_doc", description="Get one.")

    events = await _turn(_service(_history()), [tool], _calls_the_tool)

    assert [e["type"] for e in events if e["type"] == "error"] == []  # the model still answered
    assert events[-1]["data_quality"] == "low"


async def test_an_answer_nothing_saved_is_not_a_high_quality_answer():
    history = _history()
    history.save_message.return_value = None  # what the client returns on any write failure

    events = await _turn(_service(history), [], _answers)

    assert events[-1]["data_quality"] == "low"


async def test_a_session_that_could_not_be_created_is_not_a_high_quality_answer():
    history = _history()
    history.create_session.return_value = None
    service = _service(history)
    mcp = MagicMock(get_tools=AsyncMock(return_value=[]))
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp),
        patch("ai_agent.services.chat.run_agent_loop", _answers),
    ):
        events = [
            e
            async for e in service.handle_message(
                message="hi", session_id=None, context={}, user_context=UserContext(sid="abc123")
            )
        ]

    assert events[0]["id"].startswith("tmp-")
    assert events[-1]["data_quality"] == "low"


async def test_a_bug_in_the_history_read_fails_the_turn_instead_of_reading_as_an_outage():
    """Only the client decides what a Frappe outage is; a RuntimeError from it is a bug."""
    history = _history()
    history.list_messages.side_effect = RuntimeError("FrappeHistoryClient is closed")

    events = await _turn(_service(history), [], _answers)

    assert [e["type"] for e in events if e["type"] == "error"] != []
    assert events[-1]["data_quality"] == "low"


@pytest.mark.parametrize("outcome", ["timeout", "unknown tool"])
async def test_the_registry_remembers_a_call_it_could_not_answer(outcome: str):
    from ai_agent.agent.tool_registry import ToolRegistry

    async def _slow(doctype: str) -> str:
        import asyncio

        await asyncio.sleep(5)
        return ""

    tool = StructuredTool.from_function(coroutine=_slow, name="get_doc", description="Get one.")
    registry = ToolRegistry([tool], timeout_s=0.01)
    assert registry.invocations == []

    if outcome == "timeout":
        await registry.ainvoke("get_doc", {"doctype": "Customer"})
    else:
        await registry.ainvoke("no_such_tool", {})

    assert [c["ok"] for c in registry.invocations] == [False]


async def test_a_pool_closed_under_the_turn_still_ends_the_stream():
    """aclose() during shutdown; the answer still goes out and `done` says the turn fell short."""
    history = _history()
    history.save_message.side_effect = RuntimeError("FrappeHistoryClient is closed")

    events = await _turn(_service(history), [], _answers)

    assert [e["type"] for e in events][-1] == "done"
    assert events[-1]["data_quality"] == "low"
