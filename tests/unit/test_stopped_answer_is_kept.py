"""D03V: a turn stopped or cut off part-way keeps the text that arrived, marked as incomplete."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio

from ai_agent.agent.loop import REPLY_CUT_OFF, TurnFailure
from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ANSWER_STOPPED, ChatService

ANSWERED = "The three open invoices are "


def _history(saved: list[dict[str, Any]]) -> MagicMock:
    async def _save(**kwargs: Any) -> str:
        # a real write suspends, which is where the cancellation being unwound lands
        await asyncio.sleep(0.01)
        saved.append(kwargs)
        return "m-1"

    return MagicMock(
        save_message=AsyncMock(side_effect=_save),
        ensure_session=AsyncMock(side_effect=lambda *, name, **_: name),
        list_messages=AsyncMock(return_value=[]),
    )


def _turn(history: MagicMock):
    settings = Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp")  # pyright: ignore[reportCallIssue]
    service = ChatService(
        settings=settings,
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )
    return service.handle_message(
        message="how many invoices are open?",
        session_id="s-1",
        context={},
        user_context=UserContext(sid="abc"),
    )


def _loop(*, says: str = ANSWERED, then: BaseException | None = None):
    """An agent loop that calls a tool, says `says`, then raises `then` or stalls until cut."""

    def _factory(**_kwargs: Any):
        async def _gen():
            yield {"type": "tool_call", "name": "list_documents", "arguments": {}}
            if says:
                yield {"type": "content", "text": says}
            if then is not None:
                raise then
            await asyncio.sleep(60)

        return _gen()

    return _factory


def _answers(saved: list[dict[str, Any]]) -> list[str]:
    return [s["content"] for s in saved if s["role"] == "assistant"]


def _patched(loop: Any):
    mcp = MagicMock(get_tools=AsyncMock(return_value=[]))
    return (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp),
        patch("ai_agent.services.chat.run_agent_loop", loop),
    )


async def _until_type(turn: Any, kind: str) -> None:
    async for event in turn:
        if event["type"] == kind:
            return


async def test_an_answer_the_caller_stops_keeps_the_text_that_arrived():
    saved: list[dict[str, Any]] = []
    history = _history(saved)
    mcp, loop = _patched(_loop())
    with mcp, loop:
        turn = _turn(history)
        await _until_type(turn, "content")
        await turn.aclose()  # Stop, a dropped socket, a killed worker

    assert _answers(saved) == [f"{ANSWERED.rstrip()}\n\n[incomplete] {ANSWER_STOPPED}"]
    # `[error]` at the start would keep the answer out of the next turn's history
    assert not _answers(saved)[0].startswith("[error]")


async def test_a_stopped_answer_is_written_although_the_turn_is_being_cancelled():
    """The shape Starlette gives a dropped connection: the turn's task group cancels mid-stream."""
    saved: list[dict[str, Any]] = []
    history = _history(saved)
    mcp, loop = _patched(_loop())
    with mcp, loop:
        turn = _turn(history)
        async with anyio.create_task_group() as task_group:

            async def consume() -> None:
                async for event in turn:
                    if event["type"] == "content":
                        task_group.cancel_scope.cancel()

            task_group.start_soon(consume)

    assert _answers(saved) == [f"{ANSWERED.rstrip()}\n\n[incomplete] {ANSWER_STOPPED}"]


async def test_an_answer_cut_off_at_the_token_cap_keeps_the_text_that_arrived():
    saved: list[dict[str, Any]] = []
    history = _history(saved)
    mcp, loop = _patched(_loop(then=TurnFailure(REPLY_CUT_OFF)))
    with mcp, loop:
        events = [event async for event in _turn(history)]

    assert [e["message"] for e in events if e["type"] == "error"] == [REPLY_CUT_OFF]
    assert _answers(saved) == [f"{ANSWERED.rstrip()}\n\n[incomplete] {REPLY_CUT_OFF}"]


async def test_a_turn_stopped_before_it_said_anything_saves_no_answer():
    saved: list[dict[str, Any]] = []
    history = _history(saved)
    mcp, loop = _patched(_loop(says=""))
    with mcp, loop:
        turn = _turn(history)
        await _until_type(turn, "tool_call")
        await turn.aclose()

    assert _answers(saved) == []
