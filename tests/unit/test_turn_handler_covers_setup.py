"""The turn's handler covers the history setup: a failure there ends the turn, not the stream."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ANSWER_FAILED, ChatService

CLOSED = RuntimeError("FrappeHistoryClient is closed; build a new instance for further writes")


def _history() -> MagicMock:
    history = MagicMock(spec=FrappeHistoryClient)
    history.create_session.return_value = "chat-1"
    history.ensure_session.side_effect = lambda *, name, **_: name
    history.save_message.return_value = "msg-1"
    history.list_messages.return_value = []
    return history


def _answers(**_kwargs):
    async def _gen():
        yield {"type": "content", "text": "Sure."}

    return _gen()


async def _turn(history: MagicMock, session_id: str | None) -> list[dict[str, Any]]:
    service = ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp.test/mcp"),  # pyright: ignore[reportCallIssue]
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )
    mcp = MagicMock(get_tools=AsyncMock(return_value=[]))
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp),
        patch("ai_agent.services.chat.run_agent_loop", _answers),
    ):
        return [
            e
            async for e in service.handle_message(
                message="hi", session_id=session_id, context={}, user_context=UserContext(sid="s")
            )
        ]


@pytest.mark.parametrize("call", ["create_session", "ensure_session"])
async def test_a_bug_in_the_session_write_ends_the_turn_with_one_line(call: str):
    history = _history()
    getattr(history, call).side_effect = CLOSED

    events = await _turn(history, None if call == "create_session" else "chat-1")

    assert [e["type"] for e in events][-2:] == ["error", "done"]
    assert events[-2]["message"] == ANSWER_FAILED
    assert events[-1]["data_quality"] == "low"


async def test_no_session_means_nothing_is_written_into_one():
    history = _history()
    history.create_session.side_effect = CLOSED

    await _turn(history, None)

    history.save_message.assert_not_called()
