"""A turn the caller hangs up on leaves a log line, so it does not vanish without a trace."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService


def _turn(history):
    settings = Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp")  # pyright: ignore[reportCallIssue]
    service = ChatService(settings=settings, llm=MagicMock(), system_prompt_builder=lambda _ctx: "")
    service._history = history
    return service.handle_message(
        message="hi", session_id="s-1", context={}, user_context=UserContext(sid="abc")
    )


def _cancelled(log):
    return [c for c in log.info.call_args_list if c.args[:1] == ("chat_turn_cancelled",)]


async def test_a_turn_cut_mid_answer_is_logged():
    history = MagicMock(
        save_message=AsyncMock(return_value="m1"),
        ensure_session=AsyncMock(side_effect=lambda *, name, **_: name),
        list_messages=AsyncMock(return_value=[]),
    )
    client = MagicMock(get_tools=AsyncMock(return_value=[]))

    def slow_loop(**_kwargs):
        async def gen():
            yield {"type": "tool_call", "name": "list_documents", "arguments": {}}
            await asyncio.sleep(60)

        return gen()

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=client),
        patch("ai_agent.services.chat.run_agent_loop", slow_loop),
        patch("ai_agent.services.chat.logger") as log,
    ):
        turn = _turn(history)
        assert (await anext(turn))["type"] == "session"
        assert (await anext(turn))["type"] == "tool_call"
        await turn.aclose()
    assert _cancelled(log)


async def test_a_turn_cut_while_its_chat_is_created_is_logged():
    async def hang(**_kwargs):
        await asyncio.sleep(60)

    with patch("ai_agent.services.chat.logger") as log:
        first = asyncio.create_task(
            anext(_turn(MagicMock(ensure_session=AsyncMock(side_effect=hang))))
        )
        await asyncio.sleep(0.05)
        first.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await first
    assert _cancelled(log)
