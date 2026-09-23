"""C26: a failed turn shows one plain line per kind of failure; the detail stays in the log."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import structlog

from ai_agent.agent.loop import REPLY_CUT_OFF, TurnFailure
from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import (
    ANSWER_FAILED,
    TOOLS_TIMED_OUT,
    TURN_TOO_LONG,
    ChatService,
)

RAW = "ConnectError: All connection attempts failed to ollama.internal:11434"


def _service(**overrides: Any) -> tuple[ChatService, MagicMock]:
    history = MagicMock()
    history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    history.list_messages = AsyncMock(return_value=[])
    history.save_message = AsyncMock(return_value="msg-1")
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
        **overrides,
    )
    return ChatService(
        settings=settings,
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "system",
        history=history,
    ), history


def _raising_loop(exc: BaseException):
    def _factory(**_kwargs):
        async def _gen():
            raise exc
            yield  # unreachable

        return _gen()

    return _factory


async def _turn(service: ChatService, tools: Any = None) -> list[dict[str, Any]]:
    mcp = MagicMock()
    mcp.get_tools = tools or AsyncMock(return_value=[])
    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp):
        return [
            ev
            async for ev in service.handle_message(
                message="how many sales orders are open?",
                session_id="sess-1",
                context={},
                user_context=UserContext(sid="sid-1"),
            )
        ]


async def test_an_unexpected_failure_shows_one_line_and_keeps_the_detail_in_the_log():
    service, history = _service()
    with (
        structlog.testing.capture_logs() as logs,
        patch("ai_agent.services.chat.run_agent_loop", _raising_loop(RuntimeError(RAW))),
    ):
        events = await _turn(service)

    errors = [e for e in events if e["type"] == "error"]
    assert [e["message"] for e in errors] == [ANSWER_FAILED]
    assert "ollama.internal" not in json.dumps(events)
    # the failed turn is saved as the same line, so a reopened chat shows no exception text either
    assert history.save_message.call_args_list[-1].kwargs["content"] == f"[error] {ANSWER_FAILED}"
    failed = [log for log in logs if log["event"] == "chat_handle_message_failed"]
    assert failed[0]["error_type"] == "RuntimeError"
    assert RAW in failed[0]["error"]


async def test_tools_that_never_load_show_their_own_line():
    service, _ = _service(mcp_tools_load_timeout_s=0.05)

    async def _hang(*_a: Any, **_k: Any) -> list[Any]:
        await asyncio.sleep(60)
        return []

    with structlog.testing.capture_logs() as logs:
        events = await asyncio.wait_for(_turn(service, tools=_hang), 5)

    assert [e["message"] for e in events if e["type"] == "error"] == [TOOLS_TIMED_OUT]
    timed_out = [log for log in logs if log["event"] == "chat_tools_load_timed_out"]
    assert timed_out[0]["timeout_s"] == 0.05


async def test_a_turn_past_its_deadline_shows_its_own_line():
    service, _ = _service(agent_turn_timeout_s=0.2)

    def _stalls(**_kwargs):
        async def _gen():
            await asyncio.sleep(60)
            yield {}

        return _gen()

    with patch("ai_agent.services.chat.run_agent_loop", _stalls):
        events = await asyncio.wait_for(_turn(service), 5)

    assert [e["message"] for e in events if e["type"] == "error"] == [TURN_TOO_LONG]


async def test_a_cut_off_reply_shows_its_own_line():
    service, _ = _service()
    with patch("ai_agent.services.chat.run_agent_loop", _raising_loop(TurnFailure(REPLY_CUT_OFF))):
        events = await _turn(service)

    assert [e["message"] for e in events if e["type"] == "error"] == [REPLY_CUT_OFF]
