"""A session id never reaches the agent's logs, whole or in part: a log outlives its session."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import Request
from httpx import ASGITransport, AsyncClient

from ai_agent.app import create_app
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.middleware.sid import UserContext, extract_user_context
from ai_agent.services.chat import ChatService
from ai_agent.transport.sse import _require_sid

SID = "a1b2c3d4e5f6a7b8-the-users-session"


class _Answers:
    async def handle_message(self, **_kwargs):
        yield {"type": "done", "tools_called": [], "data_quality": "high", "timestamp": "t"}


def _signed_in(request: Request) -> UserContext:
    user_context = extract_user_context(request)
    assert user_context is not None
    return user_context


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


async def test_a_rate_limited_request_logs_no_sid():
    app = create_app(Settings(_env_file=None, agent_rate_limit="1/minute"))  # pyright: ignore[reportCallIssue]
    app.dependency_overrides[_require_sid] = _signed_in
    app.state.chat_service = _Answers()
    lines = _Lines()
    logging.getLogger("slowapi").addHandler(lines)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test", cookies={"sid": SID}
        ) as ac:
            statuses = [
                (await ac.post("/api/v1/chat", json={"message": "x"})).status_code for _ in range(2)
            ]
    finally:
        logging.getLogger("slowapi").removeHandler(lines)
    assert statuses == [200, 429]
    assert lines.lines
    assert not any(SID[:8] in line for line in lines.lines)


def _no_events(**_kwargs):
    async def _gen():
        return
        yield

    return _gen()


async def test_the_tools_failure_warning_holds_no_part_of_the_sid():
    history = MagicMock(spec=FrappeHistoryClient)
    history.create_session.return_value = "sess-sid-test"
    service = ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp"),  # pyright: ignore[reportCallIssue]
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )
    client = MagicMock()
    client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))
    with (
        patch("ai_agent.services.chat.logger") as logger,
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=client),
        patch("ai_agent.services.chat.run_agent_loop", _no_events),
    ):
        async for _ in service.handle_message(
            message="hi", session_id=None, context={}, user_context=UserContext(sid=SID)
        ):
            pass
    calls = [
        c
        for c in logger.warning.call_args_list
        if c.args[0] == "chat_tools_load_failed_soft_degrade"
    ]
    assert calls
    assert not any(SID[:8] in str(value) for c in calls for value in c.kwargs.values())
