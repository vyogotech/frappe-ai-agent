"""Who may ask the agent: only a sid that Frappe itself recognises as a signed-in user."""

from __future__ import annotations

import httpx
import respx
from httpx import ASGITransport, AsyncClient

from ai_agent.app import create_app
from ai_agent.config import Settings

FRAPPE = "http://frappe.test"
WHO = f"{FRAPPE}/api/method/frappe.auth.get_logged_user"


class _Answers:
    def __init__(self) -> None:
        self.asked = 0

    async def handle_message(
        self, *, message, session_id, context, user_context, confirmation=None
    ):
        self.asked += 1
        yield {"type": "content", "text": "hi"}
        yield {"type": "done", "tools_called": [], "data_quality": "high", "timestamp": "t"}


async def _chat(sid: str) -> tuple[httpx.Response, _Answers]:
    app = create_app(Settings(_env_file=None, frappe_url=FRAPPE))  # pyright: ignore[reportCallIssue]
    answers = _Answers()
    app.state.chat_service = answers
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", cookies={"sid": sid}
    ) as ac:
        response = await ac.post("/api/v1/chat", json={"message": "hello"})
    return response, answers


@respx.mock
async def test_a_sid_frappe_does_not_recognise_is_refused_before_any_work():
    respx.get(WHO).mock(return_value=httpx.Response(403, json={"exc_type": "PermissionError"}))
    response, answers = await _chat("forged-sid")
    assert response.status_code == 401
    assert answers.asked == 0


@respx.mock
async def test_a_signed_in_user_is_answered():
    who = respx.get(WHO).mock(
        return_value=httpx.Response(200, json={"message": "alice@example.test"})
    )
    response, answers = await _chat("alice-sid")
    assert response.status_code == 200
    assert answers.asked == 1
    assert who.calls.last.request.headers["cookie"] == "sid=alice-sid"


@respx.mock
async def test_the_agent_fails_closed_when_frappe_cannot_be_asked():
    respx.get(WHO).mock(side_effect=httpx.ConnectError("frappe is down"))
    response, answers = await _chat("alice-sid")
    assert response.status_code == 503
    assert answers.asked == 0
