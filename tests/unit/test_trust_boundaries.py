"""Each guard on a trust boundary, seen from the side that must be refused."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import respx
from httpx import ASGITransport, AsyncClient, Response
from langchain_core.tools import StructuredTool

from ai_agent.agent.loop import KB_TOOL, _passages, run_agent_loop
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.app import create_app
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient

FRAPPE = "http://frappe.test"
WHO = f"{FRAPPE}/api/method/frappe.auth.get_logged_user"
# Frappe serves its login page with a csrf_token of its own, for the Guest session. Short and
# word-shaped on purpose: the value only has to match _CSRF_PATTERN's hex, and a 32-character
# hex run here reads as a real credential to every secret scanner.
GUEST_TOKEN = "c0ffeedecaf"
GUEST_LOGIN_PAGE = f'<html><script>window.csrf_token = "{GUEST_TOKEN}";</script></html>'


class _Answers:
    def __init__(self) -> None:
        self.asked = 0

    async def handle_message(self, **_kwargs):
        self.asked += 1
        yield {"type": "done", "tools_called": [], "data_quality": "high", "timestamp": "t"}


@respx.mock
async def test_a_guest_session_is_not_a_signed_in_user():
    """Frappe answers 200 "Guest" to anyone; only a named user may spend the agent's tokens."""
    respx.get(WHO).mock(return_value=Response(200, json={"message": "Guest"}))
    app = create_app(Settings(_env_file=None, frappe_url=FRAPPE))  # pyright: ignore[reportCallIssue]
    answers = _Answers()
    app.state.chat_service = answers

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", cookies={"sid": "guest-sid"}
    ) as ac:
        response = await ac.post("/api/v1/chat", json={"message": "hello"})

    assert response.status_code == 401
    assert answers.asked == 0


@respx.mock
async def test_a_session_someone_else_owns_hands_over_none_of_its_messages():
    """A 409 means the row exists, not that it is the caller's: the read still goes out as them."""
    respx.get(f"{FRAPPE}/app").mock(return_value=Response(200, text='csrf_token = "abc123"'))
    respx.post(f"{FRAPPE}/api/resource/AI Chat Session").mock(return_value=Response(409))
    read = respx.get(f"{FRAPPE}/api/method/frappe.client.get_list").mock(
        return_value=Response(403, json={"exc_type": "PermissionError"})
    )
    client = FrappeHistoryClient(base_url=FRAPPE)
    try:
        name = await client.ensure_session(
            sid="mallory-sid", name="chat-alice", title="t", context_json="{}"
        )
        rows = await client.list_messages(sid="mallory-sid", session="chat-alice")
    finally:
        await client.aclose()

    assert name == "chat-alice"  # the id is kept; ownership is Frappe's to decide, not ours
    assert rows == []
    assert read.calls.last.request.headers["cookie"] == "sid=mallory-sid"


@respx.mock
async def test_an_expired_sid_never_borrows_the_guest_csrf_token():
    respx.get(f"{FRAPPE}/app").mock(
        return_value=Response(302, headers={"Location": f"{FRAPPE}/login?redirect-to=%2Fapp"})
    )
    respx.get(url__startswith=f"{FRAPPE}/login").mock(
        return_value=Response(200, text=GUEST_LOGIN_PAGE)
    )
    write = respx.post(f"{FRAPPE}/api/resource/AI Chat Message").mock(return_value=Response(403))
    client = FrappeHistoryClient(base_url=FRAPPE)
    try:
        saved = await client.save_message(
            sid="expired-sid", session="chat-1", role="user", content="hi"
        )
    finally:
        await client.aclose()

    assert saved is None
    assert "X-Frappe-CSRF-Token" not in write.calls.last.request.headers


def test_a_tool_reply_that_is_not_a_passage_list_yields_no_sources():
    """Tool output is data from documents; a broken reply is no source and ends no turn."""
    assert _passages('Found 1 passage(s)\n[{"file": "abc", "seq": 0}') == []  # truncated
    assert _passages('[{"file": "abc", "seq": 0},]') == []  # trailing comma
    assert _passages("[not json at all]") == []


SEARCHED: list[dict[str, Any]] = []
_TEXT = {"type": "text", "payload": {"content": "ok"}}


async def _search(query: str, session: str = "") -> str:
    SEARCHED.append({"query": query, "session": session})
    return "no passages"


def _llm(envelopes: list[list[dict[str, Any]]]) -> MagicMock:
    llm = MagicMock()

    def _with_structured_output(_schema, **_kw):
        structured = MagicMock()
        structured.astream = lambda _messages, **_kwargs: _replay(envelopes.pop(0))
        return structured

    llm.with_structured_output = _with_structured_output
    return llm


def _replay(envelopes: list[dict[str, Any]]):
    async def _gen():
        for envelope in envelopes:
            yield envelope

    return _gen()


async def test_the_model_cannot_aim_the_knowledge_search_at_another_chat():
    """Even with no session of its own to pin, the turn drops the one the model wrote."""
    SEARCHED.clear()
    call = {
        "type": "tool_call",
        "payload": {
            "name": KB_TOOL,
            "arguments": {"query": "salary", "session": "chat-of-someone-else"},
        },
    }
    tool = StructuredTool.from_function(
        coroutine=_search,
        name=KB_TOOL,
        description="Search.",
        metadata={"readOnlyHint": True},
    )

    await _drain(
        run_agent_loop(
            llm=_llm([[{"blocks": [call]}], [{"blocks": [_TEXT]}]]),
            tool_registry=ToolRegistry([tool]),
            user_message="what is the salary cap?",
            session=None,
        )
    )

    assert SEARCHED == [{"query": "salary", "session": ""}]


async def test_the_knowledge_search_is_pinned_to_this_turn_s_chat():
    SEARCHED.clear()
    call = {
        "type": "tool_call",
        "payload": {"name": KB_TOOL, "arguments": {"query": "salary", "session": "chat-theirs"}},
    }
    tool = StructuredTool.from_function(
        coroutine=_search,
        name=KB_TOOL,
        description="Search.",
        metadata={"readOnlyHint": True},
    )

    await _drain(
        run_agent_loop(
            llm=_llm([[{"blocks": [call]}], [{"blocks": [_TEXT]}]]),
            tool_registry=ToolRegistry([tool]),
            user_message="what is the salary cap?",
            session="chat-mine",
        )
    )

    assert SEARCHED == [{"query": "salary", "session": "chat-mine"}]


async def _drain(agen) -> list[dict[str, Any]]:
    return [ev async for ev in agen]
