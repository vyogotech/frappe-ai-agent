"""A site with no frappe_ai has no chat doctypes, and the agent must not fill its Error Log.

Frappe answers a write to a doctype that is not installed with the controller import it cannot do:
`ImportError`, which carries no `http_status_code`, so 500 (frappe/app.py, handle_exception) and an
Error Log row on that site for every one. Reading the doctype instead answers 404 only for a System
Manager -- for anyone else `handle_does_not_exist_error` turns it into the PermissionError on
`DocType`, 403 -- so the question is asked of `get_versions`, which any signed-in user may call.
"""

from unittest.mock import MagicMock

import httpx
import pytest
import respx

from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.services.chat import ChatService

BASE = "http://frappe:8000"
_VERSIONS_URL = f"{BASE}/api/method/frappe.utils.change_log.get_versions"
_SESSION_URL = f"{BASE}/api/resource/AI Chat Session"
_MESSAGE_URL = f"{BASE}/api/resource/AI Chat Message"
_CSRF_URL = f"{BASE}/app"

# frappe/utils/response.py report_error: the v1 body names the class that was raised
_NO_CONTROLLER = {
    "exc_type": "ImportError",
    "exception": "ImportError: Module import failed for AI Chat Session, "
    "the DocType you're trying to open might be deleted.",
}
_WITHOUT = {"message": {"frappe": {"title": "Frappe Framework"}, "rag": {"title": "rag"}}}
_WITH = {"message": {"frappe": {"title": "Frappe Framework"}, "frappe_ai": {"title": "Frappe AI"}}}


def _csrf() -> None:
    respx.get(_CSRF_URL).mock(
        return_value=httpx.Response(200, text='<script>csrf_token = "abc123";</script>')
    )


def _a_site_without_frappe_ai() -> None:
    _csrf()
    respx.get(_VERSIONS_URL).mock(return_value=httpx.Response(200, json=_WITHOUT))
    respx.post(_SESSION_URL).mock(return_value=httpx.Response(500, json=_NO_CONTROLLER))
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(500, json=_NO_CONTROLLER))


def _writes() -> int:
    return sum(1 for call in respx.calls if call.request.method == "POST")


@pytest.mark.asyncio
@respx.mock
async def test_the_site_is_asked_and_no_write_is_attempted():
    _a_site_without_frappe_ai()
    client = FrappeHistoryClient(base_url=BASE)

    assert await client.chat_doctypes_exist("sid-1") is False
    for _ in range(3):
        assert await client.save_message(sid="sid-1", session="s", role="user", content="q") is None
        assert (
            await client.ensure_session(sid="sid-1", name="s", title="t", context_json="{}") == "s"
        )
    assert _writes() == 0


@pytest.mark.asyncio
@respx.mock
async def test_the_question_is_asked_once():
    _a_site_without_frappe_ai()
    client = FrappeHistoryClient(base_url=BASE)

    for _ in range(3):
        await client.chat_doctypes_exist("sid-1")
    assert len(respx.calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_a_write_that_finds_the_doctype_missing_stops_the_ones_after_it():
    """The other way in: a caller supplying a session id writes before it is ever asked."""
    _a_site_without_frappe_ai()
    client = FrappeHistoryClient(base_url=BASE)

    for _ in range(3):
        await client.ensure_session(sid="sid-1", name="s", title="t", context_json="{}")
        await client.save_message(sid="sid-1", session="s", role="user", content="q")
    assert _writes() == 1


@pytest.mark.asyncio
@respx.mock
async def test_a_site_that_has_the_app_is_written_to_as_before():
    _csrf()
    respx.get(_VERSIONS_URL).mock(return_value=httpx.Response(200, json=_WITH))
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(200, json={"data": {"name": "m1"}}))
    client = FrappeHistoryClient(base_url=BASE)

    assert await client.chat_doctypes_exist("sid-1") is True
    assert await client.save_message(sid="sid-1", session="s", role="user", content="q") == "m1"
    assert _writes() == 1


@pytest.mark.asyncio
@respx.mock
async def test_a_frappe_that_will_not_answer_is_not_a_frappe_without_frappe_ai():
    """An outage is not an answer, or one of them would end history for good."""
    _csrf()
    respx.get(_VERSIONS_URL).mock(side_effect=httpx.ConnectError("frappe is down"))
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(200, json={"data": {"name": "m1"}}))
    client = FrappeHistoryClient(base_url=BASE)

    assert await client.chat_doctypes_exist("sid-1") is True
    assert await client.save_message(sid="sid-1", session="s", role="user", content="q") == "m1"


@pytest.mark.asyncio
@respx.mock
async def test_a_refusal_is_not_an_answer_either():
    _csrf()
    respx.get(_VERSIONS_URL).mock(
        return_value=httpx.Response(403, json={"exc_type": "PermissionError"})
    )
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(200, json={"data": {"name": "m1"}}))
    client = FrappeHistoryClient(base_url=BASE)

    assert await client.chat_doctypes_exist("sid-1") is True
    assert await client.save_message(sid="sid-1", session="s", role="user", content="q") == "m1"


@pytest.mark.asyncio
async def test_the_turn_asks_before_it_creates_a_chat():
    history = MagicMock(spec=FrappeHistoryClient)
    history.chat_doctypes_exist.return_value = False
    service = ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp"),  # pyright: ignore[reportCallIssue]
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )

    session_id, saved = await service._open_session(
        sid="sid-1", session_id=None, title="t", context={}
    )

    assert session_id.startswith("tmp-")  # the turn reports itself degraded, and writes nothing
    assert saved is False
    history.create_session.assert_not_called()
