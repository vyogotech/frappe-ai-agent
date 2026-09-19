"""The CSRF token fetch keeps the caller's sid across Frappe's /app -> /desk redirect."""

import httpx
import respx

from ai_agent.integrations.frappe_history import FrappeHistoryClient

FRAPPE = "http://frappe:8000"


def _desk(request: httpx.Request) -> httpx.Response:
    if request.headers.get("cookie") != "sid=alice-sid":
        return httpx.Response(302, headers={"Location": "/login?redirect-to=%2Fdesk"})
    return httpx.Response(200, text='<script>csrf_token = "a11ce0123456789abcdef";</script>')


@respx.mock
async def test_a_write_gets_its_csrf_token_through_the_desk_redirect():
    respx.get(f"{FRAPPE}/app").mock(return_value=httpx.Response(301, headers={"Location": "/desk"}))
    respx.get(f"{FRAPPE}/desk").mock(side_effect=_desk)
    respx.get(f"{FRAPPE}/login").mock(return_value=httpx.Response(200, text="<form>login</form>"))
    saved = respx.post(f"{FRAPPE}/api/resource/AI Chat Message").mock(
        return_value=httpx.Response(200, json={"data": {"name": "m1"}})
    )
    history = FrappeHistoryClient(base_url=FRAPPE)
    name = await history.save_message(sid="alice-sid", session="s1", role="user", content="hi")
    await history.aclose()
    assert name == "m1"
    assert saved.calls.last.request.headers["X-Frappe-CSRF-Token"] == "a11ce0123456789abcdef"
