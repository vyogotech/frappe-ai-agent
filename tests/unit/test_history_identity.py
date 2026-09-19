"""One history client serves every user, so no response may leave a session behind in it."""

from __future__ import annotations

import httpx
import respx

from ai_agent.integrations.frappe_history import FrappeHistoryClient

FRAPPE = "http://frappe.test"


def _answer_as_frappe_does(request: httpx.Request) -> httpx.Response:
    sid = request.headers.get("cookie", "").split("sid=")[1].split(";")[0]
    return httpx.Response(200, json={"message": []}, headers={"Set-Cookie": f"sid={sid}; Path=/"})


@respx.mock
async def test_each_history_read_goes_out_as_its_own_user_only():
    reads = respx.get(f"{FRAPPE}/api/method/frappe.client.get_list").mock(
        side_effect=_answer_as_frappe_does
    )
    client = FrappeHistoryClient(FRAPPE)
    await client.list_messages(sid="alice-sid", session="alice-chat")
    await client.list_messages(sid="bob-sid", session="bob-chat")
    await client.aclose()
    assert [call.request.headers["cookie"] for call in reads.calls] == [
        "sid=alice-sid",
        "sid=bob-sid",
    ]
