import json

import httpx
import pytest
import respx

from ai_agent.integrations.frappe_history import FrappeHistoryClient

_CSRF_URL = "http://frappe:8000/app"
_SESSION_URL = "http://frappe:8000/api/resource/AI Chat Session"
_MESSAGE_URL = "http://frappe:8000/api/resource/AI Chat Message"
_FAKE_CSRF = "abc123def456789abcdef012345"


def _csrf_page(token: str = _FAKE_CSRF) -> str:
    """A minimal /app HTML response that embeds the csrf_token JS variable."""
    return (
        "<!DOCTYPE html><html><head><script>"
        f'csrf_token = "{token}";'
        "</script></head><body></body></html>"
    )


def _mock_csrf_ok() -> None:
    """Respond to the /app GET with a page containing the csrf_token."""
    respx.get(_CSRF_URL).mock(
        return_value=httpx.Response(200, text=_csrf_page())
    )


@pytest.mark.asyncio
@respx.mock
async def test_create_session_posts_with_sid_and_csrf():
    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "sess-123"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.create_session(
        sid="abc123",
        title="first message",
        context_json="{}",
    )

    assert name == "sess-123"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert post.headers["Cookie"] == "sid=abc123"
    assert post.headers["X-Frappe-CSRF-Token"] == _FAKE_CSRF


@pytest.mark.asyncio
@respx.mock
async def test_save_message_posts_with_sid_and_csrf_and_fields():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.save_message(
        sid="abc123",
        session="sess-123",
        role="user",
        content="hello",
    )

    assert name == "msg-1"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert post.headers["Cookie"] == "sid=abc123"
    assert post.headers["X-Frappe-CSRF-Token"] == _FAKE_CSRF
    body = json.loads(post.content)
    assert body["session"] == "sess-123"
    assert body["role"] == "user"
    assert body["content"] == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_csrf_token_is_cached_across_calls():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    # Three writes with the same sid should only fetch CSRF once.
    await client.save_message(sid="abc", session="s", role="user", content="a")
    await client.save_message(sid="abc", session="s", role="user", content="b")
    await client.save_message(sid="abc", session="s", role="user", content="c")

    csrf_calls = [c for c in respx.calls if c.request.method == "GET"]
    assert len(csrf_calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_csrf_token_is_refetched_per_sid():
    """Different sids must not share a cached token."""
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    await client.save_message(sid="user-a", session="s", role="user", content="a")
    await client.save_message(sid="user-b", session="s", role="user", content="b")

    csrf_calls = [c for c in respx.calls if c.request.method == "GET"]
    assert len(csrf_calls) == 2


@pytest.mark.asyncio
@respx.mock
async def test_write_retried_once_on_csrf_error():
    """If the first POST returns a CSRF error, the client should refresh
    its token and retry the write exactly once."""
    # First CSRF fetch returns token-a. A second fetch (after invalidation)
    # returns token-b. The POST rejects token-a with CSRF error, accepts b.
    csrf_route = respx.get(_CSRF_URL).mock(
        side_effect=[
            httpx.Response(200, text=_csrf_page("aaaaaaaa11111111")),
            httpx.Response(200, text=_csrf_page("bbbbbbbb22222222")),
        ]
    )
    post_route = respx.post(_MESSAGE_URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={"exc_type": "CSRFTokenError"},
                text='{"exc_type":"CSRFTokenError","message":"Invalid CSRF token"}',
            ),
            httpx.Response(200, json={"data": {"name": "msg-1"}}),
        ]
    )

    client = FrappeHistoryClient(base_url="http://frappe:8000")
    name = await client.save_message(
        sid="abc", session="s", role="user", content="hi"
    )

    assert name == "msg-1"
    assert csrf_route.call_count == 2
    assert post_route.call_count == 2
    # The retry must use the fresh token, not the stale one.
    second_post = post_route.calls[1].request
    assert second_post.headers["X-Frappe-CSRF-Token"] == "bbbbbbbb22222222"


@pytest.mark.asyncio
@respx.mock
async def test_create_session_returns_none_on_http_error(caplog):
    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    result = await client.create_session(
        sid="abc123",
        title="t",
        context_json="{}",
    )
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_save_message_returns_none_on_http_error():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    result = await client.save_message(
        sid="abc", session="sess-1", role="user", content="hi"
    )
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_save_message_forwards_optional_tool_fields():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-2"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    await client.save_message(
        sid="abc",
        session="sess-1",
        role="assistant",
        content="final answer",
        tool_name="list_invoices",
        tool_args_json='{"status": "unpaid"}',
        tool_result_json='{"count": 3}',
    )

    post = next(c.request for c in respx.calls if c.request.method == "POST")
    body = json.loads(post.content)
    assert body["tool_name"] == "list_invoices"
    assert body["tool_args_json"] == '{"status": "unpaid"}'
    assert body["tool_result_json"] == '{"count": 3}'


@pytest.mark.asyncio
@respx.mock
async def test_write_proceeds_without_token_when_csrf_fetch_fails():
    """If Frappe's /app page is unreachable, the client should still try
    to POST (Frappe may reject with 400, which becomes a None return)."""
    respx.get(_CSRF_URL).mock(return_value=httpx.Response(500))
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.save_message(
        sid="abc", session="s", role="user", content="hi"
    )

    assert name == "msg-1"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert "X-Frappe-CSRF-Token" not in post.headers


@pytest.mark.asyncio
@respx.mock
async def test_write_proceeds_without_token_when_csrf_not_in_html():
    """If /app returns 200 but the HTML has no csrf_token JS variable,
    the client should still attempt the POST (some Frappe versions may
    omit it in certain contexts)."""
    respx.get(_CSRF_URL).mock(
        return_value=httpx.Response(200, text="<html><body>No token here</body></html>")
    )
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.save_message(
        sid="abc", session="s", role="user", content="hi"
    )

    assert name == "msg-1"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert "X-Frappe-CSRF-Token" not in post.headers
