import httpx
import pytest
import respx

from ai_agent.integrations.frappe_history import FrappeHistoryClient


@pytest.mark.asyncio
@respx.mock
async def test_create_session_posts_with_sid_cookie():
    respx.post("http://frappe:8000/api/resource/AI Chat Session").mock(
        return_value=httpx.Response(200, json={"data": {"name": "sess-123"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")
    name = await client.create_session(
        sid="abc123",
        title="first message",
        context_json="{}",
    )
    assert name == "sess-123"
    called = respx.calls[-1].request
    assert called.headers["Cookie"] == "sid=abc123"


@pytest.mark.asyncio
@respx.mock
async def test_save_message_posts_with_sid_cookie_and_fields():
    respx.post("http://frappe:8000/api/resource/AI Chat Message").mock(
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
    called = respx.calls[-1].request
    assert called.headers["Cookie"] == "sid=abc123"
    # Assert the body contains the right fields
    import json
    body = json.loads(called.content)
    assert body["session"] == "sess-123"
    assert body["role"] == "user"
    assert body["content"] == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_create_session_returns_none_on_http_error(caplog):
    respx.post("http://frappe:8000/api/resource/AI Chat Session").mock(
        return_value=httpx.Response(500)
    )
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
    respx.post("http://frappe:8000/api/resource/AI Chat Message").mock(
        return_value=httpx.Response(500)
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")
    result = await client.save_message(
        sid="abc", session="sess-1", role="user", content="hi"
    )
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_save_message_forwards_optional_tool_fields():
    respx.post("http://frappe:8000/api/resource/AI Chat Message").mock(
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
    import json
    body = json.loads(respx.calls[-1].request.content)
    assert body["tool_name"] == "list_invoices"
    assert body["tool_args_json"] == '{"status": "unpaid"}'
    assert body["tool_result_json"] == '{"count": 3}'
