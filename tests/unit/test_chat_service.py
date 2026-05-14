"""Tests for the per-request ChatService shape (Phase 6b)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import structlog
from langchain_core.messages import AIMessageChunk

from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService, _BlockStreamSplitter


def _make_settings() -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
    )


def _make_llm(formatter_yields: list[dict[str, Any]] | None = None) -> MagicMock:
    """Build an LLM mock that supports `.with_structured_output(...).astream(...)`.

    `formatter_yields` is the list of partial-dict snapshots the Pass-2
    formatter should emit. Default `[]` means Pass 2 produces no envelope
    blocks — useful for tests that only care about Pass-1 / tool-call /
    session / error behaviour.
    """
    llm = MagicMock()
    yields = list(formatter_yields or [])

    def _astream(_messages):
        async def _gen():
            for y in yields:
                yield y

        return _gen()

    formatter = MagicMock()
    formatter.astream = _astream
    llm.with_structured_output.return_value = formatter
    return llm


def _make_service(llm: Any | None = None) -> ChatService:
    return ChatService(
        settings=_make_settings(),
        llm=llm if llm is not None else _make_llm(),
        checkpointer=MagicMock(),
        system_prompt_builder=lambda _ctx: "you are helpful",
    )


async def _drain(agen) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for ev in agen:
        out.append(ev)
    return out


class _StreamFactory:
    """Helper: builds an async `astream_events` stand-in from a list of events."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    def __call__(self, *_args, **_kwargs):  # matches graph.astream_events signature
        events = self._events

        async def _gen():
            for ev in events:
                yield ev

        return _gen()


@pytest.mark.asyncio
async def test_handle_message_builds_mcp_client_with_caller_sid():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch(
            "ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client
        ) as mock_builder,
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    mock_builder.assert_called_once()
    # settings is the first positional arg; sid is the second
    args, _kwargs = mock_builder.call_args
    assert args[1] == "abc123"


@pytest.mark.asyncio
async def test_handle_message_yields_session_then_done_envelope():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    # Minimum envelope: session announced first, done last. Generic
    # "Loading tools" / "Thinking" status events were removed — the FE
    # placeholder bubble is the loading indicator on its own.
    assert len(events) >= 2
    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"
    assert events[-1]["tools_called"] == []
    assert "timestamp" in events[-1]
    assert events[-1]["timestamp"].endswith("Z")
    assert events[-1]["data_quality"] == "high"


class _RecordingStreamFactory:
    """Like _StreamFactory but records the kwargs of each astream_events call."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *_args, **kwargs):
        self.calls.append(kwargs)
        events = self._events

        async def _gen():
            for ev in events:
                yield ev

        return _gen()


@pytest.mark.asyncio
async def test_handle_message_uses_recursion_limit_from_settings():
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
        agent_recursion_limit=123,
    )
    service = ChatService(
        settings=settings,
        llm=MagicMock(),
        checkpointer=MagicMock(),
        system_prompt_builder=lambda _ctx: "you are helpful",
    )
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    recording = _RecordingStreamFactory([])
    mock_graph = MagicMock()
    mock_graph.astream_events = recording

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id="s1",
                context={},
                user_context=user_context,
            )
        )

    assert len(recording.calls) == 1
    assert recording.calls[0]["config"]["recursion_limit"] == 123


@pytest.mark.asyncio
async def test_handle_message_translates_tool_start_to_tool_call_event():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_tool_start",
                "name": "list_documents",
                "data": {"input": {"doctype": "Customer"}},
            },
            {
                "event": "on_tool_end",
                "name": "list_documents",
                "data": {"output": "ok"},
            },
        ]
    )

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="list customers",
                session_id="s1",
                context={},
                user_context=user_context,
            )
        )

    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "list_documents"
    assert tool_calls[0]["arguments"] == {"doctype": "Customer"}

    done = events[-1]
    assert done["type"] == "done"
    assert done["tools_called"] == ["list_documents"]


@pytest.mark.asyncio
async def test_pass2_text_block_becomes_content_event():
    """The envelope formatter pass emits a `text` block — the service
    translates it to a single `content` event (matches the wire protocol
    where prose is `content`, not a content_block of type text)."""
    llm = _make_llm(
        formatter_yields=[{"blocks": [{"type": "text", "payload": {"content": "hello there"}}]}]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s2",
                context={},
                user_context=user_context,
            )
        )

    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) == 1
    assert content_events[0]["text"] == "hello there"


@pytest.mark.asyncio
async def test_pass1_text_stream_is_suppressed():
    """Pass-1 `on_chat_model_stream` chunks must NOT reach the FE as
    content events — they are captured as the formatter's draft input
    only. The user-visible content comes from the envelope pass."""
    service = _make_service()  # default formatter yields nothing
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="leak ")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="me")},
            },
        ]
    )

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-suppress",
                context={},
                user_context=user_context,
            )
        )

    assert [e for e in events if e["type"] == "content"] == []


@pytest.mark.asyncio
async def test_handle_message_ignores_ai_message_with_tool_calls():
    """Intermediate AI messages that only carry tool_calls should not
    surface as content (or as draft text — they're the model's
    invocation, not its answer)."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    intermediate = AIMessageChunk(
        content="",
        tool_call_chunks=[{"id": "1", "name": "list_documents", "args": "{}", "index": 0}],
    )
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [{"event": "on_chat_model_stream", "data": {"chunk": intermediate}}]
    )

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s3",
                context={},
                user_context=user_context,
            )
        )

    assert [e for e in events if e["type"] == "content"] == []


@pytest.mark.asyncio
async def test_handle_message_surfaces_tool_load_failure_as_tools_unavailable():
    """Any error during MCP tool loading (other than TimeoutError, which
    has its own message) is surfaced to the client as a clean
    'Tools unavailable' message. The underlying cause stays in a
    single-line warning log — no traceback, no raw exception class
    leaked to the SSE stream."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with (
        structlog.testing.capture_logs() as logs,
        patch(
            "ai_agent.services.chat.build_mcp_client_for_sid",
            return_value=mock_client,
        ),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    msg = error_events[0]["message"]
    assert "Tools unavailable" in msg
    # No raw exception class leaks to the client.
    assert "RuntimeError" not in msg
    # Generator still emits a terminal `done` even on failure.
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"

    # Cause is captured in a single warning, not the exception-level
    # chat_handle_message_failed log (which dumps a full traceback).
    warns = [e for e in logs if e["event"] == "chat_tools_load_failed"]
    assert len(warns) == 1
    assert warns[0]["log_level"] == "warning"
    assert warns[0]["error_type"] == "RuntimeError"
    assert warns[0]["error"] == "mcp down"
    assert not any(e["event"] == "chat_handle_message_failed" for e in logs)


@pytest.mark.asyncio
async def test_tool_load_warning_unwraps_exception_group_to_root_cause():
    """anyio's TaskGroup wraps the real MCP error in BaseExceptionGroup
    one or more layers deep. The warning log must surface the root cause
    type/message, not the opaque outer wrapper."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    inner = RuntimeError("Session terminated")
    group = ExceptionGroup("anyio TaskGroup", [inner])

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=group)

    with (
        structlog.testing.capture_logs() as logs,
        patch(
            "ai_agent.services.chat.build_mcp_client_for_sid",
            return_value=mock_client,
        ),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    warns = [e for e in logs if e["event"] == "chat_tools_load_failed"]
    assert len(warns) == 1
    assert warns[0]["error_type"] == "RuntimeError"
    assert warns[0]["error"] == "Session terminated"


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://localhost:8000/mcp")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(
        f"Client error '{status}' for url '{request.url}'",
        request=request,
        response=response,
    )


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.asyncio
async def test_tool_load_auth_rejection_yields_authentication_message(status):
    """401/403 from the MCP server is reported as an auth-failure, not
    'cannot reach' — otherwise an operator chasing a stale sid wastes
    time looking at networking."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=_http_status_error(status))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-auth",
                context={},
                user_context=user_context,
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    msg = error_events[0]["message"]
    assert "authentication failed" in msg
    assert "cannot reach" not in msg


@pytest.mark.asyncio
async def test_tool_load_transport_error_yields_unreachable_message():
    """ConnectError / DNS failure / refused: report as unreachable so
    the operator knows to check whether the MCP process is up."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(
        side_effect=httpx.ConnectError("All connection attempts failed")
    )

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-conn",
                context={},
                user_context=user_context,
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    msg = error_events[0]["message"]
    assert "cannot reach" in msg
    assert "authentication" not in msg


@pytest.mark.asyncio
async def test_tool_load_unknown_http_status_yields_status_code_in_message():
    """An unexpected HTTP status (e.g. 500) gets a status-coded message
    so the operator can tell 'server is broken' from 'server rejected
    auth' without opening the log."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=_http_status_error(500))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-500",
                context={},
                user_context=user_context,
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "HTTP 500" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_handle_message_installs_error_handler_on_each_tool():
    """Every tool returned from MCP must go through install_tool_error_handler,
    which both wraps the coroutine and sets handle_tool_error. Checking the
    handle_tool_error attribute alone is sufficient to confirm the call ran
    (see test_tool_errors.py for the wrap behaviour itself)."""
    from ai_agent.agent.tool_errors import to_tool_result_message

    service = _make_service()
    user_context = UserContext(sid="abc123")

    tool_a = MagicMock()
    tool_a.coroutine = AsyncMock(return_value="ok")
    tool_a.handle_tool_error = None
    tool_b = MagicMock()
    tool_b.coroutine = AsyncMock(return_value="ok")
    tool_b.handle_tool_error = None

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[tool_a, tool_b])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    assert tool_a.handle_tool_error is to_tool_result_message
    assert tool_b.handle_tool_error is to_tool_result_message


@pytest.mark.asyncio
async def test_handle_message_uses_session_id_as_thread_id():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    captured: dict[str, Any] = {}

    def _capture(*_args, **kwargs):
        captured.update(kwargs)

        async def _empty():
            return
            yield  # pragma: no cover

        return _empty()

    mock_graph = MagicMock()
    mock_graph.astream_events = _capture

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id="sess-42",
                context={},
                user_context=user_context,
            )
        )

    assert captured["config"]["configurable"]["thread_id"] == "sess-42"


@pytest.mark.asyncio
async def test_handle_message_creates_session_when_session_id_is_none():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-42")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history  # inject directly

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="hello world",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    fake_history.create_session.assert_called_once()
    create_call = fake_history.create_session.call_args
    assert create_call.kwargs["sid"] == "abc123"
    assert "hello world" in create_call.kwargs["title"]

    # User message should be persisted
    save_calls = fake_history.save_message.call_args_list
    user_calls = [c for c in save_calls if c.kwargs.get("role") == "user"]
    assert len(user_calls) == 1
    assert user_calls[0].kwargs["session"] == "sess-42"
    assert user_calls[0].kwargs["content"] == "hello world"


@pytest.mark.asyncio
async def test_handle_message_uses_provided_session_id_without_creating():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="should-not-use")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="follow up",
                session_id="existing-sess",
                context={},
                user_context=user_context,
            )
        )

    fake_history.create_session.assert_not_called()
    # User message attached to the existing session id
    user_calls = [
        c for c in fake_history.save_message.call_args_list if c.kwargs.get("role") == "user"
    ]
    assert user_calls[0].kwargs["session"] == "existing-sess"


@pytest.mark.asyncio
async def test_handle_message_continues_when_history_writes_fail():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value=None)  # Frappe down
    fake_history.save_message = AsyncMock(return_value=None)
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    # Conversation must still finish cleanly with a done event
    assert events[-1]["type"] == "done"
    # And no error event was emitted just because Frappe was down
    assert [e for e in events if e["type"] == "error"] == []


@pytest.mark.asyncio
async def test_envelope_text_then_table_emits_content_then_content_block():
    """A multi-block envelope (text + table) becomes one content event
    (for the prose) and one content_block event (for the table), in
    order. The text block at index 0 emits when the table block at
    index 1 starts in the partial dict — `iter_complete_blocks`'
    next-block-started signal."""
    llm = _make_llm(
        formatter_yields=[
            # First yield: just the text block (table not started — text
            # waits because no next-block signal yet).
            {"blocks": [{"type": "text", "payload": {"content": "Here are the users:"}}]},
            # Second yield: table block has started → text block emits.
            # Table itself still waits (it's the last block in this yield).
            {
                "blocks": [
                    {"type": "text", "payload": {"content": "Here are the users:"}},
                    {
                        "type": "table",
                        "payload": {
                            "title": "Users",
                            "columns": [{"key": "name", "label": "Name"}],
                            "rows": [{"values": {"name": "Admin"}}],
                        },
                    },
                ]
            },
            # Final yield (same shape) — flushes the last block via
            # iter_complete_blocks(final=True) inside _run_envelope_formatter.
            {
                "blocks": [
                    {"type": "text", "payload": {"content": "Here are the users:"}},
                    {
                        "type": "table",
                        "payload": {
                            "title": "Users",
                            "columns": [{"key": "name", "label": "Name"}],
                            "rows": [{"values": {"name": "Admin"}}],
                        },
                    },
                ]
            },
        ]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="list users",
                session_id="sess-1",
                context={},
                user_context=user_context,
            )
        )

    content_events = [e for e in events if e["type"] == "content"]
    block_events = [e for e in events if e["type"] == "content_block"]

    assert len(content_events) == 1
    assert content_events[0]["text"] == "Here are the users:"
    assert len(block_events) == 1
    assert block_events[0]["block"]["type"] == "table"
    assert block_events[0]["block"]["title"] == "Users"

    # Order across the whole stream: content arrives before content_block.
    content_idx = next(i for i, e in enumerate(events) if e["type"] == "content")
    block_idx = next(i for i, e in enumerate(events) if e["type"] == "content_block")
    assert content_idx < block_idx

    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_session_event_announces_created_session_id():
    """When session_id is None the service creates one and immediately
    emits a session event so the frontend can remember it."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-created")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id=None,
                context={},
                user_context=user_context,
            )
        )

    session_events = [e for e in events if e["type"] == "session"]
    assert len(session_events) == 1
    assert session_events[0]["id"] == "sess-created"


@pytest.mark.asyncio
async def test_session_event_echoes_existing_session_id():
    """When session_id is already known the service echoes it back so the
    frontend can confirm continuity (and rehydrated UI state can bind)."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="should-not-use")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="follow up",
                session_id="sess-existing",
                context={},
                user_context=user_context,
            )
        )

    fake_history.create_session.assert_not_called()
    session_events = [e for e in events if e["type"] == "session"]
    assert len(session_events) == 1
    assert session_events[0]["id"] == "sess-existing"


@pytest.mark.asyncio
async def test_envelope_text_only_response_yields_one_content_event():
    """Conversational answers come back as a single-block envelope
    `{type:"text"}` — one content event, zero content_block events."""
    llm = _make_llm(
        formatter_yields=[
            {"blocks": [{"type": "text", "payload": {"content": "Hello! How can I help?"}}]}
        ]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="sess-1",
                context={},
                user_context=user_context,
            )
        )

    content_events = [e for e in events if e["type"] == "content"]
    assert len(content_events) == 1
    assert content_events[0]["text"] == "Hello! How can I help?"
    assert [e for e in events if e["type"] == "content_block"] == []


@pytest.mark.asyncio
async def test_handle_message_surfaces_mcp_tools_timeout_as_error_event():
    """When MCP tools/list exceeds the bound, the user gets a clear error
    event rather than a wedged stream. asyncio.wait_for re-raises
    TimeoutError from the inner coroutine, which chat.py maps to a
    RuntimeError carrying "timed out"."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=TimeoutError)

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-timeout",
                context={},
                user_context=user_context,
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    msg = error_events[0]["message"]
    assert msg.startswith("RuntimeError")
    assert "timed out" in msg.lower()
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"


@pytest.mark.asyncio
async def test_handle_message_aclose_mid_stream_does_not_raise():
    """Starlette calls aclose() on the SSE generator when the client
    disconnects. The generator must not raise on cleanup — yielding from
    a `finally` clause during generator cleanup raises
    RuntimeError("async generator ignored GeneratorExit"), so the
    implementation must not use that pattern."""
    import asyncio as _asyncio

    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="s-aclose")
    fake_history.save_message = AsyncMock(return_value="m1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    async def _stream(*_a, **_k):
        yield {
            "event": "on_tool_start",
            "name": "list_documents",
            "data": {"input": {"doctype": "Customer"}},
        }
        await _asyncio.sleep(60)  # never reached — consumer drops first

    mock_graph = MagicMock()
    mock_graph.astream_events = _stream

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        agen = service.handle_message(
            message="hi",
            session_id="s-aclose",
            context={},
            user_context=user_context,
        )
        # Drive past session and the first tool_call event.
        ev1 = await agen.__anext__()
        assert ev1["type"] == "session"
        ev2 = await agen.__anext__()
        assert ev2["type"] == "tool_call"
        # Suspended in the middle of Pass-1; simulate client disconnect.
        try:
            await agen.aclose()
        except RuntimeError as e:  # pragma: no cover — only fires on regression
            pytest.fail(f"aclose raised RuntimeError: {e}")


@pytest.mark.asyncio
async def test_handle_message_pass1_exception_surfaces_as_error_event():
    """If the Pass-1 graph stream raises, the failure surfaces as an
    `error` event followed by `done` with data_quality=low. No content
    was emitted (Pass-1 prose is captured-not-streamed), so there's
    nothing to flush; the contract is still error+done."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    def _raising_stream(*_args, **_kwargs):
        async def _gen():
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="draft start")},
            }
            raise RuntimeError("stream blew up")

        return _gen()

    mock_graph = MagicMock()
    mock_graph.astream_events = _raising_stream

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-flush",
                context={},
                user_context=user_context,
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "stream blew up" in error_events[0]["message"]
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"
    # No content events were emitted — Pass-1 prose is suppressed.
    assert [e for e in events if e["type"] == "content"] == []


# ─── Turn-summary structured log ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_message_emits_turn_summary_log_on_success():
    """Every successful chat turn should emit exactly one info-level
    `chat_turn_completed` event carrying duration_ms, tools_called,
    content_chars, block_events_emitted, and failed=False. This is the
    single audit-trail entry an operator can grep for to answer
    'what happened on /api/v1/chat for this user' without reading
    three different log streams."""
    llm = _make_llm(
        formatter_yields=[{"blocks": [{"type": "text", "payload": {"content": "hello there"}}]}]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_tool_start",
                "name": "list_documents",
                "data": {"input": {"doctype": "Customer"}},
            },
        ]
    )

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
        structlog.testing.capture_logs() as logs,
    ):
        await _drain(
            service.handle_message(
                message="list customers",
                session_id="s1",
                context={},
                user_context=user_context,
            )
        )

    summaries = [r for r in logs if r.get("event") == "chat_turn_completed"]
    assert len(summaries) == 1, f"expected 1 turn-summary log, got {len(summaries)}: {logs!r}"
    s = summaries[0]
    assert s["log_level"] == "info"
    assert s["failed"] is False
    assert s["tools_called"] == ["list_documents"]
    assert s["tools_called_count"] == 1
    assert s["content_chars"] == len("hello there")
    assert s["block_events_emitted"] == 0
    assert s["session_id"] == "s1"
    assert isinstance(s["duration_ms"], int | float)
    assert s["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_handle_message_emits_turn_summary_log_on_failure():
    """Failure path must also emit a turn-summary, with failed=True and
    error_type carrying the originating exception class. Without this,
    an operator counting failed turns has to grep two events."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        structlog.testing.capture_logs() as logs,
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id="s-fail",
                context={},
                user_context=user_context,
            )
        )

    summaries = [r for r in logs if r.get("event") == "chat_turn_completed"]
    assert len(summaries) == 1
    s = summaries[0]
    assert s["failed"] is True
    assert s["error_type"] == "RuntimeError"
    assert s["tools_called"] == []
    assert s["tools_called_count"] == 0
    assert s["session_id"] == "s-fail"


@pytest.mark.asyncio
async def test_handle_message_turn_summary_counts_block_events():
    """When the LLM emits an <ai-block>, the summary records it. This
    is the signal an operator uses to ask 'are users actually seeing
    structured blocks or just prose?' over a population of turns."""
    llm = _make_llm(
        formatter_yields=[
            {
                "blocks": [
                    {
                        "type": "kpi",
                        "payload": {"metrics": [{"label": "Rev", "value": 1, "format": "number"}]},
                    }
                ]
            }
        ]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
        structlog.testing.capture_logs() as logs,
    ):
        await _drain(
            service.handle_message(
                message="kpi please",
                session_id="s-blocks",
                context={},
                user_context=user_context,
            )
        )

    summaries = [r for r in logs if r.get("event") == "chat_turn_completed"]
    assert len(summaries) == 1
    assert summaries[0]["block_events_emitted"] == 1


# ─── Custom OTEL spans ────────────────────────────────────────────────────
# `otel_spans` fixture is defined in tests/unit/conftest.py — shared
# session-scope TracerProvider with per-test exporter clearing.


@pytest.mark.asyncio
async def test_handle_message_emits_chat_turn_span(otel_spans):
    """`agent.chat_turn` wraps the whole handler. Attributes carry the
    session id, the tools-called count, the content-chars count, and
    failed=False on the happy path. This is the single span an operator
    follows in a trace UI to answer 'where did those 18 seconds go?'"""
    llm = _make_llm(
        formatter_yields=[{"blocks": [{"type": "text", "payload": {"content": "reply"}}]}]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_tool_start",
                "name": "list_documents",
                "data": {"input": {"doctype": "Customer"}},
            },
        ]
    )

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id="s-trace",
                context={},
                user_context=user_context,
            )
        )

    spans = otel_spans.get_finished_spans()
    chat_turn = [s for s in spans if s.name == "agent.chat_turn"]
    assert len(chat_turn) == 1, f"expected 1 agent.chat_turn span, got {[s.name for s in spans]}"
    attrs = dict(chat_turn[0].attributes or {})
    assert attrs["session_id"] == "s-trace"
    assert attrs["tools_called_count"] == 1
    assert attrs["content_chars"] == len("reply")
    assert attrs["failed"] is False


@pytest.mark.asyncio
async def test_handle_message_emits_load_tools_and_graph_run_spans(otel_spans):
    """The chat-turn anatomy decomposes into load_tools (MCP handshake +
    tools/list) and graph_run (the actual LLM streaming loop). Without
    these inner spans, 'why was this turn slow?' can't be answered."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[MagicMock(), MagicMock(), MagicMock()])

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
        patch("ai_agent.services.chat.install_tool_error_handler"),
    ):
        await _drain(
            service.handle_message(
                message="hi",
                session_id="s-anat",
                context={},
                user_context=user_context,
            )
        )

    span_names = [s.name for s in otel_spans.get_finished_spans()]
    assert "agent.load_tools" in span_names
    assert "agent.graph_run" in span_names

    load_tools = next(s for s in otel_spans.get_finished_spans() if s.name == "agent.load_tools")
    assert dict(load_tools.attributes or {}).get("tool_count") == 3


@pytest.mark.asyncio
async def test_handle_message_failure_marks_chat_turn_span_error(otel_spans):
    """On the failure path the chat_turn span must carry an ERROR status
    and the original exception class name as an attribute, so dashboards
    can bucket failures by class without trawling logs."""
    from opentelemetry.trace import StatusCode

    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        await _drain(
            service.handle_message(
                message="hi",
                session_id="s-err",
                context={},
                user_context=user_context,
            )
        )

    chat_turn = next(s for s in otel_spans.get_finished_spans() if s.name == "agent.chat_turn")
    assert chat_turn.status.status_code == StatusCode.ERROR
    attrs = dict(chat_turn.attributes or {})
    assert attrs["failed"] is True
    assert attrs["error_type"] == "RuntimeError"


# ─── _BlockStreamSplitter ─────────────────────────────────────────────────


def _drain_splitter(splitter: _BlockStreamSplitter, chunks: list[str]):
    out: list[tuple[str, str]] = []
    for c in chunks:
        for kind, payload in splitter.feed(c):
            out.append((kind, payload))
    for kind, payload in splitter.flush():
        out.append((kind, payload))
    return out


def test_splitter_streams_pure_prose_unchanged():
    out = _drain_splitter(
        _BlockStreamSplitter(),
        ["Hello, ", "world", "."],
    )
    assert out == [("content", "Hello, "), ("content", "world"), ("content", ".")]


def test_splitter_holds_back_partial_open_tag_suffix():
    """If a chunk ends with the start of `<ai-block`, the splitter must not
    leak the suffix as content — it could complete in the next chunk."""
    s = _BlockStreamSplitter()
    out1 = list(s.feed("hello <ai-bl"))
    assert out1 == [("content", "hello ")]
    out2 = list(s.feed('ock type="kpi">{}</ai-block>'))
    # full block markup arrived
    block_events = [(k, p) for (k, p) in out2 if k == "block"]
    assert len(block_events) == 1
    assert block_events[0][1].startswith("<ai-block")
    assert block_events[0][1].endswith("</ai-block>")


def test_splitter_buffers_block_across_many_chunks():
    chunks = [
        "Prose before. ",
        "<ai-block ",
        'type="kpi">',
        '{"metrics": [',
        '{"label": "X", "value": 1, "format": "number"}]}',
        "</ai-block>",
        " Prose after.",
    ]
    out = _drain_splitter(_BlockStreamSplitter(), chunks)
    # Prose before streams; block is one event; prose after streams.
    assert ("content", "Prose before. ") in out
    blocks = [p for (k, p) in out if k == "block"]
    assert len(blocks) == 1
    assert "<ai-block" in blocks[0] and "</ai-block>" in blocks[0]
    assert ("content", " Prose after.") in out


def test_splitter_does_not_leak_lt_that_isnt_an_ai_block():
    """Chunks containing `<` that aren't `<ai-block` must stream through."""
    out = _drain_splitter(_BlockStreamSplitter(), ["look at <p>this</p> tag"])
    assert out == [("content", "look at <p>this</p> tag")]


def test_splitter_flush_emits_residual_partial_block():
    """If the LLM cuts off mid-tag, flush emits whatever was buffered as
    content (better than silently dropping the trailing markup)."""
    s = _BlockStreamSplitter()
    out = list(s.feed('<ai-block type="kpi">{partial'))
    assert out == []  # nothing emitted while inside an unclosed block
    flushed = list(s.flush())
    assert len(flushed) == 1
    assert flushed[0][0] == "content"
    assert flushed[0][1].startswith("<ai-block")


# ─── SSE contract enforcement ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_every_emitted_event_matches_sse_contract():
    """Walk a chat turn that exercises tool_call, content, content_block,
    and done events; run validate_event on each. This is the
    regression guard for "someone added a new event field and forgot
    to update the TypedDict in transport/sse_events.py" drift."""
    from ai_agent.transport.sse_events import validate_event

    # Envelope yields text + kpi blocks so the formatter pass produces
    # both a content event (for the text block) and a content_block
    # event (for the kpi block) — exercising all five SSE kinds across
    # the full turn.
    llm = _make_llm(
        formatter_yields=[
            # First yield: text block only — text needs a next-block signal,
            # so it doesn't emit yet.
            {"blocks": [{"type": "text", "payload": {"content": "Customer count:"}}]},
            # Second yield: kpi started → text emits as content.
            {
                "blocks": [
                    {"type": "text", "payload": {"content": "Customer count:"}},
                    {
                        "type": "kpi",
                        "payload": {
                            "metrics": [{"label": "Total", "value": 42, "format": "number"}]
                        },
                    },
                ]
            },
        ]
    )
    service = _make_service(llm=llm)
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_tool_start",
                "name": "list_documents",
                "data": {"input": {"doctype": "Customer"}},
            },
        ]
    )

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="count customers",
                session_id="s-contract",
                context={},
                user_context=user_context,
            )
        )

    # Every event the service emitted must satisfy the SSE contract.
    seen_kinds = set()
    for ev in events:
        validate_event(ev)
        seen_kinds.add(ev["type"])
    assert {"session", "tool_call", "content", "content_block", "done"} <= seen_kinds, (
        f"scenario didn't exercise all expected kinds: got {seen_kinds!r}"
    )


@pytest.mark.asyncio
async def test_error_path_events_match_sse_contract():
    """The failure branch emits `error` + `done`. Both must satisfy the
    contract."""
    from ai_agent.transport.sse_events import validate_event

    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="s-err",
                context={},
                user_context=user_context,
            )
        )

    seen_kinds = set()
    for ev in events:
        validate_event(ev)
        seen_kinds.add(ev["type"])
    assert {"session", "error", "done"} <= seen_kinds


# ─── Ceiling-hit paths ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_graph_recursion_limit_surfaces_clean_error_event():
    """When the LangGraph recursion ceiling is hit (small models that
    loop without converging on schema exploration are the usual culprit),
    chat.py must emit one SSE error event with the class name in the
    message — no stack trace leakage, no half-written stream — and
    follow it with a done event carrying data_quality: low. This is
    the ceiling-hit path the AI_AGENT_AGENT_RECURSION_LIMIT env var
    was made configurable for; it must fail cleanly when hit."""
    from langgraph.errors import GraphRecursionError

    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    def _recursion_explosion(*_args, **_kwargs):
        async def _gen():
            # Emit a couple of plausible-looking tool starts before the
            # ceiling hits — matches the real shape where the agent has
            # already worked a while before tripping the limit.
            yield {
                "event": "on_tool_start",
                "name": "list_documents",
                "data": {"input": {"doctype": "Customer"}},
            }
            raise GraphRecursionError("Recursion limit of 50 reached without hitting a stop")
            yield  # pragma: no cover — unreachable but keeps the function an asyncgen

        return _gen()

    mock_graph = MagicMock()
    mock_graph.astream_events = _recursion_explosion

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="please loop forever",
                session_id="s-recursion",
                context={},
                user_context=user_context,
            )
        )

    errors = [e for e in events if e["type"] == "error"]
    assert len(errors) == 1, f"expected exactly 1 error event, got {len(errors)}: {events!r}"
    msg = errors[0]["message"]
    assert msg.startswith("GraphRecursionError"), f"error message should name the class: {msg!r}"
    assert "Recursion limit" in msg, msg
    # Stack-trace leakage check: the error message must not contain a
    # 'Traceback' marker or newlines from a frame summary. The first-line
    # cap in chat.py is what enforces this.
    assert "Traceback" not in msg
    assert "\n" not in msg
    # Terminal frame: done + low quality, regardless of failure cause.
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"
    # Tool started before the ceiling hit was recorded.
    assert events[-1]["tools_called"] == ["list_documents"]


# ─── Concurrent-sid isolation ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_concurrent_turns_get_two_distinct_mcp_clients_with_their_own_sids():
    """Two chat turns inflight with different sids must each see their
    own sid threaded into the MCP client. A regression where a per-request
    builder closed over the wrong sid would 'leak' user A's session to
    user B's tool calls — the load-bearing security property of the
    permissions-stay-in-Frappe design point.

    The interleaving uses asyncio.gather to start both turns; the
    `build_mcp_client_for_sid` stub records (sid, time_called) so we
    can assert each turn used its own sid even when the calls
    overlap in the event loop."""
    import asyncio

    service = _make_service()
    user_a = UserContext(sid="sid-a")
    user_b = UserContext(sid="sid-b")

    # Per-sid mock clients; build_mcp_client_for_sid routes by sid so
    # we can later assert the right one was used for the right turn.
    client_a = MagicMock()
    client_a.get_tools = AsyncMock(return_value=[])
    client_b = MagicMock()
    client_b.get_tools = AsyncMock(return_value=[])
    clients = {"sid-a": client_a, "sid-b": client_b}
    builder_calls: list[str] = []

    def _builder(_settings, sid):
        builder_calls.append(sid)
        return clients[sid]

    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory([])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", side_effect=_builder),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        # Run both turns concurrently and drain each.
        async def _run(uc: UserContext, session: str):
            return await _drain(
                service.handle_message(
                    message="hi",
                    session_id=session,
                    context={},
                    user_context=uc,
                )
            )

        events_a, events_b = await asyncio.gather(
            _run(user_a, "sess-a"),
            _run(user_b, "sess-b"),
        )

    # Each turn must have invoked the builder with its own sid exactly
    # once — no cross-contamination.
    assert builder_calls.count("sid-a") == 1, builder_calls
    assert builder_calls.count("sid-b") == 1, builder_calls
    # And each got its own client (proving the per-sid mock above
    # was wired correctly).
    client_a.get_tools.assert_called_once()
    client_b.get_tools.assert_called_once()
    # Both turns produced clean envelopes.
    assert events_a[0]["type"] == "session" and events_a[-1]["type"] == "done"
    assert events_b[0]["type"] == "session" and events_b[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_non_json_serialisable_tool_args_drop_silently_not_explode():
    """`tool_args_json = json.dumps(tool_invocations)` is wrapped in a
    try/except (TypeError, ValueError) so a tool whose args contain
    a non-serialisable value (e.g. a datetime that escaped formatting,
    a bytes object, a custom class instance) doesn't abort the
    final history write. Covered the previously-uncovered exception
    arm in chat.py."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    # A `bytes` value is not JSON-serialisable by the stdlib encoder and
    # raises TypeError. The on_tool_start event's input dict is forwarded
    # verbatim to tool_invocations.
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_tool_start",
                "name": "weird_tool",
                "data": {"input": {"blob": b"\x01\x02\x03"}},
            }
        ]
    )

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.create_agent_graph", return_value=mock_graph),
    ):
        events = await _drain(
            service.handle_message(
                message="invoke weird_tool",
                session_id="sess-nonserialisable",
                context={},
                user_context=user_context,
            )
        )

    # Turn must still finish cleanly with `done`; no error event despite
    # the JSON-dump exception.
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "high"
    # The assistant message write got tool_args_json=None (silently dropped).
    assistant_writes = [
        c for c in fake_history.save_message.call_args_list if c.kwargs.get("role") == "assistant"
    ]
    assert len(assistant_writes) == 1
    assert assistant_writes[0].kwargs["tool_args_json"] is None
