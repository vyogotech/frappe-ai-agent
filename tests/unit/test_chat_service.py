"""Tests for the per-request ChatService shape (Phase 6b)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService, _BlockStreamSplitter


def _make_settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
    )


def _make_service() -> ChatService:
    return ChatService(
        settings=_make_settings(),
        llm=MagicMock(),
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
async def test_handle_message_translates_final_llm_message_to_content_event():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    # Token-by-token streaming via on_chat_model_stream — concatenated, the
    # chunks form the final assistant message ("hello there").
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="hello ")},
            },
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="there")},
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
                session_id="s2",
                context={},
                user_context=user_context,
            )
        )

    content_events = [e for e in events if e["type"] == "content"]
    # One event per streamed chunk; concatenated they form the final reply.
    assert len(content_events) == 2
    assert "".join(e["text"] for e in content_events) == "hello there"


@pytest.mark.asyncio
async def test_handle_message_ignores_ai_message_with_tool_calls():
    """Intermediate AI messages that only carry tool_calls should not surface as content."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    # The streaming counterpart to AIMessage.tool_calls is
    # AIMessageChunk.tool_call_chunks; chunks of a tool-calling step carry
    # this and must not surface as user-visible content.
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
async def test_handle_message_yields_error_on_exception_and_still_finishes_with_done():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
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
    # Message is prefixed with the exception class and carries the detail
    # (first line, truncated) so developers can actually debug.
    msg = error_events[0]["message"]
    assert msg.startswith("RuntimeError")
    assert "mcp down" in msg
    # Generator still emits a terminal `done` even on failure, with low quality.
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"


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
async def test_content_with_ai_block_emits_content_block_events():
    """When the LLM's final message contains <ai-block> tags, the
    service must emit one content_block event per block (including text
    blocks for prose between tags), preserving order. A plain `content`
    event is NOT emitted in this case."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    final_text = (
        "Here are the users:\n"
        '<ai-block type="table">'
        '{"title": "Users", "columns": [{"key": "name", "label": "Name"}], '
        '"rows": [{"values": {"name": "Admin"}}]}'
        "</ai-block>\n"
        "That's 1 user."
    )

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    # In real streaming the model emits many small chunks; for this test
    # we deliver the entire <ai-block>-tagged response as one chunk so
    # block parsing has the full text in a single content event.
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=final_text)},
            }
        ]
    )

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.save_message = AsyncMock(return_value="msg-1")
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

    # With the BlockStreamSplitter, prose around the block streams as
    # content events; only the structured `<ai-block>` becomes a
    # content_block. The FE renders prose (via markdown) above the block.
    content_events = [e for e in events if e["type"] == "content"]
    block_events = [e for e in events if e["type"] == "content_block"]

    assert len(block_events) == 1
    assert block_events[0]["block"]["type"] == "table"
    assert block_events[0]["block"]["title"] == "Users"

    # Prose-before and prose-after the block both arrive as content.
    streamed_text = "".join(e["text"] for e in content_events)
    assert "Here are the users:" in streamed_text
    assert "That's 1 user." in streamed_text

    # Stream still terminates with done
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
async def test_content_without_blocks_keeps_single_content_event():
    """Plain-text responses (no <ai-block> tags) still emit exactly
    one `content` event — block parsing is opt-in by the LLM's output."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])
    # Single-chunk stream — the whole reply arrives in one chunk, so we
    # get exactly one content event (matches the test's assertion below).
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content="Hello! How can I help?")},
            }
        ]
    )

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.save_message = AsyncMock(return_value="msg-1")
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
async def test_handle_message_flushes_splitter_when_stream_raises_mid_block():
    """If the graph stream raises while the splitter is buffering partial
    block markup, the buffered text must still reach the FE (as content)
    before the error event — flush is in a finally for this reason."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    def _raising_stream(*_args, **_kwargs):
        async def _gen():
            # Drive the splitter into in_block state with a partial tag,
            # then raise mid-stream.
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content='<ai-block type="kpi">{partial')},
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

    content_texts = [e["text"] for e in events if e["type"] == "content"]
    assert any("<ai-block" in t for t in content_texts), (
        "splitter buffer was dropped on exception path"
    )
    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "stream blew up" in error_events[0]["message"]
    assert events[-1]["type"] == "done"


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
