"""Tests for the per-request ChatService shape (Phase 6b)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService


def _make_settings() -> Settings:
    return Settings(
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
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client) as mock_builder,
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
async def test_handle_message_yields_status_then_done_envelope():
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

    assert len(events) >= 2
    assert events[0]["type"] == "status"
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

    final_msg = AIMessage(content="hello there")
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [
            {
                "event": "on_chat_model_end",
                "data": {"output": final_msg},
            }
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
    assert len(content_events) == 1
    assert content_events[0]["text"] == "hello there"


@pytest.mark.asyncio
async def test_handle_message_ignores_ai_message_with_tool_calls():
    """Intermediate AI messages that only carry tool_calls should not surface as content."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    intermediate = AIMessage(
        content="",
        tool_calls=[{"id": "1", "name": "list_documents", "args": {}}],
    )
    mock_graph = MagicMock()
    mock_graph.astream_events = _StreamFactory(
        [{"event": "on_chat_model_end", "data": {"output": intermediate}}]
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
    assert "mcp down" in error_events[0]["message"]
    # Generator still emits a terminal `done` even on failure, with low quality.
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"


@pytest.mark.asyncio
async def test_handle_message_sets_handle_tool_error_on_each_tool():
    """Every tool returned from MCP must have handle_tool_error installed
    before being handed to the graph factory."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    # Build a couple of fake tools that have the handle_tool_error attribute
    tool_a = MagicMock()
    tool_a.handle_tool_error = None
    tool_b = MagicMock()
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

    # Both tools must now have the error handler set
    from ai_agent.agent.tool_errors import to_tool_result_message

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
