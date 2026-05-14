"""Tests for ChatService against the unified-loop architecture.

ChatService orchestrates: history session, MCP tool load, agent loop,
error translation, span emission, history persistence, turn summary.
The agent loop itself (`ai_agent.agent.loop.run_agent_loop`) is mocked
via a fake async-generator helper so these tests exercise only the
orchestration. Loop behaviour has its own test file.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import structlog

from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService


def _make_settings() -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
    )


def _make_service() -> ChatService:
    return ChatService(
        settings=_make_settings(),
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "you are helpful",
    )


async def _drain(agen) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for ev in agen:
        out.append(ev)
    return out


def _loop_factory(events: list[dict[str, Any]]):
    """Build a stand-in for `run_agent_loop` that yields the given events.

    Matches the production signature: an async generator that yields
    SSE-schema dicts (tool_call / content / content_block).
    """

    def _factory(**_kwargs):
        async def _gen():
            for ev in events:
                yield ev

        return _gen()

    return _factory


# --------------------------------------------------------------------------- #
# MCP boundary
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_builds_mcp_client_with_caller_sid():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch(
            "ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client
        ) as mock_builder,
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
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
    args, _kwargs = mock_builder.call_args
    assert args[1] == "abc123"


@pytest.mark.asyncio
async def test_handle_message_surfaces_tool_load_failure_as_tools_unavailable():
    """Any error during MCP tool loading (other than TimeoutError) is
    surfaced as a clean 'Tools unavailable' message. The underlying
    cause stays in a single-line warning log — no traceback."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with (
        structlog.testing.capture_logs() as logs,
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
    ):
        events = await _drain(
            service.handle_message(
                message="hi", session_id=None, context={}, user_context=user_context
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "Tools unavailable" in error_events[0]["message"]
    assert "RuntimeError" not in error_events[0]["message"]
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"

    warns = [e for e in logs if e["event"] == "chat_tools_load_failed"]
    assert len(warns) == 1
    assert warns[0]["log_level"] == "warning"
    assert warns[0]["error_type"] == "RuntimeError"
    assert warns[0]["error"] == "mcp down"
    assert not any(e["event"] == "chat_handle_message_failed" for e in logs)


@pytest.mark.asyncio
async def test_tool_load_warning_unwraps_exception_group_to_root_cause():
    """anyio's TaskGroup wraps the real MCP error in BaseExceptionGroup
    one or more layers deep. The warning log must surface the root cause."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    inner = RuntimeError("Session terminated")
    group = ExceptionGroup("anyio TaskGroup", [inner])

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=group)

    with (
        structlog.testing.capture_logs() as logs,
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
    ):
        await _drain(
            service.handle_message(
                message="hi", session_id=None, context={}, user_context=user_context
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
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=_http_status_error(status))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi", session_id="s-auth", context={}, user_context=user_context
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "authentication failed" in error_events[0]["message"]
    assert "cannot reach" not in error_events[0]["message"]


@pytest.mark.asyncio
async def test_tool_load_transport_error_yields_unreachable_message():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(
        side_effect=httpx.ConnectError("All connection attempts failed")
    )

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi", session_id="s-conn", context={}, user_context=user_context
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "cannot reach" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_tool_load_unknown_http_status_yields_status_code_in_message():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=_http_status_error(500))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi", session_id="s-500", context={}, user_context=user_context
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "HTTP 500" in error_events[0]["message"]


@pytest.mark.asyncio
async def test_handle_message_surfaces_mcp_tools_timeout_as_error_event():
    """When MCP tools/list exceeds the bound, the user gets a clear
    error event with the timeout cause."""
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()

    async def _hang(*_a, **_k):
        import asyncio as _asyncio

        await _asyncio.sleep(60)
        return []

    mock_client.get_tools = _hang

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat._MCP_TOOLS_LOAD_TIMEOUT_S", 0.05),
    ):
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
    assert "timed out" in error_events[0]["message"].lower()
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"


# --------------------------------------------------------------------------- #
# Session events
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_yields_session_then_done_envelope():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
    ):
        events = await _drain(
            service.handle_message(
                message="hi", session_id=None, context={}, user_context=user_context
            )
        )

    assert events[0]["type"] == "session"
    assert events[-1]["type"] == "done"
    assert events[-1]["tools_called"] == []
    assert events[-1]["timestamp"].endswith("Z")
    assert events[-1]["data_quality"] == "high"


@pytest.mark.asyncio
async def test_session_event_announces_created_session_id():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-created")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
    ):
        events = await _drain(
            service.handle_message(
                message="hi", session_id=None, context={}, user_context=user_context
            )
        )

    session_events = [e for e in events if e["type"] == "session"]
    assert len(session_events) == 1
    assert session_events[0]["id"] == "sess-created"


@pytest.mark.asyncio
async def test_session_event_echoes_existing_session_id():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="should-not-use")
    fake_history.save_message = AsyncMock(return_value="msg-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
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


# --------------------------------------------------------------------------- #
# Loop integration — tool_call / content / content_block surface through
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_loop_tool_call_surfaces_as_tool_call_event():
    service = _make_service()
    user_context = UserContext(sid="abc123")
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    loop_events = [
        {"type": "tool_call", "name": "list_documents", "arguments": {"doctype": "Customer"}},
        {"type": "content", "text": "Found 3 customers."},
    ]
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory(loop_events)),
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
async def test_loop_content_event_passes_through():
    service = _make_service()
    user_context = UserContext(sid="abc123")
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    loop_events = [{"type": "content", "text": "Hello there"}]
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory(loop_events)),
    ):
        events = await _drain(
            service.handle_message(
                message="hi", session_id="s2", context={}, user_context=user_context
            )
        )

    content = [e for e in events if e["type"] == "content"]
    assert len(content) == 1
    assert content[0]["text"] == "Hello there"


@pytest.mark.asyncio
async def test_loop_content_block_event_passes_through():
    service = _make_service()
    user_context = UserContext(sid="abc123")
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    block_event = {
        "type": "content_block",
        "block": {
            "type": "kpi",
            "metrics": [{"label": "Revenue", "value": 100, "format": "number"}],
        },
    }
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([block_event])),
    ):
        events = await _drain(
            service.handle_message(
                message="kpi please",
                session_id="s3",
                context={},
                user_context=user_context,
            )
        )

    blocks = [e for e in events if e["type"] == "content_block"]
    assert len(blocks) == 1
    assert blocks[0]["block"]["type"] == "kpi"


@pytest.mark.asyncio
async def test_loop_exception_surfaces_as_error_event():
    service = _make_service()
    user_context = UserContext(sid="abc123")
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    def _raising_loop(**_kwargs):
        async def _gen():
            yield {"type": "tool_call", "name": "x", "arguments": {}}
            raise RuntimeError("loop blew up")

        return _gen()

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _raising_loop),
    ):
        events = await _drain(
            service.handle_message(
                message="hi", session_id="s-err", context={}, user_context=user_context
            )
        )

    error_events = [e for e in events if e["type"] == "error"]
    assert len(error_events) == 1
    assert "loop blew up" in error_events[0]["message"]
    assert events[-1]["type"] == "done"
    assert events[-1]["data_quality"] == "low"


# --------------------------------------------------------------------------- #
# History persistence
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_persists_user_message_before_loop():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    fake_history.save_message = AsyncMock(return_value="msg-1")
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
    ):
        await _drain(
            service.handle_message(
                message="hello",
                session_id="sess-1",
                context={},
                user_context=user_context,
            )
        )

    calls = fake_history.save_message.call_args_list
    assert len(calls) == 2  # user + assistant
    assert calls[0].kwargs["role"] == "user"
    assert calls[0].kwargs["content"] == "hello"
    assert calls[1].kwargs["role"] == "assistant"


@pytest.mark.asyncio
async def test_handle_message_continues_when_history_writes_fail():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    fake_history = MagicMock()
    fake_history.create_session = AsyncMock(return_value="sess-1")
    fake_history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    fake_history.save_message = AsyncMock(side_effect=RuntimeError("frappe down"))
    service._history = fake_history

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
    ):
        events = await _drain(
            service.handle_message(
                message="hi",
                session_id="sess-1",
                context={},
                user_context=user_context,
            )
        )

    # Stream finishes despite history write failures.
    assert events[-1]["type"] == "done"
    assert [e for e in events if e["type"] == "error"] == []


# --------------------------------------------------------------------------- #
# Turn-summary log
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_emits_turn_summary_log_on_success():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    loop_events = [
        {"type": "tool_call", "name": "list_documents", "arguments": {"doctype": "Customer"}},
        {"type": "content", "text": "hello there"},
    ]
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory(loop_events)),
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
    assert len(summaries) == 1
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
    assert s["session_id"] == "s-fail"


@pytest.mark.asyncio
async def test_handle_message_turn_summary_counts_block_events():
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    loop_events = [
        {
            "type": "content_block",
            "block": {
                "type": "kpi",
                "metrics": [{"label": "Rev", "value": 1, "format": "number"}],
            },
        },
    ]
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory(loop_events)),
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


# --------------------------------------------------------------------------- #
# OTEL spans
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_emits_chat_turn_span(otel_spans):
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    loop_events = [
        {"type": "tool_call", "name": "list_documents", "arguments": {"doctype": "Customer"}},
        {"type": "content", "text": "reply"},
    ]
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory(loop_events)),
    ):
        await _drain(
            service.handle_message(
                message="hi", session_id="s-trace", context={}, user_context=user_context
            )
        )

    spans = otel_spans.get_finished_spans()
    chat_turn = [s for s in spans if s.name == "agent.chat_turn"]
    assert len(chat_turn) == 1
    attrs = dict(chat_turn[0].attributes or {})
    assert attrs["session_id"] == "s-trace"
    assert attrs["tools_called_count"] == 1
    assert attrs["content_chars"] == len("reply")
    assert attrs["failed"] is False


@pytest.mark.asyncio
async def test_handle_message_emits_load_tools_and_run_spans(otel_spans):
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory([])),
    ):
        await _drain(
            service.handle_message(
                message="hi", session_id="s-anat", context={}, user_context=user_context
            )
        )

    span_names = [s.name for s in otel_spans.get_finished_spans()]
    assert "agent.load_tools" in span_names
    assert "agent.run" in span_names


@pytest.mark.asyncio
async def test_handle_message_failure_marks_chat_turn_span_error(otel_spans):
    service = _make_service()
    user_context = UserContext(sid="abc123")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        await _drain(
            service.handle_message(
                message="hi", session_id="s-fail", context={}, user_context=user_context
            )
        )

    spans = otel_spans.get_finished_spans()
    chat_turn = [s for s in spans if s.name == "agent.chat_turn"]
    assert len(chat_turn) == 1
    assert chat_turn[0].status.status_code.name == "ERROR"


# --------------------------------------------------------------------------- #
# SSE contract
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_every_emitted_event_matches_sse_contract():
    """All event kinds the service emits must satisfy the SSE contract."""
    from ai_agent.transport.sse_events import validate_event

    service = _make_service()
    user_context = UserContext(sid="abc123")
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[])

    loop_events = [
        {"type": "tool_call", "name": "list_documents", "arguments": {"doctype": "Customer"}},
        {"type": "content", "text": "Customer count:"},
        {
            "type": "content_block",
            "block": {
                "type": "kpi",
                "metrics": [{"label": "Total", "value": 42, "format": "number"}],
            },
        },
    ]
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _loop_factory(loop_events)),
    ):
        events = await _drain(
            service.handle_message(
                message="count customers",
                session_id="s-contract",
                context={},
                user_context=user_context,
            )
        )

    seen_kinds = set()
    for ev in events:
        validate_event(ev)
        seen_kinds.add(ev["type"])
    assert {"session", "tool_call", "content", "content_block", "done"} <= seen_kinds


@pytest.mark.asyncio
async def test_error_path_events_match_sse_contract():
    from ai_agent.transport.sse_events import validate_event

    service = _make_service()
    user_context = UserContext(sid="abc123")
    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=RuntimeError("mcp down"))

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client):
        events = await _drain(
            service.handle_message(
                message="hi", session_id="s-err", context={}, user_context=user_context
            )
        )

    seen_kinds = set()
    for ev in events:
        validate_event(ev)
        seen_kinds.add(ev["type"])
    assert {"session", "error", "done"} <= seen_kinds


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_handle_message_aclose_mid_stream_does_not_raise():
    """Starlette calls aclose() on the SSE generator on client disconnect.
    The generator must not yield from a `finally` clause."""
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

    def _slow_loop(**_kwargs):
        async def _gen():
            yield {"type": "tool_call", "name": "list_documents", "arguments": {}}
            await _asyncio.sleep(60)  # never reached — consumer drops first

        return _gen()

    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mock_client),
        patch("ai_agent.services.chat.run_agent_loop", _slow_loop),
    ):
        agen = service.handle_message(
            message="hi",
            session_id="s-aclose",
            context={},
            user_context=user_context,
        )
        ev1 = await agen.__anext__()
        assert ev1["type"] == "session"
        ev2 = await agen.__anext__()
        assert ev2["type"] == "tool_call"
        try:
            await agen.aclose()
        except RuntimeError as e:  # pragma: no cover
            pytest.fail(f"aclose raised: {e}")
