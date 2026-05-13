import pytest

from ai_agent.transport.sse_events import serialize, validate_event


def test_serialize_status_event():
    event = {"type": "status", "message": "thinking..."}
    assert serialize(event) == b'data: {"type":"status","message":"thinking..."}\n\n'


def test_serialize_content_event():
    event = {"type": "content", "text": "Hello"}
    assert serialize(event) == b'data: {"type":"content","text":"Hello"}\n\n'


def test_serialize_tool_call_event():
    event = {"type": "tool_call", "name": "list_invoices", "arguments": {"status": "unpaid"}}
    expected = (
        b'data: {"type":"tool_call","name":"list_invoices","arguments":{"status":"unpaid"}}\n\n'
    )
    assert serialize(event) == expected


def test_serialize_done_event():
    event = {
        "type": "done",
        "tools_called": ["list_invoices"],
        "data_quality": "high",
        "timestamp": "2026-04-14T00:00:00Z",
    }
    expected = (
        b'data: {"type":"done","tools_called":["list_invoices"],'
        b'"data_quality":"high","timestamp":"2026-04-14T00:00:00Z"}\n\n'
    )
    assert serialize(event) == expected


def test_serialize_error_event():
    event = {"type": "error", "message": "Ollama is unreachable"}
    assert serialize(event) == b'data: {"type":"error","message":"Ollama is unreachable"}\n\n'


def test_serialize_session_event():
    event = {"type": "session", "id": "sess-42"}
    assert serialize(event) == b'data: {"type":"session","id":"sess-42"}\n\n'


# ─── Typed event contract ────────────────────────────────────────────────


class TestValidateEvent:
    """`validate_event` is the runtime check that backs the SSEEvent
    TypedDict contract. Tests cover every accepted shape plus the
    common drift modes."""

    def test_accepts_all_seven_event_kinds(self):
        # Each kind, fully populated. Adding a new event type requires
        # updating this list AND the TypedDict above.
        valid_events = [
            {"type": "session", "id": "sess-1"},
            {"type": "status", "message": "thinking"},
            {"type": "tool_call", "name": "list_invoices", "arguments": {"status": "unpaid"}},
            {"type": "content", "text": "hello"},
            {"type": "content_block", "block": {"type": "kpi", "metrics": []}},
            {"type": "error", "message": "Ollama unreachable"},
            {
                "type": "done",
                "tools_called": ["list_invoices"],
                "data_quality": "high",
                "timestamp": "2026-04-14T00:00:00Z",
            },
        ]
        for ev in valid_events:
            validate_event(ev)  # must not raise

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing required 'type'"):
            validate_event({"id": "sess-1"})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown event type"):
            validate_event({"type": "thinking", "message": "foo"})

    def test_missing_required_field_raises(self):
        with pytest.raises(ValueError, match="missing required fields"):
            validate_event({"type": "tool_call", "name": "list_invoices"})

    def test_done_without_timestamp_raises(self):
        with pytest.raises(ValueError, match="timestamp"):
            validate_event({"type": "done", "tools_called": [], "data_quality": "high"})
