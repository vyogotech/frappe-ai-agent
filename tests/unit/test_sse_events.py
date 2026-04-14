from ai_agent.transport.sse_events import (
    content_event,
    done_event,
    error_event,
    serialize,
    status_event,
    tool_call_event,
)


def test_status_event_serialization():
    e = status_event("thinking...")
    assert serialize(e) == b'data: {"type":"status","message":"thinking..."}\n\n'


def test_content_event_serialization():
    e = content_event("Hello")
    assert serialize(e) == b'data: {"type":"content","text":"Hello"}\n\n'


def test_tool_call_event_serialization():
    e = tool_call_event("list_invoices", {"status": "unpaid"})
    expected = b'data: {"type":"tool_call","name":"list_invoices","arguments":{"status":"unpaid"}}\n\n'
    assert serialize(e) == expected


def test_done_event_serialization():
    e = done_event(
        tools_called=["list_invoices"],
        data_quality="high",
        timestamp="2026-04-14T00:00:00Z",
    )
    expected = (
        b'data: {"type":"done","tools_called":["list_invoices"],'
        b'"data_quality":"high","timestamp":"2026-04-14T00:00:00Z"}\n\n'
    )
    assert serialize(e) == expected


def test_error_event_serialization():
    e = error_event("Ollama is unreachable")
    assert serialize(e) == b'data: {"type":"error","message":"Ollama is unreachable"}\n\n'
