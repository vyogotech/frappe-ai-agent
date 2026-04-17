from ai_agent.transport.sse_events import serialize


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
