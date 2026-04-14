"""SSE event helpers matching the frappe-mcp-server schema byte-for-byte."""

from __future__ import annotations

import json
from typing import Any

Event = dict[str, Any]


def status_event(message: str) -> Event:
    return {"type": "status", "message": message}


def tool_call_event(name: str, arguments: dict[str, Any]) -> Event:
    return {"type": "tool_call", "name": name, "arguments": arguments}


def content_event(text: str) -> Event:
    return {"type": "content", "text": text}


def done_event(tools_called: list[str], data_quality: str, timestamp: str) -> Event:
    return {
        "type": "done",
        "tools_called": tools_called,
        "data_quality": data_quality,
        "timestamp": timestamp,
    }


def error_event(message: str) -> Event:
    return {"type": "error", "message": message}


def serialize(event: Event) -> bytes:
    # Python json.dumps does not HTML-escape '<', '>', '&' the way Go's
    # json.Marshal does. Byte-level consumers that round-trip through a Go
    # server on the same schema may see diverging escapes for these chars.
    # Decoded JSON objects are identical either way.
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
