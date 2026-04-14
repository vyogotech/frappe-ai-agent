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


def content_block_event(block: dict[str, Any]) -> Event:
    """A structured block extracted from the LLM's response.

    Emitted for each <copilot-block> tag the LLM produces, plus a text
    block for each plain-text segment between them. Clients push these
    into message.blocks in arrival order, preserving interleaving.
    """
    return {"type": "content_block", "block": block}


def session_event(session_id: str) -> Event:
    """Announce the session id for the current conversation.

    Emitted once at the very start of every request, after the server
    resolves (or creates) the session. Clients store the id so that
    subsequent messages in the same conversation can pass it back in
    the request body — without this round-trip every message would
    create a brand-new AI Chat Session row.
    """
    return {"type": "session", "id": session_id}


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
