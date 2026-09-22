"""The SSE event contract: `session` first, `done` last, and every `error` followed by `done`."""

from __future__ import annotations

import json
from typing import Any, Literal, NotRequired, TypedDict


class SessionEvent(TypedDict):
    type: Literal["session"]
    id: str


class ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    name: str
    arguments: dict[str, Any]


class ContentEvent(TypedDict):
    type: Literal["content"]
    text: str


class ContentBlockEvent(TypedDict):
    type: Literal["content_block"]
    block: dict[str, Any]


class SourcesEvent(TypedDict):
    type: Literal["sources"]
    items: list[dict[str, Any]]


class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


class DoneEvent(TypedDict):
    type: Literal["done"]
    tools_called: list[str]
    data_quality: Literal["high", "low"]
    timestamp: str
    # Ollama's output_tokens and output_seconds, when the model reports them, and
    # first_token_s: the seconds from the question to the first answer text
    usage: NotRequired[dict[str, float]]


SSEEvent = (
    SessionEvent
    | ToolCallEvent
    | SourcesEvent
    | ContentEvent
    | ContentBlockEvent
    | ErrorEvent
    | DoneEvent
)

# Runtime-check schema: type → set of required field names (excluding "type"
# itself). Kept as a plain dict so it's introspectable from tests and from
# the FE if it ever needs to mirror this validation client-side.
_REQUIRED_FIELDS: dict[str, set[str]] = {
    "session": {"id"},
    "tool_call": {"name", "arguments"},
    "content": {"text"},
    "content_block": {"block"},
    "sources": {"items"},
    "error": {"message"},
    "done": {"tools_called", "data_quality", "timestamp"},
}


def validate_event(event: dict[str, Any]) -> None:
    """Raise ValueError unless `event` has a known `type` and its fields; values are unchecked."""
    if "type" not in event:
        raise ValueError(f"event missing required 'type' field: {event!r}")
    kind = event["type"]
    if kind not in _REQUIRED_FIELDS:
        raise ValueError(f"unknown event type {kind!r}; expected one of {sorted(_REQUIRED_FIELDS)}")
    missing = _REQUIRED_FIELDS[kind] - set(event)
    if missing:
        raise ValueError(
            f"event of type {kind!r} missing required fields {sorted(missing)}: {event!r}"
        )


Event = dict[str, Any]


def serialize(event: Event) -> bytes:
    """Encode an event dict as an SSE data line."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
