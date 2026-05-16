"""SSE serialization + typed event contract.

The agent emits seven event kinds on `POST /api/v1/chat`. Each is shipped
as a `data: <json>\\n\\n` SSE frame; the frontend parses each frame and
dispatches on the `type` field.

The `TypedDict` definitions below are the wire contract. The FE can copy
this file (or a generated TS equivalent) to get full type safety against
the agent's actual emissions. `validate_event` runtime-checks an event
dict against the union and raises `ValueError` on a mismatch — used by
tests to catch contract drift without paying the cost on every emit.

Six event kinds (in emission order over a turn):

- `session`       — `id: str`. First frame, always.
- `tool_call`     — `name: str`, `arguments: dict`. One per agent tool invocation.
- `content`       — `text: str`. Prose token chunks; streamed.
- `content_block` — `block: dict`. Complete parsed structured-block payload.
- `error`         — `message: str`. Fatal; followed by `done`.
- `done`          — `tools_called: list[str]`, `data_quality`, `timestamp: str`.
  Terminal frame; always last.
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict


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


class ErrorEvent(TypedDict):
    type: Literal["error"]
    message: str


class DoneEvent(TypedDict):
    type: Literal["done"]
    tools_called: list[str]
    data_quality: Literal["high", "low"]
    timestamp: str


SSEEvent = SessionEvent | ToolCallEvent | ContentEvent | ContentBlockEvent | ErrorEvent | DoneEvent

# Runtime-check schema: type → set of required field names (excluding "type"
# itself). Kept as a plain dict so it's introspectable from tests and from
# the FE if it ever needs to mirror this validation client-side.
_REQUIRED_FIELDS: dict[str, set[str]] = {
    "session": {"id"},
    "tool_call": {"name", "arguments"},
    "content": {"text"},
    "content_block": {"block"},
    "error": {"message"},
    "done": {"tools_called", "data_quality", "timestamp"},
}


def validate_event(event: dict[str, Any]) -> None:
    """Raise ValueError if `event` doesn't match the SSEEvent contract.

    Checks (in order): the `type` field is present, it's a known kind,
    and every required field for that kind is present. The exact value
    types are not deep-checked here — the typed TypedDict declaration
    above is the source of truth for that, and pyright enforces it at
    the emit sites.

    Intended for use in tests asserting that ChatService's emissions
    match the contract; not called from the hot path.
    """
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
