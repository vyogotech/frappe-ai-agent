"""The SSE event contract: `session` first, `done` last, and every `error` followed by `done`."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import ConfigDict, Field, TypeAdapter


class SessionEvent(TypedDict):
    type: Literal["session"]
    id: str


class ToolCallEvent(TypedDict):
    type: Literal["tool_call"]
    name: str
    arguments: dict[str, Any]


class ToolConfirmEvent(TypedDict):
    """A write the user has to allow; the turn ends on it and the tool has not run."""

    type: Literal["tool_confirm"]
    id: str
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


# Tagged on `type`, so a bad frame is one error naming the field that is wrong instead of one per
# branch, and the published schema carries the tag-to-frame mapping a consumer's switch mirrors.
SSEEvent = Annotated[
    SessionEvent
    | ToolCallEvent
    | ToolConfirmEvent
    | SourcesEvent
    | ContentEvent
    | ContentBlockEvent
    | ErrorEvent
    | DoneEvent,
    Field(discriminator="type"),
]

# extra=forbid: a field this module does not declare is drift, not an extension — a consumer's
# own copy of the contract cannot see it, so the schema refuses it instead of dropping it.
_CONTRACT_ADAPTER = TypeAdapter(SSEEvent, config=ConfigDict(extra="forbid"))


# `make contract` writes this to contract/sse-event.schema.json, the published artefact frappe_ai
# and Metis check their own hand-written copies of these frames against (ADR-004).
def contract_schema() -> dict[str, Any]:
    """The envelope as JSON Schema: what `make contract` publishes and the consumers check."""
    return _CONTRACT_ADAPTER.json_schema()


def validate_event(event: dict[str, Any]) -> None:
    """Raise ValueError unless `event` is exactly one declared frame, every field and no other."""
    _CONTRACT_ADAPTER.validate_python(event)


Event = dict[str, Any]


def serialize(event: Event) -> bytes:
    """Encode an event dict as an SSE data line."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
