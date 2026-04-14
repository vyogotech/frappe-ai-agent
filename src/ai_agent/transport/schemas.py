"""Transport-specific request schemas and re-exports of shared event schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ai_agent.schemas import (
    AckEvent,
    ContentBlockEvent,
    DoneEvent,
    ErrorEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)

# Re-export for backward compatibility
__all__ = [
    "AckEvent",
    "ChatMessage",
    "ContentBlockEvent",
    "DoneEvent",
    "ErrorEvent",
    "TokenEvent",
    "ToolEndEvent",
    "ToolStartEvent",
]


class ChatMessage(BaseModel):
    """Incoming chat message from WebSocket client."""

    type: Literal["chat"] = "chat"
    content: str
    context: dict[str, Any] = Field(default_factory=dict)
    session_id: str
    request_id: str
