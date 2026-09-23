"""SSE chat transport — POST /api/v1/chat streaming text/event-stream."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator
from slowapi import Limiter

# tests override the agent's auth by this name; keep it importable from here
from ai_agent.middleware.sid import UserContext
from ai_agent.middleware.sid import require_sid as _require_sid
from ai_agent.transport.sse_events import serialize

# context goes into the system prompt, so this caps the tokens one request can spend.
_MAX_CONTEXT_BYTES = 8 * 1024


def _capped(value: dict[str, Any], field: str) -> dict[str, Any]:
    """Reject a JSON object over the prompt's byte budget: one request cannot fill the context."""
    try:
        encoded = json.dumps(value).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON-serialisable: {exc}") from exc
    if len(encoded) > _MAX_CONTEXT_BYTES:
        raise ValueError(
            f"{field} exceeds {_MAX_CONTEXT_BYTES} bytes (got {len(encoded)} bytes serialised)"
        )
    return value


class Confirmation(BaseModel):
    """A write the user allowed: the call frappe_ai recorded, and the one-time token for it."""

    tool: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any] = Field(default_factory=dict)
    token: str = Field(min_length=1, max_length=512)

    @field_validator("arguments")
    @classmethod
    def _cap_arguments_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _capped(v, "confirmation.arguments")


class ChatRequest(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=32_000)] | None = None
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    confirmation: Confirmation | None = None

    @field_validator("message")
    @classmethod
    def _reject_whitespace_only(cls, v: str | None) -> str | None:
        # `min_length=1` alone accepts "   "; strip-check rejects it without
        # mutating the value (so the LLM sees exactly what the user typed).
        if v is not None and not v.strip():
            raise ValueError("message must not be whitespace-only")
        return v

    @field_validator("context")
    @classmethod
    def _cap_context_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _capped(v, "context")

    @model_validator(mode="after")
    def _one_of_message_or_confirmation(self) -> ChatRequest:
        if (self.message is None) == (self.confirmation is None):
            raise ValueError("send either message or confirmation, not both and not neither")
        return self


def _noop_limit(_fn: Callable) -> Callable:
    """Identity decorator used when no limiter is provided (BDD tests)."""
    return _fn


def create_sse_router(
    *,
    limiter: Limiter | None = None,
    rate_limit: str = "30/minute",
) -> APIRouter:
    """Build the SSE chat router; `limiter` is optional only for tests that skip create_app."""
    router = APIRouter()
    limit = limiter.limit(rate_limit) if limiter is not None else _noop_limit

    @router.post("/api/v1/chat")
    @limit
    async def chat(
        request: Request,
        body: ChatRequest,
        user_context: Annotated[UserContext, Depends(_require_sid)],
    ):
        # `request` (not `req`) is required by slowapi's Limiter.limit
        # decorator: it inspects the wrapped function's signature for a
        # parameter named exactly `request`.
        chat_service = request.app.state.chat_service

        async def event_stream():
            async for event in chat_service.handle_message(
                message=body.message,
                session_id=body.session_id,
                context=body.context,
                user_context=user_context,
                confirmation=body.confirmation.model_dump() if body.confirmation else None,
            ):
                yield serialize(event)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
