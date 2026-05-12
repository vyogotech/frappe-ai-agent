"""SSE chat transport — POST /api/v1/chat streaming text/event-stream."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter

from ai_agent.middleware.sid import UserContext, extract_user_context
from ai_agent.transport.sse_events import serialize


def _require_sid(request: Request) -> UserContext:
    """Why: FastAPI Depends() runs before the @limiter.limit decorator's
    rate-limit check, so unauthenticated callers 401 without consuming a
    token from the (IP-keyed) bucket — preventing one bad actor from
    locking out a shared NAT.
    """
    user_context = extract_user_context(request)
    if user_context is None:
        raise HTTPException(status_code=401, detail="Missing sid cookie")
    return user_context


# Why: `context` is forwarded into the system prompt; an unbounded dict lets
# a misbehaving frontend (or a malicious caller) blow up token usage per
# request. 8 KB is enough for page context (doctype/docname/route/currency
# and a few extras) while keeping the prompt budget predictable.
_MAX_CONTEXT_BYTES = 8 * 1024


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=32_000)
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message")
    @classmethod
    def _reject_whitespace_only(cls, v: str) -> str:
        # `min_length=1` alone accepts "   "; strip-check rejects it without
        # mutating the value (so the LLM sees exactly what the user typed).
        if not v.strip():
            raise ValueError("message must not be whitespace-only")
        return v

    @field_validator("context")
    @classmethod
    def _cap_context_size(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(v).encode()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"context must be JSON-serialisable: {exc}") from exc
        if len(encoded) > _MAX_CONTEXT_BYTES:
            raise ValueError(
                f"context exceeds {_MAX_CONTEXT_BYTES} bytes (got {len(encoded)} bytes serialised)"
            )
        return v


def _noop_limit(_fn: Callable) -> Callable:
    """Identity decorator used when no limiter is provided (BDD tests)."""
    return _fn


def create_sse_router(
    *,
    limiter: Limiter | None = None,
    rate_limit: str = "30/minute",
) -> APIRouter:
    """Build the SSE chat router.

    `limiter` is optional so test bootstrap code (BDD scenarios that don't
    go through `create_app`) can wire the router directly. In production
    `create_app` always passes a real limiter.
    """
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
