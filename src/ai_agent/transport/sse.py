"""SSE chat transport — POST /api/v1/chat streaming text/event-stream."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from ai_agent.middleware.sid import extract_user_context
from ai_agent.transport.sse_events import serialize


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


def create_sse_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/chat")
    async def chat(req: Request, body: ChatRequest):
        user_context = extract_user_context(req)
        if user_context is None:
            raise HTTPException(status_code=401, detail="Missing sid cookie")

        chat_service = req.app.state.chat_service

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
