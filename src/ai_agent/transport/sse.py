"""SSE chat transport — POST /api/v1/chat streaming text/event-stream."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai_agent.transport.sse_events import serialize


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)


def create_sse_router() -> APIRouter:
    router = APIRouter()

    @router.post("/api/v1/chat")
    async def chat(req: Request, body: ChatRequest):
        sid = req.cookies.get("sid")
        if not sid:
            raise HTTPException(status_code=401, detail="Missing sid cookie")

        chat_service = req.app.state.chat_service

        async def event_stream():
            async for event in chat_service.handle_message(
                message=body.message,
                session_id=body.session_id,
                context=body.context,
                user_context=sid,  # Phase 5 will replace this with a UserContext object
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
