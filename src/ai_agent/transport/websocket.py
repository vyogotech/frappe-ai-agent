"""WebSocket endpoint for chat streaming."""

from __future__ import annotations

import json

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_agent.middleware.auth import verify_token
from ai_agent.middleware.rate_limit import RateLimiter
from ai_agent.services.chat import ChatService
from ai_agent.transport.schemas import ChatMessage, ErrorEvent

logger = structlog.get_logger()


def create_ws_router(
    chat_service: ChatService,
    rate_limiter: RateLimiter,
    jwt_secret: str,
    jwt_algorithm: str,
) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/chat")
    async def chat_ws(ws: WebSocket):
        # Extract and verify JWT from query params
        token = ws.query_params.get("token", "")
        try:
            payload = verify_token(token, secret=jwt_secret, algorithm=jwt_algorithm)
        except Exception:
            await ws.close(code=4001, reason="Invalid or missing token")
            return

        user = payload.get("sub", "unknown")
        await ws.accept()
        logger.info("ws_connected", user=user)

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                msg = ChatMessage.model_validate(data)

                # Rate limit check
                if not await rate_limiter.is_allowed(user):
                    await ws.send_json(
                        ErrorEvent(
                            code="RATE_LIMITED",
                            message="Too many requests. Please wait.",
                            request_id=msg.request_id,
                        ).model_dump()
                    )
                    continue

                # Stream response
                async for event in chat_service.handle_message(
                    content=msg.content,
                    context=msg.context,
                    session_id=msg.session_id,
                    request_id=msg.request_id,
                ):
                    await ws.send_json(event)

        except WebSocketDisconnect:
            logger.info("ws_disconnected", user=user)
        except Exception as exc:
            logger.error("ws_error", user=user, error=str(exc))

    return router
