from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


class FakeChatService:
    """Minimal stand-in for ChatService so the route can stream without an LLM."""

    async def handle_message(self, *, message, session_id, context, user_context):
        yield {"type": "status", "message": "thinking..."}
        yield {"type": "content", "text": "hi"}
        yield {
            "type": "done",
            "tools_called": [],
            "data_quality": "high",
            "timestamp": "2026-04-14T00:00:00Z",
        }


def _build_app() -> FastAPI:
    from ai_agent.transport.sse import create_sse_router

    app = FastAPI()
    app.state.chat_service = FakeChatService()
    app.include_router(create_sse_router())
    return app


async def test_sse_chat_route_missing_sid_returns_401():
    app = _build_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Accept": "text/event-stream"},
        )
    assert response.status_code == 401


async def test_sse_chat_route_with_sid_returns_event_stream():
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "valid-sid"},
    ) as ac:
        response = await ac.post(
            "/api/v1/chat",
            json={"message": "hello"},
            headers={"Accept": "text/event-stream"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type":"status"' in body
    assert '"type":"content"' in body
    assert '"type":"done"' in body
    assert "data: " in body
