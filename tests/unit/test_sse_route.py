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


async def test_sse_chat_route_rejects_empty_message():
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "valid-sid"},
    ) as ac:
        response = await ac.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422


async def test_sse_chat_route_rejects_whitespace_only_message():
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "valid-sid"},
    ) as ac:
        response = await ac.post("/api/v1/chat", json={"message": "   \n\t"})
    assert response.status_code == 422


async def test_sse_chat_route_accepts_context_just_under_cap():
    # 8 KB cap minus a few bytes of JSON envelope (`{"k": "...."}` overhead).
    payload = {"k": "x" * (8 * 1024 - 16)}
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "valid-sid"},
    ) as ac:
        response = await ac.post(
            "/api/v1/chat",
            json={"message": "hello", "context": payload},
            headers={"Accept": "text/event-stream"},
        )
    assert response.status_code == 200


async def test_sse_chat_route_rejects_oversize_context():
    payload = {"k": "x" * (8 * 1024 + 1)}
    app = _build_app()
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "valid-sid"},
    ) as ac:
        response = await ac.post(
            "/api/v1/chat",
            json={"message": "hello", "context": payload},
        )
    assert response.status_code == 422
    assert "context" in response.text.lower()


async def test_sse_chat_route_no_sid_does_not_burn_rate_limit():
    # Why: regression guard — unauthenticated callers must 401 without
    # consuming a token from the (IP-keyed) bucket. Otherwise one bad actor
    # could lock out a shared NAT by spraying anonymous requests.
    from ai_agent.app import create_app
    from ai_agent.config import Settings

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        agent_rate_limit="2/minute",
    )
    app = create_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        # 5 no-cookie requests — well over the 2/minute cap.
        statuses = [
            (await ac.post("/api/v1/chat", json={"message": "x"})).status_code for _ in range(5)
        ]

    assert all(s == 401 for s in statuses), f"expected all 5 to 401, got {statuses}"


async def test_sse_chat_route_rate_limits_after_burst():
    # Why: regression test for the slowapi wiring — burst above the limit
    # within a single minute must produce 429. The full app (built via
    # create_app) is what the limiter middleware is attached to.
    from ai_agent.app import create_app
    from ai_agent.config import Settings

    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        agent_rate_limit="2/minute",
    )
    app = create_app(settings)

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "burst-sid"},
    ) as ac:
        first = await ac.post("/api/v1/chat", json={"message": "x"})
        second = await ac.post("/api/v1/chat", json={"message": "x"})
        third = await ac.post("/api/v1/chat", json={"message": "x"})

    # First two should pass the limiter (they may then fail downstream — e.g.
    # MCP/LLM not running — but that's after the limiter check). The third
    # must be cut off with 429 regardless.
    assert first.status_code != 429
    assert second.status_code != 429
    assert third.status_code == 429
