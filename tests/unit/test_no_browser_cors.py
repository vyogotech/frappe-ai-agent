"""No CORS layer (ADR-023): every caller is server to server, so no preflight is answered."""

from fastapi.testclient import TestClient

from ai_agent.app import create_app
from ai_agent.config import Settings


def _app():
    return create_app(Settings(_env_file=None))  # pyright: ignore[reportCallIssue]


def test_the_factory_adds_no_cors_middleware():
    assert [getattr(m.cls, "__name__", m.cls) for m in _app().user_middleware] == [
        "RequestIDMiddleware"
    ]


def test_a_browser_preflight_is_answered_without_cors_headers():
    response = TestClient(_app()).options(
        "/api/v1/chat",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    cors = [name for name in response.headers if name.lower().startswith("access-control-")]
    assert cors == [], cors
