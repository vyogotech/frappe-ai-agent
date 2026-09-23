"""GET /config names the model and both peer URLs, so it takes the same sid as the chat route."""

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_agent.config import Settings
from ai_agent.services.health import HealthService
from ai_agent.transport.rest import create_rest_router


def _client(**kwargs) -> TestClient:
    settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    app = FastAPI()
    app.include_router(
        create_rest_router(settings=settings, health_service=HealthService(settings))
    )
    app.state.settings = settings  # require_sid asks Frappe at settings.frappe_url
    return TestClient(app, **kwargs)


def test_config_requires_a_signed_in_sid():
    client = _client()
    resp = client.get("/config")
    assert resp.status_code == 401
    assert "llm_base_url" not in resp.text
    assert "mcp_server_url" not in resp.text
    # the liveness probe the container runs stays open, and answers up or down only
    assert client.get("/health").status_code == 200


def test_config_answers_a_signed_in_caller():
    signed_in = AsyncMock(return_value="a@example.com")
    with patch("ai_agent.middleware.sid.signed_in_user", signed_in):
        resp = _client(cookies={"sid": "a-signed-in-session"}).get("/config")
    assert resp.status_code == 200
    assert resp.json()["llm_model"]
