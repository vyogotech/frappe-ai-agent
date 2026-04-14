from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from copilot_agent.config import Settings
from copilot_agent.services.health import HealthService
from copilot_agent.transport.rest import create_rest_router


@pytest.fixture
def settings():
    return Settings(jwt_secret="test-secret")


@pytest.fixture
def health_service():
    mock = AsyncMock(spec=HealthService)
    mock.check_all.return_value = {
        "redis": {"ok": True},
        "mcp": {"ok": True},
        "llm": {"ok": True},
        "healthy": True,
    }
    return mock


@pytest.fixture
def app(settings, health_service):
    app = FastAPI()
    router = create_rest_router(settings=settings, health_service=health_service, tools=[])
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_details(self, client):
        resp = client.get("/health?detail=true")
        assert resp.status_code == 200
        data = resp.json()
        assert "redis" in data


class TestConfigEndpoint:
    def test_config(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_model"] == "llama3.2:3b"
        assert "jwt_secret" not in data


class TestToolsEndpoint:
    def test_tools_empty(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["tools"] == []
