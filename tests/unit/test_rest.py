from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_agent.config import Settings
from ai_agent.services.health import HealthService
from ai_agent.transport.rest import create_rest_router


@pytest.fixture
def settings():
    return Settings()


@pytest.fixture
def health_service():
    mock = AsyncMock(spec=HealthService)
    mock.check_all.return_value = {
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
        assert "mcp" in data
        assert "llm" in data


class TestConfigEndpoint:
    def test_config(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["llm_model"] == "qwen3.5:9b"


class TestToolsEndpoint:
    def test_tools_empty(self, client):
        resp = client.get("/tools")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["tools"] == []
