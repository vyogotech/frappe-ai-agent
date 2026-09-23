"""The id the middleware binds must leave the agent on the calls it makes (ADR-008)."""

import json
import uuid

import httpx
import pytest
import respx
import structlog
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.request_id import RequestIDMiddleware

_LIST_URL = "http://frappe:8000/api/method/frappe.client.get_list"


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
    )


def test_mcp_client_sends_the_id_the_middleware_bound():
    """The whole mechanism in one: the middleware binds the inbound id, the binding survives into
    the SSE body generator where `handle_message` builds the client, and the client sends it on."""
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.post("/api/v1/chat")
    async def chat():
        async def body():
            client = build_mcp_client_for_sid(_settings(), sid="abc123")
            yield json.dumps(client.connections["frappe"].get("headers")).encode()  # pyright: ignore[reportTypedDictNotRequiredAccess]

        return StreamingResponse(body(), media_type="text/event-stream")

    resp = TestClient(app).post("/api/v1/chat", headers={"X-Request-ID": "rid-from-frappe-1"})
    assert json.loads(resp.text).get("X-Request-ID") == "rid-from-frappe-1"


def test_mcp_client_sends_no_request_id_outside_a_request():
    structlog.contextvars.clear_contextvars()
    client = build_mcp_client_for_sid(_settings(), sid="abc123")
    headers = client.connections["frappe"].get("headers") or {}  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "X-Request-ID" not in headers


@pytest.mark.asyncio
@respx.mock
async def test_history_reads_send_the_bound_request_id_as_frappes_own_header():
    respx.get(_LIST_URL).mock(return_value=httpx.Response(200, json={"message": []}))
    client = FrappeHistoryClient(base_url="http://frappe:8000")
    token = structlog.contextvars.bind_contextvars(request_id="rid-from-frappe-2")
    try:
        await client.list_messages(sid="abc123", session="sess-1")
    finally:
        structlog.contextvars.reset_contextvars(**token)
        await client.aclose()
    assert respx.calls.last.request.headers["X-Frappe-Request-Id"] == "rid-from-frappe-2"


@pytest.mark.parametrize("hostile", ["has spaces", "a" * 65, "sid=stolen;path=/", ""])
def test_an_inbound_id_of_the_wrong_shape_is_replaced(hostile: str):
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/noop")
    def noop():
        return {"ok": True}

    resp = TestClient(app).get("/noop", headers={"X-Request-ID": hostile})
    assert resp.headers["X-Request-ID"] != hostile
    uuid.UUID(resp.headers["X-Request-ID"])
