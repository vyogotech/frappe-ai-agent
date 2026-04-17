"""BDD step definitions for tests/features/chat.feature.

These are smoke scenarios: they exercise the real SSE route in-process via
httpx ASGITransport, with a stubbed ChatService so no Ollama / MCP / Frappe
is actually touched. They prove end-to-end that the route wiring works.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pytest_bdd import given, parsers, scenarios, then, when

from ai_agent.transport.sse import create_sse_router

scenarios("../chat.feature")


class _StubChatService:
    """Yields whatever event list is configured on it, in order."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def handle_message(self, **_kwargs: Any):
        for event in self.events:
            yield event


@pytest.fixture
def ctx() -> dict[str, Any]:
    """Scenario-scoped mutable context carrying the app, the stub, and results."""
    stub = _StubChatService()
    app = FastAPI()
    app.state.chat_service = stub
    app.include_router(create_sse_router())
    return {
        "app": app,
        "stub": stub,
        "response_status": None,
        "response_headers": None,
        "body_text": "",
    }


# ---------- Given ----------


@given("a running agent app with a stubbed chat service")
def _bg(ctx: dict[str, Any]) -> None:
    # The fixture already built the app + stub. Nothing more to do.
    assert ctx["stub"] is not None
    assert ctx["app"] is not None


@given("the stub chat service will yield status, a tool_call, content, and done")
def _happy(ctx: dict[str, Any]) -> None:
    ctx["stub"].events = [
        {"type": "status", "message": "thinking..."},
        {"type": "tool_call", "name": "list_invoices", "arguments": {"status": "unpaid"}},
        {"type": "content", "text": "You have 3 unpaid invoices."},
        {
            "type": "done",
            "tools_called": ["list_invoices"],
            "data_quality": "high",
            "timestamp": "2026-04-14T00:00:00Z",
        },
    ]


@given(
    "the stub chat service will yield a content event explaining access was denied and then done"
)
def _denied(ctx: dict[str, Any]) -> None:
    ctx["stub"].events = [
        {"type": "status", "message": "thinking..."},
        {
            "type": "content",
            "text": "Access denied: you do not have permission to read Sales Invoice.",
        },
        {
            "type": "done",
            "tools_called": ["list_invoices"],
            "data_quality": "high",
            "timestamp": "2026-04-14T00:00:00Z",
        },
    ]


@given("the stub chat service will yield an error event and then done")
def _err(ctx: dict[str, Any]) -> None:
    ctx["stub"].events = [
        {"type": "error", "message": "The AI service is currently unavailable."},
        {
            "type": "done",
            "tools_called": [],
            "data_quality": "low",
            "timestamp": "2026-04-14T00:00:00Z",
        },
    ]


# ---------- When ----------


async def _do_post(app: FastAPI, message: str, sid: str | None) -> tuple[int, Any, str]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        kwargs: dict[str, Any] = {
            "json": {"message": message},
            "headers": {"Accept": "text/event-stream"},
        }
        if sid is not None:
            kwargs["cookies"] = {"sid": sid}
        resp = await client.post("/api/v1/chat", **kwargs)
    return resp.status_code, resp.headers, resp.text


@when(parsers.parse('I POST "{message}" to /api/v1/chat without a sid cookie'))
def _post_no_sid(ctx: dict[str, Any], message: str) -> None:
    status, headers, body = asyncio.run(_do_post(ctx["app"], message, None))
    ctx["response_status"] = status
    ctx["response_headers"] = headers
    ctx["body_text"] = body


@when(parsers.parse('I POST "{message}" to /api/v1/chat with sid "{sid}"'))
def _post_with_sid(ctx: dict[str, Any], message: str, sid: str) -> None:
    status, headers, body = asyncio.run(_do_post(ctx["app"], message, sid))
    ctx["response_status"] = status
    ctx["response_headers"] = headers
    ctx["body_text"] = body


# ---------- Then ----------


@then("I receive HTTP 401")
def _status_401(ctx: dict[str, Any]) -> None:
    assert ctx["response_status"] == 401


@then("I receive HTTP 200")
def _status_200(ctx: dict[str, Any]) -> None:
    assert ctx["response_status"] == 200


@then(parsers.parse('the response content-type is "{mime}"'))
def _content_type(ctx: dict[str, Any], mime: str) -> None:
    assert ctx["response_headers"]["content-type"].startswith(mime)


@then("no SSE stream is opened")
def _no_stream(ctx: dict[str, Any]) -> None:
    # When the route returns 401 before the StreamingResponse is constructed,
    # FastAPI sends a normal JSON error body. Assert no "data: " frames.
    assert "data: " not in ctx["body_text"]


@then("the stream contains a status event")
def _has_status(ctx: dict[str, Any]) -> None:
    assert '"type":"status"' in ctx["body_text"]


@then("the stream contains a tool_call event")
def _has_tool_call(ctx: dict[str, Any]) -> None:
    assert '"type":"tool_call"' in ctx["body_text"]


@then("the stream contains at least one content event")
def _has_content(ctx: dict[str, Any]) -> None:
    assert '"type":"content"' in ctx["body_text"]


@then("the stream ends with a done event")
def _ends_with_done(ctx: dict[str, Any]) -> None:
    # The last non-empty frame should be a done event.
    frames = [f for f in ctx["body_text"].split("\n\n") if f.strip()]
    assert frames, "stream was empty"
    assert '"type":"done"' in frames[-1]


@then("the stream does not contain an error event")
def _no_error(ctx: dict[str, Any]) -> None:
    assert '"type":"error"' not in ctx["body_text"]


@then("the stream contains a content event mentioning access denied")
def _denied_content(ctx: dict[str, Any]) -> None:
    body_lower = ctx["body_text"].lower()
    assert '"type":"content"' in ctx["body_text"]
    assert "access denied" in body_lower


@then("the stream contains exactly one error event")
def _one_error(ctx: dict[str, Any]) -> None:
    count = ctx["body_text"].count('"type":"error"')
    assert count == 1, f"expected exactly 1 error event, got {count}"


@then("the error message mentions unavailable")
def _error_unavailable(ctx: dict[str, Any]) -> None:
    assert "unavailable" in ctx["body_text"].lower()
