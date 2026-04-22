import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from ai_agent.middleware.request_id import RequestIDMiddleware


def _app_with_middleware() -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestIDMiddleware)

    @app.get("/echo")
    def echo(request: Request):
        return {"rid": request.state.request_id}

    return app


class TestRequestIDMiddleware:
    def test_generates_id_when_header_absent(self):
        client = TestClient(_app_with_middleware())
        resp = client.get("/echo")
        assert resp.status_code == 200
        header = resp.headers.get("X-Request-ID")
        assert header, "middleware must set response X-Request-ID"
        uuid.UUID(header)  # raises if not a valid uuid

    def test_echoes_incoming_request_id(self):
        client = TestClient(_app_with_middleware())
        incoming = "trace-abc-123"
        resp = client.get("/echo", headers={"X-Request-ID": incoming})
        assert resp.headers["X-Request-ID"] == incoming
        assert resp.json()["rid"] == incoming
