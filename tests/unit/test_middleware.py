import uuid

import structlog
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

    def test_binds_request_id_into_structlog_context(self):
        """The whole point of request-id correlation: a log emitted from
        inside the handler must carry the same request_id that ends up on
        the response header. Without bind_contextvars, structlog's
        merge_contextvars processor has nothing to merge."""
        captured: list[dict] = []

        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/log")
        def emit_log(request: Request):
            structlog.contextvars.get_contextvars()  # touch contextvars
            captured.append(dict(structlog.contextvars.get_contextvars()))
            return {"ok": True}

        client = TestClient(app)
        incoming = "rid-under-test"
        resp = client.get("/log", headers={"X-Request-ID": incoming})
        assert resp.status_code == 200
        assert captured, "handler did not emit a context capture"
        assert captured[0].get("request_id") == incoming, (
            f"expected request_id={incoming!r} in structlog contextvars, got {captured[0]!r}"
        )

    def test_clears_request_id_after_request(self):
        """contextvars are ASGI-task-scoped; the binding must be cleared
        after the response so a worker that handles request A then request
        B without an X-Request-ID header doesn't accidentally carry A's id
        into B's logs."""
        app = FastAPI()
        app.add_middleware(RequestIDMiddleware)

        @app.get("/noop")
        def noop():
            return {"ok": True}

        client = TestClient(app)
        client.get("/noop", headers={"X-Request-ID": "leaky-rid"})
        # After the request returns, the current task's contextvars should
        # not still carry request_id from the prior request.
        assert "request_id" not in structlog.contextvars.get_contextvars()
