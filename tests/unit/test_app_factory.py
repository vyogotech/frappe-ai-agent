from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_agent.app import create_app
from ai_agent.config import Settings


class TestCreateApp:
    def test_without_otel_endpoint_does_not_instrument(self):
        settings = Settings(_env_file=None, otel_endpoint="")  # pyright: ignore[reportCallIssue]
        with patch("ai_agent.app.FastAPIInstrumentor") as mock_instrumentor:
            app = create_app(settings)
            with TestClient(app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200

        mock_instrumentor.instrument_app.assert_not_called()

    def test_with_otel_endpoint_calls_instrumentor(self):
        settings = Settings(_env_file=None, otel_endpoint="http://localhost:4317")  # pyright: ignore[reportCallIssue]
        # Patch both the instrumentor and the tracer provider so we don't
        # spin up a real OTLP exporter during the test.
        with (
            patch("ai_agent.app.FastAPIInstrumentor") as mock_instrumentor,
            patch("ai_agent.app.create_tracer_provider") as mock_tracer,
        ):
            app = create_app(settings)
            with TestClient(app) as client:
                resp = client.get("/health")
                assert resp.status_code == 200

        mock_tracer.assert_called_once_with(
            endpoint="http://localhost:4317",
            service_name=settings.otel_service_name,
        )
        mock_instrumentor.instrument_app.assert_called_once_with(app)

    def test_exposes_settings_and_chat_service_on_app_state(self):
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        app = create_app(settings)
        # State is populated at factory time, NOT lifespan — services are
        # constructed in `create_app` so route introspection (tests, OpenAPI
        # scrapers) sees them without needing to enter the TestClient context.
        assert app.state.settings is settings
        assert app.state.chat_service is not None

    def test_routes_registered_before_lifespan_startup(self):
        # Why: routers wired inside `lifespan` are invisible until first
        # request — regression guard for that mistake.
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        app = create_app(settings)

        chat_route = next(
            (r for r in app.routes if getattr(r, "path", None) == "/api/v1/chat"),
            None,
        )
        assert chat_route is not None, "POST /api/v1/chat not registered at factory time"
        assert "POST" in getattr(chat_route, "methods", set())

        # Sanity: REST routes too.
        rest_paths = {getattr(r, "path", None) for r in app.routes}
        assert "/health" in rest_paths
        assert "/config" in rest_paths

    def test_sid_or_ip_key_falls_back_to_ip_when_sid_missing(self):
        """`_sid_or_ip_key` is the slowapi key function. On the chat
        route a missing sid 401s before the key function runs, but the
        IP fallback is the safety net for any future @limit-decorated
        route without a sid-required dependency. Tested directly here
        because no current route exercises the fallback path."""
        from starlette.requests import Request

        from ai_agent.app import _sid_or_ip_key

        # No cookie header → IP fallback.
        scope_no_sid = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [],
            "client": ("203.0.113.42", 12345),
        }
        key = _sid_or_ip_key(Request(scope_no_sid))  # type: ignore[arg-type]
        assert key.startswith("ip:"), key
        assert "203.0.113.42" in key

        # Whitespace-only sid is treated as missing (the same rule
        # extract_user_context applies for the auth path).
        scope_whitespace_sid = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"cookie", b"sid=   ")],
            "client": ("203.0.113.42", 12345),
        }
        key_ws = _sid_or_ip_key(Request(scope_whitespace_sid))  # type: ignore[arg-type]
        assert key_ws.startswith("ip:"), key_ws

        # Sanity: a real sid → "sid:" prefix.
        scope_with_sid = {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(b"cookie", b"sid=real-sid-value")],
            "client": ("203.0.113.42", 12345),
        }
        key_sid = _sid_or_ip_key(Request(scope_with_sid))  # type: ignore[arg-type]
        assert key_sid == "sid:real-sid-value"
