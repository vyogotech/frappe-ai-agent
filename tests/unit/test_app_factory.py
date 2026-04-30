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
        with TestClient(app):
            # state is populated during lifespan startup
            assert app.state.settings is settings
            assert app.state.chat_service is not None
