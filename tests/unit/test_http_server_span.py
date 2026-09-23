from unittest.mock import patch

import pytest
import structlog
from fastapi.testclient import TestClient

from ai_agent.app import create_app
from ai_agent.config import Settings


@pytest.fixture(autouse=True)
def _restore_structlog_config():
    # create_app calls setup_logging, which installs a new processor list, and
    # structlog.testing.capture_logs mutates whichever list earlier tests cached against.
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


def test_a_request_produces_an_http_server_span(otel_spans):
    """`otel_spans` is the session-scoped InMemorySpanExporter from conftest; the real
    tracer provider is patched out so no OTLP exporter thread starts."""
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        otel_endpoint="http://localhost:4317",
    )
    with patch("ai_agent.app.create_tracer_provider"):
        app = create_app(settings)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200

    names = [span.name for span in otel_spans.get_finished_spans()]
    assert "GET /health" in names, names
