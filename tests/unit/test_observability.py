from unittest.mock import patch

import structlog

from ai_agent.observability.logging import setup_logging
from ai_agent.observability.tracing import create_tracer_provider


class TestLogging:
    def test_setup_json_format(self):
        setup_logging(level="info", log_format="json")
        logger = structlog.get_logger()
        assert logger is not None

    def test_setup_console_format(self):
        setup_logging(level="debug", log_format="console")
        logger = structlog.get_logger()
        assert logger is not None


class TestTracing:
    # OTEL's `set_tracer_provider` is one-shot per process — the first real
    # provider wins; subsequent calls warn and are dropped. If these tests
    # install a provider, the session-scoped `otel_spans` fixture in
    # tests/unit/conftest.py cannot install its InMemorySpanExporter on a
    # later test, so order-dependent failures appear (e.g. chat_turn span
    # assertions). We only need `create_tracer_provider` to return a
    # configured TracerProvider; the install side-effect is irrelevant to
    # what these tests assert, so patch it out.

    def test_create_tracer_disabled(self):
        with patch("ai_agent.observability.tracing.trace.set_tracer_provider"):
            provider = create_tracer_provider(endpoint="", service_name="test")
        assert provider is not None

    def test_create_tracer_with_endpoint(self):
        # Does not connect, just configures
        with patch("ai_agent.observability.tracing.trace.set_tracer_provider"):
            provider = create_tracer_provider(endpoint="http://localhost:4317", service_name="test")
        assert provider is not None
