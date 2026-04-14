import structlog

from ai_agent.observability.logging import setup_logging
from ai_agent.observability.metrics import create_meter
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
    def test_create_tracer_disabled(self):
        provider = create_tracer_provider(endpoint="", service_name="test")
        assert provider is not None

    def test_create_tracer_with_endpoint(self):
        # Does not connect, just configures
        provider = create_tracer_provider(endpoint="http://localhost:4317", service_name="test")
        assert provider is not None


class TestMetrics:
    def test_create_meter(self):
        meter = create_meter(service_name="test")
        assert meter is not None
