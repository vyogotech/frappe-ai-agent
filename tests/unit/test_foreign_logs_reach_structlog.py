import io
import json
import logging
import logging.config

import pytest
import structlog
import uvicorn.config

from ai_agent.observability.logging import setup_logging


@pytest.fixture(autouse=True)
def _restore_structlog_config():
    # setup_logging installs a new processor list, and structlog.testing.capture_logs
    # mutates whichever list the loggers of earlier tests were cached against.
    saved = structlog.get_config()
    yield
    structlog.configure(**saved)


def _capture() -> io.StringIO:
    """Point the handler setup_logging installed at a buffer and return it."""
    buf = io.StringIO()
    logging.getLogger().handlers[0].stream = buf  # type: ignore[attr-defined]
    return buf


class TestForeignLogsReachStructlog:
    def test_a_plain_stdlib_record_is_rendered_like_a_structlog_one(self):
        # slowapi and mcp log through logging.getLogger(__name__), not structlog.
        setup_logging(level="info", log_format="json")
        buf = _capture()
        structlog.contextvars.bind_contextvars(request_id="f244c8f6-uuid")
        try:
            logging.getLogger("slowapi").warning("ratelimit exceeded")
        finally:
            structlog.contextvars.clear_contextvars()

        line = json.loads(buf.getvalue())
        assert line["event"] == "ratelimit exceeded"
        assert line["level"] == "warning"
        assert line["logger"] == "slowapi"
        assert line["request_id"] == "f244c8f6-uuid"
        assert "timestamp" in line

    def test_uvicorns_own_lines_reach_the_structlog_handler(self):
        # What uvicorn does at startup, before it calls the app factory.
        logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
        setup_logging(level="debug", log_format="json")
        buf = _capture()

        logging.getLogger("uvicorn.error").info("Application startup complete.")
        logging.getLogger("uvicorn.error").debug("only at the configured level")

        events = [json.loads(line) for line in buf.getvalue().splitlines()]
        assert [e["event"] for e in events] == [
            "Application startup complete.",
            "only at the configured level",
        ]
        assert [e["level"] for e in events] == ["info", "debug"]

    def test_the_mcp_clients_protocol_chatter_is_below_the_default_level(self):
        setup_logging(level="info", log_format="json")
        buf = _capture()
        logging.getLogger("mcp.client.streamable_http").info("Negotiated protocol version: x")
        assert buf.getvalue() == ""
