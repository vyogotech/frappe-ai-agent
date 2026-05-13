import io
import logging as stdlib_logging
import re
from unittest.mock import patch

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

import structlog

from ai_agent.observability.logging import (
    _drop_request_id,
    _short_local_timestamp,
    _strip_ai_agent_logger_prefix,
    setup_logging,
)
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


class TestConsoleProcessors:
    def test_short_local_timestamp_format(self):
        result = _short_local_timestamp(None, "info", {})
        ts = result["timestamp"]
        assert len(ts) == 12, ts
        assert ts[2] == ":" and ts[5] == ":" and ts[8] == "."
        assert ts[:2].isdigit() and ts[3:5].isdigit() and ts[6:8].isdigit()
        assert ts[9:].isdigit()

    def test_strip_ai_agent_logger_prefix_strips_when_present(self):
        result = _strip_ai_agent_logger_prefix(
            None, "info", {"logger": "ai_agent.services.chat"}
        )
        assert result["logger"] == "services.chat"

    def test_strip_ai_agent_logger_prefix_leaves_others_alone(self):
        result = _strip_ai_agent_logger_prefix(
            None, "info", {"logger": "uvicorn.error"}
        )
        assert result["logger"] == "uvicorn.error"

    def test_strip_ai_agent_logger_prefix_no_logger_key(self):
        result = _strip_ai_agent_logger_prefix(None, "info", {"event": "x"})
        assert result == {"event": "x"}

    def test_drop_request_id_removes_key(self):
        result = _drop_request_id(
            None, "info", {"event": "x", "request_id": "abc-123"}
        )
        assert "request_id" not in result
        assert result["event"] == "x"

    def test_drop_request_id_absent_key_is_noop(self):
        result = _drop_request_id(None, "info", {"event": "x"})
        assert result == {"event": "x"}


class TestConsoleRendering:
    def _capture(self, log_format: str, **fields) -> str:
        setup_logging(level="debug", log_format=log_format)
        buf = io.StringIO()
        root = stdlib_logging.getLogger()
        for h in root.handlers:
            h.stream = buf  # type: ignore[attr-defined]
        # Bind request_id via contextvars so we exercise the production
        # path (RequestIDMiddleware uses contextvars, not kwargs).
        structlog.contextvars.bind_contextvars(request_id="f244c8f6-uuid")
        try:
            log = structlog.get_logger("ai_agent.services.chat")
            log.info("sample_event", extra_key="value", **fields)
        finally:
            structlog.contextvars.clear_contextvars()
        return buf.getvalue()

    def test_console_timestamp_is_short(self):
        out = _ANSI_RE.sub("", self._capture("console"))
        assert "2026-" not in out
        first = out.split()[0]
        assert len(first) == 12 and first[2] == ":" and first[8] == "."

    def test_console_logger_prefix_stripped(self):
        out = self._capture("console")
        assert "ai_agent.services.chat" not in out
        assert "services.chat" in out

    def test_console_hides_contextvar_request_id(self):
        out = self._capture("console")
        assert "request_id" not in out
        assert "f244c8f6-uuid" not in out
        assert "extra_key" in out

    def test_json_mode_keeps_request_id_and_full_logger(self):
        out = self._capture("json")
        assert "request_id" in out
        assert "f244c8f6-uuid" in out
        assert "ai_agent.services.chat" in out


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
