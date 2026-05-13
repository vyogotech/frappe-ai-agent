"""Shared fixtures for unit tests.

OTEL's `trace.set_tracer_provider` only accepts the first real provider
per process (subsequent calls warn and are silently dropped). If two
test modules each try to set up their own InMemorySpanExporter, the
second one's spans go nowhere. Session-scoped here means one provider
per pytest run; the per-test `otel_spans` fixture clears the shared
exporter so each test sees only its own spans.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def _session_otel_exporter():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    yield exporter
    provider.shutdown()


@pytest.fixture
def otel_spans(_session_otel_exporter):
    _session_otel_exporter.clear()
    return _session_otel_exporter
