"""Shared fixtures for unit tests.

OTEL's `trace.set_tracer_provider` only accepts the first real provider
per process (subsequent calls warn and are silently dropped). If two
test modules each try to set up their own InMemorySpanExporter, the
second one's spans go nowhere. Session-scoped here means one provider
per pytest run; the per-test `otel_spans` fixture clears the shared
exporter so each test sees only its own spans.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_sockets():
    """Fail a unit test that opens a socket.

    A unit test that reaches the network is not a unit test: it passes or fails on whatever
    happens to be listening on the machine that runs it, and it writes to it. `connect` is
    patched rather than one HTTP client, so the guard covers every library at once.
    """

    def _refuse(_self, address, *_args):
        # pytest.fail raises a BaseException, so the code under test cannot catch this the way
        # it rightly catches a connection error and carry on as if nothing had happened.
        pytest.fail(
            f"a unit test tried to connect to {address!r}: stub the client. "
            "Tests that need real services live in tests/integration."
        )

    with (
        patch.object(socket.socket, "connect", _refuse),
        patch.object(socket.socket, "connect_ex", _refuse),
    ):
        yield


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
