import json

import httpx
import pytest
import respx
import structlog

from ai_agent.integrations.frappe_history import FrappeHistoryClient

_CSRF_URL = "http://frappe:8000/app"
_SESSION_URL = "http://frappe:8000/api/resource/AI Chat Session"
_MESSAGE_URL = "http://frappe:8000/api/resource/AI Chat Message"
_FAKE_CSRF = "abc123def456789abcdef012345"


def _csrf_page(token: str = _FAKE_CSRF) -> str:
    """A minimal /app HTML response that embeds the csrf_token JS variable."""
    return (
        "<!DOCTYPE html><html><head><script>"
        f'csrf_token = "{token}";'
        "</script></head><body></body></html>"
    )


def _mock_csrf_ok() -> None:
    """Respond to the /app GET with a page containing the csrf_token."""
    respx.get(_CSRF_URL).mock(return_value=httpx.Response(200, text=_csrf_page()))


@pytest.mark.asyncio
@respx.mock
async def test_create_session_posts_with_sid_and_csrf():
    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "sess-123"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.create_session(
        sid="abc123",
        title="first message",
        context_json="{}",
    )

    assert name == "sess-123"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert post.headers["Cookie"] == "sid=abc123"
    assert post.headers["X-Frappe-CSRF-Token"] == _FAKE_CSRF


@pytest.mark.asyncio
@respx.mock
async def test_save_message_posts_with_sid_and_csrf_and_fields():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.save_message(
        sid="abc123",
        session="sess-123",
        role="user",
        content="hello",
    )

    assert name == "msg-1"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert post.headers["Cookie"] == "sid=abc123"
    assert post.headers["X-Frappe-CSRF-Token"] == _FAKE_CSRF
    body = json.loads(post.content)
    assert body["session"] == "sess-123"
    assert body["role"] == "user"
    assert body["content"] == "hello"


@pytest.mark.asyncio
@respx.mock
async def test_csrf_token_is_cached_across_calls():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    # Three writes with the same sid should only fetch CSRF once.
    await client.save_message(sid="abc", session="s", role="user", content="a")
    await client.save_message(sid="abc", session="s", role="user", content="b")
    await client.save_message(sid="abc", session="s", role="user", content="c")

    csrf_calls = [c for c in respx.calls if c.request.method == "GET"]
    assert len(csrf_calls) == 1


@pytest.mark.asyncio
@respx.mock
async def test_csrf_token_is_refetched_per_sid():
    """Different sids must not share a cached token."""
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    await client.save_message(sid="user-a", session="s", role="user", content="a")
    await client.save_message(sid="user-b", session="s", role="user", content="b")

    csrf_calls = [c for c in respx.calls if c.request.method == "GET"]
    assert len(csrf_calls) == 2


@pytest.mark.asyncio
@respx.mock
async def test_write_retried_once_on_csrf_error():
    """If the first POST returns a CSRF error, the client should refresh
    its token and retry the write exactly once."""
    # First CSRF fetch returns token-a. A second fetch (after invalidation)
    # returns token-b. The POST rejects token-a with CSRF error, accepts b.
    csrf_route = respx.get(_CSRF_URL).mock(
        side_effect=[
            httpx.Response(200, text=_csrf_page("aaaaaaaa11111111")),
            httpx.Response(200, text=_csrf_page("bbbbbbbb22222222")),
        ]
    )
    post_route = respx.post(_MESSAGE_URL).mock(
        side_effect=[
            httpx.Response(
                400,
                json={"exc_type": "CSRFTokenError"},
                text='{"exc_type":"CSRFTokenError","message":"Invalid CSRF token"}',
            ),
            httpx.Response(200, json={"data": {"name": "msg-1"}}),
        ]
    )

    client = FrappeHistoryClient(base_url="http://frappe:8000")
    name = await client.save_message(sid="abc", session="s", role="user", content="hi")

    assert name == "msg-1"
    assert csrf_route.call_count == 2
    assert post_route.call_count == 2
    # The retry must use the fresh token, not the stale one.
    second_post = post_route.calls[1].request
    assert second_post.headers["X-Frappe-CSRF-Token"] == "bbbbbbbb22222222"


@pytest.mark.asyncio
@respx.mock
async def test_create_session_returns_none_on_http_error(caplog):
    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    result = await client.create_session(
        sid="abc123",
        title="t",
        context_json="{}",
    )
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_save_message_returns_none_on_http_error():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    result = await client.save_message(sid="abc", session="sess-1", role="user", content="hi")
    assert result is None


@pytest.mark.asyncio
@respx.mock
async def test_save_message_forwards_optional_tool_fields():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-2"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    await client.save_message(
        sid="abc",
        session="sess-1",
        role="assistant",
        content="final answer",
        tool_name="list_invoices",
        tool_args_json='{"status": "unpaid"}',
        tool_result_json='{"count": 3}',
    )

    post = next(c.request for c in respx.calls if c.request.method == "POST")
    body = json.loads(post.content)
    assert body["tool_name"] == "list_invoices"
    assert body["tool_args_json"] == '{"status": "unpaid"}'
    assert body["tool_result_json"] == '{"count": 3}'


@pytest.mark.asyncio
@respx.mock
async def test_write_proceeds_without_token_when_csrf_fetch_fails():
    """If Frappe's /app page is unreachable, the client should still try
    to POST (Frappe may reject with 400, which becomes a None return)."""
    respx.get(_CSRF_URL).mock(return_value=httpx.Response(500))
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.save_message(sid="abc", session="s", role="user", content="hi")

    assert name == "msg-1"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert "X-Frappe-CSRF-Token" not in post.headers


# ─── Connection reuse ────────────────────────────────────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_writes_reuse_single_async_client_per_instance():
    """Per chat turn we make 1 CSRF GET + 2-3 POST writes. Opening a
    fresh AsyncClient for each was wasteful — every call paid a new
    connection setup (TLS handshake when behind HTTPS) instead of
    reusing the persistent connection. After this fix, all writes
    from one FrappeHistoryClient instance share one underlying
    AsyncClient — verified by patching the constructor and counting
    instantiations."""
    import unittest.mock as _mock

    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )

    real_init = httpx.AsyncClient.__init__
    init_calls: list[dict] = []

    def _spy_init(self, *args, **kwargs):
        init_calls.append(kwargs)
        return real_init(self, *args, **kwargs)

    with _mock.patch.object(httpx.AsyncClient, "__init__", _spy_init):
        client = FrappeHistoryClient(base_url="http://frappe:8000")
        await client.save_message(sid="abc", session="s", role="user", content="a")
        await client.save_message(sid="abc", session="s", role="user", content="b")
        await client.save_message(sid="abc", session="s", role="user", content="c")
        await client.aclose()

    # One client for all four HTTP calls (1 CSRF GET + 3 POSTs).
    assert len(init_calls) == 1, (
        f"expected 1 AsyncClient instantiation, got {len(init_calls)}: {init_calls!r}"
    )


@pytest.mark.asyncio
@respx.mock
async def test_aclose_is_idempotent():
    """aclose() must be safe to call multiple times — the lifespan
    teardown may run it after a context manager already cleaned up."""
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")
    await client.save_message(sid="abc", session="s", role="user", content="hi")
    await client.aclose()
    await client.aclose()  # idempotent — must not raise


@pytest.mark.asyncio
@respx.mock
async def test_use_after_close_raises_instead_of_leaking_new_client():
    """Regression guard. The earlier implementation reset `_client = None`
    on aclose() but left `_closed = True`, and `_get_client()` would
    happily build a fresh AsyncClient on the next write — leaving
    `_closed = True` meaning the next aclose() would return early and
    never close that new pool. Reproduction on the prior code:

        client1 = c._get_client()      # _closed=False
        await c.aclose()               # _closed=True, _client=None
        client2 = c._get_client()      # builds a NEW client, _closed stays True
        await c.aclose()               # returns early, client2.is_closed=False  ← LEAK

    Now `_get_client()` raises RuntimeError after close so the leak
    is impossible. Lifespan teardown runs aclose() once at process
    exit; any write after that is a programming error and must fail
    loudly, not silently leak a connection pool."""
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")
    await client.save_message(sid="abc", session="s", role="user", content="hi")
    await client.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await client.save_message(sid="abc", session="s", role="user", content="too-late")


def test_looks_like_csrf_error_returns_false_when_text_access_raises():
    """The helper reads `response.text` which can in pathological cases
    raise (e.g. malformed encoding declarations). The except branch
    must swallow the exception and return False — a non-CSRF error
    should fall through to the standard write-failed handling, not
    trigger an unbounded retry loop."""
    import httpx

    from ai_agent.integrations.frappe_history import _looks_like_csrf_error

    class _ExplodingResponse:
        @property
        def text(self) -> str:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    # _looks_like_csrf_error is duck-typed against `response.text` so a
    # stand-in that raises on .text drives the exception arm cleanly.
    assert _looks_like_csrf_error(_ExplodingResponse()) is False  # type: ignore[arg-type]

    # Sanity: the happy-path branches still work.
    ok_response = httpx.Response(400, json={"exc_type": "CSRFTokenError"})
    assert _looks_like_csrf_error(ok_response) is True
    not_csrf = httpx.Response(400, json={"exc_type": "ValidationError"})
    assert _looks_like_csrf_error(not_csrf) is False


@pytest.mark.asyncio
@respx.mock
async def test_write_proceeds_without_token_when_csrf_not_in_html():
    """If /app returns 200 but the HTML has no csrf_token JS variable,
    the client should still attempt the POST (some Frappe versions may
    omit it in certain contexts)."""
    respx.get(_CSRF_URL).mock(
        return_value=httpx.Response(200, text="<html><body>No token here</body></html>")
    )
    respx.post(_MESSAGE_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "msg-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    name = await client.save_message(sid="abc", session="s", role="user", content="hi")

    assert name == "msg-1"
    post = next(c.request for c in respx.calls if c.request.method == "POST")
    assert "X-Frappe-CSRF-Token" not in post.headers


# ─── Failure signal: structured log + OTEL counter ───────────────────────


@pytest.mark.asyncio
@respx.mock
async def test_session_write_failure_emits_structured_event():
    """A history-write failure must surface as a structured
    `frappe_history_write_failed` event with `kind` and `status_code`
    fields. Without these aggregable fields, a sustained Frappe-down
    state is invisible to dashboards."""
    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    with structlog.testing.capture_logs() as logs:
        result = await client.create_session(sid="abc", title="t", context_json="{}")

    assert result is None
    failures = [r for r in logs if r.get("event") == "frappe_history_write_failed"]
    assert len(failures) == 1, f"expected 1 structured failure event, got {logs!r}"
    assert failures[0]["kind"] == "session"


@pytest.mark.asyncio
@respx.mock
async def test_message_write_failure_emits_structured_event():
    _mock_csrf_ok()
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    with structlog.testing.capture_logs() as logs:
        result = await client.save_message(sid="abc", session="s", role="user", content="x")

    assert result is None
    failures = [r for r in logs if r.get("event") == "frappe_history_write_failed"]
    assert len(failures) == 1
    assert failures[0]["kind"] == "message"


@pytest.mark.asyncio
@respx.mock
async def test_session_write_emits_history_span_with_kind(otel_spans):
    """Each write is wrapped in an `agent.history.write` span carrying
    a `kind` attribute, so a trace UI can show the per-kind breakdown
    under the parent agent.chat_turn span without consulting metric
    cardinality. `otel_spans` is the shared session-scoped exporter
    defined in tests/unit/conftest.py."""
    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(
        return_value=httpx.Response(200, json={"data": {"name": "sess-1"}})
    )
    client = FrappeHistoryClient(base_url="http://frappe:8000")
    await client.create_session(sid="abc", title="t", context_json="{}")

    spans = otel_spans.get_finished_spans()
    history_spans = [s for s in spans if s.name == "agent.history.write"]
    assert len(history_spans) == 1
    assert dict(history_spans[0].attributes or {}).get("kind") == "session"


@pytest.mark.asyncio
@respx.mock
async def test_write_failures_increment_otel_counter():
    """`agent.history.write_failures` is the metric a Prometheus/OTLP
    collector scrapes to alert on Frappe-write outages. Without it, a
    100%-failure rate looks identical to 0% from outside the process."""
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    _mock_csrf_ok()
    respx.post(_SESSION_URL).mock(return_value=httpx.Response(500))
    respx.post(_MESSAGE_URL).mock(return_value=httpx.Response(500))
    client = FrappeHistoryClient(base_url="http://frappe:8000")

    await client.create_session(sid="abc", title="t", context_json="{}")
    await client.save_message(sid="abc", session="s", role="user", content="x")
    await client.save_message(sid="abc", session="s", role="user", content="y")

    # OTEL's MetricsData union (Sum / Gauge / Histogram / ExponentialHistogram)
    # makes pyright unhappy without narrowing; we only emit a Counter so the
    # data points are NumberDataPoint with a numeric .value. The narrowing
    # below is type-runtime-safe; the cast quiets pyright on the union access.
    from opentelemetry.sdk.metrics.export import MetricsData, NumberDataPoint, Sum

    md = reader.get_metrics_data()
    assert isinstance(md, MetricsData)
    points: dict[str, int] = {}
    for rm in md.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name != "agent.history.write_failures":
                    continue
                data = metric.data
                assert isinstance(data, Sum)
                for dp in data.data_points:
                    assert isinstance(dp, NumberDataPoint)
                    attrs = dict(dp.attributes) if dp.attributes else {}
                    kind = str(attrs.get("kind", "?"))
                    points[kind] = points.get(kind, 0) + int(dp.value)
    assert points.get("session") == 1, f"expected 1 session failure, got {points!r}"
    assert points.get("message") == 2, f"expected 2 message failures, got {points!r}"
