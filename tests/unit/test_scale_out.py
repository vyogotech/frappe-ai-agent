"""What a second worker or a second replica does to state the process keeps to itself."""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import ValidationError

from ai_agent.app import create_app
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import _CSRF_CACHE_MAX, FrappeHistoryClient

FRAPPE = "http://frappe.test"


def test_the_rate_limit_counters_go_where_the_setting_says():
    """Without a storage uri slowapi keeps them in the process, so N workers give N budgets."""
    app = create_app(
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            frappe_url=FRAPPE,
            rate_limit_storage_uri="memory://",
        )
    )

    assert app.state.limiter._storage_uri == "memory://"


def test_a_storage_the_limiter_cannot_open_is_refused_at_startup():
    """A typo in the redis url must not fall back to a per-process count that looks fine."""
    with pytest.raises(ValidationError, match="rate_limit_storage_uri"):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            rate_limit_storage_uri="rediss//typo.invalid:6379",
        )


@respx.mock
async def test_the_csrf_cache_holds_no_more_session_ids_than_it_is_allowed():
    """Each entry is a live session id; a cache that only grows is an outage and a disclosure."""
    respx.get(f"{FRAPPE}/app").mock(return_value=httpx.Response(200, text='csrf_token = "c0ffee"'))
    client = FrappeHistoryClient(base_url=FRAPPE)
    try:
        for i in range(_CSRF_CACHE_MAX + 50):
            await client._csrf_token_for(f"sid-of-user-{i}")
    finally:
        await client.aclose()

    assert len(client._csrf_cache) <= _CSRF_CACHE_MAX
    assert f"sid-of-user-{_CSRF_CACHE_MAX + 49}" in client._csrf_cache  # the newest is kept
    assert "sid-of-user-0" not in client._csrf_cache  # the oldest went
