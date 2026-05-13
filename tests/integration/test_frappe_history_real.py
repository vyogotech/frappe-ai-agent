"""Real Frappe REST write through FrappeHistoryClient.

Creates an AI Chat Session row on a live Frappe site using the Administrator
sid. Exercises the CSRF-scrape path (`GET /app` → regex token → cached) and
the resource POST. Asserts a server-assigned `name` came back, which is the
only signal that the write actually landed.
"""

from __future__ import annotations

import os
import uuid

import pytest

from ai_agent.integrations.frappe_history import FrappeHistoryClient

pytestmark = pytest.mark.integration

_FRAPPE_URL = os.getenv("AI_AGENT_INTEGRATION_FRAPPE_URL")


@pytest.mark.skipif(
    not _FRAPPE_URL,
    reason="set AI_AGENT_INTEGRATION_FRAPPE_URL to run",
)
async def test_create_session_writes_to_frappe(frappe_sid: str):
    assert _FRAPPE_URL is not None  # narrowed by skipif; for pyright
    client = FrappeHistoryClient(base_url=_FRAPPE_URL)
    title = f"integration-test-{uuid.uuid4().hex[:8]}"
    name = await client.create_session(
        sid=frappe_sid,
        title=title,
        context_json="{}",
    )
    assert name, f"create_session returned no name for title={title!r}"
