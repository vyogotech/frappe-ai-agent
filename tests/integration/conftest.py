"""Shared fixtures for integration tests.

These tests run against real services (Ollama, MCP, Frappe). Endpoints and
credentials come from `AI_AGENT_INTEGRATION_*` env vars set by the CI
workflow. When a required var is missing the relevant tests skip, so
`pytest -m integration` is safe to run locally even without the stack up.
"""

from __future__ import annotations

import os

import httpx
import pytest


def env(name: str) -> str | None:
    return os.getenv(name) or None


@pytest.fixture(scope="session")
def frappe_url() -> str:
    url = env("AI_AGENT_INTEGRATION_FRAPPE_URL")
    if not url:
        pytest.skip("AI_AGENT_INTEGRATION_FRAPPE_URL not set")
    return url


@pytest.fixture(scope="session")
def frappe_admin_password() -> str:
    pwd = env("AI_AGENT_INTEGRATION_FRAPPE_ADMIN_PASSWORD")
    if not pwd:
        pytest.skip("AI_AGENT_INTEGRATION_FRAPPE_ADMIN_PASSWORD not set")
    return pwd


@pytest.fixture(scope="session")
def frappe_sid(frappe_url: str, frappe_admin_password: str) -> str:
    """Log in to Frappe as Administrator and return the session cookie.

    Frappe's `/api/method/login` returns a `Set-Cookie: sid=...` on success;
    that sid is the credential the agent forwards to MCP and Frappe REST.
    """
    response = httpx.post(
        f"{frappe_url}/api/method/login",
        data={"usr": "Administrator", "pwd": frappe_admin_password},
        timeout=30.0,
    )
    response.raise_for_status()
    sid = response.cookies.get("sid")
    if not sid:
        pytest.fail(f"Frappe login did not return a sid cookie: {response.text!r}")
    return sid
