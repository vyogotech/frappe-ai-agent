# src/ai_agent/middleware/sid.py
"""sid cookie auth — single source of truth for who is calling the agent."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from starlette.requests import Request


@dataclass(frozen=True, slots=True)
class UserContext:
    """The caller's Frappe session; `sid` goes on every call made for this user."""

    sid: str

    def __post_init__(self) -> None:
        if not self.sid or not self.sid.strip():
            raise ValueError("UserContext.sid must be a non-empty, non-whitespace string")


def extract_user_context(request: Request) -> UserContext | None:
    """Return a UserContext if the request carries a non-empty sid cookie."""
    sid = request.cookies.get("sid")
    if not sid or not sid.strip():
        return None
    return UserContext(sid=sid)


async def signed_in_user(frappe_url: str, sid: str) -> str | None:
    """The user signed in under `sid`, or None if unknown; a fresh client, so no cookie is kept.

    Raises:
        httpx.HTTPError: Frappe cannot be asked.
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(
            f"{frappe_url}/api/method/frappe.auth.get_logged_user",
            headers={"Cookie": f"sid={sid}", "Accept": "application/json"},
        )
    if response.status_code in (401, 403):
        return None
    response.raise_for_status()
    user = response.json().get("message")
    return user if isinstance(user, str) and user and user != "Guest" else None
