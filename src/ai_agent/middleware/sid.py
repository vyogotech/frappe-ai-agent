# src/ai_agent/middleware/sid.py
"""sid cookie auth — single source of truth for who is calling the agent."""

from __future__ import annotations

from dataclasses import dataclass

from starlette.requests import Request


@dataclass(frozen=True, slots=True)
class UserContext:
    """The caller's Frappe session.

    The sid is the Frappe session cookie that authorises every downstream
    call (MCP tool calls, Frappe REST writes). It must be forwarded on every
    request made on behalf of this user.
    """

    sid: str

    def __post_init__(self) -> None:
        if not self.sid or not self.sid.strip():
            raise ValueError("UserContext.sid must be a non-empty, non-whitespace string")


def extract_user_context(request: Request) -> UserContext | None:
    """Return a UserContext if the request carries a non-empty sid cookie.

    Returns None if the cookie is missing, empty, or whitespace-only.
    """
    sid = request.cookies.get("sid")
    if not sid or not sid.strip():
        return None
    return UserContext(sid=sid)
