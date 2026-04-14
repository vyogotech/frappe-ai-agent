# tests/unit/test_sid_middleware.py
from dataclasses import FrozenInstanceError

import pytest
from starlette.requests import Request

from ai_agent.middleware.sid import UserContext, extract_user_context


def _make_request(cookie_value: str | None) -> Request:
    headers = []
    if cookie_value is not None:
        headers.append((b"cookie", f"sid={cookie_value}".encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/chat",
        "headers": headers,
        "query_string": b"",
    }
    return Request(scope=scope)


def test_extract_user_context_returns_userctx_when_sid_present():
    req = _make_request("abc123")
    ctx = extract_user_context(req)
    assert isinstance(ctx, UserContext)
    assert ctx.sid == "abc123"


def test_extract_user_context_returns_none_when_sid_missing():
    req = _make_request(None)
    ctx = extract_user_context(req)
    assert ctx is None


def test_extract_user_context_returns_none_when_sid_empty():
    req = _make_request("")
    ctx = extract_user_context(req)
    assert ctx is None


def test_extract_user_context_returns_none_when_sid_whitespace_only():
    req = _make_request("   ")
    ctx = extract_user_context(req)
    assert ctx is None


def test_user_context_is_frozen():
    ctx = UserContext(sid="abc123")
    with pytest.raises(FrozenInstanceError):
        ctx.sid = "other"  # type: ignore[misc]


def test_user_context_rejects_empty_sid_at_construction():
    with pytest.raises(ValueError):
        UserContext(sid="")


def test_user_context_rejects_whitespace_sid_at_construction():
    with pytest.raises(ValueError):
        UserContext(sid="   ")
