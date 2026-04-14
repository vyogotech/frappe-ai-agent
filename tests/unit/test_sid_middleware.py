# tests/unit/test_sid_middleware.py
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
