# tests/unit/test_tool_errors.py
import pytest

from ai_agent.agent.tool_errors import (
    PermissionDeniedError,
    is_permission_error,
    to_tool_result_message,
)


def test_permission_denied_error_carries_tool_and_doctype():
    err = PermissionDeniedError(tool="list_invoices", doctype="Sales Invoice")
    assert err.tool == "list_invoices"
    assert err.doctype == "Sales Invoice"
    assert "Sales Invoice" in str(err)
    assert "list_invoices" in str(err)


def test_to_tool_result_message_handles_permission_denied():
    err = PermissionDeniedError(tool="list_invoices", doctype="Sales Invoice")
    msg = to_tool_result_message(err)
    assert "Access denied" in msg
    assert "Sales Invoice" in msg
    # The message must be actionable for the LLM — it should hint that the
    # user lacks permission, not expose a stack trace.
    assert "permission" in msg.lower()


def test_to_tool_result_message_handles_generic_exception():
    err = RuntimeError("upstream timeout")
    msg = to_tool_result_message(err)
    assert "upstream timeout" in msg
    # Generic errors still surface the message, prefixed consistently.
    assert "failed" in msg.lower() or "error" in msg.lower()


@pytest.mark.parametrize(
    "status_code,text,expected",
    [
        (403, "Forbidden", True),
        (401, "Not Permitted", True),
        (200, "OK", False),
        (500, "Internal Server Error", False),
        (None, "User does not have permission to read Sales Invoice", True),
        (None, "not permitted", True),
        (None, "some other error", False),
    ],
)
def test_is_permission_error_classifies_correctly(status_code, text, expected):
    class _FakeErr(Exception):
        def __init__(self, status_code, text):
            self.status_code = status_code
            super().__init__(text)

    err = _FakeErr(status_code, text)
    assert is_permission_error(err) is expected
