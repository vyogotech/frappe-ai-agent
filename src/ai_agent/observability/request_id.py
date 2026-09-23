"""The correlation id one answer carries from the browser through Frappe to MCP (ADR-008)."""

from __future__ import annotations

import re
import uuid

import structlog

# the one correlation header frappe.monitor parses (frappe/monitor.py:79)
_FRAPPE_HEADER = "X-Frappe-Request-Id"

# Frappe mints the id as a uuid4 and its own monitor accepts [0-9a-fA-F-]{8,64}; this is the wider
# shape ADR-008 settled on. An inbound id is another service's word, and unvalidated it would put an
# unbounded caller-chosen string on every log line of the request.
_SHAPE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def accept(inbound: str | None) -> str:
    """`inbound` when it has the agreed shape, else a fresh uuid4."""
    return inbound if inbound and _SHAPE.fullmatch(inbound) else str(uuid.uuid4())


def current() -> str:
    """The id the middleware bound for this request, or "" outside one."""
    return structlog.contextvars.get_contextvars().get("request_id", "")


def frappe_header() -> dict[str, str]:
    """The header a call to Frappe carries, or nothing outside a request."""
    request_id = current()
    return {_FRAPPE_HEADER: request_id} if request_id else {}
