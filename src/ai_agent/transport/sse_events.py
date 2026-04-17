"""SSE serialization for the frappe-mcp-server-compatible event stream."""

from __future__ import annotations

import json
from typing import Any

Event = dict[str, Any]


def serialize(event: Event) -> bytes:
    """Encode an event dict as an SSE data line."""
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n".encode()
