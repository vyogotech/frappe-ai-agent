"""Parse <copilot-block> tags out of assistant text into structured blocks.

The LLM is instructed (via the system prompt) to wrap structured data in
<copilot-block type="X">JSON</copilot-block> tags. This module extracts
those tags and splits the surrounding prose into an ordered list of
blocks that preserve the original interleaving.

Output shape matches the frontend's ContentBlock union:
    {"type": "text",       "content": "..."}
    {"type": "table",      "columns": [...], "rows": [...], ...}
    {"type": "chart",      "chart_type": "bar", "data": {...}, ...}
    {"type": "kpi",        "metrics": [...]}
    {"type": "status_list","items": [...], ...}

Malformed block payloads fall back to a text block containing the raw
body so no content is silently dropped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_BLOCK_RE = re.compile(
    r'<copilot-block\s+type="([^"]+)"\s*>(.*?)</copilot-block>',
    re.DOTALL,
)


def parse_blocks(text: str) -> list[dict[str, Any]]:
    """Split text into an ordered list of blocks.

    Plain text segments become ``{"type": "text", "content": ...}``.
    Tagged blocks become ``{"type": <declared>, **json_payload}``.
    Returns an empty list for empty/whitespace input.
    """
    if not text or not text.strip():
        return []

    blocks: list[dict[str, Any]] = []
    last_end = 0

    for match in _BLOCK_RE.finditer(text):
        block_type = match.group(1)
        body = match.group(2).strip()

        # Text between the previous block (or start) and this one
        if match.start() > last_end:
            prose = text[last_end : match.start()].strip()
            if prose:
                blocks.append({"type": "text", "content": prose})

        # The block payload
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                blocks.append({"type": block_type, **payload})
            else:
                raise ValueError(f"block payload is not an object: {type(payload).__name__}")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "copilot_block_parse_failed", extra={"type": block_type, "error": str(exc)}
            )
            blocks.append({"type": "text", "content": body})

        last_end = match.end()

    # Trailing text after the last block
    if last_end < len(text):
        prose = text[last_end:].strip()
        if prose:
            blocks.append({"type": "text", "content": prose})

    return blocks
