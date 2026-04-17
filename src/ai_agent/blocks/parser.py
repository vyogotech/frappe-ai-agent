"""Parse LLM structured output into content block instances."""

from __future__ import annotations

import json
import re

import structlog
from pydantic import ValidationError

from ai_agent.blocks.models import (
    ChartBlock,
    ContentBlock,
    KPIBlock,
    StatusListBlock,
    TableBlock,
    TextBlock,
)
from ai_agent.blocks.validators import validate_block

logger = structlog.get_logger()

_BLOCK_TYPE_MAP: dict[str, type[ContentBlock]] = {
    "text": TextBlock,
    "chart": ChartBlock,
    "table": TableBlock,
    "kpi": KPIBlock,
    "status_list": StatusListBlock,
}

_BLOCK_PATTERN = re.compile(
    r"<ai-block\s+type=\"(\w+)\">\s*(.*?)\s*</ai-block>",
    re.DOTALL,
)


def parse_blocks(text: str) -> list[ContentBlock]:
    """Extract content blocks from LLM output text.

    Falls back to TextBlock for malformed JSON or unknown types.
    """
    blocks: list[ContentBlock] = []
    last_end = 0

    for match in _BLOCK_PATTERN.finditer(text):
        # Capture any plain text before this block
        prefix = text[last_end : match.start()].strip()
        if prefix:
            blocks.append(TextBlock(content=prefix))

        block_type = match.group(1)
        json_str = match.group(2)
        last_end = match.end()

        model_cls = _BLOCK_TYPE_MAP.get(block_type)
        if not model_cls:
            logger.warning("unknown_block_type", block_type=block_type)
            blocks.append(TextBlock(content=f"[Unknown block type: {block_type}]"))
            continue

        try:
            data = json.loads(json_str)
            block = model_cls.model_validate(data)
            blocks.append(validate_block(block))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("block_parse_error", block_type=block_type, error=str(exc))
            blocks.append(TextBlock(content="[Could not render block]"))

    # Capture trailing text
    trailing = text[last_end:].strip()
    if trailing:
        blocks.append(TextBlock(content=trailing))

    # If no blocks found at all, treat entire text as a TextBlock
    if not blocks and text.strip():
        blocks.append(TextBlock(content=text.strip()))

    return blocks
