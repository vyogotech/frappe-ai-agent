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

# Some smaller models (qwen3.5:9b, etc.) emit chart subtypes as the top-level
# block type (e.g. <ai-block type="pie">) instead of <ai-block type="chart">
# with chart_type:"pie" inside. Treat those as chart aliases — the inner JSON
# is forwarded to ChartBlock with the alias filled in as chart_type when the
# model omits it. Keep this list in sync with ChartBlock.chart_type Literal.
_CHART_ALIASES: dict[str, str] = {
    "bar": "bar",
    "line": "line",
    "pie": "pie",
    "funnel": "funnel",
    "heatmap": "heatmap",
    "calendar": "calendar",
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

        chart_alias = _CHART_ALIASES.get(block_type)
        if chart_alias is not None:
            model_cls: type[ContentBlock] = ChartBlock
        elif (mapped := _BLOCK_TYPE_MAP.get(block_type)) is not None:
            model_cls = mapped
        else:
            logger.warning("unknown_block_type", block_type=block_type)
            blocks.append(TextBlock(content=f"[Unknown block type: {block_type}]"))
            continue

        try:
            data = json.loads(json_str)
            if chart_alias is not None:
                # Alias path: the LLM used <ai-block type="pie"> instead of
                # the canonical <ai-block type="chart"> with chart_type:"pie".
                # Populate chart_type from the alias if the inner JSON omitted
                # it; otherwise trust whatever the inner JSON says.
                data.setdefault("chart_type", chart_alias)
            block = model_cls.model_validate(data)
            blocks.append(validate_block(block))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "block_parse_error",
                block_type=block_type,
                error=str(exc),
                raw=json_str[:500] if json_str else None,
            )
            # Surface the raw content as plain text so the user sees what the
            # agent tried to send. Truncate to keep the chat bubble readable.
            fallback_content = json_str if json_str else "[block parse error]"
            if len(fallback_content) > 1000:
                fallback_content = fallback_content[:1000] + "\n…(truncated)"
            blocks.append(TextBlock(content=fallback_content))

    # Capture trailing text
    trailing = text[last_end:].strip()
    if trailing:
        blocks.append(TextBlock(content=trailing))

    # If no blocks found at all, treat entire text as a TextBlock
    if not blocks and text.strip():
        blocks.append(TextBlock(content=text.strip()))

    return blocks
