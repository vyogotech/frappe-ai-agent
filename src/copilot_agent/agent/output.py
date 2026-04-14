"""Parse agent LLM output into content blocks."""

from __future__ import annotations

from copilot_agent.blocks.models import ContentBlock
from copilot_agent.blocks.parser import parse_blocks


def parse_agent_output(text: str) -> list[ContentBlock]:
    """Convert raw LLM text output to a list of content blocks."""
    return parse_blocks(text)
