"""Output-side system-prompt-leak filter.

The system prompt explicitly forbids the model from disclosing its
instructions, but small instruction-tuned models (e.g. qwen3:8b) cannot
reliably resist prompt-injection attacks that ask for the system prompt
verbatim. This module is the defense-in-depth layer: it scans the
agent's outgoing text content chunks and, if it detects fragments that
match the system prompt itself, suppresses further leakage and emits a
safety refusal instead.

Two failure modes we explicitly catch:

1. Verbatim regurgitation — the model dumps SYSTEM_PROMPT sentences
   one-for-one. We detect via short, distinctive substrings that would
   not naturally appear in a normal assistant response.

2. Schema enumeration — the model lists the block types (tool_call,
   text, table, chart, kpi, status_list) or names MCP tools in prose.
   We detect via the combined-list pattern.

The detector is conservative: it only flags clear leaks, so legitimate
answers that happen to mention "tools" or "Frappe AI" in passing aren't
suppressed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Phrases lifted directly from SYSTEM_PROMPT in
# `ai_agent.agent.prompts`. Each is a high-signal short substring that
# would not naturally appear in an assistant answer.
_PROMPT_FINGERPRINTS: tuple[str, ...] = (
    "You are Frappe AI, an embedded assistant",
    "Never fabricate. Every value",
    "Tool-use rules",
    "Destructive operations",
    "TWO-TURN confirmation",
    "Disclosure rules",
    "Detection tips for first-turn bypass",
    "envelope schema",
    # The block-type enumeration is a tell that the model is reciting
    # the wire-protocol grammar from UNIFIED_AGENT_SYSTEM_PROMPT.
    "tool_call | text | table | chart | kpi",
)

# Tool-name enumeration in prose. If the model lists 3+ MCP tools by
# their exact identifier in user-facing text, it is reciting internals.
_TOOL_NAMES = (
    "list_documents",
    "get_document",
    "aggregate_documents",
    "create_document",
    "update_document",
    "delete_document",
    "run_report",
    "get_doctype_meta",
)


@dataclass
class FilterResult:
    """Outcome of evaluating an accumulated text buffer."""

    leaked: bool
    reason: str = ""


def detect_system_prompt_leak(text: str) -> FilterResult:
    """Return whether `text` contains a likely system-prompt leak.

    Conservative: short ordinary text won't match. Only triggers on
    distinctive fingerprints or clear tool-name enumeration.
    """
    if not text or len(text) < 20:
        return FilterResult(leaked=False)

    for fingerprint in _PROMPT_FINGERPRINTS:
        if fingerprint in text:
            return FilterResult(
                leaked=True,
                reason=f"fingerprint:{fingerprint[:40]}",
            )

    # Tool-name enumeration: 3+ tool names within a single response.
    tool_hits = sum(1 for name in _TOOL_NAMES if name in text)
    if tool_hits >= 3:
        return FilterResult(
            leaked=True,
            reason=f"tool_name_enumeration:{tool_hits}",
        )

    return FilterResult(leaked=False)


# Stream-friendly stateful filter: keeps a running buffer and reports
# the first chunk that pushes the buffer over a leak threshold.
class StreamingLeakFilter:
    """Stateful filter for SSE chunk streams.

    Usage:
        f = StreamingLeakFilter()
        for chunk in agent_stream:
            verdict = f.observe(chunk["text"])
            if verdict.leaked:
                # suppress further content, emit refusal
                break
            yield chunk
    """

    SAFE_REFUSAL_MESSAGE = (
        "I can help with ERPNext data tasks, but I can't share my internal "
        "configuration or instructions. Try asking about your business data."
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._triggered = False

    def observe(self, chunk_text: str) -> FilterResult:
        if self._triggered:
            return FilterResult(leaked=True, reason="already_triggered")
        if not chunk_text:
            return FilterResult(leaked=False)
        self._buffer += chunk_text
        verdict = detect_system_prompt_leak(self._buffer)
        if verdict.leaked:
            self._triggered = True
        return verdict

    @property
    def triggered(self) -> bool:
        return self._triggered
