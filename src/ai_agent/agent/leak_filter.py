"""Output-side filter that suppresses a reply reciting the system prompt."""

from __future__ import annotations

from dataclasses import dataclass

# Phrases lifted directly from SYSTEM_PROMPT in
# `ai_agent.agent.prompts`. Each is a high-signal short substring that
# would not naturally appear in an assistant answer.
_PROMPT_FINGERPRINTS: tuple[str, ...] = (
    "You are Frappe AI, an embedded assistant",
    "Never fabricate. Every value",
    "Tool-use rules",
    "Writes need the user's confirmation",
    "never ask the user to type a confirmation phrase",
    "Disclosure rules",
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
    leaked: bool
    reason: str = ""


# Below this, a fragment is too short to hold a fingerprint in a meaningful context.
_MIN_LEN = 20
# The longest pattern, less one character: a window that carries this much of what came
# before is wide enough for a pattern split across two chunks to be whole in it.
_OVERLAP = max(len(p) for p in (*_PROMPT_FINGERPRINTS, *_TOOL_NAMES)) - 1


def _fingerprint_in(text: str) -> str | None:
    return next((f for f in _PROMPT_FINGERPRINTS if f in text), None)


def detect_system_prompt_leak(text: str) -> FilterResult:
    """Return whether `text` likely leaks the system prompt; a passing mention of tools does not."""
    if not text or len(text) < _MIN_LEN:
        return FilterResult(leaked=False)

    fingerprint = _fingerprint_in(text)
    if fingerprint is not None:
        return FilterResult(leaked=True, reason=f"fingerprint:{fingerprint[:40]}")

    # Tool-name enumeration: 3+ tool names within a single response.
    tool_hits = sum(1 for name in _TOOL_NAMES if name in text)
    if tool_hits >= 3:
        return FilterResult(
            leaked=True,
            reason=f"tool_name_enumeration:{tool_hits}",
        )

    return FilterResult(leaked=False)


class StreamingLeakFilter:
    """Leak check over a stream: once a chunk trips it, every later chunk reports a leak."""

    SAFE_REFUSAL_MESSAGE = (
        "I can help with ERPNext data tasks, but I can't share my internal "
        "configuration or instructions. Try asking about your business data."
    )

    def __init__(self) -> None:
        # The tail of what has arrived, not all of it: scanning the whole answer again on
        # every delta costs the square of its length, on the loop that is streaming it.
        self._tail = ""
        self._chars = 0
        self._tools_seen: set[str] = set()
        self._triggered = False

    def observe(self, chunk_text: str) -> FilterResult:
        if self._triggered:
            return FilterResult(leaked=True, reason="already_triggered")
        if not chunk_text:
            return FilterResult(leaked=False)
        window = self._tail + chunk_text
        self._chars += len(chunk_text)
        self._tail = window[-_OVERLAP:]
        if self._chars < _MIN_LEN:
            return FilterResult(leaked=False)

        fingerprint = _fingerprint_in(window)
        if fingerprint is not None:
            self._triggered = True
            return FilterResult(leaked=True, reason=f"fingerprint:{fingerprint[:40]}")

        # Names are counted over the whole answer, so each one is remembered once it is seen.
        self._tools_seen.update(name for name in _TOOL_NAMES if name in window)
        if len(self._tools_seen) >= 3:
            self._triggered = True
            return FilterResult(
                leaked=True, reason=f"tool_name_enumeration:{len(self._tools_seen)}"
            )
        return FilterResult(leaked=False)

    @property
    def triggered(self) -> bool:
        return self._triggered
