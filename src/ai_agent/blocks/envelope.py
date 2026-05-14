"""JSON-envelope strategy for constrained-decoding LLMs.

Small Ollama models (≤4B) cannot reliably emit `<ai-block type="...">{...JSON...}
</ai-block>` markup directly — they leak markdown fences, miss closing tags, and
mangle inline JSON. To make them production-safe we instead ask them to emit a
JSON envelope shaped like:

    {"blocks": [{"type": "table", "payload": {...}}, ...]}

and enforce it at the token level via Ollama's `format=<schema>` constrained-
decoding hook. The envelope's per-type payload schemas mirror the pydantic
block models (ChartBlock, TableBlock, etc.) closely enough that downstream
validation rarely catches anything.

`envelope_to_markup` translates a model response (raw JSON or JSON wrapped in
prose / `​```json` fence) back into the `<ai-block>` markup that
`ai_agent.blocks.parser.parse_blocks` expects, so the downstream parser and
streaming splitter remain unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

__all__ = [
    "BLOCK_ENVELOPE_SCHEMA",
    "ENVELOPE_FORMATTER_SYSTEM_PROMPT",
    "build_formatter_messages",
    "envelope_to_markup",
    "iter_complete_blocks",
]


_TEXT_PAYLOAD: dict = {
    "type": "object",
    "properties": {"content": {"type": "string"}},
    "required": ["content"],
    "additionalProperties": False,
}

_COLUMN: dict = {
    "type": "object",
    "properties": {
        "key": {"type": "string"},
        "label": {"type": "string"},
        "format": {
            "type": "string",
            "enum": ["text", "currency", "number", "percent", "date"],
        },
    },
    "required": ["key", "label"],
    "additionalProperties": False,
}

_DOC_ROUTE: dict = {
    "type": "object",
    "properties": {"doctype": {"type": "string"}, "name": {"type": "string"}},
    "required": ["doctype", "name"],
    "additionalProperties": False,
}

_TABLE_PAYLOAD: dict = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "columns": {"type": "array", "items": _COLUMN, "minItems": 1, "maxItems": 12},
        "rows": {
            "type": "array",
            "maxItems": 100,
            "items": {
                "type": "object",
                "properties": {
                    "values": {"type": "object"},
                    "route": _DOC_ROUTE,
                },
                "required": ["values"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "columns", "rows"],
    "additionalProperties": False,
}

_DATASET: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "values": {"type": "array", "items": {"type": ["number", "null"]}},
    },
    "required": ["name", "values"],
    "additionalProperties": False,
}

_CHART_PAYLOAD: dict = {
    "type": "object",
    "properties": {
        "chart_type": {
            "type": "string",
            "enum": ["bar", "line", "pie", "funnel", "heatmap", "calendar"],
        },
        "title": {"type": "string"},
        "data": {
            "type": "object",
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}},
                "datasets": {"type": "array", "items": _DATASET, "minItems": 1},
            },
            "required": ["labels", "datasets"],
            "additionalProperties": False,
        },
        "options": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["number", "currency", "percent"]},
                "currency": {"type": "string"},
                "stacked": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "required": ["chart_type", "title", "data"],
    "additionalProperties": False,
}

_KPI_PAYLOAD: dict = {
    "type": "object",
    "properties": {
        "metrics": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": ["number", "string"]},
                    "format": {
                        "type": "string",
                        "enum": ["number", "currency", "percent", "text"],
                    },
                    "trend": {"type": "string", "enum": ["up", "down", "flat"]},
                    "trend_value": {"type": "string"},
                },
                "required": ["label", "value"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["metrics"],
    "additionalProperties": False,
}

_STATUS_LIST_PAYLOAD: dict = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "items": {
            "type": "array",
            "minItems": 1,
            "maxItems": 50,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "status": {"type": "string"},
                    "color": {
                        "type": "string",
                        "enum": ["green", "red", "yellow", "blue", "gray"],
                    },
                    "route": _DOC_ROUTE,
                },
                "required": ["label", "status", "color"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "items"],
    "additionalProperties": False,
}

# The envelope uses `oneOf` per block type so Ollama (via llama.cpp's JSON-
# schema-to-GBNF grammar) enforces per-type payload shape at every token. This
# is what stops small models from echoing JSON-schema-like nested objects into
# payloads (a real failure mode on smollm2:1.7b at A5 in the prior baseline).
BLOCK_ENVELOPE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "blocks": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"type": {"const": "text"}, "payload": _TEXT_PAYLOAD},
                        "required": ["type", "payload"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"type": {"const": "table"}, "payload": _TABLE_PAYLOAD},
                        "required": ["type", "payload"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"type": {"const": "chart"}, "payload": _CHART_PAYLOAD},
                        "required": ["type", "payload"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"type": {"const": "kpi"}, "payload": _KPI_PAYLOAD},
                        "required": ["type", "payload"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "status_list"},
                            "payload": _STATUS_LIST_PAYLOAD,
                        },
                        "required": ["type", "payload"],
                        "additionalProperties": False,
                    },
                ]
            },
        }
    },
    "required": ["blocks"],
    "additionalProperties": False,
}


_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_STRUCTURED_TYPES = frozenset({"table", "chart", "kpi", "status_list"})


def envelope_to_markup(raw: str) -> str:
    """Translate a JSON envelope LLM response into `<ai-block>` markup.

    Handles three real model behaviors:
    - bare JSON (the happy path under `format=` constrained decoding)
    - JSON wrapped in `​```json` fences or prose (Gemma 3 4B is documented to
      add commentary around the JSON even with constrained decoding)
    - text-block-only envelopes (translate to bare prose; the parser treats
      orphan prose as a TextBlock)

    Raises ValueError on empty input or input that contains no JSON object.
    """
    if not raw or not raw.strip():
        raise ValueError("empty response")
    s = raw.strip()
    s = _FENCE_OPEN.sub("", s)
    s = _FENCE_CLOSE.sub("", s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        m = _JSON_OBJECT.search(s)
        if not m:
            raise
        data = json.loads(m.group(0))
    if not isinstance(data, dict) or "blocks" not in data:
        raise ValueError("envelope missing 'blocks'")

    parts: list[str] = []
    for entry in data.get("blocks", []):
        if not isinstance(entry, dict):
            continue
        btype = entry.get("type")
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        if btype == "text":
            content = str(payload.get("content") or "")
            if content:
                parts.append(content)
            continue
        if btype not in _STRUCTURED_TYPES:
            continue
        json_payload = json.dumps(payload, ensure_ascii=False)
        parts.append(f'<ai-block type="{btype}">\n{json_payload}\n</ai-block>')
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Pass-2 (envelope formatter) prompt + message builder
# --------------------------------------------------------------------------- #

ENVELOPE_FORMATTER_SYSTEM_PROMPT = """\
You are a FORMATTER. Take the AGENT DRAFT (the model's free-form answer
from the agent loop) and the TOOL RESULTS (data the agent fetched) and
emit one JSON object matching this envelope:

{"blocks":[{"type":"text|table|chart|kpi|status_list","payload":{...}}, ...]}

Emit only that JSON object — no prose around it, no markdown fence.

## Payload by type

text:
  {"content":"narrative markdown"}

kpi:
  {"metrics":[
    {"label","value","format":"currency|number|percent|text",
     "trend":"up|down|flat","trend_value"}
  ]}

table:
  {"title","columns":[
    {"key","label","format":"text|currency|number|percent|date"}
  ],"rows":[
    {"values":{"<key>":<value>},"route":{"doctype","name"}}
  ]}

chart:
  {"chart_type":"bar|line|pie|funnel|heatmap|calendar","title",
   "data":{"labels":["..."],
           "datasets":[{"name","values":[1,2,3]}]},
   "options":{"format":"number|currency|percent"}}

status_list:
  {"title","items":[
    {"label","status","color":"green|red|yellow|blue|gray",
     "route":{"doctype","name"}}
  ]}

## How to choose blocks

- 3+ rows or 2+ columns of tabular data → table.
- A trend or comparison across labels → chart.
- One or a few headline numbers → kpi.
- A small list of items with state/status → status_list.
- Conversational answer or a "no data" message → text.
- Multi-block answers (e.g., narrative + table) emit each piece in
  display order.

Numbers in currency-formatted cells are RAW (the frontend formats them).
For chart datasets, `values` length MUST equal `labels` length; use null
for missing points."""


def build_formatter_messages(
    *,
    user_message: str,
    tool_results: list[dict[str, Any]],
    draft: str,
    system_prompt: str = ENVELOPE_FORMATTER_SYSTEM_PROMPT,
) -> list[SystemMessage | HumanMessage]:
    """Compose the messages for the Pass-2 envelope formatter call.

    `tool_results` is a list of `{"name","args","result"}` dicts captured
    from `on_tool_end` events during the Pass-1 graph run. `draft` is the
    Pass-1 model's free-form final message (its text was suppressed from
    user-visible streaming and accumulated here).
    """
    if tool_results:
        tool_log = "\n".join(
            f"- {tr.get('name', '?')}({json.dumps(tr.get('args') or {}, ensure_ascii=False)})"
            f" → {_truncate(str(tr.get('result') or ''), 2000)}"
            for tr in tool_results
        )
    else:
        tool_log = "(no tools called)"

    body = (
        f"USER ASKED:\n{user_message}\n\n"
        f"TOOL RESULTS:\n{tool_log}\n\n"
        f"AGENT DRAFT:\n{draft or '(empty)'}"
    )
    return [SystemMessage(content=system_prompt), HumanMessage(content=body)]


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "…(truncated)"


# --------------------------------------------------------------------------- #
# Streaming: detect newly-completed blocks across partial-dict yields
# --------------------------------------------------------------------------- #


def iter_complete_blocks(
    partial: dict[str, Any] | None,
    state: dict[str, Any],
    *,
    final: bool = False,
) -> Iterator[dict[str, Any]]:
    """Yield block dicts that have JUST completed in `partial`.

    The `with_structured_output().astream()` API emits a dict snapshot on
    every token. We want to detect when a particular block index has
    transitioned from "still streaming" to "definitely closed" so we can
    translate it to ai-block markup and emit a content_block event.

    The reliable signal is "the NEXT block has started" (or it's the
    final yield). At that point the closing brace of the current block
    has arrived; partial-JSON parsing for numbers (digit-by-digit) is
    only known-done when followed by a structural token.

    `state` is a mutable dict the caller passes across yields. It tracks
    which indices have been emitted so we don't re-yield.
    """
    state.setdefault("emitted", set())
    if not isinstance(partial, dict):
        return
    blocks = partial.get("blocks")
    if not isinstance(blocks, list):
        return
    n = len(blocks)
    for i, b in enumerate(blocks):
        if i in state["emitted"]:
            continue
        # Block at index i is "done" when:
        # - a later block (i+1, etc.) has started in the partial — its
        #   closing brace must have arrived for the next object to open,
        # - OR we're at the final yield (entire envelope closed).
        next_started = i < n - 1
        if not (next_started or final):
            continue
        if not isinstance(b, dict) or "type" not in b or "payload" not in b:
            # Malformed block — skip but mark emitted so we don't loop.
            state["emitted"].add(i)
            continue
        state["emitted"].add(i)
        yield b
