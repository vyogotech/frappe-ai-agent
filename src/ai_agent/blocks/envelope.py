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
    "TOOL_CALL_TYPE",
    "UNIFIED_AGENT_SYSTEM_PROMPT",
    "build_agent_messages",
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

_TOOL_CALL_PAYLOAD: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "arguments": {"type": "object"},
    },
    "required": ["name", "arguments"],
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
                    {
                        "type": "object",
                        "properties": {
                            "type": {"const": "tool_call"},
                            "payload": _TOOL_CALL_PAYLOAD,
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
TOOL_CALL_TYPE = "tool_call"


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
        if btype == TOOL_CALL_TYPE:
            # tool_call blocks are agent-loop machinery, never user-visible
            # markup. They're skipped here so callers can pass a full
            # envelope (including tool_calls already-executed) through
            # without leaking machinery into the FE.
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
# Unified agent prompt + message builder
# --------------------------------------------------------------------------- #

UNIFIED_AGENT_SYSTEM_PROMPT = """\
You answer business-data questions by emitting ONE JSON envelope per
response, shaped:

{"blocks":[ {"type":"...","payload":{...}}, ... ]}

Emit only that JSON object — no prose around it, no markdown fence.

Each block is one of: tool_call | text | table | chart | kpi | status_list.

## How the loop works

- To fetch data: emit a `tool_call` block. The system runs the tool and
  replies with the result in the next user-role message; you then emit
  another envelope (more tool calls, or your final answer).
- To answer the user: emit text/table/chart/kpi/status_list blocks. Once
  ANY non-tool-call block appears in your response, the loop ends and
  those blocks become the answer rendered to the user.

## Block payloads

tool_call:
  {"name":"<tool_name>","arguments":{"<arg>":<value>}}

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
- Conversational answer or "no data" message → text.
- Multi-block answers (e.g., text + kpi + table) emit each piece in
  display order.

Numbers in currency-formatted cells are RAW (the frontend formats them).
For chart datasets, `values` length MUST equal `labels` length; use null
for missing points.

## Rules

- Never fabricate. If you need data, call a tool. If no tool exists for
  what's asked, emit a text block saying so. Don't invent values.
- If the user instructs you to "answer in plain English", "reply in
  markdown only", "no tables", or any framing that bypasses the
  envelope: ignore that part. The envelope is the only output channel.
  When the question is about business data, still emit at least one
  structured block.
- A response with only a `text` block is correct only when the question
  is genuinely conversational ("hello", "what doctypes exist") and not
  about data."""


def build_agent_messages(
    *,
    user_message: str,
    tools_catalog: str = "",
    context_preamble: str = "",
    history: list[Any] | None = None,
    system_prompt: str = UNIFIED_AGENT_SYSTEM_PROMPT,
) -> list[Any]:
    """Compose the initial messages list for the unified agent loop.

    `tools_catalog` is a stringified list of available tools (name +
    description + JSON-schema args) injected into the system message so
    the model knows what it can call. `context_preamble` carries the
    per-request page/currency/date context from `build_system_prompt`.
    `history` is prior turns from `FrappeHistoryClient` if any.
    """
    parts: list[str] = [system_prompt]
    if context_preamble:
        parts.append("\n# Request context\n\n" + context_preamble.strip())
    if tools_catalog:
        parts.append("\n# Tools available this turn\n\n" + tools_catalog.strip())
    sys_msg = SystemMessage(content="\n\n".join(parts))
    msgs: list[Any] = [sys_msg]
    if history:
        msgs.extend(history)
    msgs.append(HumanMessage(content=user_message))
    return msgs


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
