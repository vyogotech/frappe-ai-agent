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

__all__ = ["BLOCK_ENVELOPE_SCHEMA", "envelope_to_markup"]


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
