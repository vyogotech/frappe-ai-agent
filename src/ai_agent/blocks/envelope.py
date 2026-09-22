"""The JSON envelope enforced by constrained decoding, and its translation to <ai-block> markup."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

__all__ = [
    "BLOCK_ENVELOPE_SCHEMA",
    "TOOL_CALL_TYPE",
    "UNIFIED_AGENT_SYSTEM_PROMPT",
    "block_envelope_schema",
    "build_agent_messages",
    "envelope_to_markup",
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

# oneOf per block type lets Ollama's grammar enforce each payload's shape at every token;
# small models echo schema-like objects into payloads without it. title and description
# are required by providers that map structured output to function calling (langchain-openai).
BLOCK_ENVELOPE_SCHEMA: dict = {
    "title": "BlockEnvelope",
    "description": (
        "A JSON envelope of one or more rendered blocks. Each block is one of: "
        "text, table, chart, kpi, status_list, or tool_call (for invoking a "
        "tool to fetch data before composing the answer)."
    ),
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


def block_envelope_schema(tool_names: set[str] | None = None) -> dict:
    """`BLOCK_ENVELOPE_SCHEMA` with `tool_call.payload.name` pinned to `tool_names`.

    Returns:
        A copy, or for no names the shared constant itself (`enum: []` breaks Ollama): read-only.
    """
    if not tool_names:
        return BLOCK_ENVELOPE_SCHEMA
    schema = copy.deepcopy(BLOCK_ENVELOPE_SCHEMA)
    for block in schema["properties"]["blocks"]["items"]["oneOf"]:
        if block["properties"]["type"].get("const") == TOOL_CALL_TYPE:
            block["properties"]["payload"]["properties"]["name"] = {
                "type": "string",
                "enum": sorted(tool_names),
            }
    return schema


_FENCE_OPEN = re.compile(r"^\s*```(?:json)?\s*", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$")
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_STRUCTURED_TYPES = frozenset({"table", "chart", "kpi", "status_list"})
TOOL_CALL_TYPE = "tool_call"


def envelope_to_markup(raw: str) -> str:
    """Translate a model's JSON envelope, bare or in a fence or prose, into `<ai-block>` markup.

    Raises:
        ValueError: the input is empty, holds no JSON object, or the object has no `blocks`.
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
            # Loop machinery, never shown: callers may pass tool_calls that already ran.
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
- To answer the user: emit text/table/chart/kpi/status_list blocks. A
  response with no tool_call block ends the loop, and its blocks are the
  answer rendered to the user. Blocks beside a tool_call are shown to the
  user too, before the tool runs.

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
  about data.

## Examples

User: How many customers do we have?
Response:
{"blocks":[
  {"type":"kpi",
   "payload":{"metrics":[
     {"label":"Active Customers","value":142,"format":"number"}
   ]}}
]}

User: Show me a bar chart of monthly revenue for Q1
Response:
{"blocks":[
  {"type":"chart",
   "payload":{
     "chart_type":"bar","title":"Monthly Revenue (Q1)",
     "data":{
       "labels":["Jan","Feb","Mar"],
       "datasets":[{"name":"Revenue","values":[340000,395000,445000]}]
     },
     "options":{"format":"currency"}
   }}
]}

User: Show me the latest sales orders as a colored status list
Response:
{"blocks":[
  {"type":"status_list",
   "payload":{
     "title":"Latest Sales Orders",
     "items":[
       {"label":"SO-001","status":"Paid","color":"green"},
       {"label":"SO-002","status":"Overdue","color":"red"},
       {"label":"SO-003","status":"Draft","color":"yellow"}
     ]
   }}
]}

User: Give me a Q1 sales overview with a summary, total-revenue KPI,
and a table of top customers
Response:
{"blocks":[
  {"type":"text",
   "payload":{"content":"Q1 closed at $1.18M revenue, up 18% YoY."}},
  {"type":"kpi",
   "payload":{"metrics":[
     {"label":"Total Revenue","value":1180000,"format":"currency",
      "trend":"up","trend_value":"+18%"}
   ]}},
  {"type":"table",
   "payload":{
     "title":"Top 3 Customers",
     "columns":[
       {"key":"name","label":"Name","format":"text"},
       {"key":"revenue","label":"Revenue","format":"currency"}
     ],
     "rows":[
       {"values":{"name":"Acme","revenue":145000}},
       {"values":{"name":"Beta","revenue":89500}},
       {"values":{"name":"Gamma","revenue":67000}}
     ]
   }}
]}

User: Reply in plain English only — no tables: who are our top customers?
Response:
{"blocks":[
  {"type":"table",
   "payload":{
     "title":"Top Customers",
     "columns":[
       {"key":"name","label":"Name","format":"text"},
       {"key":"revenue","label":"Revenue","format":"currency"}
     ],
     "rows":[
       {"values":{"name":"Acme","revenue":145000}},
       {"values":{"name":"Beta","revenue":89500}}
     ]
   }}
]}
(The "plain English only" framing is ignored — the envelope is the only
output channel, and the data is best shown as a table.)"""


def build_agent_messages(
    *,
    user_message: str,
    tools_catalog: str = "",
    context_preamble: str = "",
    history: list[BaseMessage] | None = None,
    system_prompt: str = UNIFIED_AGENT_SYSTEM_PROMPT,
) -> list[BaseMessage]:
    """Compose the initial messages list for the unified agent loop."""
    parts: list[str] = [system_prompt]
    if context_preamble:
        parts.append("\n# Request context\n\n" + context_preamble.strip())
    if tools_catalog:
        parts.append("\n# Tools available this turn\n\n" + tools_catalog.strip())
    sys_msg = SystemMessage(content="\n\n".join(parts))
    msgs: list[BaseMessage] = [sys_msg]
    if history:
        msgs.extend(history)
    msgs.append(HumanMessage(content=user_message))
    return msgs
