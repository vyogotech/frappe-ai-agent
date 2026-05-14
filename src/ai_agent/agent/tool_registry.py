"""Thin tool registry over MCP / LangChain `BaseTool` instances.

The agent loop in `ai_agent.agent.loop` drives a unified envelope schema
where tool calls are JUST a block type — there is no LangGraph
`ToolNode`, no `bind_tools`, no per-provider tool-call wire format. All
the loop needs is "what tools exist, what do they do, and how do I call
one of them by name and get a stringified result back."

`ToolRegistry` is that surface. It wraps the `list[BaseTool]` returned
by `MultiServerMCPClient.get_tools()` and exposes:

- `schemas()` — a catalog rendering for the system prompt
- `names()` — for quick existence checks
- `ainvoke(name, args)` — the only entry point the loop calls; folds any
  exception into a tool-result string so a tool error is data the LLM
  reasons about, not an abort that kills the SSE stream

Per-tool error handling lives here instead of being patched onto each
`BaseTool` (the old `install_tool_error_handler` approach) so the
registry is the single chokepoint for both error policy and the
LangChain dependency.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool


def _is_permission_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like a Frappe permission denial?"""
    status_code = getattr(exc, "status_code", None)
    if status_code in (401, 403):
        return True
    text = str(exc).lower()
    return any(
        phrase in text
        for phrase in (
            "permission denied",
            "not permitted",
            "forbidden",
            "does not have permission",
            "not authorized",
            "unauthorized",
        )
    )


def _exception_to_result(exc: Exception) -> str:
    """Convert any exception into a tool-result string the LLM can act on.

    Free of stack traces / internal details. Permission errors are
    distinguished from generic failures so the LLM can surface the
    right user-facing message.
    """
    if _is_permission_error(exc):
        return f"Access denied: permission error — {exc}"
    return f"Tool call failed: {exc}"


class ToolRegistry:
    """Wraps a list of LangChain BaseTool into a name → tool lookup."""

    def __init__(self, tools: list[BaseTool]) -> None:
        # Dedup-by-name (last-write-wins) — paranoia for MCP servers that
        # surface two tools with the same name across namespaces.
        self._by_name: dict[str, BaseTool] = {t.name: t for t in tools}

    def names(self) -> set[str]:
        return set(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def schemas(self) -> str:
        """Render the tool catalog as a string for the system prompt.

        Format: one tool per block. `name` + first-paragraph description
        + arg schema. Kept compact so the model can scan it; long
        descriptions are truncated.
        """
        if not self._by_name:
            return "(no tools available this turn)"
        lines: list[str] = []
        for name in sorted(self._by_name):
            tool = self._by_name[name]
            description = (tool.description or "").strip()
            # First paragraph only, capped at 200 chars.
            first_para = description.split("\n\n", 1)[0].strip()
            if len(first_para) > 200:
                first_para = first_para[:200].rstrip() + "…"
            args_schema = _render_args_schema(tool)
            lines.append(f"- {name}: {first_para}\n  args: {args_schema}")
        return "\n".join(lines)

    async def ainvoke(self, name: str, args: dict[str, Any] | None) -> str:
        """Run the named tool with `args`; always return a stringified result.

        Any exception becomes a tool-result string. Unknown tool names
        return an `error:` string so the model can correct itself.
        """
        if name not in self._by_name:
            return f"error: unknown tool {name!r}; available: {sorted(self._by_name)}"
        tool = self._by_name[name]
        try:
            raw = await tool.ainvoke(args or {})
        except Exception as exc:
            return _exception_to_result(exc)
        # MCP tools return strings or pydantic models; coerce uniformly.
        if isinstance(raw, str):
            return raw
        if hasattr(raw, "model_dump"):
            return json.dumps(raw.model_dump(), ensure_ascii=False)
        return str(raw)


def _render_args_schema(tool: BaseTool) -> str:
    """Render a tool's argument schema as compact JSON for the prompt.

    `BaseTool.args_schema` is typically a pydantic model. We render its
    JSON schema (just the `properties` map) so the LLM sees field names,
    types, and descriptions in a familiar shape.
    """
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return "{}"
    try:
        # Pydantic v2 BaseModel
        if hasattr(args_schema, "model_json_schema"):
            schema = args_schema.model_json_schema()
        else:
            # Already a dict — MCP adapter sometimes hands us raw JSON-schema
            schema = dict(args_schema)
    except Exception:
        return "{}"
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    if not properties:
        return "{}"
    parts: list[str] = []
    for field_name, field_schema in properties.items():
        t = field_schema.get("type", "any")
        if t == "array":
            inner = (field_schema.get("items") or {}).get("type", "any")
            t = f"array<{inner}>"
        mark = "" if field_name in required else "?"
        desc = field_schema.get("description")
        if desc:
            parts.append(f"{field_name}{mark}: {t} — {desc[:80]}")
        else:
            parts.append(f"{field_name}{mark}: {t}")
    return "{" + "; ".join(parts) + "}"
