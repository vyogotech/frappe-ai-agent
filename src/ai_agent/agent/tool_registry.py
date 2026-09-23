"""Name-to-tool registry over the MCP tools, and the one place a tool error becomes data."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from langchain_core.tools import BaseTool

logger = structlog.get_logger(__name__)


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
    """Convert any exception into a tool-result string the LLM can act on."""
    if _is_permission_error(exc):
        return f"Access denied: permission error — {exc}"
    return f"Tool call failed: {exc}"


class ToolRegistry:
    """Wraps a list of LangChain BaseTool into a name → tool lookup."""

    def __init__(self, tools: list[BaseTool], timeout_s: float = 30.0) -> None:
        # Dedup-by-name (last-write-wins) — paranoia for MCP servers that
        # surface two tools with the same name across namespaces.
        self._by_name: dict[str, BaseTool] = {t.name: t for t in tools}
        self._timeout_s = timeout_s
        for t in tools:
            # langchain-mcp-adapters 0.3 returns an MCP isError result as ordinary output
            t.handle_tool_error = False

    def names(self) -> set[str]:
        return set(self._by_name)

    def __len__(self) -> int:
        return len(self._by_name)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def writes(self, name: str) -> bool:
        """Whether the tool needs confirmation: anything but a declared read does."""
        tool = self._by_name.get(name)
        if tool is None:
            return True
        return (tool.metadata or {}).get("readOnlyHint") is not True

    def schemas(self) -> str:
        """Render the tool catalog as a string for the system prompt."""
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
        """Run the named tool; never raises: an unknown name or an error comes back as a string."""
        if name not in self._by_name:
            return f"error: unknown tool {name!r}; available: {sorted(self._by_name)}"
        tool = self._by_name[name]
        try:
            raw = await asyncio.wait_for(tool.ainvoke(args or {}), self._timeout_s)
        except TimeoutError:
            logger.warning("tool_call_timed_out", tool=name, timeout_s=self._timeout_s)
            return f"Tool call failed: {name} timed out after {self._timeout_s:.0f}s"
        except Exception as exc:  # noqa: BLE001 - every tool failure is a result the model reads
            logger.warning(
                "tool_call_failed", tool=name, error_type=type(exc).__name__, error=str(exc)[:200]
            )
            return _exception_to_result(exc)
        # MCP tools return strings, pydantic models, or (langchain-mcp-adapters) a list of
        # content blocks; coerce uniformly. Text blocks become their text: str() of the list
        # handed the model a Python repr with block ids in it.
        if isinstance(raw, str):
            return raw
        if (
            isinstance(raw, list)
            and raw
            and all(isinstance(b, dict) and b.get("type") == "text" for b in raw)
        ):
            return "\n".join(str(b.get("text", "")) for b in raw)
        model_dump = getattr(raw, "model_dump", None)
        if callable(model_dump):
            return json.dumps(model_dump(), ensure_ascii=False)
        return str(raw)


# Arguments the agent fills itself, never the model: a knowledge search is scoped to the chat
# the question came from, and offering `session` would let the model aim it at another chat.
AGENT_ARGS = {"search_knowledge_base": {"session"}}


def _render_args_schema(tool: BaseTool) -> str:
    """Render a tool's argument schema as compact JSON for the prompt."""
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
    except (TypeError, ValueError):  # pydantic's PydanticUserError is a TypeError
        return "{}"
    hidden = AGENT_ARGS.get(tool.name, set())
    properties = {k: v for k, v in (schema.get("properties") or {}).items() if k not in hidden}
    required = set(schema.get("required") or []) - hidden
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
