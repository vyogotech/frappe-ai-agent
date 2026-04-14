"""Per-request chat orchestration.

`ChatService` holds the long-lived pieces (settings, llm, checkpointer, and a
system-prompt builder) and builds a fresh MCP client + LangGraph agent per call
to `handle_message`. Every request uses the caller's sid to authenticate with
the MCP server so tool calls run under that Frappe user's permissions.

Events yielded here must match the SSE schema in `transport.sse_events`:
`status`, `tool_call`, `content`, `done`, `error`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any, Callable

import structlog
from langchain_core.messages import AIMessage, HumanMessage

from ai_agent.agent.graph import create_agent_graph
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.agent.tool_errors import to_tool_result_message
from ai_agent.config import Settings
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.sid import UserContext

logger = structlog.get_logger(__name__)


SystemPromptBuilder = Callable[[dict[str, Any]], str]


def _utcnow_rfc3339_z() -> str:
    """RFC3339 timestamp ending in `Z` (matches frappe-mcp-server format)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ChatService:
    """Per-request agent invocation.

    Instances are shared across requests but carry no per-user state. The
    per-request graph + MCP client are built inside `handle_message`.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        llm: Any,
        checkpointer: Any,
        system_prompt_builder: SystemPromptBuilder = build_system_prompt,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._checkpointer = checkpointer
        self._build_system_prompt = system_prompt_builder

    async def handle_message(
        self,
        *,
        message: str,
        session_id: str | None,
        context: dict[str, Any],
        user_context: UserContext,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run the graph for one message, yielding SSE-schema events."""
        tools_called: list[str] = []
        failed = False

        try:
            yield {"type": "status", "message": "loading tools..."}

            # Per-request MCP client carrying the caller's sid cookie.
            mcp_client = build_mcp_client_for_sid(self._settings, user_context.sid)
            tools = await mcp_client.get_tools()

            # Install an error handler on every tool so exceptions raised by
            # individual tool calls become LLM-visible tool observations
            # instead of aborting the whole graph run. The ToolNode in
            # LangGraph catches the exception and returns the handler's
            # string as the tool result; the LLM then responds gracefully.
            for tool in tools:
                tool.handle_tool_error = to_tool_result_message

            logger.debug(
                "chat_tools_loaded",
                count=len(tools),
                session_id=session_id,
            )

            # Per-request prompt lets the UI pass page context per message.
            system_prompt = self._build_system_prompt(context or {})

            # Cheap: create_react_agent just wires a graph around the model
            # and tool list. No network calls here.
            graph = create_agent_graph(
                llm=self._llm,
                tools=tools,
                system_prompt=system_prompt,
                checkpointer=self._checkpointer,
            )

            yield {"type": "status", "message": "thinking..."}

            graph_input = {"messages": [HumanMessage(content=message)]}
            graph_config = {
                "configurable": {"thread_id": session_id or "default"},
            }

            async for event in graph.astream_events(
                graph_input,
                config=graph_config,
                version="v2",
            ):
                translated = self._translate_event(event, tools_called)
                if translated is not None:
                    yield translated

        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            failed = True
            logger.exception(
                "chat_handle_message_failed",
                session_id=session_id,
                sid_present=bool(user_context.sid),
            )
            yield {"type": "error", "message": str(exc)}

        yield {
            "type": "done",
            "tools_called": tools_called,
            "data_quality": "low" if failed else "high",
            "timestamp": _utcnow_rfc3339_z(),
        }

    # ------------------------------------------------------------------ #
    # Event translation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _translate_event(
        event: dict[str, Any],
        tools_called: list[str],
    ) -> dict[str, Any] | None:
        """Map one LangGraph v2 event to an SSE-schema dict, or None to skip."""
        kind = event.get("event")

        if kind == "on_tool_start":
            name = event.get("name") or "unknown"
            args = event.get("data", {}).get("input") or {}
            tools_called.append(name)
            return {"type": "tool_call", "name": name, "arguments": args}

        if kind == "on_chat_model_end":
            message = event.get("data", {}).get("output")
            if not isinstance(message, AIMessage):
                return None
            # Skip intermediate AI messages that exist only to call tools —
            # only the final natural-language answer should reach the user.
            if getattr(message, "tool_calls", None):
                return None
            content = message.content
            if not content:
                return None
            text = content if isinstance(content, str) else str(content)
            if not text.strip():
                return None
            return {"type": "content", "text": text}

        # on_tool_end, on_chat_model_start, on_chain_*, etc. are swallowed.
        return None
