"""Per-request chat orchestration.

`ChatService` holds the long-lived pieces (settings, llm, checkpointer, and a
system-prompt builder) and builds a fresh MCP client + LangGraph agent per call
to `handle_message`. Every request uses the caller's sid to authenticate with
the MCP server so tool calls run under that Frappe user's permissions.

Events yielded here must match the SSE schema in `transport.sse_events`:
`status`, `tool_call`, `content`, `done`, `error`.

Chat history is persisted best-effort to Frappe via `FrappeHistoryClient`:
we create a session if none is supplied, record the user's message before
the graph runs, and record the final assistant message when it finishes.
History write failures are logged but never abort the conversation.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent

from ai_agent.agent.graph import create_agent_graph
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.agent.tool_errors import to_tool_result_message
from ai_agent.blocks.parser import parse_blocks
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.sid import UserContext

logger = structlog.get_logger(__name__)


SystemPromptBuilder = Callable[[dict[str, Any]], str]

_TITLE_MAX_LEN = 60


def _utcnow_rfc3339_z() -> str:
    """RFC3339 timestamp ending in `Z` (matches frappe-mcp-server format)."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _derive_title(message: str) -> str:
    """First ~60 chars of the user's message, trimmed, for session title."""
    stripped = message.strip()
    if len(stripped) <= _TITLE_MAX_LEN:
        return stripped
    return stripped[:_TITLE_MAX_LEN]


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
        history: FrappeHistoryClient | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._checkpointer = checkpointer
        self._build_system_prompt = system_prompt_builder
        self._history = history or FrappeHistoryClient(base_url=settings.frappe_url)

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
        tool_invocations: list[dict[str, Any]] = []
        assistant_text_parts: list[str] = []
        failed = False
        error_message: str | None = None

        # Resolve / create the history session BEFORE the graph runs so the
        # user's message and the eventual assistant reply can both be stored.
        # If Frappe is unreachable, fall back to a client-side id so the rest
        # of the request still works; the history is just lost for this turn.
        if session_id is None:
            created = await self._history.create_session(
                sid=user_context.sid,
                title=_derive_title(message),
                context_json=json.dumps(context or {}),
            )
            if created is None:
                session_id = f"tmp-{uuid4().hex[:8]}"
                logger.warning(
                    "chat_history_session_create_failed_using_tmp",
                    session_id=session_id,
                )
            else:
                session_id = created

        # Announce the session id so the frontend can remember it and
        # pass it back on subsequent messages in the same conversation.
        # Without this round-trip every user message would land in a
        # brand-new AI Chat Session row.
        yield {"type": "session", "id": session_id}

        # Persist the user's message. Best-effort; failures do not abort.
        await self._history.save_message(
            sid=user_context.sid,
            session=session_id,
            role="user",
            content=message,
        )

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
            graph_config: RunnableConfig = {
                "configurable": {"thread_id": session_id or "default"},
                # Smaller local models (qwen3.5:9b, llama3.1:8b, ...) are
                # prone to tool-call loops — they'll list the same doctype
                # over and over exploring the schema. The LangGraph default
                # of 25 trips before they converge. 50 is enough headroom
                # without letting a truly-stuck agent run forever.
                "recursion_limit": 50,
            }

            async for event in graph.astream_events(
                graph_input,
                config=graph_config,
                version="v2",
            ):
                translated = self._translate_event(event, tools_called, tool_invocations)
                if translated is None:
                    continue
                if translated["type"] != "content":
                    yield translated
                    continue
                # Content events may carry inlined <ai-block> tags.
                # Parse into an ordered list of blocks. If the text has at
                # least one real block, emit them as content_block events so
                # the frontend can render each via its block component. Pure
                # prose (no tags) is a single text block — fall back to a
                # normal content event so simple answers keep the plain path.
                assistant_text_parts.append(translated["text"])
                parsed = parse_blocks(translated["text"])
                has_real_blocks = any(b.type != "text" for b in parsed)
                if not has_real_blocks:
                    yield translated
                    continue
                for block in parsed:
                    yield {"type": "content_block", "block": block.model_dump()}

        except Exception as exc:
            failed = True
            error_message = str(exc)
            logger.exception(
                "chat_handle_message_failed",
                session_id=session_id,
                sid_present=bool(user_context.sid),
            )
            yield {"type": "error", "message": error_message}

        # Persist the final assistant message (success or error). Best-effort:
        # if this fails it is logged inside the client and we still emit `done`.
        assistant_content = f"[error] {error_message}" if failed else "".join(assistant_text_parts)
        tool_args_json: str | None = None
        if tool_invocations:
            try:
                tool_args_json = json.dumps(tool_invocations)
            except (TypeError, ValueError):
                # Arguments weren't JSON-serialisable — drop them silently.
                tool_args_json = None
        await self._history.save_message(
            sid=user_context.sid,
            session=session_id,
            role="assistant",
            content=assistant_content,
            tool_args_json=tool_args_json,
        )

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
        event: StreamEvent,
        tools_called: list[str],
        tool_invocations: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Map one LangGraph v2 event to an SSE-schema dict, or None to skip."""
        kind = event.get("event")

        if kind == "on_tool_start":
            name = event.get("name") or "unknown"
            args = event.get("data", {}).get("input") or {}
            tools_called.append(name)
            tool_invocations.append({"name": name, "args": args})
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
