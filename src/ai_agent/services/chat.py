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

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import structlog
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.schema import StreamEvent

from ai_agent.agent.graph import create_agent_graph
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.agent.tool_errors import install_tool_error_handler
from ai_agent.blocks.parser import parse_blocks
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.sid import UserContext

logger = structlog.get_logger(__name__)


SystemPromptBuilder = Callable[[dict[str, Any]], str]

_TITLE_MAX_LEN = 60

# Upper bound for the MCP tools/list call. If the MCP server is unreachable or
# hung, we must not wedge the SSE generator forever — 20 s is enough for a
# cold-start listing over a slow link while being short enough that the user
# gets a clear error rather than a dead stream.
_MCP_TOOLS_LOAD_TIMEOUT_S = 20.0


_AI_BLOCK_OPEN = "<ai-block"
_AI_BLOCK_CLOSE = "</ai-block>"


class _BlockStreamSplitter:
    """State machine that splits streamed LLM text into prose and block markup.

    The LLM emits per-token chunks via `on_chat_model_stream`. Streaming each
    chunk as a content event would leak partial `<ai-block ...>...</ai-block>`
    markup into the FE bubble (the user sees raw HTML scrolling in until the
    closing tag arrives). Instead, this splitter:

    - Streams chunks of prose as soon as they're "safe" (not a partial open
      tag), as ("content", text) events.
    - Buffers chunks once an `<ai-block` is detected, until the matching
      `</ai-block>` arrives, then emits the complete markup as a single
      ("block", markup) event for parse_blocks() to handle.

    Yields `(kind, payload)` tuples where `kind` is `"content"` or `"block"`.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_block = False

    def feed(self, chunk: str):
        self._buf += chunk
        while True:
            if not self._in_block:
                idx = self._buf.find(_AI_BLOCK_OPEN)
                if idx >= 0:
                    if idx > 0:
                        yield ("content", self._buf[:idx])
                    self._buf = self._buf[idx:]
                    self._in_block = True
                    continue
                # No open tag yet. Hold back any suffix that matches a
                # prefix of "<ai-block" so we don't leak a partial open
                # tag into the FE bubble.
                safe_end = len(self._buf)
                for n in range(min(len(_AI_BLOCK_OPEN) - 1, len(self._buf)), 0, -1):
                    if _AI_BLOCK_OPEN.startswith(self._buf[-n:]):
                        safe_end = len(self._buf) - n
                        break
                if safe_end > 0:
                    yield ("content", self._buf[:safe_end])
                    self._buf = self._buf[safe_end:]
                break
            else:
                end_idx = self._buf.find(_AI_BLOCK_CLOSE)
                if end_idx < 0:
                    break
                end = end_idx + len(_AI_BLOCK_CLOSE)
                yield ("block", self._buf[:end])
                self._buf = self._buf[end:]
                self._in_block = False

    def flush(self):
        """Emit any residual buffered text. Called once the LLM stream ends.

        If we're stuck inside a block (LLM cut off mid-tag), the partial
        markup is emitted as content so the user at least sees what
        arrived, instead of silently losing it.
        """
        if self._buf:
            yield ("content", self._buf)
            self._buf = ""
            self._in_block = False


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
        error_message = ""

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
            # Per-request MCP client carrying the caller's sid cookie.
            mcp_client = build_mcp_client_for_sid(self._settings, user_context.sid)
            try:
                tools = await asyncio.wait_for(
                    mcp_client.get_tools(),
                    timeout=_MCP_TOOLS_LOAD_TIMEOUT_S,
                )
            except TimeoutError as exc:
                raise RuntimeError(
                    f"MCP tools/list timed out after {_MCP_TOOLS_LOAD_TIMEOUT_S:.0f}s"
                ) from exc

            # Install an error handler on every tool so exceptions raised by
            # individual tool calls become LLM-visible tool observations
            # instead of aborting the whole graph run. `install_tool_error_handler`
            # both wraps the coroutine (so non-ToolException errors are
            # re-raised as ToolException) and sets `handle_tool_error` —
            # both are needed because LangChain's built-in hook only catches
            # ToolException, and MCP/Frappe errors don't subclass it.
            for tool in tools:
                install_tool_error_handler(tool)

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

            # Splitter buffers `<ai-block>...</ai-block>` markup across token
            # chunks so the FE never sees partial HTML. Prose tokens stream
            # through; complete block tags are parsed and emitted as
            # content_block events that the FE renders via getBlockComponent.
            splitter = _BlockStreamSplitter()

            def _emit_split(kind: str, payload: str):
                if kind == "content":
                    return [{"type": "content", "text": payload}]
                # Block markup. parse_blocks may return TextBlock segments
                # if the LLM nested prose-like content inside the tag; emit
                # those as plain content too so the FE markdown path
                # picks them up.
                events_out: list[dict[str, Any]] = []
                for block in parse_blocks(payload):
                    if block.type == "text":
                        events_out.append({"type": "content", "text": block.content})
                    else:
                        events_out.append({"type": "content_block", "block": block.model_dump()})
                return events_out

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
                assistant_text_parts.append(translated["text"])
                for kind, payload in splitter.feed(translated["text"]):
                    for ev in _emit_split(kind, payload):
                        yield ev

            # Stream ended. Flush any text the splitter is still holding
            # (only happens if the LLM cut off mid-tag).
            for kind, payload in splitter.flush():
                for ev in _emit_split(kind, payload):
                    yield ev

        except Exception as exc:
            failed = True
            logger.exception(
                "chat_handle_message_failed",
                session_id=session_id,
                sid_present=bool(user_context.sid),
                error_type=type(exc).__name__,
            )
            # Show the exception type plus the first line of its message,
            # capped at 500 chars. Full tracebacks stay in the structured
            # log, but this is an internally-authenticated agent — withholding
            # the whole error breaks debugging for no real security gain.
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            detail = first_line[:500]
            error_message = f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
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

        if kind == "on_chat_model_stream":
            # Per-token streaming. Each event carries an AIMessageChunk;
            # concatenated, the chunks form the final assistant message.
            # We emit only chunks that have actual text and are NOT part of
            # a tool-calling step. tool_call_chunks is the streaming
            # counterpart to tool_calls — its presence means the model is
            # currently emitting a tool invocation, not a user-visible reply.
            chunk = event.get("data", {}).get("chunk")
            if not isinstance(chunk, AIMessageChunk):
                return None
            if getattr(chunk, "tool_call_chunks", None):
                return None
            content = chunk.content
            text = content if isinstance(content, str) else ""
            if not text:
                return None
            return {"type": "content", "text": text}

        # on_tool_end, on_chat_model_start, on_chat_model_end, on_chain_*,
        # etc. are swallowed.
        return None
