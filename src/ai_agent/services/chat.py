"""Per-request chat orchestration.

`ChatService` holds the long-lived pieces (settings, llm, and a system-
prompt builder) and builds a fresh MCP client + tool registry per call
to `handle_message`. Every request uses the caller's sid to authenticate
with the MCP server so tool calls run under that Frappe user's
permissions.

The agent execution itself runs through `ai_agent.agent.loop.run_agent_loop`,
which drives a unified envelope schema (tool_call is a block type) via
`llm.with_structured_output(...).astream(...)`.

Events yielded here must match the SSE schema in `transport.sse_events`:
`session` (announced first), `tool_call`, `content`, `content_block`,
`error`, `done`.

Chat history is persisted best-effort to Frappe via `FrappeHistoryClient`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

from ai_agent.agent.leak_filter import StreamingLeakFilter
from ai_agent.agent.loop import run_agent_loop
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.integrations.mcp import build_mcp_client_for_sid, filter_deprecated
from ai_agent.middleware.sid import UserContext

logger = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)


SystemPromptBuilder = Callable[[dict[str, Any]], str]

_TITLE_MAX_LEN = 60

# MCP tools/list timeout moved to Settings.mcp_tools_load_timeout_s so ops
# can tune it per-environment (slow LAN, busy MCP). The constant lookup
# stays local to keep the call site readable.


def _tools_unavailable_message(root: BaseException) -> str:
    """Map a tool-load root cause to a user-facing SSE error string.

    Three buckets — the structured warning log carries the exact
    type + message for operators; this string is what reaches the FE
    bubble. Picked so the user can tell "the server is down" from
    "my session expired" without reading exception classes.
    """
    if isinstance(root, httpx.HTTPStatusError):
        status = root.response.status_code
        if status in (401, 403):
            return "Tools unavailable: MCP server rejected the session (authentication failed)."
        return f"Tools unavailable: MCP server returned HTTP {status}."
    if isinstance(root, httpx.TransportError):
        return "Tools unavailable: cannot reach the MCP server."
    return "Tools unavailable."


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
        llm: BaseChatModel,
        system_prompt_builder: SystemPromptBuilder = build_system_prompt,
        history: FrappeHistoryClient | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._build_system_prompt = system_prompt_builder
        self._history = history or FrappeHistoryClient(base_url=settings.frappe_url)

    async def aclose(self) -> None:
        """Release any owned async resources (HTTP connection pools, etc.).

        Called from the FastAPI lifespan teardown so the FrappeHistoryClient's
        shared AsyncClient drops its sockets before the process exits.
        """
        await self._history.aclose()

    async def handle_message(
        self,
        *,
        message: str,
        session_id: str | None,
        context: dict[str, Any],
        user_context: UserContext,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Run the graph for one message, yielding SSE-schema events.

        Returns an AsyncGenerator (not AsyncIterator) so callers can call
        aclose() to cancel the stream cleanly on client disconnect.
        """
        tools_called: list[str] = []
        tool_invocations: list[dict[str, Any]] = []
        # Final user-visible content (text blocks from the agent loop's
        # last iteration), accumulated for history persistence.
        assistant_text_parts: list[str] = []
        block_events_emitted = 0
        failed = False
        error_message = ""
        error_type: str | None = None
        # Wall-clock timer for the turn-summary log. perf_counter is
        # monotonic and the right primitive for a duration measurement
        # (immune to system clock jumps).
        t0 = time.perf_counter()

        # `agent.chat_turn` wraps the entire turn. Inner spans
        # (load_tools, graph_run) nest under it so a trace UI shows the
        # anatomy at a glance: "the 18s turn was 0.8s load_tools +
        # 16.9s graph_run + 0.3s history writes". get_tracer returns a
        # ProxyTracer that defers to the global provider at use-time,
        # so this is a no-op when OTEL is disabled.
        with _tracer.start_as_current_span("agent.chat_turn") as turn_span:
            # Resolve / create the history session BEFORE the graph runs so
            # the user's message and the eventual assistant reply can both
            # be stored. If Frappe is unreachable, fall back to a
            # client-side id so the rest of the request still works; the
            # history is just lost for this turn.
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
            else:
                # Caller supplied an id (e.g. Frappe forwarded the browser's
                # conversation id). Ensure a matching AI Chat Session row
                # exists so the upcoming save_message calls' Link validation
                # doesn't 417. Idempotent: a duplicate-name create is
                # treated as success.
                await self._history.ensure_session(
                    sid=user_context.sid,
                    name=session_id,
                    title=_derive_title(message),
                    context_json=json.dumps(context or {}),
                )

            turn_span.set_attribute("session_id", session_id)

            # Announce the session id so the frontend can remember it and
            # pass it back on subsequent messages in the same conversation.
            # Without this round-trip every user message would land in a
            # brand-new AI Chat Session row.
            yield {"type": "session", "id": session_id}

            # Persist the user's message. Best-effort: a Frappe outage must
            # not abort the chat turn — log and continue.
            try:
                await self._history.save_message(
                    sid=user_context.sid,
                    session=session_id,
                    role="user",
                    content=message,
                )
            except Exception as exc:
                logger.warning(
                    "chat_history_user_message_write_failed",
                    session_id=session_id,
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )

            try:
                # Per-request MCP client carrying the caller's sid cookie.
                mcp_client = build_mcp_client_for_sid(self._settings, user_context.sid)
                tools: list[Any] = []
                tools_unavailable_reason: str | None = None
                tools_load_timeout_s = self._settings.mcp_tools_load_timeout_s
                with _tracer.start_as_current_span("agent.load_tools") as load_span:
                    try:
                        tools = await asyncio.wait_for(
                            mcp_client.get_tools(),
                            timeout=tools_load_timeout_s,
                        )
                    except TimeoutError as exc:
                        # Timeout deserves an explicit error event — the user
                        # likely waited the full window and is still owed a
                        # response.
                        raise RuntimeError(
                            f"MCP tools/list timed out after {tools_load_timeout_s:.0f}s"
                        ) from exc
                    except Exception as exc:
                        # MCP server unreachable / refusing the handshake /
                        # returning errors. Soft-fail: log a warning and let
                        # the agent run with an empty tool registry. The
                        # model can still answer conversational queries
                        # ("hi", "what doctypes exist") and emit "I need
                        # data tools to answer that" for data questions —
                        # both better UX than a hard 'Tools unavailable'
                        # that blocks every turn including the ones tools
                        # weren't needed for.
                        root: BaseException = exc
                        while isinstance(root, BaseExceptionGroup) and root.exceptions:
                            root = root.exceptions[0]
                        sid_prefix = user_context.sid[:8] if user_context.sid else None
                        tools_unavailable_reason = _tools_unavailable_message(root)
                        logger.warning(
                            "chat_tools_load_failed_soft_degrade",
                            session_id=session_id,
                            sid_prefix=sid_prefix,
                            mcp_url=self._settings.mcp_server_url,
                            error_type=type(root).__name__,
                            error=str(root)[:200],
                        )
                        load_span.set_attribute("failed", True)
                        load_span.set_attribute("error_type", type(root).__name__)
                        load_span.set_attribute("degraded", True)
                        tools = []
                    load_span.set_attribute("tool_count", len(tools))

                # Drop deprecated MCP tools (project-status family — see
                # `integrations.mcp.DEPRECATED_TOOLS` for the policy and
                # the list). They confuse the LLM and surface as failed
                # tool_call cards on doctypes the user doesn't even use.
                pre_count = len(tools)
                tools = filter_deprecated(tools)
                if pre_count != len(tools):
                    load_span.set_attribute("tools_filtered", pre_count - len(tools))

                logger.debug(
                    "chat_tools_loaded",
                    count=len(tools),
                    session_id=session_id,
                )

                # Per-request preamble — page context + currency + date
                # conventions + tool-use rules. Injected into the unified
                # agent system message by `build_agent_messages`. The
                # envelope schema itself is fixed in `ai_agent.blocks.envelope`.
                context_preamble = self._build_system_prompt(context or {})
                if tools_unavailable_reason is not None:
                    # Soft-degraded: tell the model so it answers data
                    # questions with a "I can't fetch that right now"
                    # text block instead of hallucinating values.
                    context_preamble += (
                        "\n\n# Tool status\n\n"
                        f"NOTE: {tools_unavailable_reason} "
                        "Answer conversational questions normally, but for "
                        "questions that need real data emit a text block "
                        "explaining tools are temporarily unavailable. "
                        "Do not fabricate data."
                    )
                tool_registry = ToolRegistry(tools)

                # Pull prior turns from this session so the LLM can resolve
                # references ("the first one", "sort by name", "yes, delete")
                # against the conversation it's actually in. Best-effort — a
                # history-load failure logs and proceeds with an empty list
                # rather than aborting the whole turn. `tmp-*` ids are unsaved
                # sessions (history is by definition empty); skip the
                # round-trip.
                history_messages: list[BaseMessage] = []
                if session_id and not session_id.startswith("tmp-"):
                    try:
                        rows = await self._history.list_messages(
                            sid=user_context.sid,
                            session=session_id,
                            limit=20,
                        )
                    except Exception as exc:
                        logger.warning(
                            "chat_history_load_failed_using_empty",
                            session_id=session_id,
                            error_type=type(exc).__name__,
                            error=str(exc)[:200],
                        )
                        rows = []
                    for row in rows:
                        if row["role"] == "user":
                            history_messages.append(HumanMessage(content=row["content"]))
                        else:
                            history_messages.append(AIMessage(content=row["content"]))

                # Unified agent loop. tool_call is a block type in the
                # envelope; the loop drives the LLM via
                # `with_structured_output(...).astream(...)` and yields
                # SSE-schema events directly.
                # `max_steps` derived from settings.agent_recursion_limit
                # (each step is at most one LLM call + tool fan-out).
                max_steps = max(1, self._settings.agent_recursion_limit // 2)
                with _tracer.start_as_current_span("agent.run") as run_span:
                    run_span.set_attribute("tool_count", len(tool_registry))
                    run_span.set_attribute("max_steps", max_steps)
                    run_span.set_attribute("history_turns", len(history_messages))
                    # Defense-in-depth: scan outgoing text chunks for
                    # system-prompt leakage. The system prompt explicitly
                    # forbids disclosure, but small instruct models can be
                    # talked past that rule. If a leak is detected mid-stream,
                    # we stop forwarding LLM output and emit a safe refusal.
                    # See BUG-019 + tests/unit/test_leak_filter.py.
                    leak_filter = StreamingLeakFilter()
                    leak_triggered = False
                    async for ev in run_agent_loop(
                        llm=self._llm,
                        tool_registry=tool_registry,
                        user_message=message,
                        context_preamble=context_preamble,
                        history=history_messages or None,
                        max_steps=max_steps,
                    ):
                        if leak_triggered:
                            # Drain remaining events without emitting them so
                            # the LLM can complete the turn but the user only
                            # sees the refusal we already published.
                            continue
                        if ev["type"] == "tool_call":
                            tools_called.append(ev["name"])
                            tool_invocations.append({"name": ev["name"], "args": ev["arguments"]})
                        elif ev["type"] == "content_block":
                            block_events_emitted += 1
                        elif ev["type"] == "content":
                            verdict = leak_filter.observe(ev.get("text", ""))
                            if verdict.leaked:
                                leak_triggered = True
                                logger.warning(
                                    "system_prompt_leak_suppressed",
                                    session_id=session_id,
                                    reason=verdict.reason,
                                )
                                # Replace whatever the model was streaming
                                # with a safe refusal. The frontend appends
                                # content chunks in arrival order; emitting
                                # an `error` chunk would settle the stream
                                # and discard text already published, so we
                                # emit refusal text and a `done`.
                                refusal = StreamingLeakFilter.SAFE_REFUSAL_MESSAGE
                                assistant_text_parts.append("\n\n" + refusal)
                                yield {"type": "content", "text": "\n\n" + refusal}
                                continue
                            assistant_text_parts.append(ev["text"])
                        yield ev

            except Exception as exc:
                failed = True
                error_type = type(exc).__name__
                # Unwrap ExceptionGroup (from anyio/asyncio TaskGroup) to
                # the real cause — otherwise the FE shows the opaque outer
                # "unhandled errors in a TaskGroup (N sub-exceptions)"
                # instead of the actual auth/MCP/LLM failure underneath.
                display_exc: BaseException = exc
                while isinstance(display_exc, BaseExceptionGroup) and display_exc.exceptions:
                    display_exc = display_exc.exceptions[0]
                logger.exception(
                    "chat_handle_message_failed",
                    session_id=session_id,
                    sid_present=bool(user_context.sid),
                    error_type=type(exc).__name__,
                    root_cause_type=type(display_exc).__name__,
                    error=str(display_exc)[:300],
                    llm_provider=self._settings.llm_provider,
                    llm_model=self._settings.llm_model,
                    llm_base_url=self._settings.llm_base_url,
                )
                # Record on the active span so a trace UI shows ERROR
                # status without consulting the log. record_exception
                # captures the type/message/stacktrace as a span event.
                turn_span.record_exception(exc)
                turn_span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
                # Show the exception type plus the first line of its
                # message, capped at 500 chars. Full tracebacks stay in
                # the structured log, but this is an
                # internally-authenticated agent — withholding the whole
                # error breaks debugging for no real security gain.
                first_line = str(display_exc).splitlines()[0] if str(display_exc) else ""
                detail = first_line[:500]
                error_message = (
                    f"{type(display_exc).__name__}: {detail}"
                    if detail
                    else type(display_exc).__name__
                )
                yield {"type": "error", "message": error_message}

            # Persist the final assistant message (success or error).
            # Best-effort: if this fails it is logged inside the client and
            # we still emit `done`.
            assistant_content = (
                f"[error] {error_message}" if failed else "".join(assistant_text_parts)
            )
            tool_args_json: str | None = None
            if tool_invocations:
                try:
                    tool_args_json = json.dumps(tool_invocations)
                except (TypeError, ValueError):
                    # Arguments weren't JSON-serialisable — drop them silently.
                    tool_args_json = None
            try:
                await self._history.save_message(
                    sid=user_context.sid,
                    session=session_id,
                    role="assistant",
                    content=assistant_content,
                    tool_args_json=tool_args_json,
                )
            except Exception as exc:
                logger.warning(
                    "chat_history_assistant_message_write_failed",
                    session_id=session_id,
                    error_type=type(exc).__name__,
                    error=str(exc)[:200],
                )

            yield {
                "type": "done",
                "tools_called": tools_called,
                "data_quality": "low" if failed else "high",
                "timestamp": _utcnow_rfc3339_z(),
            }

            # Final summary attributes on the chat_turn span — these are
            # what a trace UI shows as the per-turn rollup. Mirrors the
            # turn-summary log fields below; the log is for stdout-based
            # aggregation, the span is for trace-UI navigation. Both are
            # kept because operators reach for whichever tool is in front
            # of them.
            content_chars = sum(len(p) for p in assistant_text_parts)
            turn_span.set_attribute("tools_called_count", len(tools_called))
            turn_span.set_attribute("content_chars", content_chars)
            turn_span.set_attribute("block_events_emitted", block_events_emitted)
            turn_span.set_attribute("failed", failed)
            if error_type is not None:
                turn_span.set_attribute("error_type", error_type)

            # Single info-level audit event per turn. One log line answers
            # "what happened on this chat call" without grepping multiple
            # streams; failed=True funnels error_type so dashboards can
            # bucket failures by class. Emitted after `done` so a cancelled
            # turn (client aclose) is not summarised as completed.
            duration_ms = (time.perf_counter() - t0) * 1000.0
            summary: dict[str, Any] = {
                "session_id": session_id,
                "duration_ms": duration_ms,
                "tools_called": tools_called,
                "tools_called_count": len(tools_called),
                "content_chars": content_chars,
                "block_events_emitted": block_events_emitted,
                "failed": failed,
            }
            if error_type is not None:
                summary["error_type"] = error_type
            logger.info("chat_turn_completed", **summary)
