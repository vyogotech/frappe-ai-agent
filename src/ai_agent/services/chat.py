"""Per-request chat orchestration.

`ChatService` holds the long-lived pieces (settings, llm, and a system-
prompt builder) and builds a fresh MCP client + tool registry per call
to `handle_message`. Every request uses the caller's sid to authenticate
with the MCP server so tool calls run under that Frappe user's
permissions.

The agent execution itself runs through `ai_agent.agent.loop.run_agent_loop`,
which drives a unified envelope schema (tool_call is a block type) via
`llm.with_structured_output(...).astream(...)`. This replaces the prior
LangGraph react-agent + envelope-formatter two-pass — that design forced
Pass-2 to mirror Pass-1's block-type choices, regressing rich-block UX on
small models.

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
from typing import Any
from uuid import uuid4

import httpx
import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from ai_agent.agent.loop import run_agent_loop
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.sid import UserContext

logger = structlog.get_logger(__name__)
_tracer = trace.get_tracer(__name__)


SystemPromptBuilder = Callable[[dict[str, Any]], str]

_TITLE_MAX_LEN = 60

# Upper bound for the MCP tools/list call. If the MCP server is unreachable or
# hung, we must not wedge the SSE generator forever — 20 s is enough for a
# cold-start listing over a slow link while being short enough that the user
# gets a clear error rather than a dead stream.
_MCP_TOOLS_LOAD_TIMEOUT_S = 20.0


# MCP tools that frappe-mcp-server keeps for backward compatibility but
# that we don't want the LLM to invoke. The project-status family was
# replaced by generic aggregate_documents / run_report flows; surfacing
# them just gives the LLM a tempting wrong-path option that fails on
# sites without the corresponding doctypes.
_DEPRECATED_TOOLS = frozenset(
    {
        "get_project_status",
        "analyze_project_timeline",
        "get_resource_allocation",
        "generate_project_report",
        "resource_utilization_analysis",
        "budget_variance_analysis",
    }
)


class _ToolsUnavailable(RuntimeError):
    """Raised at the tool-load boundary when MCP can't serve tools.

    The args carry a client-safe message. Already logged as a warning
    at the boundary; the outer chat-turn handler must NOT re-log it as
    an exception traceback — MCP being unreachable is a known
    operational state, not a programming error.
    """


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
        llm: Any,
        system_prompt_builder: SystemPromptBuilder = build_system_prompt,
        history: FrappeHistoryClient | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._build_system_prompt = system_prompt_builder
        self._history = history or FrappeHistoryClient(base_url=settings.frappe_url)

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
                with _tracer.start_as_current_span("agent.load_tools") as load_span:
                    try:
                        tools = await asyncio.wait_for(
                            mcp_client.get_tools(),
                            timeout=_MCP_TOOLS_LOAD_TIMEOUT_S,
                        )
                    except TimeoutError as exc:
                        raise RuntimeError(
                            f"MCP tools/list timed out after {_MCP_TOOLS_LOAD_TIMEOUT_S:.0f}s"
                        ) from exc
                    except Exception as exc:
                        # MCP server unreachable / refusing the handshake /
                        # returning errors. Unwrap ExceptionGroup (anyio
                        # TaskGroup wraps the real cause one or more layers
                        # deep) so the warning log records the root type
                        # and message, not the opaque outer wrapper.
                        root: BaseException = exc
                        while isinstance(root, BaseExceptionGroup) and root.exceptions:
                            root = root.exceptions[0]
                        # sid_prefix (first 8 chars) lets the operator
                        # cross-reference a failing MCP load against the
                        # Frappe session log without exposing the full sid
                        # in plaintext. A 401 here almost always means the
                        # sid we forwarded was technically non-empty but
                        # invalid/expired (the empty-sid case is caught
                        # earlier by build_mcp_client_for_sid's ValueError).
                        sid_prefix = user_context.sid[:8] if user_context.sid else None
                        logger.warning(
                            "chat_tools_load_failed",
                            session_id=session_id,
                            sid_prefix=sid_prefix,
                            error_type=type(root).__name__,
                            error=str(root)[:200],
                        )
                        load_span.set_attribute("failed", True)
                        load_span.set_attribute("error_type", type(root).__name__)
                        raise _ToolsUnavailable(_tools_unavailable_message(root)) from exc
                    load_span.set_attribute("tool_count", len(tools))

                # Drop deprecated MCP tools (project-status family, kept
                # in frappe-mcp-server for backward compat but no longer
                # documented). They confuse the LLM and surface as failed
                # tool_call cards on doctypes the user doesn't even use.
                # Filter here rather than in MCP itself so the server can
                # keep serving older clients that still depend on them.
                pre_count = len(tools)
                tools = [t for t in tools if t.name not in _DEPRECATED_TOOLS]
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
                tool_registry = ToolRegistry(tools)

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
                    async for ev in run_agent_loop(
                        llm=self._llm,
                        tool_registry=tool_registry,
                        user_message=message,
                        context_preamble=context_preamble,
                        history=None,
                        max_steps=max_steps,
                    ):
                        if ev["type"] == "tool_call":
                            tools_called.append(ev["name"])
                            tool_invocations.append({"name": ev["name"], "args": ev["arguments"]})
                        elif ev["type"] == "content_block":
                            block_events_emitted += 1
                        elif ev["type"] == "content":
                            assistant_text_parts.append(ev["text"])
                        yield ev

            except _ToolsUnavailable as exc:
                # Tool-load boundary already emitted a single-line WARNING
                # with the underlying cause; surfacing a full traceback
                # here would double-log a known operational state. Just
                # mark the turn failed and yield the pre-baked
                # client-safe message. Use the underlying cause's class
                # for error_type so the turn summary + span attrs let
                # operators bucket by real failure mode (McpError,
                # ConnectError, ...) instead of the internal wrapper.
                failed = True
                root_cause: BaseException | None = exc.__cause__
                while isinstance(root_cause, BaseExceptionGroup) and root_cause.exceptions:
                    root_cause = root_cause.exceptions[0]
                error_type = (
                    type(root_cause).__name__ if root_cause is not None else type(exc).__name__
                )
                turn_span.set_status(Status(StatusCode.ERROR, error_type))
                error_message = str(exc)
                yield {"type": "error", "message": error_message}
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
