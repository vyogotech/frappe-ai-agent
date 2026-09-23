"""A chat turn per request, its MCP client built with the caller's sid so tools run as them."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import anyio
import httpx
import structlog
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import LLMResult
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

from ai_agent.agent.leak_filter import StreamingLeakFilter
from ai_agent.agent.loop import TurnFailure, cap_for_prompt, run_agent_loop
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

TURN_TOO_LONG = "This took too long. Try a shorter question."
TOOLS_TIMED_OUT = "The assistant's tools timed out. Try again."
ANSWER_FAILED = "The answer could not be completed. Try again."
ANSWER_STOPPED = "The answer was stopped. Ask again for a full answer."

# The confirmed turn's question, written here rather than by the model: the user clicked a button,
# so there is no message of theirs to replay and nothing of the model's may stand in for one.
CONFIRMED_TURN = "The user allowed this action and it has run. Tell them the result."

# The turn is already over when a stopped answer is written, so that save gets its own budget.
_STOPPED_SAVE_TIMEOUT_S = 5


async def _until(
    deadline: float, events: AsyncGenerator[dict[str, Any], None]
) -> AsyncGenerator[dict[str, Any], None]:
    """Yield the turn's events, failing it once `deadline` (a loop clock reading) has passed."""
    while True:
        try:
            # armed around the loop's own await, never around the yield below: a timeout there
            # would cancel whoever is reading the stream instead of the turn
            async with asyncio.timeout_at(deadline):
                event = await anext(events)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise TurnFailure(TURN_TOO_LONG) from exc
        yield event


def _root_cause(exc: BaseException) -> BaseException:
    """The first concrete exception under any ExceptionGroup, as anyio's TaskGroup raises."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _tools_unavailable_message(root: BaseException) -> str:
    """Map a tool-load root cause to the tool-status note put in the model's prompt."""
    if isinstance(root, httpx.HTTPStatusError):
        status = root.response.status_code
        if status in (401, 403):
            return "Tools unavailable: MCP server rejected the session (authentication failed)."
        return f"Tools unavailable: MCP server returned HTTP {status}."
    if isinstance(root, httpx.TransportError):
        return "Tools unavailable: cannot reach the MCP server."
    return "Tools unavailable."


def _source_ref(item: dict[str, Any]) -> dict[str, Any]:
    """A source as it is saved: the file and the passage's place in it, never the passage itself."""
    ref: dict[str, Any] = {"file": item.get("file"), "seq": item.get("seq")}
    if item.get("attachment"):
        # a chat's attachment is in no Drive listing, so this row is the only copy of its name
        ref |= {"file_name": item.get("file_name"), "attachment": True}
    # no score rather than no field: a reader computes relevance from it, and a missing one is NaN
    return ref | {"distance": None}


def _saved_answer(parts: list[str], note: str) -> str:
    """The row a turn that did not finish leaves: the text that arrived and why it stops there."""
    text = "".join(parts).rstrip()
    # list_messages drops an assistant row that starts with `[error]`, so a kept answer must not
    return f"{text}\n\n[incomplete] {note}" if text else f"[error] {note}"


def _failed_calls(invocations: list[dict[str, Any]]) -> int:
    """How many of a turn's tool calls ended in an error."""
    return sum(1 for call in invocations if not call["ok"])


def _utcnow_rfc3339_z() -> str:
    """RFC3339 timestamp ending in `Z` (matches frappe-mcp-server format)."""
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _derive_title(message: str) -> str:
    stripped = message.strip()
    if len(stripped) <= _TITLE_MAX_LEN:
        return stripped
    return stripped[:_TITLE_MAX_LEN]


class _DecodeUsage(AsyncCallbackHandler):
    """Ollama's own count and timing of the tokens it writes, summed over a turn's model calls."""

    def __init__(self) -> None:
        self.tokens = 0
        self.nanos = 0

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for g in generations:
                meta = getattr(getattr(g, "message", None), "response_metadata", None) or {}
                self.tokens += meta.get("eval_count") or 0
                self.nanos += meta.get("eval_duration") or 0

    def summary(self) -> dict[str, float] | None:
        # hosted providers report no decode time, and a speed without one would be a guess
        if not self.nanos:
            return None
        return {"output_tokens": self.tokens, "output_seconds": round(self.nanos / 1e9, 3)}


def _usage(decode: _DecodeUsage, first_token: float | None) -> dict[str, float]:
    """The model's decode counts plus the seconds to the first answer text, where each is known."""
    return (decode.summary() or {}) | (
        {"first_token_s": first_token} if first_token is not None else {}
    )


def _confirmed_call(confirmation: dict[str, Any] | None) -> dict[str, Any] | None:
    """The write the user allowed, as the agent loop's `confirmed` argument."""
    if not confirmation:
        return None
    return {"name": confirmation["tool"], "arguments": confirmation.get("arguments") or {}}


def _done_event(
    tools_called: list[str], *, failed: bool, degraded: bool, usage: dict[str, float]
) -> dict[str, Any]:
    """The turn's closing frame: what ran, and what the answer it produced is worth."""
    return {
        "type": "done",
        "tools_called": tools_called,
        "data_quality": "low" if failed or degraded else "high",
        "timestamp": _utcnow_rfc3339_z(),
        **({"usage": usage} if usage else {}),
    }


class _TurnEvents:
    """One turn's events as they arrive: what each leaves behind, and the leak filter's veto."""

    def __init__(self) -> None:
        self.tools_called: list[str] = []
        # Final user-visible content (text blocks from the agent loop's
        # last iteration), accumulated for history persistence.
        self.text_parts: list[str] = []
        # Kept with the assistant message so a reopened chat shows what the live one did.
        self.sources: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []
        self.first_token: float | None = None  # seconds until the first answer text went out
        # The prompt forbids disclosure; small models get talked past it.
        self._leak_filter = StreamingLeakFilter()
        self._leaked = False

    def observe(self, ev: dict[str, Any], *, t0: float, session_id: str) -> list[dict[str, Any]]:
        """What to send on for `ev`: itself, a refusal in its place, or nothing at all."""
        if self._leaked:
            # Drain remaining events without emitting them so the LLM can complete the turn but
            # the user only sees the refusal we already published.
            return []
        if ev["type"] == "tool_call":
            self.tools_called.append(ev["name"])
        elif ev["type"] == "content_block":
            self.blocks.append(ev["block"])
        elif ev["type"] == "sources":
            self.sources.extend(ev["items"])
        elif ev["type"] == "content":
            return self._observe_content(ev, t0=t0, session_id=session_id)
        return [ev]

    def _observe_content(
        self, ev: dict[str, Any], *, t0: float, session_id: str
    ) -> list[dict[str, Any]]:
        verdict = self._leak_filter.observe(ev.get("text", ""))
        if verdict.leaked:
            self._leaked = True
            logger.warning(
                "system_prompt_leak_suppressed", session_id=session_id, reason=verdict.reason
            )
            # Refusal text, not an error event, which drops the text shown.
            refusal = "\n\n" + StreamingLeakFilter.SAFE_REFUSAL_MESSAGE
            self.text_parts.append(refusal)
            return [{"type": "content", "text": refusal}]
        self.text_parts.append(ev["text"])
        if self.first_token is None:
            self.first_token = round(time.perf_counter() - t0, 3)
        return [ev]


def _log_completed(
    span: trace.Span,
    events: _TurnEvents,
    *,
    session_id: str | None,
    t0: float,
    failed: bool,
    degraded: bool,
    tools_failed: int,
    error_type: str | None,
) -> None:
    """The turn's record on its span and in the log, once the client has been told it is done."""
    content_chars = sum(len(p) for p in events.text_parts)
    span.set_attribute("tools_called_count", len(events.tools_called))
    span.set_attribute("tools_failed_count", tools_failed)
    span.set_attribute("content_chars", content_chars)
    span.set_attribute("block_events_emitted", len(events.blocks))
    span.set_attribute("failed", failed)
    if error_type is not None:
        span.set_attribute("error_type", error_type)

    summary: dict[str, Any] = {
        "session_id": session_id,
        "duration_ms": (time.perf_counter() - t0) * 1000.0,
        "tools_called": events.tools_called,
        "tools_called_count": len(events.tools_called),
        "tools_failed_count": tools_failed,
        "content_chars": content_chars,
        "block_events_emitted": len(events.blocks),
        "failed": failed,
        "degraded": degraded,
    }
    if error_type is not None:
        summary["error_type"] = error_type
    logger.info("chat_turn_completed", **summary)


@contextmanager
def _log_when_cancelled(
    turn: dict[str, Any], tools_called: list[str], parts: list[str]
) -> Iterator[None]:
    """Log a turn the caller dropped (Stop, timeout, killed worker); it must not await or yield."""
    try:
        yield
    except (asyncio.CancelledError, GeneratorExit):
        logger.info(
            "chat_turn_cancelled",
            session_id=turn["session_id"],
            duration_ms=(time.perf_counter() - turn["t0"]) * 1000.0,
            tools_called_count=len(tools_called),
            content_chars=sum(len(p) for p in parts),
        )
        raise


class ChatService:
    """Shared by every request and holds no per-user state: each turn builds its own MCP client."""

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
        """Release any owned async resources (HTTP connection pools, etc.)."""
        await self._history.aclose()

    async def _open_session(
        self, *, sid: str, session_id: str | None, title: str, context: dict[str, Any]
    ) -> tuple[str, bool]:
        """The session row this turn writes to, and whether Frappe took it."""
        context_json = json.dumps(context or {})
        if session_id is not None:
            # save_message 417s until this row exists; a duplicate create is a no-op.
            await self._history.ensure_session(
                sid=sid, name=session_id, title=_derive_title(title), context_json=context_json
            )
            return session_id, True
        # asked on the read route before the turn's first write: a site without the chat doctypes
        # answers a write with a 500 and an Error Log row of its own for each one
        if not await self._history.chat_doctypes_exist(sid):
            return f"tmp-{uuid4().hex[:8]}", False
        created = await self._history.create_session(
            sid=sid, title=_derive_title(title), context_json=context_json
        )
        if created is not None:
            return created, True
        temporary = f"tmp-{uuid4().hex[:8]}"
        logger.warning("chat_history_session_create_failed_using_tmp", session_id=temporary)
        return temporary, False

    async def _history_messages(self, *, sid: str, session_id: str) -> list[BaseMessage]:
        """The session's earlier turns as prompt messages, each capped to what a prompt may hold."""
        if not session_id or session_id.startswith("tmp-"):
            return []
        rows = await self._history.list_messages(sid=sid, session=session_id, limit=20)
        limit = self._settings.agent_prompt_text_max_chars
        return [
            HumanMessage(content=cap_for_prompt(row["content"], limit))
            if row["role"] == "user"
            else AIMessage(content=cap_for_prompt(row["content"], limit))
            for row in rows
        ]

    async def _load_tools(
        self, *, sid: str, session_id: str, confirmation: dict[str, Any] | None
    ) -> tuple[list[Any], str | None]:
        """The caller's tools, or none of them and the line telling the model why it has none."""
        # Per-request MCP client carrying the caller's sid cookie.
        mcp_client = build_mcp_client_for_sid(
            self._settings, sid, confirmation["token"] if confirmation else None
        )
        tools: list[Any] = []
        unavailable_reason: str | None = None
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
                logger.warning(
                    "chat_tools_load_timed_out",
                    session_id=session_id,
                    mcp_url=self._settings.mcp_server_url,
                    timeout_s=tools_load_timeout_s,
                )
                raise TurnFailure(TOOLS_TIMED_OUT) from exc
            except Exception as exc:  # noqa: BLE001 - without tools the model still answers
                root = _root_cause(exc)
                unavailable_reason = _tools_unavailable_message(root)
                logger.warning(
                    "chat_tools_load_failed_soft_degrade",
                    session_id=session_id,
                    mcp_url=self._settings.mcp_server_url,
                    error_type=type(root).__name__,
                    error=str(root)[:200],
                )
                load_span.set_attribute("failed", True)
                load_span.set_attribute("error_type", type(root).__name__)
                load_span.set_attribute("degraded", True)
                tools = []
            load_span.set_attribute("tool_count", len(tools))

        pre_count = len(tools)
        tools = filter_deprecated(tools)
        if pre_count != len(tools):
            load_span.set_attribute("tools_filtered", pre_count - len(tools))

        logger.debug(
            "chat_tools_loaded",
            count=len(tools),
            session_id=session_id,
        )
        return tools, unavailable_reason

    def _system_prompt(self, context: dict[str, Any], tools_unavailable_reason: str | None) -> str:
        """The turn's system prompt, carrying the note the model needs when its tools are down."""
        preamble = self._build_system_prompt(context or {})
        if tools_unavailable_reason is None:
            return preamble
        # Soft-degraded: tell the model so it answers data
        # questions with a "I can't fetch that right now"
        # text block instead of hallucinating values.
        return preamble + (
            "\n\n# Tool status\n\n"
            f"NOTE: {tools_unavailable_reason} "
            "Answer conversational questions normally, but for "
            "questions that need real data emit a text block "
            "explaining tools are temporarily unavailable. "
            "Do not fabricate data."
        )

    async def handle_message(
        self,
        *,
        message: str | None = None,
        session_id: str | None,
        context: dict[str, Any],
        user_context: UserContext,
        confirmation: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Yield one turn's SSE events; an AsyncGenerator, so aclose() can stop it on disconnect."""
        user_message = CONFIRMED_TURN if confirmation else (message or "")
        events = _TurnEvents()
        # Rebound below to the registry's own rows, which carry each call's outcome; empty
        # until then, so a turn that never gets a registry still saves and still ends.
        tool_invocations: list[dict[str, Any]] = []
        decode = _DecodeUsage()
        # One clock for the turn: perf_counter, not time(), so a system-clock jump cannot move it.
        t0 = time.perf_counter()
        # The whole turn's budget, so the history read and the tool load spend it too.
        deadline = asyncio.get_running_loop().time() + self._settings.agent_turn_timeout_s
        failed = False
        # Set where the turn falls short of what its answer implies: a tool call that came back
        # an error, a row Frappe would not take. `done` carries it, so nothing reports high
        # quality after nothing worked.
        degraded = False
        error_message = ""
        error_type: str | None = None

        async def save_answer(content: str, session: str) -> None:
            """Write the assistant row; best-effort, so a history outage never ends the turn."""
            nonlocal degraded
            tool_args_json: str | None = None
            if tool_invocations:
                try:
                    tool_args_json = json.dumps(tool_invocations)
                except (TypeError, ValueError):
                    # Arguments weren't JSON-serialisable — drop them silently.
                    tool_args_json = None
            usage = _usage(decode, events.first_token)
            saved_sources = list(
                {(s.get("file"), s.get("seq")): _source_ref(s) for s in events.sources}.values()
            )
            tool_result_json = (
                json.dumps(
                    {"sources": saved_sources, "blocks": events.blocks}
                    | ({"usage": usage} if usage else {})
                )
                if events.sources or events.blocks or usage
                else None
            )
            # The client turns every Frappe failure into None and logs it; nothing is left here
            # to handle, only to report. The one thing it does raise is a closed pool, and this
            # save runs after the turn's own handler, so it is caught here or it ends the stream.
            try:
                saved = await self._history.save_message(
                    sid=user_context.sid,
                    session=session,
                    role="assistant",
                    content=content,
                    tool_args_json=tool_args_json,
                    tool_result_json=tool_result_json,
                )
            except RuntimeError as exc:
                logger.warning(
                    "chat_history_closed_before_the_answer_was_saved",
                    session_id=session,
                    error=str(exc)[:200],
                )
                saved = None
            if saved is None:
                degraded = True

        turn: dict[str, Any] = {"session_id": session_id, "t0": t0}
        with (
            _tracer.start_as_current_span("agent.chat_turn") as turn_span,
            _log_when_cancelled(turn, events.tools_called, events.text_parts),
        ):
            try:
                session_id, session_saved = await self._open_session(
                    sid=user_context.sid,
                    session_id=session_id,
                    title=user_message,
                    context=context,
                )
                if not session_saved:
                    degraded = True

                turn_span.set_attribute("session_id", session_id)
                turn["session_id"] = session_id

                # The frontend sends this id back; without it every message opens a new session.
                yield {"type": "session", "id": session_id}

                # Read before the question is saved, so the history holds only earlier turns.
                history_messages = await self._history_messages(
                    sid=user_context.sid, session_id=session_id
                )

                # A confirmed turn has no message of the user's, and saving the agent's
                # stand-in line would replay it as one next turn.
                if confirmation is None and (
                    await self._history.save_message(
                        sid=user_context.sid,
                        session=session_id,
                        role="user",
                        content=user_message,
                    )
                    is None
                ):
                    degraded = True

                tools, tools_unavailable_reason = await self._load_tools(
                    sid=user_context.sid, session_id=session_id, confirmation=confirmation
                )
                context_preamble = self._system_prompt(context, tools_unavailable_reason)
                tool_registry = ToolRegistry(tools, self._settings.mcp_tool_timeout_s)
                tool_invocations = tool_registry.invocations

                # A step is one LLM call plus its tool calls: two units of the recursion limit.
                max_steps = max(1, self._settings.agent_recursion_limit // 2)
                with _tracer.start_as_current_span("agent.run") as run_span:
                    run_span.set_attribute("tool_count", len(tool_registry))
                    run_span.set_attribute("max_steps", max_steps)
                    run_span.set_attribute("history_turns", len(history_messages))
                    async for ev in _until(
                        deadline,
                        run_agent_loop(
                            llm=self._llm,
                            tool_registry=tool_registry,
                            user_message=user_message,
                            context_preamble=context_preamble,
                            history=history_messages or None,
                            max_steps=max_steps,
                            tool_result_max_chars=self._settings.agent_prompt_text_max_chars,
                            callbacks=[decode],
                            session=session_id,
                            confirmed=_confirmed_call(confirmation),
                        ),
                    ):
                        for out in events.observe(ev, t0=t0, session_id=session_id):
                            yield out

                    # a tool the turn asked for and did not get: the answer is not fully backed
                    degraded = degraded or bool(_failed_calls(tool_registry.invocations))

            except (asyncio.CancelledError, GeneratorExit):
                # Stop, a dropped socket or a killed worker: the user read the text that arrived,
                # so keep it before the cancellation finishes unwinding this turn.
                if events.text_parts and session_id is not None:
                    with anyio.move_on_after(_STOPPED_SAVE_TIMEOUT_S, shield=True):
                        await save_answer(
                            _saved_answer(events.text_parts, ANSWER_STOPPED), session_id
                        )
                raise

            except Exception as exc:
                failed = True
                error_type = type(exc).__name__
                display_exc = _root_cause(exc)
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
                # One plain line per kind of failure: only a TurnFailure carries a line written for
                # the user, and every other exception's type and text stay in the log above.
                error_message = (
                    str(display_exc) if isinstance(display_exc, TurnFailure) else ANSWER_FAILED
                )
                yield {"type": "error", "message": error_message}

            assistant_content = (
                _saved_answer(events.text_parts, error_message)
                if failed
                else "".join(events.text_parts)
            )
            if session_id is not None:  # the session write itself is what failed; nowhere to save
                await save_answer(assistant_content, session_id)

            # A tool failure never raises — it comes back as a result the model reads — so a
            # turn built on one is still answered, and it is the answer that is worth less.
            tools_failed = _failed_calls(tool_invocations)
            yield _done_event(
                events.tools_called,
                failed=failed,
                degraded=degraded,
                usage=_usage(decode, events.first_token),
            )

            # After `done`, so a turn the client cancels is not logged as completed.
            _log_completed(
                turn_span,
                events,
                session_id=session_id,
                t0=t0,
                failed=failed,
                degraded=degraded,
                tools_failed=tools_failed,
                error_type=error_type,
            )
