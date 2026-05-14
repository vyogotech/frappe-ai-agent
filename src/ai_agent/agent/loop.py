"""Unified-schema agent loop.

ONE envelope schema, ONE system prompt, ONE custom loop. `tool_call` is a
block type alongside text/table/chart/kpi/status_list. The loop drives
the LLM via `with_structured_output(BLOCK_ENVELOPE_SCHEMA,
method="json_schema")`, which routes to:

- Ollama → `format=<schema>` (token-level grammar enforcement)
- OpenAI → `response_format={"type":"json_schema",...}` (strict)
- Anthropic → tool-input enforcement
- Google → `response_schema=`

…uniformly. Tool calls are *data the model writes into the envelope*,
not a separate provider API surface — which is what lets the same code
path work on tiny local models (where `format=` defeats native
tool-calling) and on hosted models alike.

Each iteration:
  - calls `.astream(messages)`, buffering partial dicts
  - on stream-end, checks the final envelope for tool_call blocks
  - if there are tool_calls: emits `tool_call` SSE events, runs them,
    appends results to `messages`, loops
  - if there are no tool_calls: re-emits the buffered envelope as
    streamed content/content_block SSE events (so the FE sees the
    incremental render UX) and returns

Convergence guards: `max_steps` cap, same-tool-same-args repeat
detection (3x → terminate with a "no progress" text block).
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from ai_agent.blocks.envelope import (
    BLOCK_ENVELOPE_SCHEMA,
    TOOL_CALL_TYPE,
    build_agent_messages,
    envelope_to_markup,
    iter_complete_blocks,
)
from ai_agent.blocks.parser import parse_blocks

logger = structlog.get_logger(__name__)


# Maximum number of model→tool→model round-trips per user turn. The Plan
# review recommended `settings.agent_recursion_limit // 2` (≈ 25). We don't
# read settings here so the loop stays decoupled; the caller passes the cap.
_DEFAULT_MAX_STEPS = 25

# Repeat-detection trip count: if the model emits the same (tool, args) this
# many times in one turn, we bail out. Prevents infinite tool loops on a
# model that's confused about how to use the result.
_REPEAT_LIMIT = 3


async def run_agent_loop(
    *,
    llm: BaseChatModel,
    # `tool_registry` is `ai_agent.agent.tool_registry.ToolRegistry`; typed
    # as Any to avoid an import cycle with tool_registry → loop.
    tool_registry: Any,
    user_message: str,
    context_preamble: str = "",
    history: list[Any] | None = None,
    max_steps: int = _DEFAULT_MAX_STEPS,
) -> AsyncGenerator[dict[str, Any], None]:
    """Drive the unified-schema agent loop, yielding SSE-schema events.

    Events yielded (matching `transport.sse_events`):
      - `tool_call`  — once per tool the model invokes (before execution)
      - `content`    — once per text block in the final envelope
      - `content_block` — once per structured block in the final envelope

    Yields nothing for session/done/error — those are the orchestrator's
    job (see `services/chat.py`).
    """
    structured_llm = llm.with_structured_output(BLOCK_ENVELOPE_SCHEMA, method="json_schema")
    messages = build_agent_messages(
        user_message=user_message,
        tools_catalog=tool_registry.schemas(),
        context_preamble=context_preamble,
        history=history,
    )

    seen_calls: dict[tuple[str, str], int] = {}

    for step in range(max_steps):
        # Stream the iteration, buffering partial dicts. At stream end we
        # decide whether it was a tool-calling iteration or the final one.
        last_partial: dict[str, Any] | None = None
        try:
            async for partial in structured_llm.astream(messages):
                if isinstance(partial, dict):
                    last_partial = partial
        except Exception as exc:
            # Surface the *endpoint* in the warning so a DNS/host
            # misconfig is one log line, not an unwound traceback. Attr
            # names differ across LangChain chat-model classes; we read
            # whichever happens to exist.
            llm_endpoint = (
                getattr(llm, "openai_api_base", None)
                or getattr(llm, "base_url", None)
                or "<unknown>"
            )
            llm_model = (
                getattr(llm, "model_name", None)
                or getattr(llm, "model", None)
                or "<unknown>"
            )
            logger.warning(
                "agent_loop_llm_error",
                step=step,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
                llm_endpoint=str(llm_endpoint),
                llm_model=str(llm_model),
            )
            raise

        final_envelope = last_partial or {"blocks": []}
        blocks: list[dict[str, Any]] = list(final_envelope.get("blocks") or [])
        tool_blocks = [b for b in blocks if _is_tool_call(b)]
        non_tool_blocks = [b for b in blocks if not _is_tool_call(b)]

        # Defense: empty envelope (no blocks at all). The schema's
        # `minItems: 1` should make this unreachable, but a stalled
        # provider stream can land us here. Retry the same iteration
        # once with an explicit "emit any block" prod.
        if not blocks:
            if step == 0:  # only one retry, on the first iteration
                logger.warning("agent_loop_empty_envelope_retry", step=step)
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous response was empty. Emit a JSON "
                            "envelope with at least one block — a text block "
                            "is fine if you're unsure."
                        )
                    )
                )
                continue
            # Already retried — surface a placeholder text block instead
            # of hanging. The FE still gets a renderable envelope.
            logger.warning("agent_loop_empty_envelope_final", step=step)
            yield {
                "type": "content",
                "text": "I wasn't able to compose a response. Please try rephrasing.",
            }
            return

        if not tool_blocks:
            # Final iteration — emit the buffered envelope as SSE events
            # using iter_complete_blocks so the FE gets the same
            # incremental render as a true astream replay.
            state: dict[str, Any] = {}
            emitted_any = False
            for block in iter_complete_blocks(final_envelope, state, final=True):
                if _is_tool_call(block):
                    continue
                for ev in _events_from_block(block):
                    emitted_any = True
                    yield ev
            if not emitted_any:
                # All non-tool blocks were malformed and dropped by
                # parse_blocks / envelope_to_markup. Surface a fallback.
                logger.warning(
                    "agent_loop_no_emittable_blocks",
                    step=step,
                    block_types=[b.get("type") for b in non_tool_blocks],
                )
                yield {
                    "type": "content",
                    "text": "I produced a response but it couldn't be rendered.",
                }
            return

        # Tool-calling iteration. Emit synthetic tool_call SSE events so
        # the FE can render "fetching..." UI, then execute each tool and
        # feed results back into the message list.
        for tb in tool_blocks:
            payload = tb.get("payload") or {}
            name = str(payload.get("name") or "")
            args = payload.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            yield {"type": "tool_call", "name": name, "arguments": args}

        # Repeat-detection: same tool with same args N+ times = giving up.
        repeated = False
        for tb in tool_blocks:
            payload = tb.get("payload") or {}
            key = (
                str(payload.get("name") or ""),
                json.dumps(payload.get("arguments") or {}, sort_keys=True),
            )
            seen_calls[key] = seen_calls.get(key, 0) + 1
            if seen_calls[key] >= _REPEAT_LIMIT:
                repeated = True
        if repeated:
            yield {
                "type": "content",
                "text": (
                    "I'm not making progress on this — the same tool with the same "
                    "arguments hasn't returned new information. Try rephrasing or "
                    "narrowing the question."
                ),
            }
            return

        # Execute tools and append results to the message stream.
        result_lines: list[str] = []
        for tb in tool_blocks:
            payload = tb.get("payload") or {}
            name = str(payload.get("name") or "")
            args = payload.get("arguments") or {}
            if not isinstance(args, dict):
                args = {}
            result = await tool_registry.ainvoke(name, args)
            result_lines.append(f"tool {name}({json.dumps(args, ensure_ascii=False)}) → {result}")

        # Replay the model's tool_call envelope as an AIMessage so the
        # context shows what was attempted; then feed back the results.
        messages.append(AIMessage(content=json.dumps(final_envelope, ensure_ascii=False)))
        messages.append(HumanMessage(content="\n".join(result_lines)))

    # Loop fell off the bottom — hit the step cap.
    logger.warning("agent_loop_max_steps_exhausted", max_steps=max_steps)
    yield {
        "type": "content",
        "text": (
            f"I couldn't converge on an answer within {max_steps} steps. "
            "Try a more specific question."
        ),
    }


def _is_tool_call(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == TOOL_CALL_TYPE


def _events_from_block(block: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one completed envelope block into SSE-schema events.

    Mirrors the helper that used to live in `services/chat.py`:
    structured blocks go through the spec's `parse_blocks` for pydantic
    validation before becoming content_block events; text blocks become
    content events; tool_call blocks (which shouldn't arrive here) are
    silently dropped.
    """
    if _is_tool_call(block):
        return []
    try:
        markup = envelope_to_markup(json.dumps({"blocks": [block]}, ensure_ascii=False))
    except (ValueError, TypeError):
        return []
    if not markup:
        return []
    events: list[dict[str, Any]] = []
    for parsed in parse_blocks(markup):
        if parsed.type == "text":
            events.append({"type": "content", "text": parsed.content})
        else:
            events.append({"type": "content_block", "block": parsed.model_dump()})
    return events
