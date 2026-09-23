"""C34: a reply the provider cut off at its token cap runs no tool and is not shown as final."""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ai_agent.agent.loop import REPLY_CUT_OFF, run_agent_loop
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService

_CUT_TOOL_CALL = {
    "blocks": [
        {
            "type": "tool_call",
            "payload": {
                "name": "delete_document",
                # the model was writing "Annual Report 2025 final.pdf"
                "arguments": {"doctype": "File", "name": "Annual Report 2025 fi"},
            },
        }
    ]
}
_CUT_ANSWER = {
    "blocks": [{"type": "text", "payload": {"content": "The total outstanding is A$6,"}}]
}


def _llm(partials: list[dict[str, Any]], stop_reason: str) -> MagicMock:
    """A model that streams `partials` and reports `stop_reason` the way langchain-ollama does."""
    llm = MagicMock()

    def _astream(_messages, config=None):
        handlers = list((config or {}).get("callbacks") or [])

        async def _gen():
            for partial in partials:
                yield partial
            result = LLMResult(
                generations=[
                    [
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                response_metadata={"done": True, "done_reason": stop_reason},
                            )
                        )
                    ]
                ]
            )
            for handler in handlers:
                out = handler.on_llm_end(result)
                if inspect.isawaitable(out):
                    await out

        return _gen()

    structured = MagicMock()
    structured.astream = _astream
    llm.with_structured_output.return_value = structured
    return llm


class _RecordingRegistry(ToolRegistry):
    def __init__(self) -> None:
        super().__init__([])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def names(self) -> set[str]:
        return {"delete_document"}

    def schemas(self) -> str:
        return "- delete_document: delete one document"

    async def ainvoke(self, name: str, args: dict[str, Any] | None) -> str:
        self.calls.append((name, args or {}))
        return "deleted"


async def _drain(agen) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for event in agen:
        out.append(event)
    return out


@pytest.mark.asyncio
async def test_a_cut_off_tool_call_is_not_run():
    registry = _RecordingRegistry()

    with pytest.raises(RuntimeError, match=REPLY_CUT_OFF):
        await _drain(
            run_agent_loop(
                llm=_llm([_CUT_TOOL_CALL], "length"),
                tool_registry=registry,
                user_message="delete the file called Annual Report 2025 final.pdf",
                max_steps=1,
            )
        )

    assert registry.calls == []


@pytest.mark.asyncio
async def test_a_finished_tool_call_still_runs():
    registry = _RecordingRegistry()

    events = await _drain(
        run_agent_loop(
            llm=_llm([_CUT_TOOL_CALL], "stop"),
            tool_registry=registry,
            user_message="delete that file",
            max_steps=1,
        )
    )

    assert [name for name, _args in registry.calls] == ["delete_document"]
    assert [e["type"] for e in events] == ["tool_call", "content"]


@pytest.mark.asyncio
async def test_a_cut_off_answer_ends_the_turn_as_an_error():
    history = MagicMock()
    history.create_session = AsyncMock(return_value="sess-1")
    history.ensure_session = AsyncMock(return_value=None)
    history.list_messages = AsyncMock(return_value=[])
    history.save_message = AsyncMock(return_value=None)
    history.aclose = AsyncMock(return_value=None)
    service = ChatService(
        settings=Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            llm_provider="ollama",
            llm_model="qwen3.5:9b",
            llm_base_url="http://localhost:11434",
            mcp_server_url="http://mcp:8080/mcp",
        ),
        llm=_llm([_CUT_ANSWER], "length"),
        system_prompt_builder=lambda _ctx: "system",
        history=history,
    )

    mcp = MagicMock()
    mcp.get_tools = AsyncMock(return_value=[])
    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp):
        events = await _drain(
            service.handle_message(
                message="what is outstanding?",
                session_id="sess-1",
                context={},
                user_context=UserContext(sid="sid-1"),
            )
        )

    kinds = [e["type"] for e in events]
    assert kinds[-2:] == ["error", "done"]
    assert events[-2]["message"].endswith(REPLY_CUT_OFF)
    assert events[-1]["data_quality"] == "low"
    # the text that did arrive was streamed before the cut was known
    assert {"type": "content", "text": "The total outstanding is A$6,"} in events
