"""C03: the turn deadline, the model-call timeout, the tool-call timeout and the prompt caps."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from ai_agent.agent.loop import cap_for_prompt, run_agent_loop
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import TURN_TOO_LONG, ChatService


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_provider": "ollama",
        "llm_model": "qwen3.5:9b",
        "llm_base_url": "http://localhost:11434",
        "mcp_server_url": "http://mcp:8080/mcp",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # pyright: ignore[reportCallIssue]


def _history(rows: list[dict[str, str]] | None = None) -> MagicMock:
    history = MagicMock()
    history.create_session = AsyncMock(return_value="sess-1")
    history.ensure_session = AsyncMock(return_value=None)
    history.list_messages = AsyncMock(return_value=rows or [])
    history.save_message = AsyncMock(return_value=None)
    history.aclose = AsyncMock(return_value=None)
    return history


def _llm(iterations: list[list[dict[str, Any]]], prompts: list[str] | None = None) -> MagicMock:
    """A model that replays scripted envelope snapshots, recording each prompt it was sent."""
    llm = MagicMock()
    queue = list(iterations)

    def _astream(messages, config=None):
        if prompts is not None:
            prompts.append("\n".join(str(m.content) for m in messages))
        partials = queue.pop(0)

        async def _gen():
            for partial in partials:
                yield partial

        return _gen()

    structured = MagicMock()
    structured.astream = _astream
    llm.with_structured_output.return_value = structured
    return llm


def _stalled_llm() -> MagicMock:
    llm = MagicMock()

    def _astream(_messages, config=None):
        async def _gen():
            await asyncio.sleep(3600)
            yield {}

        return _gen()

    structured = MagicMock()
    structured.astream = _astream
    llm.with_structured_output.return_value = structured
    return llm


class _FakeRegistry(ToolRegistry):
    def __init__(self, result: str) -> None:
        super().__init__([])
        self._result = result

    def names(self) -> set[str]:
        return {"get_document"}

    def schemas(self) -> str:
        return "- get_document: one document"

    def writes(self, name: str) -> bool:
        # scripted as reads: the pause a write needs has its own tests
        return False

    async def ainvoke(self, name: str, args: dict[str, Any] | None) -> str:
        return self._result


_TOOL_CALL = {
    "blocks": [
        {
            "type": "tool_call",
            "payload": {"name": "get_document", "arguments": {"doctype": "File"}},
        }
    ]
}
_ANSWER = {"blocks": [{"type": "text", "payload": {"content": "done"}}]}


async def _drain(agen) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    async for event in agen:
        out.append(event)
    return out


async def _run_turn(service: ChatService) -> list[dict[str, Any]]:
    mcp = MagicMock()
    mcp.get_tools = AsyncMock(return_value=[])
    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp):
        return await _drain(
            service.handle_message(
                message="how many sales orders are open?",
                session_id="sess-1",
                context={},
                user_context=UserContext(sid="sid-1"),
            )
        )


@pytest.mark.asyncio
async def test_stalled_model_ends_the_turn_at_the_deadline():
    service = ChatService(
        settings=_settings(agent_turn_timeout_s=0.2),
        llm=_stalled_llm(),
        system_prompt_builder=lambda _ctx: "system",
        history=_history(),
    )
    # 5 s, not forever: without the deadline this turn never ends and the test would hang.
    events = await asyncio.wait_for(_run_turn(service), 5)

    kinds = [e["type"] for e in events]
    assert kinds[-2:] == ["error", "done"]
    assert events[-2]["message"].endswith(TURN_TOO_LONG)
    assert events[-1]["data_quality"] == "low"


@pytest.mark.asyncio
async def test_a_turn_inside_the_deadline_is_untouched():
    service = ChatService(
        settings=_settings(agent_turn_timeout_s=30.0),
        llm=_llm([[_ANSWER]]),
        system_prompt_builder=lambda _ctx: "system",
        history=_history(),
    )
    events = await _run_turn(service)

    assert [e["type"] for e in events] == ["session", "content", "done"]
    assert events[-1]["data_quality"] == "high"


def test_ollama_client_is_built_with_the_model_call_timeout():
    llm = create_llm(_settings(llm_request_timeout_s=12.5))

    assert isinstance(llm, ChatOllama)
    assert (llm.client_kwargs or {})["timeout"] == 12.5


def test_mcp_session_gives_up_no_later_than_the_tool_call():
    client = build_mcp_client_for_sid(_settings(mcp_tool_timeout_s=7.0), sid="sid-1")

    connection: dict[str, Any] = dict(client.connections["frappe"])
    assert connection["timeout"].total_seconds() == 7.0
    assert connection["sse_read_timeout"].total_seconds() == 7.0


@pytest.mark.asyncio
async def test_a_tool_that_never_returns_becomes_a_result_the_model_can_read():
    @tool
    async def _stalls() -> str:
        """A tool that never answers."""
        await asyncio.sleep(3600)
        return "never"

    registry = ToolRegistry([_stalls], timeout_s=0.1)

    result = await asyncio.wait_for(registry.ainvoke("_stalls", {}), 5)

    assert result.startswith("Tool call failed: _stalls timed out after")


@pytest.mark.asyncio
async def test_a_long_tool_result_is_capped_before_it_enters_the_prompt():
    prompts: list[str] = []
    await _drain(
        run_agent_loop(
            llm=_llm([[_TOOL_CALL], [_ANSWER]], prompts),
            tool_registry=_FakeRegistry("x" * 50_000),
            user_message="show me that file",
            tool_result_max_chars=500,
        )
    )

    assert "x" * 501 not in prompts[1]
    assert "[truncated to 500 characters]" in prompts[1]


@pytest.mark.asyncio
async def test_a_long_history_row_is_capped_before_it_enters_the_prompt():
    prompts: list[str] = []
    service = ChatService(
        settings=_settings(agent_prompt_text_max_chars=500),
        llm=_llm([[_ANSWER]], prompts),
        system_prompt_builder=lambda _ctx: "system",
        history=_history([{"role": "user", "content": "y" * 50_000}]),
    )
    await _run_turn(service)

    assert "y" * 501 not in prompts[0]
    assert "[truncated to 500 characters]" in prompts[0]


def test_cap_for_prompt_leaves_text_within_the_limit_alone():
    assert cap_for_prompt("short", 500) == "short"
