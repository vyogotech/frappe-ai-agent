"""The repeat guard stops a call before announcing it, and only a call repeated back to back."""

from __future__ import annotations

import pytest
from test_agent_loop import _drain, _fake_llm_with_yields, _FakeRegistry

from ai_agent.agent.loop import run_agent_loop


def _text(content):
    return {"type": "text", "payload": {"content": content}}


def _call(name, **args):
    return {"type": "tool_call", "payload": {"name": name, "arguments": args}}


async def _events(*iterations):
    llm = _fake_llm_with_yields([[{"blocks": list(blocks)}] for blocks in iterations])
    return await _drain(run_agent_loop(llm=llm, tool_registry=_FakeRegistry(), user_message="hi"))


@pytest.mark.asyncio
async def test_a_call_stopped_as_a_repeat_is_not_announced():
    same = [_call("stuck", q=1)]
    events = await _events(same, same, same)
    assert [e["type"] for e in events].count("tool_call") == 2


@pytest.mark.asyncio
async def test_a_call_repeated_with_other_calls_between_is_not_a_repeat():
    events = await _events(
        [_call("list")],
        [_call("update")],
        [_call("list")],
        [_call("update")],
        [_call("list")],
        [_text("ok")],
    )
    assert events[-1] == {"type": "content", "text": "ok"}
