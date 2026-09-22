"""A block the model writes beside a tool call is shown, not only replayed to the model."""

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


def _said(events):
    return "".join(e["text"] for e in events if e["type"] == "content")


@pytest.mark.asyncio
async def test_a_block_written_after_a_tool_call_is_shown():
    events = await _events([_call("ledger"), _text("Checking the ledger.")], [_text("Done.")])
    assert "Checking the ledger." in _said(events)
