"""Two text blocks in one reply reach the user as two paragraphs."""

from __future__ import annotations

import pytest
from test_agent_loop import _drain, _fake_llm_with_yields, _FakeRegistry

from ai_agent.agent.loop import run_agent_loop


def _text(content):
    return {"type": "text", "payload": {"content": content}}


async def _events(*iterations):
    llm = _fake_llm_with_yields([[{"blocks": list(blocks)}] for blocks in iterations])
    return await _drain(run_agent_loop(llm=llm, tool_registry=_FakeRegistry(), user_message="hi"))


def _said(events):
    return "".join(e["text"] for e in events if e["type"] == "content")


@pytest.mark.asyncio
async def test_two_text_blocks_are_two_paragraphs():
    events = await _events([_text("Revenue rose 18%."), _text("Acme led the quarter.")])
    assert _said(events) == "Revenue rose 18%.\n\nAcme led the quarter."
