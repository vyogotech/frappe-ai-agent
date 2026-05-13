"""Real LLM round-trip against a live Ollama instance.

Proves that `create_llm` produces a usable model that can reach the configured
base URL and return a completion. Tiny prompt, tiny model — the assertion is
"got non-empty content back", not anything about quality.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.messages import HumanMessage

from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm

pytestmark = pytest.mark.integration

_BASE_URL = os.getenv("AI_AGENT_INTEGRATION_LLM_BASE_URL")
_MODEL = os.getenv("AI_AGENT_INTEGRATION_LLM_MODEL")


@pytest.mark.skipif(
    not _BASE_URL or not _MODEL,
    reason="set AI_AGENT_INTEGRATION_LLM_BASE_URL and AI_AGENT_INTEGRATION_LLM_MODEL to run",
)
async def test_ollama_completes_a_prompt():
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_base_url=_BASE_URL,
        llm_model=_MODEL,
        llm_temperature=0.0,
        # qwen3 family emits <think>...</think> reasoning before the answer;
        # 32 tokens runs out mid-think and yields empty content. 512 gives
        # the smallest variant (0.6b) enough room to finish reasoning AND
        # produce a visible reply.
        llm_max_tokens=512,
        llm_num_ctx=2048,
    )
    llm = create_llm(settings)
    response = await llm.ainvoke([HumanMessage(content="Reply with the single word: pong")])
    content = response.content if isinstance(response.content, str) else str(response.content)
    assert content.strip(), f"empty completion from Ollama: {response!r}"
