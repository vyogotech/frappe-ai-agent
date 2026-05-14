"""LLM factory using langchain's universal init_chat_model."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from ai_agent.config import Settings


def create_llm(settings: Settings) -> BaseChatModel:
    """Create a chat model from settings. Provider-agnostic."""
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        # Explicit branch for Ollama: avoids coupling to init_chat_model's
        # provider routing and keeps the dependency on langchain-ollama explicit.
        # `num_predict` is Ollama's max-output-tokens knob; `num_ctx` is the
        # context window. Without setting num_ctx, Ollama defaults to 2048
        # which is too small for our system prompt + tool results + answer
        # and causes the model to silently truncate earlier context, producing
        # garbled mid-response output (e.g. "A$6,neakers" splice bug).
        #
        # `repeat_penalty=1.0` disables Ollama's repeated-token penalty
        # (default 1.1). For structured JSON output that penalty hurts —
        # closing braces, commas, and quote characters legitimately recur
        # and the down-weighting can push the sampler off the correct
        # next token. `top_k=1` forces pure greedy decoding even when
        # the backend still samples at temperature=0. Combined, these
        # make small-model structured output reliably deterministic.
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            num_ctx=settings.llm_num_ctx,
            repeat_penalty=1.0,
            top_k=1,
        )

    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
