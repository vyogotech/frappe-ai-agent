"""LLM factory using langchain's universal init_chat_model."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from copilot_agent.config import Settings


def create_llm(settings: Settings) -> BaseChatModel:
    """Create a chat model from settings. Provider-agnostic."""
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key or None,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
