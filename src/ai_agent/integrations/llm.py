"""LLM factory using langchain's universal init_chat_model."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from ai_agent.config import Settings


def create_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        # repeat_penalty=1.0: a Modelfile's penalty (lfm2.5 sets 1.05) down-weights the braces,
        # commas and quotes JSON repeats. top_k=1 makes decoding greedy whatever the temperature.
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
