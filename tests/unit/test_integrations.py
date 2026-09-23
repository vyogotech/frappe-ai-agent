"""The LLM factory, seen through the client it builds rather than the call it makes."""

import pytest
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from pydantic import SecretStr

from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm


def test_a_hosted_provider_gets_the_model_the_key_and_the_call_bound():
    """The ollama branch has its own test; this is the init_chat_model one, built for real."""
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_base_url="http://openai.test/v1",
        llm_api_key="sk-not-a-real-key",
        llm_temperature=0.2,
        llm_max_tokens=8192,
        llm_request_timeout_s=60.0,
    )

    llm = create_llm(settings)

    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"
    assert str(llm.openai_api_base) == "http://openai.test/v1"
    key = llm.openai_api_key
    assert isinstance(key, SecretStr)
    assert key.get_secret_value() == "sk-not-a-real-key"
    assert llm.temperature == 0.2
    assert llm.max_tokens == 8192
    assert llm.request_timeout == 60.0


def test_an_empty_key_is_not_passed_off_as_a_key(monkeypatch):
    """`api_key=""` would go out as a credential; the factory sends None, so the client refuses."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_api_key="",
    )

    with pytest.raises(OpenAIError, match="api_key"):
        create_llm(settings)
