from unittest.mock import patch

from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm
from langchain_ollama import ChatOllama


def test_create_llm_uses_explicit_ollama_branch_not_init_chat_model():
    settings = Settings(
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        jwt_secret="unused-but-required-by-settings",
    )
    with patch("ai_agent.integrations.llm.init_chat_model") as mock_init:
        llm = create_llm(settings)
    mock_init.assert_not_called()
    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3.5:9b"
    assert llm.base_url == "http://localhost:11434"
