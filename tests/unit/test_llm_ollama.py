from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm
from langchain_ollama import ChatOllama


def test_create_llm_returns_chat_ollama_when_provider_is_ollama():
    settings = Settings(
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        jwt_secret="unused-but-required-by-settings",
    )
    llm = create_llm(settings)
    assert isinstance(llm, ChatOllama)
    assert llm.model == "qwen3.5:9b"
    assert llm.base_url == "http://localhost:11434"
