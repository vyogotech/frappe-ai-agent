from unittest.mock import patch

from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm


class TestCreateLLM:
    @patch("ai_agent.integrations.llm.init_chat_model")
    def test_creates_model_with_settings(self, mock_init):
        settings = Settings(
            llm_provider="openai",
            llm_base_url="http://localhost:11434/v1",
            llm_model="llama3.2:3b",
        )
        create_llm(settings)
        mock_init.assert_called_once_with(
            "llama3.2:3b",
            model_provider="openai",
            base_url="http://localhost:11434/v1",
            api_key=None,
            temperature=0.7,
            max_tokens=4096,
        )

    @patch("ai_agent.integrations.llm.init_chat_model")
    def test_passes_api_key_when_set(self, mock_init):
        settings = Settings(
            llm_provider="openai",
            llm_api_key="sk-123",
        )
        create_llm(settings)
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["api_key"] == "sk-123"
