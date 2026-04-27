from ai_agent.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(_env_file=None)
        assert settings.host == "0.0.0.0"
        assert settings.port == 8484
        assert settings.workers == 4
        assert settings.llm_provider == "ollama"
        assert settings.llm_base_url == "http://localhost:11434"
        assert settings.llm_model == "qwen3.5:9b"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 8192
        assert settings.llm_num_ctx == 16384
        assert settings.mcp_server_url == "http://localhost:8080/mcp"
        assert settings.frappe_url == "http://localhost:8000"
        assert settings.log_level == "info"
        assert settings.log_format == "json"
        assert settings.otel_endpoint == ""
        assert settings.otel_service_name == "frappe-ai-agent"

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("AI_AGENT_PORT", "9999")
        monkeypatch.setenv("AI_AGENT_LLM_MODEL", "mistral:7b")
        settings = Settings(_env_file=None)
        assert settings.port == 9999
        assert settings.llm_model == "mistral:7b"
