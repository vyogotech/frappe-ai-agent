import pytest
from pydantic import ValidationError

from ai_agent.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        assert settings.host == "0.0.0.0"
        assert settings.port == 8484
        assert settings.workers == 1
        assert settings.llm_provider == "ollama"
        assert settings.llm_base_url == "http://localhost:11434"
        assert settings.llm_model == "qwen3.5:9b"
        assert settings.llm_temperature == 0.2
        assert settings.llm_max_tokens == 8192
        assert settings.llm_num_ctx == 16384
        assert settings.agent_recursion_limit == 50
        assert settings.agent_rate_limit == "30/minute"
        assert settings.mcp_server_url == "http://localhost:8080/mcp"
        assert settings.frappe_url == "http://localhost:8000"
        assert settings.log_level == "info"
        assert settings.log_format == "json"
        assert settings.otel_endpoint == ""
        assert settings.otel_service_name == "frappe-ai-agent"

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("AI_AGENT_PORT", "9999")
        monkeypatch.setenv("AI_AGENT_LLM_MODEL", "mistral:7b")
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        assert settings.port == 9999
        assert settings.llm_model == "mistral:7b"

    def test_cors_origins_rejects_wildcard(self):
        # Why: allow_credentials=True + "*" silently fails in Starlette;
        # fail at startup instead.
        with pytest.raises(ValidationError) as excinfo:
            Settings(_env_file=None, cors_origins=["*"])  # pyright: ignore[reportCallIssue]
        assert "cors_origins" in str(excinfo.value).lower()

    def test_cors_origins_rejects_wildcard_mixed_with_specific(self):
        with pytest.raises(ValidationError):
            Settings(
                _env_file=None,  # pyright: ignore[reportCallIssue]
                cors_origins=["http://localhost:8000", "*"],
            )

    def test_agent_recursion_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_AGENT_AGENT_RECURSION_LIMIT", "75")
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        assert settings.agent_recursion_limit == 75

    def test_agent_rate_limit_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_AGENT_AGENT_RATE_LIMIT", "10/second")
        settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
        assert settings.agent_rate_limit == "10/second"
