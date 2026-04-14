import pytest
from pydantic import ValidationError

from copilot_agent.config import Settings


class TestSettings:
    def test_defaults(self):
        settings = Settings(jwt_secret="test-secret")
        assert settings.host == "0.0.0.0"
        assert settings.port == 8484
        assert settings.workers == 4
        assert settings.llm_provider == "openai"
        assert settings.llm_base_url == "http://localhost:11434/v1"
        assert settings.llm_model == "llama3.2:3b"
        assert settings.llm_temperature == 0.7
        assert settings.llm_max_tokens == 4096
        assert settings.mcp_server_url == "http://localhost:8080/mcp"
        assert (
            settings.database_url == "postgresql+asyncpg://copilot:copilot@localhost:5432/copilot"
        )
        assert settings.redis_url == "redis://localhost:6379/0"
        assert settings.rate_limit_requests == 60
        assert settings.rate_limit_window_seconds == 60
        assert settings.log_level == "info"
        assert settings.log_format == "json"
        assert settings.otel_endpoint == ""

    def test_jwt_secret_required(self):
        with pytest.raises(ValidationError):
            Settings()

    def test_env_prefix(self, monkeypatch):
        monkeypatch.setenv("COPILOT_JWT_SECRET", "from-env")
        monkeypatch.setenv("COPILOT_PORT", "9999")
        monkeypatch.setenv("COPILOT_LLM_MODEL", "mistral:7b")
        settings = Settings()
        assert settings.jwt_secret == "from-env"
        assert settings.port == 9999
        assert settings.llm_model == "mistral:7b"
