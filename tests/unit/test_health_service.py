import httpx
import pytest
import respx

from ai_agent.config import Settings
from ai_agent.services.health import HealthService


def _settings() -> Settings:
    return Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        mcp_server_url="http://mcp.test:8080/mcp",
        llm_base_url="http://llm.test:11434",
    )


class TestHealthService:
    @pytest.mark.asyncio
    @respx.mock
    async def test_all_ok_when_both_dependencies_return_200(self):
        respx.get("http://mcp.test:8080/health").respond(status_code=200)
        respx.get("http://llm.test:11434/api/tags").respond(status_code=200)

        result = await HealthService(_settings()).check_all()
        assert result["mcp"] == {"ok": True}
        assert result["llm"] == {"ok": True}
        assert result["healthy"] is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_mcp_non_200_reported_not_ok(self):
        respx.get("http://mcp.test:8080/health").respond(status_code=503)
        respx.get("http://llm.test:11434/api/tags").respond(status_code=200)

        result = await HealthService(_settings()).check_all()
        assert result["mcp"] == {"ok": False}
        assert result["healthy"] is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_mcp_connect_error_swallowed_and_reported(self):
        respx.get("http://mcp.test:8080/health").mock(side_effect=httpx.ConnectError("mcp down"))
        respx.get("http://llm.test:11434/api/tags").respond(status_code=200)

        result = await HealthService(_settings()).check_all()
        assert result["mcp"]["ok"] is False
        assert "mcp down" in result["mcp"]["error"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_llm_timeout_swallowed_and_reported(self):
        respx.get("http://mcp.test:8080/health").respond(status_code=200)
        respx.get("http://llm.test:11434/api/tags").mock(side_effect=httpx.ReadTimeout("slow"))

        result = await HealthService(_settings()).check_all()
        assert result["llm"]["ok"] is False
        assert "slow" in result["llm"]["error"]
        assert result["healthy"] is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_llm_base_url_with_v1_suffix_is_stripped(self):
        """llm_base_url ending in /v1 (OpenAI-compatible) is converted to
        the Ollama /api/tags probe by stripping the suffix."""
        settings = Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            mcp_server_url="http://mcp.test:8080/mcp",
            llm_base_url="http://llm.test:11434/v1",
        )
        respx.get("http://mcp.test:8080/health").respond(status_code=200)
        route = respx.get("http://llm.test:11434/api/tags").respond(status_code=200)

        result = await HealthService(settings).check_all()
        assert route.called
        assert result["llm"] == {"ok": True}

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_ollama_provider_skips_llm_probe(self):
        """Hosted providers (openai/anthropic/google) don't expose Ollama's
        /api/tags and require auth on /v1/models, so the probe is skipped
        rather than spuriously reporting ok=False."""
        settings = Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            mcp_server_url="http://mcp.test:8080/mcp",
            llm_provider="openai",
            llm_base_url="https://api.openai.com/v1",
        )
        respx.get("http://mcp.test:8080/health").respond(status_code=200)
        # No respx route for the LLM — if the probe ran, the test would fail
        # with an unmatched-request error.

        result = await HealthService(settings).check_all()
        assert result["llm"] == {
            "ok": True,
            "skipped": True,
            "reason": "no probe for provider=openai",
        }
        assert result["healthy"] is True
