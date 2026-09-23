import asyncio

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


class TestHealthProbesConcurrent:
    @pytest.mark.asyncio
    @respx.mock
    async def test_both_probes_are_in_flight_at_once(self):
        # Each probe waits at the barrier, which only opens once both have arrived:
        # run one after the other, the first waits forever and wait_for gives up.
        barrier = asyncio.Barrier(2)

        async def _wait_for_the_other(_request: httpx.Request) -> httpx.Response:
            await barrier.wait()
            return httpx.Response(200)

        respx.get("http://mcp.test:8080/health").mock(side_effect=_wait_for_the_other)
        respx.get("http://llm.test:11434/api/tags").mock(side_effect=_wait_for_the_other)

        result = await asyncio.wait_for(HealthService(_settings()).check_all(), timeout=5)
        assert result == {"mcp": {"ok": True}, "llm": {"ok": True}, "healthy": True}

    def test_probe_timeout_expires_before_the_caller_gives_up(self):
        # rag/rag/status.py calls /health?detail with timeout=5.
        assert Settings(_env_file=None).health_probe_timeout_s < 5.0  # pyright: ignore[reportCallIssue]
