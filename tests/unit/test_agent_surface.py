"""The agent is internal: no interactive API docs and no probe errors for whoever reaches it."""

import httpx
import respx
from httpx import ASGITransport, AsyncClient

from ai_agent.app import create_app
from ai_agent.config import Settings
from ai_agent.services.health import HealthService


async def test_no_interactive_docs():
    app = create_app(Settings(_env_file=None))  # pyright: ignore[reportCallIssue]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        assert (await ac.get("/docs")).status_code == 404
        assert (await ac.get("/redoc")).status_code == 404


@respx.mock
async def test_health_detail_carries_no_exception_text():
    respx.get("http://mcp.test:8080/health").mock(
        side_effect=httpx.ConnectError("mcp.internal refused")
    )
    respx.get("http://llm.test:11434/api/tags").mock(side_effect=httpx.ReadTimeout("slow"))
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        mcp_server_url="http://mcp.test:8080/mcp",
        llm_base_url="http://llm.test:11434",
    )
    result = await HealthService(settings).check_all()
    assert result == {"mcp": {"ok": False}, "llm": {"ok": False}, "healthy": False}
