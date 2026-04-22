"""Health check service — verifies external dependencies."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

import httpx
import structlog

from ai_agent.config import Settings

logger = structlog.get_logger()


class HealthService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def check_all(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        results["mcp"] = await self._check_mcp()
        results["llm"] = await self._check_llm()
        results["healthy"] = all(r.get("ok", False) for r in results.values())
        return results

    async def _check_mcp(self) -> dict[str, Any]:
        # Parse the URL instead of str.replace — `mcp_server_url` like
        # "http://mcp:8081/mcp" would otherwise have its *hostname* mangled
        # ("//mcp" is the first match of "/mcp" in the string).
        parsed = urlparse(self._settings.mcp_server_url)
        url = urlunparse(parsed._replace(path="/health", query="", fragment=""))
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return {"ok": resp.status_code == 200}
        except Exception as e:
            logger.warning("mcp_health_failed", error=str(e))
            return {"ok": False, "error": str(e)}

    async def _check_llm(self) -> dict[str, Any]:
        url = self._settings.llm_base_url.removesuffix("/v1") + "/api/tags"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                return {"ok": resp.status_code == 200}
        except Exception as e:
            logger.warning("llm_health_failed", error=str(e))
            return {"ok": False, "error": str(e)}
