"""Health check service — verifies external dependencies."""

from __future__ import annotations

import asyncio
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
        # Together, on one client: one after the other they cost the sum of their timeouts,
        # which outlives the budget the caller gave the whole request.
        async with httpx.AsyncClient(timeout=self._settings.health_probe_timeout_s) as client:
            mcp, llm = await asyncio.gather(self._check_mcp(client), self._check_llm(client))
        results: dict[str, Any] = {"mcp": mcp, "llm": llm}
        results["healthy"] = all(r.get("ok", False) for r in results.values())
        return results

    async def _check_mcp(self, client: httpx.AsyncClient) -> dict[str, Any]:
        # Parse the URL instead of str.replace — `mcp_server_url` like
        # "http://mcp:8081/mcp" would otherwise have its *hostname* mangled
        # ("//mcp" is the first match of "/mcp" in the string).
        parsed = urlparse(self._settings.mcp_server_url)
        url = urlunparse(parsed._replace(path="/health", query="", fragment=""))
        try:
            resp = await client.get(url)
            return {"ok": resp.status_code == 200}
        except Exception as e:  # noqa: BLE001 - a failed probe answers ok: False, never a 500
            logger.warning("mcp_health_failed", error_type=type(e).__name__, error=str(e))
            return {"ok": False}

    async def _check_llm(self, client: httpx.AsyncClient) -> dict[str, Any]:
        # Only Ollama has an unauthenticated probe (/api/tags); hosted /v1/models needs the key.
        provider = self._settings.llm_provider.lower()
        if provider != "ollama":
            return {"ok": True, "skipped": True, "reason": f"no probe for provider={provider}"}
        url = self._settings.llm_base_url.removesuffix("/v1") + "/api/tags"
        try:
            resp = await client.get(url)
            return {"ok": resp.status_code == 200}
        except Exception as e:  # noqa: BLE001 - a failed probe answers ok: False, never a 500
            logger.warning("llm_health_failed", error_type=type(e).__name__, error=str(e))
            return {"ok": False}
