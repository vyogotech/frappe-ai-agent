"""Health check service — verifies all dependencies."""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from ai_agent.config import Settings
from ai_agent.integrations.redis import RedisClient

logger = structlog.get_logger()


class HealthService:
    def __init__(self, settings: Settings, redis: RedisClient) -> None:
        self._settings = settings
        self._redis = redis

    async def check_all(self) -> dict[str, Any]:
        results: dict[str, Any] = {}
        results["redis"] = await self._check_redis()
        results["mcp"] = await self._check_mcp()
        results["llm"] = await self._check_llm()
        results["healthy"] = all(r.get("ok", False) for r in results.values())
        return results

    async def _check_redis(self) -> dict[str, Any]:
        try:
            await self._redis.ping()
            return {"ok": True}
        except Exception as e:
            logger.warning("redis_health_failed", error=str(e))
            return {"ok": False, "error": str(e)}

    async def _check_mcp(self) -> dict[str, Any]:
        url = self._settings.mcp_server_url.replace("/mcp", "/health")
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
