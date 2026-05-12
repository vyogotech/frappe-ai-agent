"""REST API endpoints for health and config."""

from __future__ import annotations

from fastapi import APIRouter, Query

from ai_agent.config import Settings
from ai_agent.services.health import HealthService


def create_rest_router(
    settings: Settings,
    health_service: HealthService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health(detail: bool = Query(False)):
        if detail:
            return await health_service.check_all()
        return {"status": "ok"}

    @router.get("/config")
    async def config():
        return {
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "llm_base_url": settings.llm_base_url,
            "mcp_server_url": settings.mcp_server_url,
        }

    return router
