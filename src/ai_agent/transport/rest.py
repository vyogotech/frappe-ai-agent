"""REST API endpoints for health, config, and tools."""

from __future__ import annotations

from fastapi import APIRouter, Query
from langchain_core.tools import BaseTool

from copilot_agent.config import Settings
from copilot_agent.services.health import HealthService


def create_rest_router(
    settings: Settings,
    health_service: HealthService,
    tools: list[BaseTool],
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

    @router.get("/tools")
    async def list_tools():
        return {
            "count": len(tools),
            "tools": [{"name": t.name, "description": t.description} for t in tools],
        }

    return router
