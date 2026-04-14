"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from copilot_agent.agent.graph import create_agent_graph
from copilot_agent.agent.prompts import build_system_prompt
from copilot_agent.config import Settings
from copilot_agent.integrations.llm import create_llm
from copilot_agent.integrations.mcp import create_mcp_client
from copilot_agent.integrations.postgres import create_checkpointer
from copilot_agent.integrations.redis import RedisClient
from copilot_agent.middleware.rate_limit import RateLimiter
from copilot_agent.middleware.request_id import RequestIDMiddleware
from copilot_agent.observability.logging import setup_logging
from copilot_agent.observability.tracing import create_tracer_provider
from copilot_agent.services.chat import ChatService
from copilot_agent.services.health import HealthService
from copilot_agent.services.session import SessionService
from copilot_agent.transport.rest import create_rest_router
from copilot_agent.transport.websocket import create_ws_router

logger = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""
    if settings is None:
        settings = Settings()

    setup_logging(level=settings.log_level, log_format=settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting", port=settings.port, model=settings.llm_model)

        # Initialize integrations
        redis = RedisClient(url=settings.redis_url)

        async with create_checkpointer(settings.database_url) as checkpointer:
            # MCP tools
            mcp_client = create_mcp_client(settings)
            tools = await mcp_client.get_tools()
            logger.info("mcp_tools_loaded", count=len(tools))

            # LLM + Agent graph
            llm = create_llm(settings)
            system_prompt = build_system_prompt({})
            agent_graph = create_agent_graph(
                llm=llm,
                tools=tools,
                system_prompt=system_prompt,
                checkpointer=checkpointer,
            )

            # Services
            rate_limiter = RateLimiter(
                redis=redis,
                limit=settings.rate_limit_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
            chat_service = ChatService(agent_graph=agent_graph)
            health_service = HealthService(settings=settings, redis=redis)
            session_service = SessionService(redis=redis)

            # Register routes
            app.include_router(
                create_rest_router(settings=settings, health_service=health_service, tools=tools)
            )
            app.include_router(
                create_ws_router(
                    chat_service=chat_service,
                    rate_limiter=rate_limiter,
                    jwt_secret=settings.jwt_secret,
                    jwt_algorithm=settings.jwt_algorithm,
                )
            )

            # Store for access in tests/extensions
            app.state.settings = settings
            app.state.redis = redis
            app.state.chat_service = chat_service
            app.state.session_service = session_service

            # OTEL
            if settings.otel_endpoint:
                create_tracer_provider(
                    endpoint=settings.otel_endpoint,
                    service_name=settings.otel_service_name,
                )

            logger.info("started", tools=len(tools))
            yield

        # Shutdown (checkpointer connection closed by context manager above)
        await redis.close()
        logger.info("stopped")

    app = FastAPI(
        title="Frappe Copilot Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    return app
