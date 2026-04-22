"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from ai_agent.agent.graph import build_checkpointer
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.config import Settings
from ai_agent.integrations.llm import create_llm
from ai_agent.middleware.request_id import RequestIDMiddleware
from ai_agent.observability.logging import setup_logging
from ai_agent.observability.tracing import create_tracer_provider
from ai_agent.services.chat import ChatService
from ai_agent.services.health import HealthService
from ai_agent.transport.rest import create_rest_router
from ai_agent.transport.sse import create_sse_router

logger = structlog.get_logger()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""
    if settings is None:
        settings = Settings()

    setup_logging(level=settings.log_level, log_format=settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting", port=settings.port, model=settings.llm_model)

        checkpointer = build_checkpointer()
        llm = create_llm(settings)

        # Services. ChatService builds its MCP client + agent graph per
        # request using the caller's sid, so nothing tool-related happens at
        # startup.
        chat_service = ChatService(
            settings=settings,
            llm=llm,
            checkpointer=checkpointer,
            system_prompt_builder=build_system_prompt,
        )
        health_service = HealthService(settings=settings)

        # Register routes. The /tools REST endpoint used to report the
        # startup-loaded tool list; tools are now per-user, so we expose an
        # empty list as a diagnostic stub. A future phase can re-scope it
        # behind sid auth if needed.
        app.include_router(
            create_rest_router(
                settings=settings,
                health_service=health_service,
                tools=[],
            )
        )
        app.include_router(create_sse_router())

        # Store for access in tests/extensions
        app.state.settings = settings
        app.state.chat_service = chat_service

        # OTEL
        if settings.otel_endpoint:
            create_tracer_provider(
                endpoint=settings.otel_endpoint,
                service_name=settings.otel_service_name,
            )
            FastAPIInstrumentor.instrument_app(app)

        logger.info("started")
        yield

        logger.info("stopped")

    app = FastAPI(
        title="Frappe AI Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware
    # Credentialed CORS: the Frappe frontend forwards the `sid` cookie so the
    # agent can authenticate the caller against Frappe. That requires an
    # explicit origin list (no "*") and allow_credentials=True.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    return app
