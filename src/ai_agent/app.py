"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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


def _sid_or_ip_key(request: Request) -> str:
    """Key function for slowapi — prefer caller's sid, fall back to IP."""
    # slowapi rate-limit key — return value is never rendered to a client,
    # so semgrep's Flask directly-returned-format-string rule (which fires
    # below) is a false positive. nosem suppressions kept on the same lines.
    sid = request.cookies.get("sid")
    if sid and sid.strip():
        # slowapi logs this key on every 429, so it must not be the credential itself
        return "sid:" + hashlib.sha256(sid.encode()).hexdigest()  # nosem
    return "ip:" + get_remote_address(request)  # nosem


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    setup_logging(level=settings.log_level, log_format=settings.log_format)

    # Built here, not in lifespan: routes wired there are missing from app.routes until the
    # first request. Keep this construction free of network I/O.
    llm = create_llm(settings)
    chat_service = ChatService(
        settings=settings,
        llm=llm,
        system_prompt_builder=build_system_prompt,
    )
    health_service = HealthService(settings=settings)
    limiter = Limiter(key_func=_sid_or_ip_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting", port=settings.port, model=settings.llm_model)
        logger.info("started")
        try:
            yield
        finally:
            # Release ChatService's owned async resources (history client's
            # AsyncClient pool, etc.) so sockets close on graceful shutdown.
            await chat_service.aclose()

        logger.info("stopped")

    app = FastAPI(
        title="Frappe AI Agent",
        version="0.1.0",
        lifespan=lifespan,
        openapi_url=None,
    )

    # Not in the lifespan: instrument_app only patches build_middleware_stack, and Starlette
    # has already called and cached it (starlette/applications.py:88) by then — the lifespan
    # scope is itself the first ASGI call, so no HTTP span would ever be created.
    if settings.otel_endpoint:
        create_tracer_provider(
            endpoint=settings.otel_endpoint,
            service_name=settings.otel_service_name,
        )
        FastAPIInstrumentor.instrument_app(app)

    # No SlowAPIMiddleware: it runs before Depends(), so requests without a valid sid would
    # spend tokens before their 401; @limiter.limit checks after Depends().
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Middleware
    # Credentialed, for the sid cookie the frontend forwards; config.py rejects a "*" origin.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["POST", "GET", "OPTIONS"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIDMiddleware)

    # Routers
    app.include_router(
        create_rest_router(
            settings=settings,
            health_service=health_service,
        )
    )
    app.include_router(
        create_sse_router(limiter=limiter, rate_limit=settings.agent_rate_limit),
    )

    # Stored for access in tests/extensions
    app.state.settings = settings
    app.state.chat_service = chat_service

    return app
