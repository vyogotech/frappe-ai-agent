"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from ai_agent.agent.graph import build_checkpointer, checkpointer_context
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
    """Key function for slowapi — prefer caller's sid, fall back to IP.

    Why: the chat route 401s without a sid, so sid will normally be present.
    The IP fallback covers any future endpoints we decorate.

    The returned string is a rate-limit bucket id consumed by slowapi
    internally — it is never rendered to a browser, so the Flask-route
    XSS rule semgrep flags here doesn't apply.
    """
    # slowapi rate-limit key — return value is never rendered to a client,
    # so semgrep's Flask directly-returned-format-string rule (which fires
    # below) is a false positive. nosem suppressions kept on the same lines.
    sid = request.cookies.get("sid")
    if sid and sid.strip():
        return "sid:" + sid  # nosem
    return "ip:" + get_remote_address(request)  # nosem


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI application."""
    if settings is None:
        settings = Settings()

    # Ship-blocker guard: in-memory checkpointer + multi-worker = silent
    # conversation loss. uvicorn does not pin a sid to a worker, so a
    # follow-up turn has a 1/workers chance of seeing the prior checkpoint.
    # Emit BEFORE setup_logging so the warning is visible to any structlog
    # processor active when create_app is called (e.g. capture_logs in
    # tests). Get a fresh logger here (not the module-level one) so a
    # test that wraps create_app in capture_logs gets the current
    # processor chain instead of a cached-at-import-time chain.
    if settings.workers > 1 and settings.agent_checkpointer == "memory":
        structlog.get_logger().warning(
            "checkpointer_memory_multi_worker_unsafe",
            workers=settings.workers,
            advice=(
                "Set AI_AGENT_AGENT_CHECKPOINTER=sqlite:/path/to/ckpt.db for a "
                "shared backend, or set AI_AGENT_WORKERS=1 if persistence across "
                "workers is not required."
            ),
        )

    setup_logging(level=settings.log_level, log_format=settings.log_format)

    # Services + routers are bound at factory time, not during startup.
    # Why: routers wired inside `lifespan` are invisible to anything that
    # inspects `app.routes` before the first request (tests, OpenAPI
    # scrapers). All construction here is in-process and synchronous
    # (no network I/O — ChatService builds its MCP client per request).
    checkpointer = build_checkpointer()
    llm = create_llm(settings)
    chat_service = ChatService(
        settings=settings,
        llm=llm,
        checkpointer=checkpointer,
        system_prompt_builder=build_system_prompt,
    )
    health_service = HealthService(settings=settings)
    limiter = Limiter(key_func=_sid_or_ip_key)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("starting", port=settings.port, model=settings.llm_model)

        # OTEL is the one piece that legitimately needs startup timing —
        # the exporter background thread is what we don't want spinning up
        # in tests that construct the app for introspection only.
        if settings.otel_endpoint:
            create_tracer_provider(
                endpoint=settings.otel_endpoint,
                service_name=settings.otel_service_name,
            )
            FastAPIInstrumentor.instrument_app(app)

        # Activate the configured checkpointer for the lifetime of the
        # app. For "memory" this is a no-op replacement (the same kind
        # of InMemorySaver as the factory-time default). For
        # "sqlite:<path>" the AsyncSqliteSaver is opened here and
        # closed on shutdown — without the close, langgraph's docs note
        # the process can hang on exit. ChatService reads the
        # checkpointer per request, so swapping the attribute in place
        # is safe.
        async with checkpointer_context(settings.agent_checkpointer) as saver:
            chat_service._checkpointer = saver
            logger.info("started", checkpointer=settings.agent_checkpointer)
            yield

        logger.info("stopped")

    app = FastAPI(
        title="Frappe AI Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    # slowapi: register the 429 handler. Why no SlowAPIMiddleware: the
    # middleware runs the limit check before FastAPI dependency resolution,
    # so unauthenticated requests would burn a token before the sid-check
    # 401s. Without the middleware, the @limiter.limit decorator performs
    # the check inside the wrapped function — i.e. after Depends() runs.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

    # Middleware
    # Credentialed CORS: the Frappe frontend forwards the `sid` cookie so the
    # agent can authenticate the caller against Frappe. That requires an
    # explicit origin list (no "*", enforced by config.py) and
    # allow_credentials=True.
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
