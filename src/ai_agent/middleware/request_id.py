"""X-Request-ID middleware for request correlation."""

from __future__ import annotations

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from ai_agent.observability import request_id as correlation


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = correlation.accept(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        # Bound into structlog's contextvars so every log line in this request carries the id.
        token = structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.reset_contextvars(**token)
        response.headers["X-Request-ID"] = request_id
        return response
