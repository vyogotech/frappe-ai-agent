"""X-Request-ID middleware for request correlation."""

from __future__ import annotations

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        # Bind the id into structlog's contextvars so every log emitted
        # inside this request carries it. Without this, `merge_contextvars`
        # in logging.py has nothing to merge and the response-header round
        # trip is correlation-in-name-only. Cleared in `finally` so a
        # worker reused for a subsequent request without the header doesn't
        # inherit the prior id.
        token = structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.reset_contextvars(**token)
        response.headers["X-Request-ID"] = request_id
        return response
