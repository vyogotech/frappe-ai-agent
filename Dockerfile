# Stage 1: Builder
FROM python:3.12-slim AS builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv sync --no-dev --no-install-project
COPY src/ src/
RUN uv sync --no-dev --no-editable

# Stage 2: Runtime
FROM python:3.12-slim
WORKDIR /app
RUN addgroup --gid 1001 appgroup && adduser --uid 1001 --gid 1001 --disabled-password appuser
COPY --from=builder /app /app
# the health check's python and the start command find the app's own environment first
ENV PATH=/app/.venv/bin:$PATH
USER appuser
EXPOSE 8484
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8484/health').raise_for_status()"
# Default `--workers` to 1 so a fresh deploy without AI_AGENT_WORKERS set
# is a known-safe single-worker case; each worker keeps its own rate-limit count.
# exec: uvicorn must be PID 1 to receive docker stop's SIGTERM and shut down within its 10 s
CMD ["/bin/sh", "-c", "exec uvicorn ai_agent.app:create_app --factory --host 0.0.0.0 --port 8484 --workers ${AI_AGENT_WORKERS:-1} --timeout-graceful-shutdown 8"]
