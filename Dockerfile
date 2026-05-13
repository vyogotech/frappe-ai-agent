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
USER appuser
EXPOSE 8484
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8484/health').raise_for_status()"
# `--workers ${AI_AGENT_WORKERS:-1}` makes the checkpointer-vs-workers
# warning meaningful: previously `Settings.workers=4` (config default)
# triggered the warning even though the container actually ran one
# uvicorn worker because nothing wired the env into the CMD. Default
# to 1 here so a fresh deploy without AI_AGENT_WORKERS set is the
# safe-single-worker case; deployers who want multi-worker now have
# a real knob, and the safety warning will fire correctly.
CMD ["/bin/sh", "-c", "/app/.venv/bin/uvicorn ai_agent.app:create_app --factory --host 0.0.0.0 --port 8484 --workers ${AI_AGENT_WORKERS:-1}"]
