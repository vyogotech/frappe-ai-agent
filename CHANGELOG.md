# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- MIT `LICENSE` file at repo root (matches `pyproject.toml` declaration).
- `AI_AGENT_AGENT_RECURSION_LIMIT` env var — was a hardcoded `50` in `chat.py`.
- `AI_AGENT_AGENT_RATE_LIMIT` env var + slowapi-based per-sid rate limit on `POST /api/v1/chat` (default `30/minute`).
- 8 KB cap on the `context` payload of `ChatRequest` to bound prompt growth.
- Startup-time rejection of `cors_origins=["*"]` (incompatible with `allow_credentials=True`).
- New CI job `integration` (manual `workflow_dispatch`) running `pytest -m integration` against external services.

### Changed
- Routers and services are now wired at `create_app` time instead of inside the `lifespan` context manager — `app.routes` is populated before first request.
- Parser chart-alias storage simplified from a degenerate dict to a `frozenset`.
- CI `test` job now includes `tests/features/` (BDD smoke scenarios) alongside `tests/unit/`.

### Fixed
- Health probe skips the `/api/tags` call for non-Ollama LLM providers (OpenAI / Anthropic / Google have no public model-list endpoint we want to hit).
- `tests/` cleared of latent pyright errors and re-enabled in the type-check pass.
- `chat.py` module docstring now enumerates all seven SSE event types (`session`, `status`, `tool_call`, `content`, `content_block`, `error`, `done`) — previously omitted `session` and `content_block`.
- `.env.example` defaults synced to code: `LLM_TEMPERATURE=0.2`, `LLM_MAX_TOKENS=8192`, plus new `LLM_NUM_CTX=16384`.

### Removed
- `/tools` stub REST endpoint (was returning placeholder data).
- Dead `PermissionDeniedError` class from `agent/tool_errors.py`.

Note: the `0.1.0 — 2026-04-09` entry below pre-dates the first commit in this repository by five days (first commit `16e91ba` is 2026-04-14); it is preserved verbatim for historical continuity.

## [0.1.0] - 2026-04-09

### Added
- FastAPI SSE server with streaming events
- LangGraph ReAct agent with tool-calling loop
- MCP integration via langchain-mcp-adapters (Streamable HTTP)
- Provider-agnostic LLM via init_chat_model (Ollama, OpenAI, Anthropic, Google)
- Content blocks: Text, Chart, Table, KPI, StatusList
- Frappe sid cookie authentication
- Chat history persistence to Frappe DocTypes (AI Chat Session, AI Chat Message)
- OpenTelemetry tracing and structured logging
- Docker and docker-compose for dev environment
- CI pipeline: lint, typecheck, test, security scan, Docker build
