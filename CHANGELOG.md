# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Typed SSE event contract** in `src/ai_agent/transport/sse_events.py`:
  seven `TypedDict`s (`SessionEvent`, `StatusEvent`, `ToolCallEvent`,
  `ContentEvent`, `ContentBlockEvent`, `ErrorEvent`, `DoneEvent`) plus the
  `SSEEvent` union — the wire contract a frontend can mirror in TypeScript.
  New `validate_event(event)` raises `ValueError` on any drift; a
  `test_every_emitted_event_matches_sse_contract` test in `test_chat_service.py`
  drains a real chat turn and validates every event the service emits, so
  contract drift is caught at CI rather than in a FE bug report.
- **Operational runbook** in the README covering chat 500s, MCP health
  failures, silent Frappe history loss, multi-worker conversation amnesia,
  rate-limit 429s, recursion-limit hits, and SSE stream hangs — each with
  the structured log events, OTEL spans, and remediations to consult.
- MIT `LICENSE` file at repo root (matches `pyproject.toml` declaration).
- `AI_AGENT_AGENT_RECURSION_LIMIT` env var — was a hardcoded `50` in `chat.py`.
- `AI_AGENT_AGENT_RATE_LIMIT` env var + slowapi-based per-sid rate limit on `POST /api/v1/chat` (default `30/minute`).
- 8 KB cap on the `context` payload of `ChatRequest` to bound prompt growth.
- Startup-time rejection of `cors_origins=["*"]` (incompatible with `allow_credentials=True`).
- New CI job `integration` runs on every push to `main`, spinning up MariaDB + Redis services, a real Frappe v15 bench with the `frappe_ai` app installed, the `frappe-mcp-server` Go binary built from source, and a containerised Ollama (`qwen3:0.6b`). Three integration tests exercise the LLM, MCP, and Frappe-history boundaries end-to-end via the Administrator sid.

### Changed
- Routers and services are now wired at `create_app` time instead of inside the `lifespan` context manager — `app.routes` is populated before first request.
- Parser chart-alias storage simplified from a degenerate dict to a `frozenset`.
- CI `test` job now includes `tests/features/` (BDD smoke scenarios) alongside `tests/unit/`.

### Fixed
- **Incorrect Frappe v17 references.** `FrappeHistoryClient` docstrings
  claimed the CSRF embed pattern was specific to "Frappe v17", but v17 does
  not exist — Frappe's latest releases are v15 and v16. CI pins v15, which is
  the only version we actually verify. Docstrings now say so plainly.
- **`docker-compose.dev.yml.example`** sets
  `AI_AGENT_MCP_SERVER_URL: http://mcp:8080/mcp` on the agent service so the
  agent container reaches the mcp service over the compose network instead
  of trying to dial its own localhost. The `.env.example` default
  (`http://localhost:8080/mcp`) is right for laptop dev but wrong inside
  compose.
- Health probe skips the `/api/tags` call for non-Ollama LLM providers (OpenAI / Anthropic / Google have no public model-list endpoint we want to hit).
- `tests/` cleared of latent pyright errors and re-enabled in the type-check pass.
- `chat.py` module docstring now enumerates all seven SSE event types (`session`, `status`, `tool_call`, `content`, `content_block`, `error`, `done`) — previously omitted `session` and `content_block`.
- `.env.example` defaults synced to code: `LLM_TEMPERATURE=0.2`, `LLM_MAX_TOKENS=8192`, plus new `LLM_NUM_CTX=16384`.
- `FrappeHistoryClient.create_session` now supplies a UUID-based `name` in the POST payload. The `AI Chat Session` DocType is declared `autoname: "prompt"`, so Frappe rejected nameless writes with a 417 "Please set the document name" — every first-turn session creation would have silently failed in production. Surfaced by the new integration test.

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
