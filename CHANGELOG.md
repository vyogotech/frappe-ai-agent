# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Observability — request-id correlation.** `RequestIDMiddleware` now binds
  the resolved `X-Request-ID` into `structlog.contextvars` for the duration of
  the request and resets it after the response. The `merge_contextvars`
  processor was already wired in `setup_logging`, but nothing called
  `bind_contextvars` — handler-side logs had no `request_id`. Tests assert
  the binding inside a handler and the reset between requests.
- **Observability — per-turn audit log.** `ChatService.handle_message` emits a
  single info-level `chat_turn_completed` event after the `done` SSE frame,
  carrying `session_id`, `duration_ms`, `tools_called`, `tools_called_count`,
  `content_chars`, `block_events_emitted`, `failed`, and `error_type` (on
  failure). One log line answers "what happened on this chat call". Emitted
  after `done` so a cancelled stream (client `aclose`) is not summarised
  as completed.
- **Observability — custom OTEL spans.** Adds three nested spans on every
  chat turn so a trace UI can answer "where did those 18 seconds go?"
  without consulting logs:
  - `agent.chat_turn` — full handler; carries final turn-summary attributes
    and sets ERROR status with `record_exception` on the failure path.
  - `agent.load_tools` — wraps the MCP `tools/list` call. `tool_count` attribute.
  - `agent.graph_run` — wraps the LangGraph `astream_events` loop.
  - `agent.history.write` — wraps each Frappe REST write inside
    `FrappeHistoryClient`. `kind` attribute (`session`/`message`),
    `status_code` on the happy path, `failed`/`error_type` on the error path.
  Spans are no-ops when OTEL is disabled (`trace.get_tracer` returns a
  ProxyTracer that defers to the global provider at use-time).
- **Observability — history-write failure counter.** OTEL counter
  `agent.history.write_failures` with a `kind` attribute, incremented on every
  failed Frappe REST write. Lets a Prometheus/OTLP collector alert on
  sustained outages that the per-call WARN logs alone could not surface
  (`rate(agent_history_write_failures_total[5m]) > 0`).
- **Observability — structlog migration for FrappeHistoryClient.** The module
  switched from stdlib `logging` to structlog so failures emit a consistent
  `frappe_history_write_failed` event with `kind`, `error_type`, `error`,
  and `status_code` fields. The CSRF fetch and retry logs gained event names
  too (`frappe_history_csrf_fetch_failed`,
  `frappe_history_csrf_token_rejected_refreshing`).
- **Pluggable LangGraph checkpointer.** New env var `AI_AGENT_AGENT_CHECKPOINTER`:
  - `memory` (default) — `InMemorySaver`, today's behaviour. Per-process.
  - `sqlite:<path>` — `AsyncSqliteSaver` against the given file. Shared across
    workers via the file, persists across restarts.
  - `sqlite::memory:` — in-process SQLite, useful for tests.
  Field validator rejects any other value at startup (catches `memry` /
  `Postgres://…` typos that would otherwise have silently fallen back to
  in-memory). New `checkpointer_context(setting)` async context manager in
  `agent/graph.py` opens and closes the AsyncSqliteSaver via the lifespan
  so the connection releases cleanly on shutdown (langgraph's docs note
  the process can hang otherwise).
- **Multi-worker safety warning.** A loud structured
  `checkpointer_memory_multi_worker_unsafe` warning fires at startup when
  `workers > 1` AND `agent_checkpointer == "memory"`. Without the warning,
  conversation continuity broke silently in a 4-worker default deploy
  because uvicorn does not pin a sid to a worker — the LLM "forgot" what
  was said even though the Frappe history rows preserved it. The warning
  carries remediation advice (set the sqlite URL or drop workers to 1).
- MIT `LICENSE` file at repo root (matches `pyproject.toml` declaration).
- `AI_AGENT_AGENT_RECURSION_LIMIT` env var — was a hardcoded `50` in `chat.py`.
- `AI_AGENT_AGENT_RATE_LIMIT` env var + slowapi-based per-sid rate limit on `POST /api/v1/chat` (default `30/minute`).
- 8 KB cap on the `context` payload of `ChatRequest` to bound prompt growth.
- Startup-time rejection of `cors_origins=["*"]` (incompatible with `allow_credentials=True`).
- New CI job `integration` runs on every push to `main`, spinning up MariaDB + Redis services, a real Frappe v15 bench with the `frappe_ai` app installed, the `frappe-mcp-server` Go binary built from source, and a containerised Ollama (`qwen3:0.6b`). Three integration tests exercise the LLM, MCP, and Frappe-history boundaries end-to-end via the Administrator sid.

### Changed
- **`FrappeHistoryClient` reuses one `httpx.AsyncClient` per instance** instead
  of opening a fresh one for every CSRF fetch / write. Previously a chat turn
  paid 3-4 TCP connection setups (plus TLS handshakes when behind HTTPS); now
  the writes share a connection pool. New idempotent `aclose()` method is
  invoked by the FastAPI lifespan teardown so the pool releases sockets cleanly
  on shutdown.
- Routers and services are now wired at `create_app` time instead of inside the `lifespan` context manager — `app.routes` is populated before first request.
- Parser chart-alias storage simplified from a degenerate dict to a `frozenset`.
- CI `test` job now includes `tests/features/` (BDD smoke scenarios) alongside `tests/unit/`.

### Fixed
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
