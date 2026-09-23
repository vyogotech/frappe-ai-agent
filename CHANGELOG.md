# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **The user allows or denies a write.** Before `create_document`, `update_document`,
  `delete_document` or any other tool that changes data, the turn stops and sends a
  `tool_confirm` SSE event; the tool runs only on the next turn, carrying a one-time token the
  model never sees. No tool in that reply runs, not even a read beside the write.
- **Answer text streams as the model writes it**, instead of arriving whole at the end of a turn.
- **Sources.** An answer reports the passages it used as a `sources` SSE event and keeps each one
  in `tool_result_json` as its file and position (`file`, `seq`, and `file_name` for a file
  attached to the chat), never a copy of the passage text. Searches are scoped to the chat they
  come from.
- **Speed in the `done` event.** `usage` carries the model's own `output_tokens` and
  `output_seconds` where the provider reports them (Ollama does), plus `first_token_s`, the
  seconds from the question to the first answer text.
- **An output-side leak filter** for the system prompt: verbatim recitation, and enumeration of
  the block schema or the tool names, are replaced with a refusal — defence in depth behind the
  prompt's own disclosure rule, which small models can be talked past.
- **A cut-off or stopped answer is kept**: the text that arrived is saved with an `[incomplete]`
  marker saying why it stops there, instead of being lost.
- **Observability — request-id correlation.** `RequestIDMiddleware` binds the resolved
  `X-Request-ID` into `structlog.contextvars` for the duration of the request and resets it
  afterwards, so every log line inside a handler carries `request_id`.
- **Observability — per-turn audit log.** One info-level `chat_turn_completed` event after the
  `done` frame, with `session_id`, `duration_ms`, `tools_called`, `tools_called_count`,
  `content_chars`, `block_events_emitted`, `failed` and `error_type`. Emitted after `done`, so a
  cancelled stream is not summarised as completed.
- **Observability — OTEL spans on every chat turn**: `agent.chat_turn` (the whole handler, with
  the turn's summary attributes, an ERROR status and `record_exception` on the failure path),
  `agent.load_tools` (the MCP `tools/list` call), `agent.run` (the envelope tool-use loop) and
  `agent.history.write` (each Frappe REST write). They are no-ops when OTEL is disabled.
- **Observability — history-write failure counter.** The OTEL counter
  `agent.history.write_failures`, with a `kind` attribute (`session`/`message`), so a collector
  can alert on sustained Frappe-write outages that per-call warnings cannot surface.
- **Observability — structlog for `FrappeHistoryClient`**, so a failed write emits
  `frappe_history_write_failed` with `kind`, `error_type`, `error` and `status_code`, and the
  CSRF paths emit `frappe_history_csrf_fetch_failed` and
  `frappe_history_csrf_token_rejected_refreshing`.
- **A typed SSE event contract** in `src/ai_agent/transport/sse_events.py`: `SessionEvent`,
  `ToolCallEvent`, `ToolConfirmEvent`, `ContentEvent`, `ContentBlockEvent`, `SourcesEvent`,
  `ErrorEvent` and `DoneEvent`, plus the `SSEEvent` union, with `validate_event(event)` raising
  `ValueError` on drift. A test drains a real turn and validates every event it emits, and a
  snapshot test pins the generated schema, so a frontend can take its types from there.
- **An operational runbook** in the README: chat failures, MCP health, silent Frappe history
  loss, rate-limit 429s, a turn that runs out of steps, a stream that seems to hang, a caller
  that hangs up, and a reply the model cut off — each with the log events, spans and remedy.
- `GET /config` also reports the Ollama context window (`llm_num_ctx`).
- `make audit`: ruff, bandit, pip-audit, trivy and gitleaks into `audit-out/`, each skipped with
  a warning when the tool is absent.
- An `integration` CI job: a real Frappe v15 bench with `frappe_ai` installed,
  `frappe-mcp-server` built from source and a containerised Ollama, exercising the LLM, MCP and
  Frappe-history boundaries end to end.
- `AI_AGENT_AGENT_RECURSION_LIMIT` (was a hard-coded 50) and `AI_AGENT_AGENT_RATE_LIMIT` with a
  slowapi per-sid rate limit on `POST /api/v1/chat` (default `30/minute`).
- An 8 KB cap on the `context` payload of a chat request, to bound prompt growth.
- An MIT `LICENSE` file at the repository root, matching the `pyproject.toml` declaration.

### Changed

- **LangGraph is gone from the chat path.** One JSON envelope per turn
  (`{"blocks": [{"type": …, "payload": …}]}`), enforced at the token level by the provider's own
  structured output, and one hand-written loop that runs the tool-call blocks. This is what makes
  small local models usable: with `format=<schema>` they cannot do native tool calling. The
  turn's span is `agent.run`; `agent/graph.py`, its checkpointer setting and the multi-worker
  warning that went with them no longer exist.
- **Prior turns are replayed.** Up to the last 20 rows of the conversation are read before the
  question is saved and passed to the loop as `history`, each cut to
  `AI_AGENT_AGENT_PROMPT_TEXT_MAX_CHARS`. A failed turn's `[error]` row is skipped, and an
  answer's blocks are replayed with its text, so a block-only answer still reaches the next turn.
- **Tool results are handed back marked as data**, not as the user's words, and MCP text blocks
  are read as text rather than as a Python repr. An MCP `isError` result counts as a failed call
  whatever the adapter version puts in it.
- **A failed answer says one line** ("The answer could not be completed. Try again.", or the line
  for that kind of failure); the exception type and its text stay in the log and on the span.
- **A tool-load failure soft-degrades**: only a `tools/list` timeout ends the turn; any other
  failure carries on with an empty tool registry and tells the model so, so the answer says the
  data is unavailable instead of inventing it.
- **Every wait is bounded**: the turn (`AI_AGENT_AGENT_TURN_TIMEOUT_S`), each model call
  (`AI_AGENT_LLM_REQUEST_TIMEOUT_S`), each tool call and the MCP session under it
  (`AI_AGENT_MCP_TOOL_TIMEOUT_S`) and `tools/list` (`AI_AGENT_MCP_TOOLS_LOAD_TIMEOUT_S`). A rate
  limit slowapi cannot parse is refused at startup instead of silently not limiting.
- **No OpenAPI document and no interactive API docs** are served, and `/health` no longer returns
  probe exception text.
- **The image builds from the committed `uv.lock`** (`uv sync --locked`), caches its pip and uv
  downloads between builds, runs its health check with the application's own Python, and runs
  uvicorn as PID 1 so `docker stop` reaches it.
- **The manifest declares what the code imports** — `langchain-core`, `starlette`, and
  `opentelemetry-exporter-otlp-proto-grpc` in place of the OTLP meta-package — the dev tools moved
  to a PEP 735 `[dependency-groups]` table, and each hosted provider's SDK became its own extra
  (`openai`, `anthropic`, `google`), so the default image carries Ollama only.
- **`FrappeHistoryClient` reuses one `httpx.AsyncClient`** instead of opening a fresh one for
  every CSRF fetch and write; its idempotent `aclose()` is called from the FastAPI lifespan.
- Routers and services are wired at `create_app` time instead of inside the lifespan, so
  `app.routes` is populated before the first request.
- Comments and docstrings were cut to the line at the trap and to the contract a signature does
  not show; the design reasons they held are recorded in the audit's `DECISIONS.md`.
- The CI `test` job runs `tests/unit/` and reports coverage with no `--cov-fail-under` floor: a
  percentage pays for a line executed, not for a line asserted. The four `tests/features/` BDD
  scenarios are gone — each asserted only what its own stub had been told to yield.
- A rate limit is counted in one shared store across workers (`AI_AGENT_RATE_LIMIT_STORAGE_URI`),
  and the session-to-CSRF cache is bounded, so a second worker no longer multiplies the limit and
  the cache no longer grows without end.
- The parser's chart-alias storage is a `frozenset` rather than a dictionary that mapped every
  key to the same value.
- **A currency is never guessed.** The prompt names one only when `context.currency` is three
  ASCII letters, the shape of an ISO 4217 alpha-3 code, and names it by that code; with none,
  the prompt carries no currency line and no symbol rule. The ten-entry symbol table and the
  Indian-rupee default are gone.

### Fixed

- **Sessions.** A history read can no longer carry one user's `sid` into another user's request;
  the `sid` survives each hop of the CSRF fetch; the chat route answers only a `sid` Frappe
  recognises as a signed-in user; and no session id reaches a log line or the rate-limit key,
  which holds a digest of it.
- **The loop.** A repeated call is stopped before it is announced, counting only repeats back to
  back; the blocks a model writes beside a tool call are shown before the tool runs; each text
  block starts a new paragraph; `tool_call.name` is constrained to the tools that are loaded; and
  the envelope schema carries a title, which OpenAI-style providers need to route structured
  output.
- The history is read before the question is saved, so the model does not see the question twice.
- A turn the caller hangs up on is logged (`chat_turn_cancelled`) instead of vanishing.
- **Incorrect Frappe v17 references.** There is no v17; CI pins v15, which is the only version
  verified, and the docstrings now say so.
- `docker-compose.dev.yml.example` sets `AI_AGENT_MCP_SERVER_URL: http://mcp:8080/mcp`, so the
  agent container reaches the mcp service over the compose network instead of its own localhost.
- The health probe skips Ollama's `/api/tags` for hosted providers, which have no public
  model-list endpoint worth calling from a health route.
- `.env.example` defaults match the code (`LLM_TEMPERATURE=0.2`, `LLM_MAX_TOKENS=8192`,
  `LLM_NUM_CTX=16384`).
- `FrappeHistoryClient.create_session` supplies a UUID `name`: `AI Chat Session` is declared
  `autoname: "prompt"`, so Frappe refused a nameless write with a 417 and every first-turn
  session creation failed silently. Surfaced by the integration test.
- `tests/` cleared of latent pyright errors and re-enabled in the type-check pass.

### Removed

- **The CORS layer and the `host`, `port`, `workers` and `cors_origins` settings** (ADR-023):
  every caller is server to server, and uvicorn's command line decides the bind. **Remove
  `AI_AGENT_HOST`, `AI_AGENT_PORT`, `AI_AGENT_WORKERS` and `AI_AGENT_CORS_ORIGINS` from any
  `.env` the agent reads**, or startup fails with a `ValidationError` naming the key. In the
  process environment (Docker's `--env-file`, compose's `env_file:`) unknown keys are ignored,
  and `AI_AGENT_WORKERS` still reaches the container's start command.
- `iter_complete_blocks`, which nothing called.
- The `/tools` stub REST endpoint, which returned placeholder data, and the dead
  `PermissionDeniedError` class.

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
