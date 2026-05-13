# Frappe AI Agent

AI agent service for Frappe/ERPNext — natural-language questions in, structured visual answers out, streamed over Server-Sent Events.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

`frappe-ai-agent` is the backend that powers the AI sidebar in a Frappe/ERPNext deployment. The browser POSTs a user message; the agent runs a LangGraph ReAct loop, calls ERPNext tools through `frappe-mcp-server`, and streams back a mix of prose and rich content blocks (charts, tables, KPI cards, status lists) for the frontend to render.

Three design points are load-bearing:

1. **Permissions stay in Frappe.** The browser forwards the user's `sid` cookie on every chat request. The agent authenticates the request from that cookie and forwards the same `sid` to MCP for every tool call, so each tool runs under the caller's Frappe user — no shadow admin account, no permission re-implementation.
2. **No fabrication.** The system prompt forbids inventing data; every value must come from a tool call in the same turn. Tool errors are folded back into the conversation as observations so the LLM can explain what failed instead of aborting.
3. **Streaming with typed blocks.** Prose tokens stream to the UI as they arrive. `<ai-block>` markup is buffered across token boundaries, parsed into typed Pydantic models, then emitted as a single `content_block` event — the frontend never sees half-finished tags.

## Architecture

```
Browser (Vue sidebar)
   │  POST /api/v1/chat  (SSE, Cookie: sid=...)
   ▼
frappe-ai-agent  (FastAPI + LangGraph ReAct loop)
   │
   ├──▶ LLM provider                (Ollama / OpenAI / Anthropic / Google)
   ├──▶ frappe-mcp-server           (MCP Streamable HTTP, sid forwarded)
   │       ▼
   │    ERPNext REST API            (runs as the caller)
   └──▶ Frappe REST                 (AI Chat Session / AI Chat Message)
```

Per chat request, `ChatService.handle_message` does the following:

1. Resolve or create an `AI Chat Session` in Frappe (best-effort; falls back to a temporary in-memory id if Frappe is down).
2. Persist the user message to `AI Chat Message`.
3. Build a fresh MCP client carrying the caller's `sid` and load its tools (timeout: 20 s).
4. Wrap every tool with an error handler that turns exceptions into LLM-visible observations instead of graph aborts.
5. Build a per-request system prompt with page context and currency.
6. Run the LangGraph ReAct agent and translate its event stream into SSE events.
7. Persist the final assistant message (success or error).

## Quick start

**Prerequisites**

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) for dependency management
- Ollama running locally (`http://localhost:11434`) with a tool-capable model pulled, or API credentials for OpenAI / Anthropic / Google
- `frappe-mcp-server` reachable on `http://localhost:8080/mcp`
- A Frappe/ERPNext instance reachable on `http://localhost:8000` (for chat history persistence)

```bash
uv sync --all-extras
cp .env.example .env
make serve
```

The agent listens on `http://localhost:8484`.

## API

### `POST /api/v1/chat`

Streaming chat endpoint. Returns `text/event-stream`.

**Request body** (JSON):

```json
{
  "message": "show me unpaid invoices",
  "session_id": "AI-CHAT-0001",
  "context": { "doctype": "Customer", "docname": "ACME", "currency": "USD" }
}
```

- `message` — required, 1–32 000 chars, non-whitespace.
- `session_id` — optional. Omit on the first turn; the agent creates a session and announces its id in the first SSE frame. Pass that id back on subsequent turns to continue the conversation.
- `context` — optional page context. Recognised keys: `doctype`, `docname`, `route`, `currency` (ISO 4217 code; default `INR`). Everything else is ignored.

**Authentication** — must include a Frappe `sid` cookie. Missing or empty cookie → `401`.

**Rate limit** — `30/minute` per `sid` by default, enforced by [slowapi](https://github.com/laurentS/slowapi). Exceeding the limit returns `429`. Tune via `AI_AGENT_AGENT_RATE_LIMIT`. The limit applies per request, not per concurrent connection — a single sid may hold multiple open SSE streams within its quota.

**Response stream** — newline-delimited `data: <json>\n\n` frames. Event types:

| `type`          | Payload                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `session`       | `{ id }` — sent once, before any other event                            |
| `status`        | `{ message }` — informational (reserved for future use)                 |
| `tool_call`     | `{ name, arguments }` — emitted when the agent invokes a tool           |
| `content`       | `{ text }` — prose token chunks (streams as the LLM generates)          |
| `content_block` | `{ block }` — a complete parsed content block (chart, table, kpi, …)    |
| `error`         | `{ message }` — fatal error; followed by `done`                         |
| `done`          | `{ tools_called, data_quality, timestamp }` — terminal frame            |

`data_quality` is `"high"` on success, `"low"` if the turn ended in error.

### `GET /health`

Lightweight liveness check: `{"status": "ok"}`.

`GET /health?detail=true` probes MCP (`/health`) and Ollama (`/api/tags`). Non-Ollama LLM providers are reported as `skipped: true` — the agent does not call hosted-provider model-list endpoints from a public health route.

### `GET /config`

Returns the resolved LLM provider, model, base URL, and MCP server URL. Useful for the frontend to render a "connected to: …" indicator.

## Content blocks

The LLM wraps structured data in `<ai-block type="...">{ JSON }</ai-block>` tags. The block JSON is validated against a Pydantic model, capped at sane size limits, and streamed to the frontend as a single `content_block` event.

| Block         | Purpose                       | Cap                              |
|---------------|-------------------------------|----------------------------------|
| `chart`       | Bar / line / pie / funnel / heatmap / calendar via ECharts | 500 datapoints per dataset       |
| `table`       | Sortable rows + columns with optional row→doc link | 100 rows                         |
| `kpi`         | Horizontal row of metric cards | 8 metrics                        |
| `status_list` | Colored status entries        | 50 items                         |

Tolerances:

- `<ai-block type="pie">` (or `bar`, `line`, …) is accepted as a chart alias — `chart_type` is filled in from the tag if the inner JSON omitted it. Some smaller local models reach for this shape first.
- Malformed JSON, unknown types, or oversized payloads fall back to a `TextBlock` so the user still sees something instead of a dropped message.
- Chart datasets may contain `null` to mean "no data here" — ECharts renders these as gaps.

The full schemas are in `src/ai_agent/blocks/models.py`.

## Configuration

All settings are loaded from environment or `.env` with the `AI_AGENT_` prefix. Unknown keys with the prefix cause startup to fail rather than silently being ignored.

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_AGENT_HOST` | `0.0.0.0` | Bind host |
| `AI_AGENT_PORT` | `8484` | Bind port |
| `AI_AGENT_WORKERS` | `1` | Uvicorn workers. Wired into the Dockerfile CMD as `uvicorn --workers ${AI_AGENT_WORKERS:-1}`. Raise above 1 only with a shared checkpointer (see `AI_AGENT_AGENT_CHECKPOINTER`). |
| `AI_AGENT_CORS_ORIGINS` | `["http://localhost:8000"]` | JSON list of credentialed-CORS origins. `"*"` is not allowed because cookies are forwarded |
| `AI_AGENT_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `google` |
| `AI_AGENT_LLM_BASE_URL` | `http://localhost:11434` | Provider base URL |
| `AI_AGENT_LLM_API_KEY` | _empty_ | API key for hosted providers |
| `AI_AGENT_LLM_MODEL` | `qwen3.5:9b` | Model identifier |
| `AI_AGENT_LLM_TEMPERATURE` | `0.2` | Low default tightens tool-call argument formatting on small local models |
| `AI_AGENT_LLM_MAX_TOKENS` | `8192` | Max output tokens (Ollama `num_predict`) |
| `AI_AGENT_LLM_NUM_CTX` | `16384` | Ollama context window. Ignored for hosted providers. The Ollama default of 2048 is too small for system prompt + tool results + answer |
| `AI_AGENT_AGENT_RECURSION_LIMIT` | `50` | LangGraph graph recursion ceiling — small models need headroom while exploring doctype schemas before converging |
| `AI_AGENT_AGENT_RATE_LIMIT` | `30/minute` | slowapi-format per-sid rate limit on `POST /api/v1/chat` (e.g. `100/hour`, `10/second`) |
| `AI_AGENT_AGENT_CHECKPOINTER` | `memory` | LangGraph checkpointer backend. `memory` is process-local and **not safe with `workers > 1`** (see [Multi-worker deployments](#multi-worker-deployments)). `sqlite:/abs/path/to/ckpt.db` opens an `AsyncSqliteSaver` against the file. `sqlite::memory:` is in-process SQLite. Invalid values are rejected at startup. |
| `AI_AGENT_MCP_SERVER_URL` | `http://localhost:8080/mcp` | MCP Streamable HTTP endpoint |
| `AI_AGENT_FRAPPE_URL` | `http://localhost:8000` | Frappe URL for chat history writes |
| `AI_AGENT_OTEL_ENDPOINT` | _empty_ | OTLP gRPC endpoint. Empty = tracing disabled |
| `AI_AGENT_OTEL_SERVICE_NAME` | `frappe-ai-agent` | Resource attribute on emitted spans |
| `AI_AGENT_LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |
| `AI_AGENT_LOG_FORMAT` | `json` | `json` or `console` |

See [`.env.example`](.env.example) for a working starter file.

## LLM providers

The factory in `src/ai_agent/integrations/llm.py` instantiates a chat model from settings:

- **Ollama** — direct `ChatOllama` (so `num_ctx` and `num_predict` are wired correctly).
- **OpenAI / Anthropic / Google** — via LangChain's universal `init_chat_model`. Install the optional extras (`anthropic`, `google`) if you need those.

The default config targets a local Ollama running `qwen3.5:9b`. The system prompt and the parser are designed to tolerate the kinds of mistakes small models make (chart-type aliases, occasional tool-call formatting drift, etc.), but any tool-capable model will work.

## MCP integration

Tools are loaded per-request from `frappe-mcp-server` via the Streamable HTTP transport (`langchain-mcp-adapters`). A new MCP client is built for every chat turn so the caller's `sid` cookie can be attached as a request header — sharing clients across users would leak sessions.

`tools/list` is bounded by a 20 s timeout. If MCP is unreachable, the user sees a single SSE `error` event and a `done` frame; the stream does not hang.

Every loaded tool is wrapped by `install_tool_error_handler` so that any exception (MCP errors, Frappe permission denials, httpx timeouts) becomes a string tool-observation the LLM can read and explain to the user. Permission errors get a clearer prefix (`Access denied: permission error — …`). Without this wrapping, non-`ToolException` errors escape LangChain's ToolNode and abort the whole graph run.

## Multi-worker deployments

The LangGraph agent keeps per-conversation state in a *checkpointer* — the same thread id (== Frappe chat session id) replays the conversation history on the next turn. The default `AI_AGENT_AGENT_CHECKPOINTER=memory` is process-local. With the default `AI_AGENT_WORKERS=1` (single worker) this is fine. Raise `AI_AGENT_WORKERS` above 1 and a follow-up turn has a 1-in-N chance of landing on the worker that has the prior checkpoint — the LLM "forgets" what was said even though the Frappe history rows preserve it for the UI scrollback.

If you run with `workers > 1`, set a shared backend:

```bash
AI_AGENT_AGENT_CHECKPOINTER=sqlite:/var/lib/frappe-ai-agent/checkpoints.db
```

The agent uses [`AsyncSqliteSaver`](https://langchain-ai.github.io/langgraph/reference/checkpoints/#langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver) (`langgraph-checkpoint-sqlite`). LangGraph's docs caution against SQLite under heavy write concurrency, but a single Frappe deployment with a handful of users is well inside its envelope.

A loud `checkpointer_memory_multi_worker_unsafe` warning fires at startup if `workers > 1` AND `agent_checkpointer == memory`, with remediation advice in the log fields. The warning is at WARNING level, so any aggregator with a default-severity filter will pick it up.

For ephemeral / dev use, `sqlite::memory:` keeps state inside the process (same constraint as `memory`, with the SQLite backend's overhead).

## Chat history

Chat history is persisted to two Frappe DocTypes:

- `AI Chat Session` — created on the first turn of a new conversation. `title` is the first ~60 chars of the user's message; `context_json` is the page context payload.
- `AI Chat Message` — one row per user/assistant message. Tool invocations from the assistant's turn are stored as JSON in `tool_args_json`.

Writes go through Frappe's REST API with the caller's `sid`. The client handles Frappe's HTML-embedded CSRF token: it GETs `/app`, regexes the `csrf_token = "<hex>"` JS variable, caches it per `sid`, and refreshes it once on a 400 CSRF error.

History writes are **best-effort** — failures are logged but never abort the conversation. If session creation fails, the agent falls back to a `tmp-<hex>` id so the rest of the request still works (and the user just loses persistence for that turn).

## Project layout

```
src/ai_agent/
├── app.py                       FastAPI factory + lifespan + middleware wiring
├── config.py                    Pydantic settings (env-var loader)
├── agent/
│   ├── graph.py                 LangGraph ReAct graph + in-memory checkpointer
│   ├── prompts.py               System prompt template + page/currency builder
│   └── tool_errors.py           Wrap MCP tools so errors become observations
├── blocks/
│   ├── models.py                Pydantic models for chart/table/kpi/status_list
│   ├── parser.py                Extract <ai-block> markup → typed blocks
│   └── validators.py            Size caps (rows/datapoints/metrics/items)
├── integrations/
│   ├── llm.py                   Provider-agnostic chat model factory
│   ├── mcp.py                   Per-sid MCP client builder
│   └── frappe_history.py        AI Chat Session/Message writer with CSRF
├── middleware/
│   ├── request_id.py            X-Request-ID propagation
│   └── sid.py                   sid cookie → UserContext
├── observability/
│   ├── logging.py               structlog JSON / console output
│   └── tracing.py               OTLP tracer provider
├── services/
│   ├── chat.py                  Per-request graph orchestration + SSE event mapping
│   └── health.py                MCP + Ollama probes
└── transport/
    ├── sse.py                   POST /api/v1/chat
    ├── sse_events.py            Event → SSE-frame serializer
    └── rest.py                  /health, /config
```

## Development

```bash
make install      # uv sync --all-extras
make serve        # uvicorn --reload on :8484
make test         # pytest -v (unit + BDD)
make lint         # ruff check
make format       # ruff format
make typecheck    # pyright on src/
make clean        # remove __pycache__ / *.egg-info
```

Pre-commit hooks (`.pre-commit-config.yaml`) run ruff lint + format and a handful of file hygiene checks. Install with `pre-commit install`.

## Testing

The suite is unit-first:

- **`tests/unit/`** — pure unit tests for every module. Heavy fakes live alongside the tests (e.g. `test_chat_service.py` builds an in-memory graph stub and walks the full event-translation pipeline).
- **`tests/features/`** — BDD smoke scenarios that exercise the real SSE route in-process via `httpx.ASGITransport`, with a stubbed `ChatService`. Verifies route wiring: 401 on missing sid, happy-path event ordering, exactly-one-error on upstream failure.

Tests marked `@pytest.mark.integration` need real Ollama/MCP/Frappe and are not part of the default run. To enable type-checking of tests, `pyright` is configured to scan both `src/` and `tests/`.

```bash
uv run pytest                          # unit + BDD
uv run pytest tests/unit/              # unit only
uv run pytest -m integration           # opt-in, requires real services
uv run pytest --cov=ai_agent           # coverage
```

## Observability

- **Structured logging** via `structlog`. Default JSON output to stdout; switch to a human-readable renderer with `AI_AGENT_LOG_FORMAT=console`. `uvicorn.access`, `httpx`, and `httpcore` are pinned to WARNING to keep the stream readable.
- **Request correlation** — every response carries an `X-Request-ID` header; clients can pin a value by sending the same header.
- **OpenTelemetry** — set `AI_AGENT_OTEL_ENDPOINT` to an OTLP gRPC collector to enable FastAPI instrumentation and span export. Empty endpoint = tracing off (no exporter wired in).

## Docker

```bash
docker build -t frappe-ai-agent .
docker run -p 8484:8484 --env-file .env frappe-ai-agent
```

The Dockerfile is a two-stage UV build that runs as a non-root user and ships a `HEALTHCHECK` hitting `GET /health`. `docker-compose.yml` builds the agent in isolation; `docker-compose.dev.yml.example` shows how to stack it with `frappe-mcp-server` for end-to-end dev.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs five jobs per push/PR:

- `lint` — `ruff check` + `ruff format --check`
- `typecheck` — `pyright`
- `test` — `pytest tests/unit/` with coverage upload to Codecov
- `security` — Semgrep auto config
- `build` — multi-arch Docker build, pushed to `ghcr.io/vyogotech/frappe-ai-agent` on non-PR refs

## License

MIT.
