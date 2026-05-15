# Frappe AI Agent

AI agent service for Frappe/ERPNext — natural-language questions in, structured visual answers out, streamed over Server-Sent Events.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

`frappe-ai-agent` is the backend that powers the AI sidebar in a Frappe/ERPNext deployment. The browser POSTs a user message; the agent runs a custom envelope-based tool-use loop, calls ERPNext tools through `frappe-mcp-server`, and streams back a mix of prose and rich content blocks (charts, tables, KPI cards, status lists) for the frontend to render.

Three design points are load-bearing:

1. **Permissions stay in Frappe.** The browser forwards the user's `sid` cookie on every chat request. The agent authenticates the request from that cookie and forwards the same `sid` to MCP for every tool call, so each tool runs under the caller's Frappe user — no shadow admin account, no permission re-implementation.
2. **No fabrication.** The system prompt forbids inventing data; every value must come from a tool call in the same turn. Tool errors are folded back into the conversation as observations so the LLM can explain what failed instead of aborting.
3. **Envelope-protocol streaming.** The LLM emits a single JSON envelope per turn whose blocks are one of `tool_call | text | table | chart | kpi | status_list`. The agent loop runs tool-call blocks itself and re-prompts; non-tool blocks become `content` / `content_block` SSE events. The browser sees discrete typed events, not partial markup.

## Architecture

```
Browser (Vue sidebar)
   │  POST /api/v1/chat  (SSE, Cookie: sid=...)
   ▼
frappe-ai-agent  (FastAPI + envelope tool-use loop)
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
3. Build a fresh MCP client carrying the caller's `sid` and load its tools (timeout configurable via `mcp_tools_load_timeout_s`, default 20 s).
4. Register each tool in `ToolRegistry`, which surfaces exceptions to the LLM as observations rather than aborting the loop.
5. Build a per-request system prompt with page context and currency.
6. Run `run_agent_loop`: structured-output envelope → execute any `tool_call` blocks → re-prompt with results → repeat until the envelope contains terminal blocks. Translate each block into the matching SSE event.
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
| `session`       | `{ id: str }` — sent once, before any other event                       |
| `status`        | `{ message: str }` — informational (reserved for future use)            |
| `tool_call`     | `{ name: str, arguments: dict }` — emitted when the agent invokes a tool |
| `content`       | `{ text: str }` — prose token chunks (streams as the LLM generates)     |
| `content_block` | `{ block: dict }` — a complete parsed content block (chart, table, …)   |
| `error`         | `{ message: str }` — fatal error; followed by `done`                    |
| `done`          | `{ tools_called: list[str], data_quality, timestamp: str }`             |

`data_quality` is `"high"` on success, `"low"` if the turn ended in error.

The wire contract is encoded as `TypedDict`s in [`src/ai_agent/transport/sse_events.py`](src/ai_agent/transport/sse_events.py) (`SessionEvent`, `StatusEvent`, `ToolCallEvent`, `ContentEvent`, `ContentBlockEvent`, `ErrorEvent`, `DoneEvent`, plus the `SSEEvent` union). A `validate_event(event: dict)` helper in the same module raises `ValueError` on any drift from the contract — `tests/unit/test_chat_service.py::test_every_emitted_event_matches_sse_contract` runs it over every event the service emits in a typical turn, so adding a new field server-side without updating the TypedDict is caught at CI time, not in a frontend bug report.

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
| `AI_AGENT_AGENT_RECURSION_LIMIT` | `50` | Envelope-loop recursion ceiling — small models need headroom while exploring doctype schemas before converging |
| `AI_AGENT_AGENT_RATE_LIMIT` | `30/minute` | slowapi-format per-sid rate limit on `POST /api/v1/chat` (e.g. `100/hour`, `10/second`) |
| `AI_AGENT_MCP_SERVER_URL` | `http://localhost:8080/mcp` | MCP Streamable HTTP endpoint |
| `AI_AGENT_MCP_TOOLS_LOAD_TIMEOUT_S` | `20.0` | Per-request bound on `tools/list`. A timeout becomes a single SSE `error` event, not a hung stream |
| `AI_AGENT_HEALTH_PROBE_TIMEOUT_S` | `5.0` | Timeout on the agent's own `/health` reachability pings against MCP and Ollama |
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

`tools/list` is bounded by `AI_AGENT_MCP_TOOLS_LOAD_TIMEOUT_S` (default 20 s). If MCP is unreachable, the user sees a single SSE `error` event and a `done` frame; the stream does not hang.

`ToolRegistry.ainvoke` catches every tool exception (MCP errors, Frappe permission denials, httpx timeouts) and surfaces it to the LLM as a string observation rather than aborting the loop. Permission errors get a clearer prefix (`Access denied: permission error — …`).

## Multi-turn state

The agent does NOT yet replay prior turns into the LLM context — `run_agent_loop` is called with `history=None`. Frappe still persists every turn into `AI Chat Message` (used for sidebar scrollback and offline review), but the LLM only sees the current turn's user message. Destructive operations therefore require both the request and its confirmation in the same conversation thread to be impossible — by design, the agent refuses until per-turn history wiring is added.

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
│   ├── loop.py                  Envelope-driven tool-use loop
│   ├── prompts.py               System prompt template + page/currency builder
│   └── tool_registry.py         Tool dispatcher with per-tool error handling
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
- **Request correlation** — every response carries an `X-Request-ID` header (incoming value echoed if the client sets one, otherwise a fresh UUID). The same id is bound into `structlog.contextvars` for the duration of the request, so any log emitted inside a handler carries `request_id=…` automatically.
- **Per-turn audit log** — `ChatService.handle_message` emits one info-level `chat_turn_completed` event at the end of every turn with `duration_ms`, `tools_called`, `tools_called_count`, `content_chars`, `block_events_emitted`, `failed`, `session_id`, and `error_type` on failure. One log line per turn answers "what happened on this chat call" without grepping multiple streams.
- **OpenTelemetry tracing** — set `AI_AGENT_OTEL_ENDPOINT` to an OTLP gRPC collector to enable export. Tracing is off when the env var is empty. Spans emitted per chat turn (nested under the FastAPI auto-instrumented HTTP span):
  - `agent.chat_turn` (the whole handler — `session_id`, `tools_called_count`, `content_chars`, `block_events_emitted`, `failed`, `error_type`)
  - `agent.load_tools` (MCP `tools/list` call — `tool_count`)
  - `agent.run` (the envelope tool-use loop — `tool_count`, `max_steps`)
  - `agent.history.write` (each Frappe REST write — `kind`, `status_code`, `failed`)

  On the failure path the `agent.chat_turn` span carries an ERROR status and the original exception via `record_exception`, so a trace UI bubbles it up.
- **OpenTelemetry metrics** — the OTEL Metrics API is wired with one counter: `agent.history.write_failures` (attribute `kind=session|message`). Lets a Prometheus/OTLP collector alert on sustained Frappe-write outages (e.g. `rate(agent_history_write_failures_total[5m]) > 0`) that the per-call WARN logs alone could not surface.

## Runbook

Practical "got paged at 3am" diagnosis paths. Each scenario lists the
visible symptom, the signals to consult, and the remediation. Assumes
the deployment has `AI_AGENT_LOG_FORMAT=json` (default) and the OTEL
spans / counters from the Operability branch are wired.

### Chat requests return 500 / clients see `error` events

**Symptoms:** FE shows a red error bubble; SSE stream carries one
`error` event followed by `done` with `data_quality: "low"`.

1. Grep the agent log for `chat_handle_message_failed` — the
   `root_cause_type` field names the first concrete exception
   underneath any TaskGroup wrapper.
2. Inspect the `agent.chat_turn` span (status: ERROR) in your trace UI.
   `record_exception` on the span carries the type / message / stack;
   the child spans (`agent.load_tools`, `agent.run`) show which
   phase blew up.
3. Common causes:
   - MCP unreachable → `agent.load_tools` span error;
     `RuntimeError: MCP tools/list timed out after <N>s` (N is from
     `mcp_tools_load_timeout_s`)
   - LLM unreachable → `agent.run` span error; httpx connect /
     timeout under it
   - Frappe Login expired → tool observations return
     `Access denied: permission error — …` (those are not errors,
     they're tool results; the LLM should explain them in the reply)

### MCP `/health?detail=true` reports MCP not OK

**Symptoms:** `/health?detail=true` returns `{"mcp": {"ok": false, ...}}`.

1. The probe hits `<mcp_server_url's host>/health` — it's a liveness
   probe only; it does not call `tools/list`. A frappe-mcp-server up
   on its HTTP port but failing tools/list can still report healthy.
2. Verify directly: `curl -s http://<mcp-host>:<port>/health`.
3. If health is OK but chats fail with MCP errors, the issue is
   tools/list — exercise via `curl` or the `agent.load_tools` span.

### Frappe history writes are silently dropped

**Symptoms:** Conversations work fine but AI Chat Session / AI Chat
Message rows stop appearing in Frappe.

1. Grep `frappe_history_write_failed` in agent logs — `kind`
   (session/message), `error_type`, and `status_code` (when HTTP)
   identify what's failing.
2. If you scrape OTEL metrics, `agent.history.write_failures`
   counter (labels: `kind`) is non-zero. An alert
   `rate(agent_history_write_failures_total[5m]) > 0` is the right
   shape for this.
3. Common causes: Frappe down (transport errors), session cookie
   expired (401), CSRF token wedged (400 with `CSRFTokenError` —
   the client auto-refreshes once, but a sustained block means the
   `/app` page is unreachable, see the
   `frappe_history_csrf_fetch_failed` log).

### Multi-worker conversation amnesia ("AI keeps forgetting")

**Symptoms:** First turn in a session works; second turn behaves as
if the prior context never happened, even though Frappe rows show
both turns.

1. Look for a startup `checkpointer_memory_multi_worker_unsafe`
   warning. It fires when `AI_AGENT_WORKERS > 1` and
   `AI_AGENT_AGENT_CHECKPOINTER=memory`. Each worker has its own
   in-memory checkpointer, so a follow-up turn that lands on a
   different worker sees a fresh thread state.
2. Switch to a shared backend (see
   [Multi-worker deployments](#multi-worker-deployments)) or set
   `AI_AGENT_WORKERS=1` if a single worker is enough for your load.

### Rate-limit `429`s

**Symptoms:** Specific user gets `429 Too Many Requests` from
`POST /api/v1/chat`.

1. The default limit is `30/minute` per `sid`. If that's wrong for
   your deployment, tune `AI_AGENT_AGENT_RATE_LIMIT` (slowapi
   syntax: `<count>/<period>`).
2. Anonymous callers (missing/empty `sid` cookie) get `401` and
   **do not** consume a token from the IP-keyed bucket, so a spray
   from one IP cannot lock out shared-NAT users.

### Agent loops past the recursion limit

**Symptoms:** `chat_handle_message_failed` log with `root_cause_type`
mentioning `GraphRecursionError`.

1. The agent is stuck exploring schemas without converging — common
   with small local models on complex multi-doctype queries.
2. Short-term: bump `AI_AGENT_AGENT_RECURSION_LIMIT` (default 50;
   75-100 is reasonable for a small model on a heavy query).
3. Longer-term: a larger model (qwen2.5:14b+) usually fixes this.

### SSE stream hangs / never closes

**Symptoms:** Client connection open indefinitely; no `done` frame.

1. `chat_turn_completed` log absent for that request means the
   generator never reached `done`. Check `X-Request-ID` and trace
   the matching `agent.chat_turn` span — if it's still open, the
   request is genuinely live.
2. The 20s MCP `tools/list` timeout is the only hard internal
   bound — beyond that, the LLM is presumed to be streaming.
3. If the LLM provider hangs, the request will hang too. Configure
   an httpx timeout on the LLM client via the provider's options
   if your provider supports it.

### Known limitation — cancelled turns leave no `chat_turn_completed` log

When a client disconnects mid-stream, Starlette calls `aclose()` on
the async generator, raising `GeneratorExit` at the current `yield`.
`GeneratorExit` is `BaseException`, not `Exception`, so the existing
`except Exception` in `ChatService.handle_message` does not catch
it; control unwinds through the `agent.chat_turn` span (which ends
cleanly with no ERROR status — correct, cancellation isn't an
error). The trailing `logger.info("chat_turn_completed", ...)` line
sits after `yield "done"`, so on a cancelled turn it never fires.

**Detection today:** a `request_id` value that appears in the early
request lifecycle (the `RequestIDMiddleware` bind, or the initial
`session` SSE event) but does NOT appear in a subsequent
`chat_turn_completed` line is a cancelled turn.

A future change may emit a dedicated `chat_turn_cancelled` event in
the `GeneratorExit` path. It is intentionally not implemented today
to avoid the yield-in-cancellation-path `RuntimeError` that the
explicit non-`finally` design in `src/ai_agent/services/chat.py`
(see comments around the inner `try/except Exception:` block in
`handle_message`) was added to avoid.

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
