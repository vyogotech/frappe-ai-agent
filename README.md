# Frappe AI Agent

AI agent service for Frappe/ERPNext — natural-language questions in, structured visual answers out, streamed over Server-Sent Events.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

`frappe-ai-agent` is the backend that powers the AI sidebar in a Frappe/ERPNext deployment. A Frappe background job posts the user's message on their behalf; the agent runs a custom envelope-based tool-use loop, calls ERPNext tools through `frappe-mcp-server`, and streams back a mix of prose and rich content blocks (charts, tables, KPI cards, status lists) for the frontend to render.

Three design points are load-bearing:

1. **Permissions stay in Frappe.** The caller's `sid` cookie travels with every chat request — the browser sends it to Frappe, and Frappe's relay job forwards it here. The agent authenticates the request from that cookie and forwards the same `sid` to MCP for every tool call, so each tool runs under the caller's Frappe user — no shadow admin account, no permission re-implementation.
2. **No fabrication.** The system prompt forbids inventing data; every value must come from a tool call in the same turn. Tool errors are folded back into the conversation as observations so the LLM can explain what failed instead of aborting.
3. **Envelope-protocol streaming.** The LLM emits a single JSON envelope per turn whose blocks are one of `tool_call | text | table | chart | kpi | status_list`. The agent loop runs tool-call blocks itself and re-prompts; non-tool blocks become `content` / `content_block` SSE events. The browser sees discrete typed events, not partial markup.

## Architecture

```text
Browser (the Vue sidebar in the Frappe desk, or Metis at /ask)
   │  1. subscribe to frappe_ai:chunk:<session_id>   (socket.io)
   │  2. POST frappe_ai.api.chat.start_stream        (a whitelisted method, Cookie: sid=...)
   ▼
Frappe web worker — enqueues the relay on the `long` queue and returns at once
   ▼
RQ worker
   │  POST /api/v1/chat  (SSE, the caller's sid forwarded)
   ▼
frappe-ai-agent  (FastAPI + envelope tool-use loop)
   │
   ├──▶ LLM provider                (Ollama; OpenAI / Anthropic / Google with that extra)
   ├──▶ frappe-mcp-server           (MCP Streamable HTTP, sid forwarded)
   │       ▼
   │    ERPNext REST API            (runs as the caller)
   └──▶ Frappe REST                 (AI Chat Session / AI Chat Message)
```

The RQ worker republishes each SSE frame with `frappe.publish_realtime`, and the browser reads
the answer from its subscription. No browser talks to this service directly, which is why it
carries no CORS layer.

Per chat request, `ChatService.handle_message` does the following:

1. Resolve or create an `AI Chat Session` in Frappe (best-effort; falls back to a temporary in-memory id if Frappe is down).
2. Persist the user message to `AI Chat Message`.
3. Build a fresh MCP client carrying the caller's `sid` and load its tools (timeout configurable via `mcp_tools_load_timeout_s`, default 20 s).
4. Register each tool in `ToolRegistry`, which surfaces exceptions to the LLM as observations rather than aborting the loop.
5. Build a per-request system prompt with the page context, and the currency if the caller named one.
6. Run `run_agent_loop`: structured-output envelope → execute any `tool_call` blocks → re-prompt with results → repeat until the envelope contains terminal blocks. Translate each block into the matching SSE event.
7. Persist the final assistant message (success or error).

## Quick start

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) for dependency management
- Ollama running locally (`http://localhost:11434`) with a tool-capable model pulled, or API credentials for OpenAI / Anthropic / Google
- `frappe-mcp-server` reachable on `http://localhost:8080/mcp`
- A Frappe/ERPNext instance reachable on `http://localhost:8000` (for chat history persistence).
  Frappe **version-16** is what the stack runs and what CI tests; nothing else is tested.

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
- `context` — optional page context. Recognised keys: `doctype`, `docname`, `route`, `currency` (three ASCII letters, the shape of an ISO 4217 alpha-3 code; anything else and the answer names no currency). Everything else is ignored.

**Authentication** — must include a Frappe `sid` cookie. Missing or empty cookie → `401`.

**Rate limit** — `30/minute` per `sid` by default, enforced by [slowapi](https://github.com/laurentS/slowapi). Exceeding the limit returns `429`. Tune via `AI_AGENT_AGENT_RATE_LIMIT`. The limit applies per request, not per concurrent connection — a single sid may hold multiple open SSE streams within its quota.

**Response stream** — newline-delimited `data: <json>\n\n` frames. Event types:

| `type` | Payload |
| --- | --- |
| `session` | `{ id: str }` — sent once, before any other event |
| `tool_call` | `{ name: str, arguments: dict }` — emitted when the agent invokes a tool |
| `tool_confirm` | `{ id: str, name: str, arguments: dict }` — a write the user must allow; the turn ends on it and the tool has **not** run |
| `content` | `{ text: str }` — prose token chunks (streams as the LLM generates) |
| `content_block` | `{ block: dict }` — a complete parsed content block (chart, table, …) |
| `sources` | `{ items: list[dict] }` — the passages an answer used |
| `error` | `{ message: str }` — fatal error; followed by `done` |
| `done` | `{ tools_called: list[str], data_quality, timestamp: str, usage?: dict }` |

`data_quality` is `"high"` on success, `"low"` if the turn ended in error. `usage` is present
only when there is something to report: `output_tokens` and `output_seconds` when the provider
reports its own decode counts (Ollama does; hosted providers do not), and `first_token_s`, the
seconds from the question to the first answer text.

An `error` message is one plain line per kind of failure — the turn ran past its deadline, the reply was cut off, `tools/list` did not answer in time, or anything else, which reads "The answer could not be completed. Try again." The exception type and its text never reach the client: they are in the turn's own log line (`chat_handle_message_failed`, `chat_tools_load_timed_out`) and on the `agent.chat_turn` span. The same line is what the turn saves as its answer — appended under an `[incomplete]` marker to the text that had already arrived, when there was any — so a reopened chat shows no exception text either.

The wire contract is encoded as `TypedDict`s in [`src/ai_agent/transport/sse_events.py`](src/ai_agent/transport/sse_events.py) (`SessionEvent`, `ToolCallEvent`, `ToolConfirmEvent`, `ContentEvent`, `ContentBlockEvent`, `SourcesEvent`, `ErrorEvent`, `DoneEvent`, plus the `SSEEvent` union). A `validate_event(event: dict)` helper in the same module raises `ValueError` on any drift from the contract — `tests/unit/test_chat_service.py::test_every_emitted_event_matches_sse_contract` runs it over every event the service emits in a typical turn, so adding a new field server-side without updating the TypedDict is caught at CI time, not in a frontend bug report.

### `GET /health`

Lightweight liveness check: `{"status": "ok"}`.

`GET /health?detail=true` probes MCP (`/health`) and Ollama (`/api/tags`). Non-Ollama LLM providers are reported as `skipped: true` — the agent does not call hosted-provider model-list endpoints from a public health route.

### `GET /config`

Returns the resolved LLM provider, model, base URL, Ollama context window (`llm_num_ctx`) and MCP server URL. Useful for the frontend to render a "connected to: …" indicator.

**Authentication** — the same Frappe `sid` cookie `POST /api/v1/chat` requires; missing, empty or not signed in → `401`. The route names the model and both peer URLs, so a caller that may not ask a question may not read it either. `GET /health` is the unauthenticated liveness route.

## Content blocks

The LLM wraps structured data in `<ai-block type="...">{ JSON }</ai-block>` tags. The block JSON is validated against a Pydantic model, capped at sane size limits, and streamed to the frontend as a single `content_block` event.

| Block | Purpose | Cap |
| --- | --- | --- |
| `chart` | Bar / line / pie / funnel / heatmap / calendar via ECharts | 500 datapoints per dataset |
| `table` | Sortable rows + columns with optional row→doc link | 100 rows |
| `kpi` | Horizontal row of metric cards | 8 metrics |
| `status_list` | Colored status entries | 50 items |

Tolerances:

- `<ai-block type="pie">` (or `bar`, `line`, …) is accepted as a chart alias — `chart_type` is filled in from the tag if the inner JSON omitted it. Some smaller local models reach for this shape first.
- Malformed JSON, unknown types, or oversized payloads fall back to a `TextBlock` so the user still sees something instead of a dropped message.
- Chart datasets may contain `null` to mean "no data here" — ECharts renders these as gaps.

The full schemas are in `src/ai_agent/blocks/models.py`.

## Configuration

All settings are loaded from the environment or from `.env`, with the `AI_AGENT_` prefix. The
two sources treat an unknown key differently, which is pydantic-settings' own behaviour: a key in
the `.env` file that names no setting fails startup with a `ValidationError`, while an unknown
variable in the process environment is dropped before validation and the agent starts. A
container configured with `--env-file` or compose's `env_file:` takes the second path, so a typo
there is silent — check the resolved values on `GET /config`.

| Variable | Default | Description |
| --- | --- | --- |
| `AI_AGENT_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `google` |
| `AI_AGENT_LLM_BASE_URL` | `http://localhost:11434` | Provider base URL |
| `AI_AGENT_LLM_API_KEY` | _empty_ | API key for hosted providers |
| `AI_AGENT_LLM_MODEL` | `qwen3.5:9b` | Model identifier |
| `AI_AGENT_LLM_TEMPERATURE` | `0.2` | Low default tightens tool-call argument formatting on small local models |
| `AI_AGENT_LLM_MAX_TOKENS` | `8192` | Max output tokens (Ollama `num_predict`) |
| `AI_AGENT_LLM_NUM_CTX` | `16384` | Ollama context window. Ignored for hosted providers. The Ollama default of 2048 is too small for system prompt + tool results + answer |
| `AI_AGENT_LLM_REQUEST_TIMEOUT_S` | `60.0` | Bound on one model call, as the httpx timeout under the provider client. Ollama's own client default is no timeout at all |
| `AI_AGENT_AGENT_TURN_TIMEOUT_S` | `90.0` | Wall-clock bound on one chat turn. On expiry the turn ends with an `error` event and `done` with `data_quality: "low"` |
| `AI_AGENT_AGENT_PROMPT_TEXT_MAX_CHARS` | `8000` | Longest single tool result or history row that may enter the prompt; the rest is cut with a `[truncated to N characters]` marker |
| `AI_AGENT_AGENT_RECURSION_LIMIT` | `50` | Envelope-loop ceiling. The loop runs half of it (25) model-tool rounds per turn, since a round is one model call plus its tool calls; small models need the headroom while exploring doctype schemas before converging |
| `AI_AGENT_AGENT_RATE_LIMIT` | `30/minute` | slowapi-format per-sid rate limit on `POST /api/v1/chat` (e.g. `100/hour`, `10/second`) |
| `AI_AGENT_RATE_LIMIT_STORAGE_URI` | `memory://` | Where the rate limit is counted. The default counts in the worker's own memory, so N workers let one sid make N times the limit; point it at the Redis the stack already runs (`redis://redis:6379/1`) and they share one count. A URI slowapi cannot open is refused at startup rather than silently not limiting. |
| `AI_AGENT_MCP_SERVER_URL` | `http://localhost:8080/mcp` | MCP Streamable HTTP endpoint |
| `AI_AGENT_MCP_TOOLS_LOAD_TIMEOUT_S` | `20.0` | Per-request bound on `tools/list`. A timeout becomes a single SSE `error` event, not a hung stream |
| `AI_AGENT_MCP_TOOL_TIMEOUT_S` | `30.0` | Bound on one tool call, and on the HTTP and SSE read timeouts of the MCP session under it. A timeout comes back as a tool result the model can answer around |
| `AI_AGENT_HEALTH_PROBE_TIMEOUT_S` | `5.0` | Timeout on the agent's own `/health` reachability pings against MCP and Ollama |
| `AI_AGENT_FRAPPE_URL` | `http://localhost:8000` | Frappe URL for chat history writes |
| `AI_AGENT_OTEL_ENDPOINT` | _empty_ | OTLP gRPC endpoint. Empty = tracing disabled |
| `AI_AGENT_OTEL_SERVICE_NAME` | `frappe-ai-agent` | Resource attribute on emitted spans |
| `AI_AGENT_LOG_LEVEL` | `info` | `debug` / `info` / `warning` / `error` |
| `AI_AGENT_LOG_FORMAT` | `json` | `json` or `console` |

See [`.env.example`](.env.example) for a working starter file.

### Bounds on a turn

Every wait a turn can make sits inside the one above it, so the innermost failure is
the one the user hears about:

| Bound | Setting | Default | Why it sits there |
| --- | --- | --- | --- |
| Turn deadline | `AI_AGENT_AGENT_TURN_TIMEOUT_S` | 90 s | Outermost inside the agent, and inside its caller's: `frappe_ai` reads the stream with its own timeout (120 s by default) and kills the relay job at that plus 30 s. At 90 s the agent's own `error` and `done` still reach the browser instead of the relay's generic failure. |
| Model call | `AI_AGENT_LLM_REQUEST_TIMEOUT_S` | 60 s | Inside the turn, so one stuck call fails while the turn still has time to report it. It is an httpx timeout, so on a streaming call it bounds the wait for the next chunk, not the whole answer — a model that keeps emitting is ended by the turn deadline. |
| Tool call | `AI_AGENT_MCP_TOOL_TIMEOUT_S` | 30 s | Inside the turn, and short enough that a few calls still fit in one. A timeout comes back as a tool result, not a failed turn, so the model can answer around it. |
| MCP session | `AI_AGENT_MCP_TOOL_TIMEOUT_S` | 30 s | The transport under the tool call: the adapter's own defaults are 30 s HTTP and 300 s for the SSE read, and that read would otherwise outlive both the call and the turn. `tools/list` keeps its own bound (`AI_AGENT_MCP_TOOLS_LOAD_TIMEOUT_S`, 20 s) and runs before the loop, inside the same turn deadline. |
| HTTP | — | 10 s Frappe history, 5 s session check, `AI_AGENT_HEALTH_PROBE_TIMEOUT_S` for `/health` | Innermost: one request each, and each failure is reported by the step that made it. |

Prompt size is bounded the same way: every tool result and every history row is cut to
`AI_AGENT_AGENT_PROMPT_TEXT_MAX_CHARS` characters with a marker before it enters the
prompt, so a single large document cannot fill `num_ctx` and leave the model silently
truncating its own context.

## LLM providers

The factory in `src/ai_agent/integrations/llm.py` instantiates a chat model from settings:

- **Ollama** — direct `ChatOllama` (so `num_ctx` and `num_predict` are wired correctly).
- **OpenAI / Anthropic / Google** — via LangChain's universal `init_chat_model`. Each provider's SDK is an optional extra (`openai`, `anthropic`, `google`): install the one you use, or `uv sync --all-extras` for all three. The Docker image is built without extras, so it carries Ollama only; a hosted provider needs an image built with its extra, and without it startup fails with LangChain's own `Initializing Chat… requires the langchain-… package` error.

The default config targets a local Ollama running `qwen3.5:9b`. The system prompt and the parser are designed to tolerate the kinds of mistakes small models make (chart-type aliases, occasional tool-call formatting drift, etc.), but any tool-capable model will work.

## MCP integration

Tools are loaded per-request from `frappe-mcp-server` via the Streamable HTTP transport (`langchain-mcp-adapters`). A new MCP client is built for every chat turn so the caller's `sid` cookie can be attached as a request header — sharing clients across users would leak sessions.

`tools/list` is bounded by `AI_AGENT_MCP_TOOLS_LOAD_TIMEOUT_S` (default 20 s). Only a timeout on
that call ends the turn: the user sees one SSE `error` event ("The assistant's tools timed out.
Try again.") and a `done` frame, and the stream does not hang. **Any other tool-load failure soft-
degrades**: the agent logs `chat_tools_load_failed_soft_degrade`, carries on with an empty tool
registry, and tells the model in its prompt that tools are unavailable, so a conversational
question is still answered and a data question comes back as "I can't fetch that right now"
rather than as invented numbers.

`ToolRegistry.ainvoke` catches every tool exception (MCP errors, Frappe permission denials, httpx timeouts) and surfaces it to the LLM as a string observation rather than aborting the loop. Permission errors get a clearer prefix (`Access denied: permission error — …`).

## Multi-turn state

Prior turns are replayed. Before the question is saved, the agent reads up to the last 20 rows of
the conversation from `AI Chat Message` (`list_messages(limit=20)`, newest first, then reversed,
so it keeps the latest turns) and passes them to `run_agent_loop` as `history`; each row is cut
to `AI_AGENT_AGENT_PROMPT_TEXT_MAX_CHARS` first. A failed turn's `[error]` row is skipped, so the
model never replays its own failure text. When the history read fails, the turn continues with an
empty history and logs `chat_history_load_failed_using_empty`.

## Chat history

Chat history is persisted to two Frappe DocTypes:

- `AI Chat Session` — created on the first turn of a new conversation. `title` is the first ~60 chars of the user's message; `context_json` is the page context payload.
- `AI Chat Message` — one row per user/assistant message. Tool invocations from the assistant's turn are stored as JSON in `tool_args_json`.

An answer's `tool_result_json` keeps its content blocks and, for every passage the answer used, only the file and the passage's position in it (`file`, `seq`, and for a file attached to the chat its `file_name`) — never the passage text, which is streamed live and then belongs to the document. A reader who reopens the chat therefore sees the source's file and position, not a copy of the text as it was: reading the excerpt back from the knowledge base under that reader's own permissions is Metis's half of this change and is not in place yet, so today a reopened chat shows the source without its excerpt. Rows written before this still hold the old `content`; nothing here reads it, and a chat that has them still opens.

Writes go through Frappe's REST API with the caller's `sid`. The client handles Frappe's HTML-embedded CSRF token: it GETs `/app`, regexes the `csrf_token = "<hex>"` JS variable, caches it per `sid`, and refreshes it once on a 400 CSRF error.

History writes are **best-effort** — failures are logged but never abort the conversation. If session creation fails, the agent falls back to a `tmp-<hex>` id so the rest of the request still works (and the user just loses persistence for that turn).

## Project layout

```text
src/ai_agent/
├── app.py                       FastAPI factory + lifespan + middleware wiring
├── config.py                    Pydantic settings (env-var loader)
├── agent/
│   ├── loop.py                  Envelope-driven tool-use loop
│   ├── prompts.py               System prompt template + page/currency builder
│   ├── leak_filter.py           Output-side system-prompt leak filter
│   └── tool_registry.py         Tool dispatcher with per-tool error handling
├── blocks/
│   ├── envelope.py              The JSON envelope schema and its streaming splitter
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
│   ├── chat.py                  Per-request turn orchestration + SSE event mapping
│   └── health.py                MCP + Ollama probes
└── transport/
    ├── sse.py                   POST /api/v1/chat
    ├── sse_events.py            Event → SSE-frame serializer
    └── rest.py                  /health, /config
```

## Development

```bash
make install          # uv sync --all-extras
make serve            # uvicorn --reload on :8484
make test             # pytest tests/unit/ with coverage
make test-integration # pytest tests/integration/, needs Ollama, MCP and a Frappe bench
make lint             # ruff check and ruff format --check
make format           # ruff format
make typecheck        # pyright over src and tests
make boundaries       # the .importlinter layer contracts
make security         # semgrep, bandit and pip-audit
make workflows        # zizmor over .github/workflows
make clean            # remove __pycache__ / *.egg-info
```

Pre-commit hooks (`.pre-commit-config.yaml`) run ruff lint + format and a handful of file hygiene checks. Install with `pre-commit install`.

## Testing

The suite is unit-first:

- **`tests/unit/`** — pure unit tests for every module. Heavy fakes live alongside the tests (e.g. `test_chat_service.py` stands in for `run_agent_loop` and walks the full event-translation pipeline).
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
the deployment has `AI_AGENT_LOG_FORMAT=json` (default) and an OTLP
endpoint set (`AI_AGENT_OTEL_ENDPOINT`), so the spans and counters
described under Observability are exported.

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
     `chat_tools_load_timed_out` carries the `timeout_s` that was
     spent (from `mcp_tools_load_timeout_s`); the client sees only
     "The assistant's tools timed out. Try again."
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

### Rate-limit `429`s

**Symptoms:** Specific user gets `429 Too Many Requests` from
`POST /api/v1/chat`.

1. The default limit is `30/minute` per `sid`. If that's wrong for
   your deployment, tune `AI_AGENT_AGENT_RATE_LIMIT` (slowapi
   syntax: `<count>/<period>`).
2. Anonymous callers (missing/empty `sid` cookie) get `401` and
   **do not** consume a token from the IP-keyed bucket, so a spray
   from one IP cannot lock out shared-NAT users.

### The agent runs out of steps

**Symptoms:** `agent_loop_max_steps_exhausted` in the log, and an
answer that reads "I couldn't converge on an answer within N steps.
Try a more specific question." This is not an error: the turn ends
with that text, a normal `done` frame and `data_quality: "high"`.

1. The agent is stuck exploring schemas without converging — common
   with small local models on complex multi-doctype queries.
2. Short-term: bump `AI_AGENT_AGENT_RECURSION_LIMIT` (default 50;
   75-100 is reasonable for a small model on a heavy query). The loop
   runs half that many model-tool rounds, so 50 means 25 steps.
3. Longer-term: a larger model (qwen2.5:14b+) usually fixes this.

### SSE stream hangs / never closes

**Symptoms:** Client connection open indefinitely; no `done` frame.

1. `chat_turn_completed` log absent for that request means the
   generator never reached `done`. Check `X-Request-ID` and trace
   the matching `agent.chat_turn` span — if it's still open, the
   request is genuinely live.
2. A turn is bounded by `AI_AGENT_AGENT_TURN_TIMEOUT_S` (default
   90 s), a model call by `AI_AGENT_LLM_REQUEST_TIMEOUT_S` and a
   tool call by `AI_AGENT_MCP_TOOL_TIMEOUT_S` — see "Bounds on a
   turn". A stream open past the turn deadline is the client or the
   relay holding it, not the agent.
3. On the deadline the turn sends one `error` event ("This took too
   long. Try a shorter question.") followed by `done` with
   `data_quality: "low"`, and logs `chat_turn_completed` with
   `failed=true`.

### A turn the caller hangs up on

**Symptoms:** `chat_turn_cancelled` in the log and no
`chat_turn_completed` line for that turn; in the chat, an answer that
ends "[incomplete] The answer was stopped. Ask again for a full
answer."

The user pressed Stop, or the connection died — a relay read timeout,
a killed worker, a closed laptop. Both reach the agent the same way:
Starlette calls `aclose()` on the async generator, raising
`GeneratorExit` at the current `yield`, or cancels the task, raising
`CancelledError` at the current `await`. Both are `BaseException`, not
`Exception`, so `except Exception` in `ChatService.handle_message`
never sees them; they are caught by name instead. The turn then logs
`chat_turn_cancelled` (duration, tools called, characters answered)
and, when text had already reached the user, saves that text as the
answer with the stopped line appended, so a reopened chat shows what
the live one did rather than an unanswered question. A turn cut before
it said anything saves no answer — there is none to save.

That save is awaited inside `anyio.move_on_after(5, shield=True)`, so
the cancellation being unwound cannot cut the write short. Nothing is
yielded on that path: a `yield` after `GeneratorExit` is the
`RuntimeError` the explicit non-`finally` design in
`src/ai_agent/services/chat.py` avoids. The `agent.chat_turn` span
ends cleanly with no ERROR status — cancellation isn't an error.

### The model ran out of tokens mid-answer

**Symptoms:** `agent_loop_reply_cut_off` in the log, and a turn that
ends with an `error` event and `done` with `data_quality: "low"`.

The provider reported that generation stopped because it hit the
token cap (`done_reason: "length"` on Ollama, `finish_reason` or
`stop_reason` elsewhere), so the envelope stops wherever the tokens
ran out. The loop runs no tool from that reply — a cut-off argument
would name the wrong document — and ends the turn with "The answer
was cut short. Try a narrower question." after whatever text had
already been streamed. That text is kept: the answer is saved as the
text that arrived with the same line appended under an `[incomplete]`
marker, so a reopened chat shows the part that was written.

If this is frequent, the answers are longer than the cap: raise
`AI_AGENT_LLM_MAX_TOKENS` (Ollama `num_predict`, default 8192),
keeping it inside the model's context window.

## Docker

```bash
docker build -t frappe-ai-agent .
docker run -p 8484:8484 --env-file .env frappe-ai-agent
```

The agent listens on `0.0.0.0:8484`, which the start command fixes — the Dockerfile's `CMD` and the Makefile's `serve` target both pass `--host` and `--port` to uvicorn, and no setting overrides them. One variable reaches that command: `AI_AGENT_WORKERS` (default `1`) becomes `uvicorn --workers ${AI_AGENT_WORKERS:-1}`. The agent holds no per-request state, but each worker keeps its own count for `AI_AGENT_AGENT_RATE_LIMIT` unless `AI_AGENT_RATE_LIMIT_STORAGE_URI` points them at a shared store, so N workers otherwise let one session make N times that many requests. It is read by the container's shell, not by the application, so it belongs in the environment (`--env-file`, compose `env_file:`) and not in a `.env` file the app itself reads.

The Dockerfile is a two-stage UV build that installs the committed `uv.lock`, runs as a non-root user and ships a `HEALTHCHECK` hitting `GET /health`. Its `pip` and `uv` downloads go to BuildKit cache mounts, so a rebuild after a dependency change resolves from the local cache; `docker buildx prune` and `docker system prune` wipe them with the rest of the build cache, `docker buildx prune --filter 'type!=exec.cachemount'` keeps them, and `docker build --no-cache` hands the build an empty mount rather than reusing one. `docker-compose.yml` builds the agent in isolation; `docker-compose.dev.yml.example` shows how to stack it with `frappe-mcp-server` for end-to-end dev.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs six jobs per push/PR:

Every job runs the same command a developer runs, through the Makefile.

- `lint` — `make lint` (`ruff check` and `ruff format --check`), `make boundaries`
  (the `.importlinter` layer contracts) and `pre-commit run --all-files`
- `typecheck` — `make typecheck` (`pyright` over `src` and `tests`)
- `test` — `make test` (`pytest tests/unit/` with coverage uploaded to Codecov, no floor: see the
  comment in the workflow)
- `integration` — after `lint` and `typecheck`: a real Frappe version-16 bench with `frappe_ai`
  installed, `frappe-mcp-server` built from the commit ragbot pins and a containerised Ollama,
  exercising the LLM, MCP and Frappe-history boundaries
- `security` — `make security` (Semgrep `p/python`, Bandit, pip-audit) and `make workflows` (zizmor)
- `build` — the Docker build, pushed to `ghcr.io/vyogotech/frappe-ai-agent` on non-PR refs, and only
  after all five jobs above have passed

## License

MIT.
