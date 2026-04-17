# Frappe AI Agent

AI agent service for Frappe/ERPNext — SSE streaming, tool-calling via MCP, and rich visual responses.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

Frappe AI Agent is a standalone service that powers the AI sidebar in Frappe/ERPNext. Users ask questions in natural language and receive structured visual answers — charts, tables, KPI cards, and status lists — streamed in real time over Server-Sent Events.

The agent uses a LangGraph ReAct loop to reason about questions, call ERPNext tools via the Model Context Protocol (MCP), and compose rich content blocks from the results. Every request is authenticated using the caller's Frappe session cookie (`sid`), so Frappe's permission system is enforced end-to-end.

## Architecture

```
Browser (Vue sidebar)
    │  POST /api/v1/chat  (SSE, Cookie: sid=...)
    ▼
frappe-ai-agent  (LangGraph ReAct loop)
    │
    ├──▶ Ollama / LLM           (reasoning + token stream)
    │
    ├──▶ frappe-mcp-server      (MCP tools, sid forwarded)
    │       ▼
    │    Frappe REST API         (runs as that user)
    │
    └──▶ Frappe DocType API     (save AI Chat Session/Message)
```

## Quick Start

**Prerequisites:** Python 3.12+, [UV](https://docs.astral.sh/uv/), Ollama running locally, `frappe-mcp-server` running on port 8080.

```bash
# 1. Install dependencies
uv sync --all-extras

# 2. Configure
cp .env.example .env
# Edit .env if needed (defaults work with local Ollama + MCP on port 8080)

# 3. Start the agent
make serve
```

The agent is now running at `http://localhost:8484`. The chat endpoint is `POST /api/v1/chat`.

## Configuration

All settings use the `AI_AGENT_` prefix. Configured via environment variables or `.env` file.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AI_AGENT_HOST` | `str` | `0.0.0.0` | Server bind address |
| `AI_AGENT_PORT` | `int` | `8484` | Server port |
| `AI_AGENT_WORKERS` | `int` | `4` | Uvicorn worker count |
| `AI_AGENT_CORS_ORIGINS` | `list[str]` | `["http://localhost:8000"]` | Allowed CORS origins (must be explicit, not `*`) |
| `AI_AGENT_LLM_PROVIDER` | `str` | `ollama` | LLM provider (`ollama`, `openai`, `anthropic`, `google`) |
| `AI_AGENT_LLM_BASE_URL` | `str` | `http://localhost:11434` | LLM API base URL |
| `AI_AGENT_LLM_API_KEY` | `str` | `""` | LLM API key (not needed for Ollama) |
| `AI_AGENT_LLM_MODEL` | `str` | `qwen3.5:9b` | Model name |
| `AI_AGENT_LLM_TEMPERATURE` | `float` | `0.7` | Sampling temperature |
| `AI_AGENT_LLM_MAX_TOKENS` | `int` | `4096` | Max output tokens |
| `AI_AGENT_MCP_SERVER_URL` | `str` | `http://localhost:8080/mcp` | MCP server Streamable HTTP endpoint |
| `AI_AGENT_FRAPPE_URL` | `str` | `http://localhost:8000` | Frappe instance URL (for chat history persistence) |
| `AI_AGENT_OTEL_ENDPOINT` | `str` | `""` | OTLP exporter endpoint (empty = disabled) |
| `AI_AGENT_OTEL_SERVICE_NAME` | `str` | `frappe-ai-agent` | OTEL service name |
| `AI_AGENT_LOG_LEVEL` | `str` | `info` | Log level (`debug`, `info`, `warning`, `error`) |
| `AI_AGENT_LOG_FORMAT` | `str` | `json` | Log format (`json`, `console`) |

## SSE Protocol

### Request

```
POST /api/v1/chat
Cookie: sid=<frappe-session-id>
Content-Type: application/json
Accept: text/event-stream

{"message": "Show me this month's sales", "session_id": "ses_abc123", "context": {"doctype": "Sales Invoice", "page": "list"}}
```

`session_id` is optional on the first message. The server assigns one and sends it back as the first event.

### Server Events

Each event is a JSON object on a `data:` line, followed by two newlines.

**session** — conversation ID (first event, always):
```
data: {"type":"session","id":"ses_abc123"}
```

**status** — agent is working:
```
data: {"type":"status","message":"loading tools..."}
```

**tool_call** — agent is calling an MCP tool:
```
data: {"type":"tool_call","name":"list_documents","arguments":{"doctype":"Sales Invoice"}}
```

**content** — plain text response (no structured blocks):
```
data: {"type":"content","text":"You have 42 pending invoices."}
```

**content_block** — structured visual content (chart, table, KPI, status_list):
```
data: {"type":"content_block","block":{"type":"chart","chart_type":"bar","title":"Monthly Sales","data":{"labels":["Jan","Feb","Mar"],"datasets":[{"name":"Revenue","values":[50000,62000,71000]}]}}}
```

**error** — something went wrong:
```
data: {"type":"error","message":"The AI service is currently unavailable."}
```

**done** — request complete (last event, always):
```
data: {"type":"done","tools_called":["list_documents"],"data_quality":"high","timestamp":"2026-04-17T10:00:00Z"}
```

## Content Blocks

### Text
Markdown-formatted text responses.
```json
{"type": "text", "content": "You have **42** pending invoices totaling $125,000."}
```

### Chart
ECharts visualizations. Supports bar, line, pie, funnel, heatmap, and calendar types.
```json
{"type": "chart", "chart_type": "pie", "title": "Revenue by Region", "data": {"labels": ["North", "South", "East", "West"], "datasets": [{"name": "Revenue", "values": [30000, 25000, 40000, 18000]}]}, "options": {"format": "currency", "currency": "USD"}}
```

### Table
Sortable data tables with optional ERPNext document links.
```json
{"type": "table", "title": "Top Customers", "columns": [{"key": "name", "label": "Customer", "format": "text"}, {"key": "revenue", "label": "Revenue", "format": "currency"}], "rows": [{"values": {"name": "Acme Corp", "revenue": 50000}, "route": {"doctype": "Customer", "name": "Acme Corp"}}]}
```

### KPI
Horizontal metric cards with trend indicators.
```json
{"type": "kpi", "metrics": [{"label": "Monthly Revenue", "value": 125000, "format": "currency", "trend": "up", "trend_value": "+12%"}, {"label": "Open Orders", "value": 38, "format": "number", "trend": "down", "trend_value": "-5"}]}
```

### StatusList
Colored badge items with optional document links.
```json
{"type": "status_list", "title": "Order Status", "items": [{"label": "ORD-001", "status": "Completed", "color": "green", "route": {"doctype": "Sales Order", "name": "ORD-001"}}, {"label": "ORD-002", "status": "Overdue", "color": "red"}]}
```

## Development

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) package manager
- Ollama (or another LLM provider)
- `frappe-mcp-server` running on port 8080

### Make Targets

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies (including dev) |
| `make test` | Run unit tests |
| `make test-all` | Run all tests |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |
| `make typecheck` | Run pyright type checking |
| `make serve` | Start dev server with hot reload on port 8484 |
| `make clean` | Remove build artifacts and caches |

### Project Structure

```
frappe-ai-agent/
├── src/ai_agent/
│   ├── app.py                  # FastAPI app factory with lifespan
│   ├── config.py               # Pydantic settings (env vars)
│   ├── agent/                  # LangGraph orchestration
│   │   ├── graph.py            #   ReAct agent graph builder
│   │   ├── prompts.py          #   System prompt template
│   │   └── tool_errors.py      #   Tool error → LLM message translation
│   ├── blocks/                 # Content block types
│   │   ├── models.py           #   Pydantic block models
│   │   ├── parser.py           #   <ai-block> tag extraction
│   │   └── validators.py       #   Truncation limits
│   ├── integrations/           # External service clients
│   │   ├── llm.py              #   Provider-agnostic LLM factory
│   │   ├── mcp.py              #   Per-request MCP client builder
│   │   └── frappe_history.py   #   Chat history persistence to Frappe
│   ├── middleware/             # Request processing
│   │   ├── sid.py              #   Frappe sid cookie extraction
│   │   └── request_id.py       #   Request ID injection
│   ├── observability/          # Monitoring
│   │   ├── logging.py          #   structlog configuration
│   │   └── tracing.py          #   OpenTelemetry setup
│   ├── services/               # Business logic
│   │   ├── chat.py             #   Chat orchestration + event streaming
│   │   └── health.py           #   Health checks
│   └── transport/              # API layer
│       ├── rest.py             #   REST endpoints (/health, /config)
│       ├── sse.py              #   SSE chat endpoint (POST /api/v1/chat)
│       └── sse_events.py       #   SSE wire-format serializer
├── tests/unit/                 # Unit tests (no external deps)
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Dev environment
├── pyproject.toml              # UV project config, ruff, pytest
├── Makefile                    # Build/test/serve shortcuts
└── .github/workflows/ci.yml   # CI pipeline
```

## Docker

```bash
docker build -t frappe-ai-agent .
docker run -p 8484:8484 --env-file .env frappe-ai-agent
```

Or with docker-compose:

```bash
docker compose up -d
```

## License

MIT
