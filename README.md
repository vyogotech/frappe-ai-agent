# Frappe Copilot Agent

AI agent service for Frappe/ERPNext -- WebSocket streaming, tool-calling via MCP, and rich visual responses.

![CI](https://github.com/frappe/copilot-agent/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

## Overview

Frappe Copilot Agent is a standalone AI service that powers the Copilot sidebar in Frappe/ERPNext. Users ask questions in natural language and receive structured visual answers -- charts, tables, KPI cards, and status lists -- streamed in real time over WebSocket.

The agent uses a LangGraph ReAct loop to reason about questions, call ERPNext tools via the Model Context Protocol (MCP), and compose rich content blocks from the results.

## Architecture

```
Frappe App (Vue sidebar)
    |
    | WebSocket (JSON events)
    v
+--------------------------------------------------+
|  Frappe Copilot Agent                            |
|                                                  |
|  Transport        FastAPI, WebSocket, REST       |
|  Middleware        JWT auth, rate limiting        |
|  Services          Chat, Session, Health         |
|  Orchestration     LangGraph ReAct agent         |
|  Integrations      LLM, MCP, PostgreSQL, Redis   |
|  Observability     OTEL tracing, metrics, logs   |
+--------------------------------------------------+
    |                    |
    | Streamable HTTP    | asyncpg / redis
    v                    v
MCP Server           PostgreSQL + Redis
    |
    | API calls
    v
ERPNext
```

## Features

- **WebSocket streaming** -- real-time token-by-token and event-based delivery
- **Tool-calling via MCP** -- LangGraph agent invokes ERPNext tools through Streamable HTTP
- **Content blocks** -- Text, Chart (bar/line/pie/funnel/heatmap/calendar), Table, KPI, StatusList
- **JWT authentication** -- token-based WebSocket auth with configurable expiry
- **Rate limiting** -- Redis-backed per-user request throttling
- **Redis sessions** -- persistent conversation sessions across reconnects
- **PostgreSQL checkpoints** -- LangGraph state persistence via AsyncPostgresSaver
- **Provider-agnostic LLM** -- supports OpenAI, Anthropic, Google via `init_chat_model`
- **OpenTelemetry** -- distributed tracing, metrics, and structured JSON logging
- **Docker-ready** -- multi-stage Dockerfile and docker-compose for dev/prod

## Quick Start

```bash
# 1. Clone
git clone https://github.com/frappe/copilot-agent.git
cd copilot-agent

# 2. Install dependencies
pip install uv
uv sync --all-extras

# 3. Configure
cp .env.example .env
# Edit .env: set COPILOT_JWT_SECRET to a random string

# 4. Start infrastructure
docker compose up -d postgres redis

# 5. Run the server
make serve
```

The agent is now running at `ws://localhost:8484/ws/chat`.

## Configuration

All settings are configured via environment variables with the `COPILOT_` prefix.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `COPILOT_HOST` | `str` | `0.0.0.0` | Server bind address |
| `COPILOT_PORT` | `int` | `8484` | Server port |
| `COPILOT_WORKERS` | `int` | `4` | Uvicorn worker count |
| `COPILOT_CORS_ORIGINS` | `list[str]` | `["*"]` | Allowed CORS origins |
| `COPILOT_JWT_SECRET` | `str` | *required* | Secret key for JWT signing |
| `COPILOT_JWT_ALGORITHM` | `str` | `HS256` | JWT algorithm |
| `COPILOT_JWT_EXPIRY_HOURS` | `int` | `24` | JWT token expiry in hours |
| `COPILOT_LLM_PROVIDER` | `str` | `openai` | LLM provider (openai, anthropic, google) |
| `COPILOT_LLM_BASE_URL` | `str` | `http://localhost:11434/v1` | LLM API base URL |
| `COPILOT_LLM_API_KEY` | `str` | `""` | LLM API key |
| `COPILOT_LLM_MODEL` | `str` | `llama3.2:3b` | Model name |
| `COPILOT_LLM_TEMPERATURE` | `float` | `0.7` | Sampling temperature |
| `COPILOT_LLM_MAX_TOKENS` | `int` | `4096` | Max output tokens |
| `COPILOT_MCP_SERVER_URL` | `str` | `http://localhost:8080/mcp` | MCP server endpoint |
| `COPILOT_DATABASE_URL` | `str` | `postgresql+asyncpg://copilot:copilot@localhost:5432/copilot` | PostgreSQL connection |
| `COPILOT_REDIS_URL` | `str` | `redis://localhost:6379/0` | Redis connection |
| `COPILOT_RATE_LIMIT_REQUESTS` | `int` | `60` | Max requests per window |
| `COPILOT_RATE_LIMIT_WINDOW_SECONDS` | `int` | `60` | Rate limit window in seconds |
| `COPILOT_OTEL_ENDPOINT` | `str` | `""` | OTLP exporter endpoint (empty = disabled) |
| `COPILOT_OTEL_SERVICE_NAME` | `str` | `copilot-agent` | OTEL service name |
| `COPILOT_LOG_LEVEL` | `str` | `info` | Log level (debug, info, warning, error) |
| `COPILOT_LOG_FORMAT` | `str` | `json` | Log format (json, console) |

## WebSocket Protocol

Connect to `/ws/chat` with a JWT token as a query parameter: `ws://host:8484/ws/chat?token=<jwt>`.

### Client -> Server

```json
{
  "type": "chat",
  "content": "Show me this month's sales",
  "context": {"doctype": "Sales Invoice", "page": "list"},
  "session_id": "ses_abc123",
  "request_id": "req_001"
}
```

### Server -> Client Events

**ack** -- message received, processing started:
```json
{"type": "ack", "request_id": "req_001", "session_id": "ses_abc123"}
```

**tool_start** -- agent is calling a tool:
```json
{"type": "tool_start", "call_id": "call_1", "name": "list_documents", "arguments": {"doctype": "Sales Invoice"}}
```

**tool_end** -- tool call completed:
```json
{"type": "tool_end", "call_id": "call_1", "result": "Found 42 invoices", "success": true}
```

**content_block** -- structured visual content:
```json
{"type": "content_block", "block": {"type": "chart", "chart_type": "bar", "title": "Monthly Sales", "data": {"labels": ["Jan", "Feb", "Mar"], "datasets": [{"name": "Revenue", "values": [50000, 62000, 71000]}]}, "options": {"format": "currency", "currency": "USD"}}}
```

**token** -- streaming text token:
```json
{"type": "token", "content": "Here are"}
```

**error** -- something went wrong:
```json
{"type": "error", "code": "AGENT_ERROR", "message": "MCP server unreachable", "suggestion": "Check MCP server status", "request_id": "req_001"}
```

**done** -- request complete:
```json
{"type": "done", "request_id": "req_001", "usage": {"input_tokens": 150, "output_tokens": 320}}
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
- Docker and Docker Compose
- PostgreSQL 16+ and Redis 7+ (or use docker-compose)

### Make Targets

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies (including dev) |
| `make test` | Run unit tests |
| `make test-integration` | Run integration tests (requires Docker services) |
| `make test-all` | Run all tests |
| `make lint` | Check code with ruff |
| `make format` | Auto-format code with ruff |
| `make typecheck` | Run pyright type checking |
| `make serve` | Start dev server with hot reload on port 8484 |
| `make clean` | Remove build artifacts and caches |

### Testing

```bash
# Unit tests (no external services needed)
uv run pytest tests/unit/ -v

# With coverage
uv run pytest tests/unit/ -v --cov=copilot_agent --cov-report=term-missing

# Integration tests (requires PostgreSQL + Redis)
docker compose up -d postgres redis
uv run pytest tests/integration/ -v -m integration
```

### Project Structure

```
frappe-copilot-agent/
├── src/copilot_agent/
│   ├── app.py                  # FastAPI app factory with lifespan
│   ├── config.py               # Pydantic settings (env vars)
│   ├── agent/                  # LangGraph orchestration
│   │   ├── graph.py            #   ReAct agent graph
│   │   ├── state.py            #   Agent state schema
│   │   ├── prompts.py          #   System prompt builder
│   │   └── output.py           #   Output parser
│   ├── blocks/                 # Content block types
│   │   ├── models.py           #   Pydantic models
│   │   ├── parser.py           #   Block extraction from LLM output
│   │   └── validators.py       #   Sanitize and truncate
│   ├── integrations/           # External service clients
│   │   ├── llm.py              #   Provider-agnostic LLM factory
│   │   ├── mcp.py              #   MCP client adapter
│   │   ├── postgres.py         #   AsyncPostgresSaver checkpointer
│   │   └── redis.py            #   Redis client wrapper
│   ├── middleware/             # Request processing
│   │   ├── auth.py             #   JWT authentication
│   │   ├── rate_limit.py       #   Redis-backed rate limiter
│   │   └── request_id.py       #   Request ID injection
│   ├── observability/          # Monitoring
│   │   ├── logging.py          #   structlog configuration
│   │   ├── tracing.py          #   OpenTelemetry setup
│   │   └── metrics.py          #   OTEL metrics
│   ├── services/               # Business logic
│   │   ├── chat.py             #   Chat orchestration
│   │   ├── health.py           #   Health checks
│   │   └── session.py          #   Session management
│   └── transport/              # API layer
│       ├── rest.py             #   REST endpoints
│       ├── websocket.py        #   WebSocket handler
│       └── schemas.py          #   Request/response models
├── tests/
│   ├── unit/                   # Unit tests (no external deps)
│   ├── integration/            # Integration tests (requires services)
│   └── features/               # BDD feature files
├── Dockerfile                  # Multi-stage production build
├── docker-compose.yml          # Dev environment (agent + PG + Redis)
├── pyproject.toml              # UV project config, ruff, pytest
├── Makefile                    # Build/test/serve shortcuts
└── .github/workflows/ci.yml   # CI pipeline
```

## Deployment

### Docker Build

```bash
docker build -t frappe-copilot-agent .
docker run -p 8484:8484 --env-file .env frappe-copilot-agent
```

### Docker Compose (Full Stack)

```bash
docker compose up -d
```

### Production Checklist

- [ ] Set `COPILOT_JWT_SECRET` to a strong random value
- [ ] Restrict `COPILOT_CORS_ORIGINS` to your Frappe domain
- [ ] Use production PostgreSQL and Redis instances (not the bundled containers)
- [ ] Set `COPILOT_OTEL_ENDPOINT` to your observability platform
- [ ] Set `COPILOT_LOG_FORMAT=json` for structured log aggregation
- [ ] Configure `COPILOT_LLM_API_KEY` for your chosen provider
- [ ] Run behind a reverse proxy (nginx/caddy) with TLS termination
- [ ] Set `COPILOT_RATE_LIMIT_REQUESTS` appropriate for your user base

## License

MIT
