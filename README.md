# Frappe AI Agent

AI agent service for Frappe/ERPNext — SSE streaming, tool-calling via MCP, and rich visual responses.

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

Frappe AI Agent powers the AI sidebar in Frappe/ERPNext. Users ask questions in natural language and receive structured visual answers — charts, tables, KPI cards, and status lists — streamed over Server-Sent Events.

The agent runs a LangGraph ReAct loop, calls ERPNext tools via MCP, and authenticates every request using the caller's Frappe `sid` cookie so permissions are enforced end-to-end.

## Architecture

```
Browser (Vue sidebar)
    │  POST /api/v1/chat  (SSE, Cookie: sid=...)
    ▼
frappe-ai-agent  (LangGraph ReAct loop)
    │
    ├──▶ Ollama / LLM           (reasoning + token stream)
    ├──▶ frappe-mcp-server      (MCP tools, sid forwarded)
    │       ▼
    │    Frappe REST API         (runs as that user)
    └──▶ Frappe DocType API     (save AI Chat Session/Message)
```

## Quick Start

**Prerequisites:** Python 3.12+, [UV](https://docs.astral.sh/uv/), Ollama running locally, `frappe-mcp-server` on port 8080.

```bash
uv sync --all-extras
cp .env.example .env
make serve
```

The agent runs at `http://localhost:8484`. Chat endpoint: `POST /api/v1/chat`.

## Configuration

All settings use the `AI_AGENT_` prefix. See [`.env.example`](.env.example) for the full list with defaults.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_AGENT_LLM_PROVIDER` | `ollama` | `ollama`, `openai`, `anthropic`, `google` |
| `AI_AGENT_LLM_MODEL` | `qwen3.5:9b` | Model name |
| `AI_AGENT_MCP_SERVER_URL` | `http://localhost:8080/mcp` | MCP server endpoint |
| `AI_AGENT_FRAPPE_URL` | `http://localhost:8000` | Frappe instance (for chat history) |

## Development

| Command | Description |
|---------|-------------|
| `make install` | Install all dependencies |
| `make test` | Run unit tests |
| `make lint` | Check code with ruff |
| `make typecheck` | Run pyright |
| `make serve` | Dev server with hot reload (port 8484) |

## Docker

```bash
docker build -t frappe-ai-agent .
docker run -p 8484:8484 --env-file .env frappe-ai-agent
```

## License

MIT
