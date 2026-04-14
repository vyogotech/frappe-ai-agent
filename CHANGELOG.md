# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-04-09

### Added
- FastAPI WebSocket server with streaming events
- LangGraph ReAct agent with tool-calling loop
- MCP integration via langchain-mcp-adapters (Streamable HTTP)
- Provider-agnostic LLM via init_chat_model
- Content blocks: Text, Chart, Table, KPI, StatusList
- JWT authentication on WebSocket connections
- Redis-backed sessions and rate limiting
- PostgreSQL persistence via AsyncPostgresSaver
- OpenTelemetry tracing, metrics, and structured logging
- Docker and docker-compose for dev environment
- CI pipeline: lint, typecheck, test, security scan, Docker build
