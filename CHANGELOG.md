# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
