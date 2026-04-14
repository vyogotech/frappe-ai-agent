"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_AGENT_")

    # Server
    host: str = "0.0.0.0"
    port: int = 8484
    workers: int = 4
    cors_origins: list[str] = ["http://localhost:8000"]

    # LLM
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # MCP (streamable HTTP transport is on port 8081; 8080 was the legacy REST API)
    mcp_server_url: str = "http://localhost:8081/mcp"

    # Frappe URL for chat history persistence
    frappe_url: str = "http://localhost:8000"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "frappe-ai-agent"
    log_level: str = "info"
    log_format: str = "json"
