"""Application configuration via environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_AGENT_")

    # Server
    host: str = "0.0.0.0"
    port: int = 8484
    workers: int = 4
    cors_origins: list[str] = ["http://localhost:8080"]

    # Auth
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24

    # LLM
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_model: str = "qwen3.5:9b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # MCP
    mcp_server_url: str = "http://localhost:8080/mcp"

    # PostgreSQL
    database_url: str = "postgresql+asyncpg://copilot:copilot@localhost:5432/copilot"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Rate Limiting
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "copilot-agent"
    log_level: str = "info"
    log_format: str = "json"
