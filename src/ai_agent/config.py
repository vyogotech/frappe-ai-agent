"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — the .env file sits next to pyproject.toml. Using an absolute
# path keeps loading independent of the process CWD (tests, uvicorn in any
# directory, Docker with bind-mounts, etc.). Missing .env is not an error;
# pydantic-settings silently skips it and falls back to os.environ.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # No extra="ignore": an orphan AI_AGENT_* key in .env (field removed from
    # code but never cleaned out of the file) should crash startup, not be
    # silently dropped onto the floor while the Python default takes over.
    # (Note: pydantic-settings drops unknown prefixed vars from os.environ
    # before validation, so this only catches drift in the .env file.)
    model_config = SettingsConfigDict(
        env_prefix="AI_AGENT_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

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
    llm_max_tokens: int = 8192
    # Ollama context window. Default is 2048 which is too small for our
    # system prompt + tool results + final answer with structured blocks —
    # the model silently truncates earlier context and produces garbled
    # mid-response output. Bump to 16k for headroom on multi-tool queries.
    # Ignored for non-Ollama providers.
    llm_num_ctx: int = 16384

    # MCP: Streamable HTTP endpoint. frappe-mcp-server mounts /mcp on its
    # main HTTP port (default 8080), NOT the port+1 MCP-protocol-only server.
    mcp_server_url: str = "http://localhost:8080/mcp"

    # Frappe URL for chat history persistence
    frappe_url: str = "http://localhost:8000"

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "frappe-ai-agent"
    log_level: str = "info"
    log_format: str = "json"
