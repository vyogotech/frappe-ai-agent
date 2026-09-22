"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from limits import parse_many
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute, so .env is found whatever the process CWD; a missing .env is skipped.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # No extra="ignore": an orphan AI_AGENT_* key left in .env must fail startup, not fall
    # back to the default. Unknown keys in os.environ are dropped before validation either way.
    model_config = SettingsConfigDict(
        env_prefix="AI_AGENT_",
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8484
    # Matches the Dockerfile's ${AI_AGENT_WORKERS:-1}; each worker keeps its own rate-limit count.
    workers: int = 1
    cors_origins: list[str] = ["http://localhost:8000"]

    @field_validator("agent_rate_limit")
    @classmethod
    def _rate_limit_parses(cls, v: str) -> str:
        # slowapi stops limiting on a string it cannot read, so refuse one at startup
        parse_many(v)
        return v

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard_origin(cls, v: list[str]) -> list[str]:
        # With "*" and allow_credentials=True (app.py), Starlette echoes any Origin back with
        # credentials, so any site could call the agent as the signed-in user (starlette cors.py).
        if "*" in v:
            raise ValueError(
                'cors_origins cannot contain "*" — credentialed CORS requires '
                "an explicit origin list"
            )
        return v

    # LLM
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    llm_model: str = "qwen3.5:9b"
    # Low for tool-call arguments: qwen3.5:9b wrote malformed tool-call XML at 0.7.
    llm_temperature: float = 0.2
    llm_max_tokens: int = 8192
    # Ollama only. Its default context is too small for the system prompt plus tool results
    # and truncates silently.
    llm_num_ctx: int = 16384

    # Agent
    # Small models loop exploring schema; the loop runs this // 2 model-tool rounds per turn.
    agent_recursion_limit: int = 50
    # Per-sid rate limit on POST /api/v1/chat. slowapi syntax: "<count>/<period>"
    # (minute / second / hour / day).
    agent_rate_limit: str = "30/minute"

    # MCP: Streamable HTTP endpoint. frappe-mcp-server mounts /mcp on its
    # main HTTP port (default 8080), NOT the port+1 MCP-protocol-only server.
    mcp_server_url: str = "http://localhost:8080/mcp"

    # Longer than the MCP server's sid check; a timeout fails the turn (services/chat.py).
    mcp_tools_load_timeout_s: float = 20.0

    # For /health's MCP and Ollama probes; keep it short so a slow downstream cannot stall it.
    health_probe_timeout_s: float = 5.0

    # Frappe URL for chat history persistence
    frappe_url: str = "http://localhost:8000"

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "frappe-ai-agent"
    log_level: str = "info"
    log_format: str = "json"
