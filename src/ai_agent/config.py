"""Application configuration via environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
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
    # The Dockerfile honours `${AI_AGENT_WORKERS:-1}`, so this default
    # matches actual container behaviour. The agent is stateless per
    # request (history is fetched from Frappe each turn) so raising
    # workers is safe whenever the LLM/MCP backend can keep up.
    workers: int = 1
    cors_origins: list[str] = ["http://localhost:8000"]

    @field_validator("cors_origins")
    @classmethod
    def _reject_wildcard_origin(cls, v: list[str]) -> list[str]:
        # Why: app.py sets allow_credentials=True. Starlette silently refuses
        # to send credentialed responses when allow_origins contains "*", so
        # a misconfigured deployment would 200 the request and *appear* fine
        # while the browser drops the response. Fail at startup instead.
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
    # Lower temperature for more deterministic tool-call argument formatting.
    # qwen3.5:9b at 0.7 occasionally emits malformed function-call XML
    # (e.g. "<function> closed by </parameter>") after 8+ tool calls in a
    # session. 0.2 keeps prose readable but tightens tool-arg syntax.
    llm_temperature: float = 0.2
    llm_max_tokens: int = 8192
    # Ollama context window. Default is 2048 which is too small for our
    # system prompt + tool results + final answer with structured blocks —
    # the model silently truncates earlier context and produces garbled
    # mid-response output. Bump to 16k for headroom on multi-tool queries.
    # Ignored for non-Ollama providers.
    llm_num_ctx: int = 16384

    # Agent
    # Why: small models loop while exploring schema; the envelope loop
    # divides this by 2 to bound model→tool→model round-trips per turn.
    # 50 is enough headroom without letting a truly stuck agent run forever.
    agent_recursion_limit: int = 50
    # Per-sid rate limit on POST /api/v1/chat. slowapi syntax: "<count>/<period>"
    # (minute / second / hour / day).
    agent_rate_limit: str = "30/minute"

    # MCP: Streamable HTTP endpoint. frappe-mcp-server mounts /mcp on its
    # main HTTP port (default 8080), NOT the port+1 MCP-protocol-only server.
    mcp_server_url: str = "http://localhost:8080/mcp"

    # How long the agent waits for `mcp_client.get_tools()` before giving
    # up and soft-degrading to "tools unavailable". A timeout here is
    # almost always a misconfigured MCP server or a stale sid the MCP
    # auth layer is busy validating — 20s is the upper bound on either.
    mcp_tools_load_timeout_s: float = 20.0

    # Connect/read timeout for the agent's own outbound HTTP probes
    # (`/health` reachability checks against MCP and Frappe). These are
    # cheap pings; keep them fast so a flaky downstream doesn't block
    # the liveness handler.
    health_probe_timeout_s: float = 5.0

    # Frappe URL for chat history persistence
    frappe_url: str = "http://localhost:8000"

    # Observability
    otel_endpoint: str = ""
    otel_service_name: str = "frappe-ai-agent"
    log_level: str = "info"
    log_format: str = "json"
