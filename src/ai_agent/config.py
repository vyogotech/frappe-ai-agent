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
    # Default to 1 so the checkpointer-vs-workers warning is rare and
    # intentional (operators who pick multi-worker deliberately set this).
    # The Dockerfile honours `${AI_AGENT_WORKERS:-1}`, so this default
    # matches actual container behaviour.
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
    # Why: small models loop while exploring schema and trip the LangGraph
    # default of 25 before converging. 50 is enough headroom without letting
    # a truly stuck agent run forever.
    agent_recursion_limit: int = 50
    # Why: per-sid rate limit on POST /api/v1/chat. slowapi syntax;
    # "<count>/<period>" — minute / second / hour / day.
    agent_rate_limit: str = "30/minute"
    # LangGraph checkpointer backend. Determines where per-conversation
    # state (thread history that drives multi-turn continuity) lives:
    #   "memory" — InMemorySaver. Process-local; lost on restart. The
    #     default, but NOT safe with workers > 1 because uvicorn does
    #     not pin a sid to a worker — a second turn has a 1/workers
    #     chance of seeing the prior checkpoint.
    #   "sqlite:/abs/path/to/checkpoints.db" — AsyncSqliteSaver. Shared
    #     across processes via the file. Production-acceptable for
    #     low-concurrency deployments; langgraph's docs caution against
    #     it under heavy write load.
    #   "sqlite::memory:" — in-process SQLite. Same process-local
    #     limitation as memory; useful for tests.
    agent_checkpointer: str = "memory"

    @field_validator("agent_checkpointer")
    @classmethod
    def _validate_checkpointer(cls, v: str) -> str:
        # Catches "memry" / "Postgres://..." typos that would have
        # silently fallen back to in-memory at startup.
        if v == "memory" or v.startswith("sqlite:"):
            return v
        raise ValueError(f"agent_checkpointer must be 'memory' or 'sqlite:<path>', got {v!r}")

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
