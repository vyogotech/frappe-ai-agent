"""PostgreSQL integration for LangGraph checkpoints."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


@asynccontextmanager
async def create_checkpointer(database_url: str) -> AsyncIterator[AsyncPostgresSaver]:
    """Create and initialize an async PostgreSQL checkpointer.

    Must be used as an async context manager — the connection is closed on exit.
    """
    # Convert asyncpg URL format for psycopg (required by langgraph-checkpoint-postgres)
    conn_string = database_url.replace("postgresql+asyncpg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(conn_string) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
