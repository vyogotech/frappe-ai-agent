"""Vestigial — LangGraph wrapper kept for the BDD test scaffolding only.

The agent execution path moved to `ai_agent.agent.loop.run_agent_loop`
(unified envelope schema, custom loop). LangGraph is no longer in the
chat flow; this module exists to keep the test_graph_checkpointer
helper functions available for the small set of tests that exercise
the underlying `langgraph.checkpoint.*` backends. Production code does
NOT import from this module.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Checkpointer


def build_checkpointer() -> InMemorySaver:
    """In-memory saver. Kept for tests; not used by chat.py anymore."""
    return InMemorySaver()


@asynccontextmanager
async def checkpointer_context(setting: str) -> AsyncIterator[Checkpointer]:
    """Yield a configured checkpointer for the lifetime of the context.

    Kept for tests; not used by chat.py anymore. Same grammar as
    `Settings.agent_checkpointer`: `"memory"` or `"sqlite:<path>"`.
    """
    if setting == "memory":
        yield InMemorySaver()
        return
    if setting.startswith("sqlite:"):
        conn_string = setting[len("sqlite:") :]
        async with AsyncSqliteSaver.from_conn_string(conn_string) as saver:
            yield saver
        return
    raise ValueError(
        f"Unsupported checkpointer backend {setting!r}; expected 'memory' or 'sqlite:<path>'"
    )
