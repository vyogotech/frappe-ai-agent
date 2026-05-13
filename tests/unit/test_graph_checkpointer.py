from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph

from ai_agent.agent.graph import build_checkpointer, checkpointer_context, create_agent_graph


def test_build_checkpointer_returns_in_memory_saver():
    cp = build_checkpointer()
    assert isinstance(cp, InMemorySaver)


def test_create_agent_graph_returns_compiled_graph():
    """create_agent_graph should return a compiled graph with the tools wired in."""

    @tool
    def _dummy(x: int) -> int:
        """doc"""
        return x

    llm = MagicMock()
    graph = create_agent_graph(
        llm=llm,
        tools=[_dummy],
        system_prompt="you are helpful",
        checkpointer=build_checkpointer(),
    )

    assert isinstance(graph, CompiledStateGraph)
    assert "tools" in graph.nodes


@pytest.mark.asyncio
async def test_checkpointer_context_memory_yields_in_memory_saver():
    """`memory` config keeps the current InMemorySaver behavior — the
    context manager yields a synchronously-built saver with no
    teardown work."""
    async with checkpointer_context("memory") as saver:
        assert isinstance(saver, InMemorySaver)


@pytest.mark.asyncio
async def test_checkpointer_context_sqlite_opens_async_saver(tmp_path: Path):
    """`sqlite:/path/to/db` config opens an AsyncSqliteSaver against the
    file at that path. The connection must be live during the context
    body and closed on exit (otherwise the process would hang on
    shutdown per langgraph's docs)."""
    db_path = tmp_path / "checkpoints.db"
    async with checkpointer_context(f"sqlite:{db_path}") as saver:
        assert isinstance(saver, AsyncSqliteSaver)
    # After exit the connection must be closed. aiosqlite exposes
    # `_running` on the underlying connection; checking `conn._connection`
    # is fragile, so instead we re-open with a fresh saver and verify the
    # file is usable (the previous connection must have released it).
    async with checkpointer_context(f"sqlite:{db_path}") as saver2:
        assert isinstance(saver2, AsyncSqliteSaver)


@pytest.mark.asyncio
async def test_checkpointer_context_in_memory_sqlite():
    """`sqlite::memory:` is the documented way to use an in-process
    SQLite db; useful for tests and ephemeral deployments. Without
    this branch, `:memory:` would be interpreted as a file path."""
    async with checkpointer_context("sqlite::memory:") as saver:
        assert isinstance(saver, AsyncSqliteSaver)


@pytest.mark.asyncio
async def test_checkpointer_context_invalid_backend_raises():
    """The context manager body validates the setting on entry — calling
    `checkpointer_context("redis://...")` alone wouldn't trigger the
    error (asyncgen bodies don't run until aenter). Inside `async with`
    the unknown-backend branch raises before any saver is yielded."""
    with pytest.raises(ValueError) as excinfo:
        async with checkpointer_context("redis://localhost") as _saver:
            pass  # pragma: no cover — unreachable
    assert "redis" in str(excinfo.value).lower() or "unsupported" in str(excinfo.value).lower()
