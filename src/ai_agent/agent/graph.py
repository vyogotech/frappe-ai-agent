"""LangGraph ReAct agent with checkpointer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer


def build_checkpointer() -> InMemorySaver:
    """Build an in-memory checkpointer for the agent graph.

    Kept for the synchronous factory path (tests, app construction
    before lifespan). Production app.py uses `checkpointer_context`
    inside the FastAPI lifespan to support sqlite as well.
    """
    return InMemorySaver()


@asynccontextmanager
async def checkpointer_context(setting: str) -> AsyncIterator[Checkpointer]:
    """Yield a configured checkpointer for the lifetime of the context.

    `setting` follows the same grammar as `Settings.agent_checkpointer`:
    - `"memory"` — InMemorySaver. No teardown.
    - `"sqlite:<path>"` — AsyncSqliteSaver against the given file path,
      with the connection opened on entry and closed on exit (langgraph's
      docs warn that an unclosed connection can hang the process at
      shutdown). `<path>` is forwarded to aiosqlite, so `:memory:` is
      accepted as an in-process db.

    Raises ValueError on unknown backend strings — the field validator
    on Settings is the front-line check; this is the defense-in-depth
    line for callers that build a setting string by other means.
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


def create_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: Checkpointer | None = None,
) -> CompiledStateGraph:
    """Create a LangGraph ReAct agent with optional persistence.

    Tool errors are handled per-tool via handle_tool_error set in ChatService.
    """
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )
