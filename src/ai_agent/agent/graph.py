"""LangGraph ReAct agent with checkpointer."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer


def build_checkpointer() -> InMemorySaver:
    """Build an in-memory checkpointer for the agent graph."""
    return InMemorySaver()


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
