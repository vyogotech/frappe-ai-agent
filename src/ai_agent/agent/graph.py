"""LangGraph ReAct agent with checkpointer."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

from ai_agent.agent.tool_errors import to_tool_result_message


def build_checkpointer() -> InMemorySaver:
    """Build an in-memory checkpointer for the agent graph."""
    return InMemorySaver()


def create_agent_graph(
    llm: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Create a LangGraph ReAct agent with optional persistence.

    Tool errors (including generic exceptions like Frappe 403s) are caught at
    the ToolNode level and turned into LLM-readable tool-result strings via
    to_tool_result_message. Without this, LangGraph's default handler only
    rescues ToolInvocationError and re-raises everything else, crashing the
    graph on any real Frappe permission failure.
    """
    tool_node = ToolNode(tools, handle_tool_errors=to_tool_result_message)
    return create_react_agent(
        model=llm,
        tools=tool_node,
        prompt=system_prompt,
        checkpointer=checkpointer,
    )
