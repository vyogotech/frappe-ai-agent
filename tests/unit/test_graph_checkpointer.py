from unittest.mock import MagicMock

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from ai_agent.agent.graph import build_checkpointer, create_agent_graph


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
