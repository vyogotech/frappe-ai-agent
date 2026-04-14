from unittest.mock import MagicMock

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode

from ai_agent.agent.graph import build_checkpointer, create_agent_graph
from ai_agent.agent.tool_errors import to_tool_result_message


def test_build_checkpointer_returns_in_memory_saver():
    cp = build_checkpointer()
    assert isinstance(cp, InMemorySaver)


def test_create_agent_graph_wires_to_tool_result_message_as_tool_error_handler():
    """The graph's ToolNode must use to_tool_result_message so generic
    exceptions (e.g. Frappe 403) become LLM-readable strings instead of
    propagating past the ToolNode and crashing the graph run."""

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

    # create_react_agent composes the tool node under node name "tools".
    tool_node = graph.nodes["tools"].bound
    assert isinstance(tool_node, ToolNode)
    assert tool_node._handle_tool_errors is to_tool_result_message
