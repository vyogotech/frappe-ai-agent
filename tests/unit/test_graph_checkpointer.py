from langgraph.checkpoint.memory import InMemorySaver

from ai_agent.agent.graph import build_checkpointer


def test_build_checkpointer_returns_in_memory_saver():
    cp = build_checkpointer()
    assert isinstance(cp, InMemorySaver)
