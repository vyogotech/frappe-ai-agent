"""Agent state definition for LangGraph."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State passed between graph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    context: dict
