"""Chat service — orchestrates agent invocation and streaming."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from copilot_agent.agent.output import parse_agent_output
from copilot_agent.schemas import (
    AckEvent,
    ContentBlockEvent,
    DoneEvent,
    ErrorEvent,
    ToolEndEvent,
    ToolStartEvent,
)

logger = structlog.get_logger()


class ChatService:
    def __init__(self, agent_graph: CompiledStateGraph) -> None:
        self._graph = agent_graph

    async def handle_message(
        self,
        content: str,
        context: dict[str, Any],
        session_id: str,
        request_id: str,
    ) -> AsyncIterator[dict]:
        """Process a chat message and yield streaming events."""
        # Ack
        yield AckEvent(request_id=request_id, session_id=session_id).model_dump()

        config = {"configurable": {"thread_id": session_id}}
        input_messages = {"messages": [HumanMessage(content=content)]}

        try:
            async for event in self._graph.astream_events(
                input_messages, config=config, version="v2"
            ):
                kind = event.get("event")

                if kind == "on_tool_start":
                    yield ToolStartEvent(
                        call_id=event.get("run_id", ""),
                        name=event.get("name", ""),
                        arguments=event.get("data", {}).get("input", {}),
                    ).model_dump()

                elif kind == "on_tool_end":
                    output = event.get("data", {}).get("output", "")
                    if isinstance(output, ToolMessage):
                        output = output.content
                    yield ToolEndEvent(
                        call_id=event.get("run_id", ""),
                        result=str(output),
                        success=True,
                    ).model_dump()

                elif kind == "on_chat_model_end":
                    message = event.get("data", {}).get("output")
                    if (
                        isinstance(message, AIMessage)
                        and message.content
                        and not message.tool_calls
                    ):
                        blocks = parse_agent_output(str(message.content))
                        for block in blocks:
                            yield ContentBlockEvent(block=block.model_dump()).model_dump()

        except Exception as exc:
            logger.error("chat_error", error=str(exc), session_id=session_id)
            yield ErrorEvent(
                code="AGENT_ERROR",
                message=str(exc),
                suggestion="Try again or check service logs.",
                request_id=request_id,
            ).model_dump()

        yield DoneEvent(request_id=request_id).model_dump()
