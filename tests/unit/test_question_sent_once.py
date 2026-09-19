"""The question reaches the model once, however the history store orders the save and the read."""

from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from ai_agent.blocks.envelope import build_agent_messages
from ai_agent.config import Settings
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService


async def test_the_model_gets_the_new_question_once():
    settings = Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp")  # pyright: ignore[reportCallIssue]
    service = ChatService(settings=settings, llm=MagicMock(), system_prompt_builder=lambda _ctx: "")
    saved = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]

    async def save_message(*, role, content, **_):
        saved.append({"role": role, "content": content})
        return "m"

    async def list_messages(**_):
        return list(saved)

    service._history = MagicMock(
        save_message=save_message,
        ensure_session=AsyncMock(side_effect=lambda *, name, **_: name),
        list_messages=list_messages,
    )
    seen = {}

    def loop(**kwargs):
        seen["messages"] = build_agent_messages(
            user_message=kwargs["user_message"], history=kwargs["history"]
        )

        async def gen():
            yield {"type": "done"}

        return gen()

    client = MagicMock(get_tools=AsyncMock(return_value=[]))
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=client),
        patch("ai_agent.services.chat.run_agent_loop", loop),
    ):
        turn = service.handle_message(
            message="new question",
            session_id="s-1",
            context={},
            user_context=UserContext(sid="abc"),
        )
        [ev async for ev in turn]
    questions = [m.content for m in seen["messages"] if isinstance(m, HumanMessage)]
    assert questions == ["earlier question", "new question"]
