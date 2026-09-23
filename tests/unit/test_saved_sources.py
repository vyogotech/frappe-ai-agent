"""D08: an answer saves where a passage was read, not the passage itself."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import respx
from httpx import Response

from ai_agent.config import Settings
from ai_agent.integrations.frappe_history import FrappeHistoryClient
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ChatService

EXCERPT = "Meals are capped at 45 AUD a day, and the cap is per traveller."
FRAPPE = "http://frappe.test"


def _service() -> tuple[ChatService, MagicMock]:
    history = MagicMock()
    history.ensure_session = AsyncMock(side_effect=lambda *, name, **_: name)
    history.list_messages = AsyncMock(return_value=[])
    history.save_message = AsyncMock(return_value="msg-1")
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]
        llm_provider="ollama",
        llm_model="qwen3.5:9b",
        llm_base_url="http://localhost:11434",
        mcp_server_url="http://mcp:8080/mcp",
    )
    return ChatService(
        settings=settings,
        llm=MagicMock(),
        system_prompt_builder=lambda _ctx: "system",
        history=history,
    ), history


def _loop(events: list[dict[str, Any]]):
    def _factory(**_kwargs):
        async def _gen():
            for ev in events:
                yield ev

        return _gen()

    return _factory


async def _saved(loop_events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """The sources saved with the answer, and the whole saved payload as text."""
    service, history = _service()
    mcp = MagicMock()
    mcp.get_tools = AsyncMock(return_value=[])
    with (
        patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp),
        patch("ai_agent.services.chat.run_agent_loop", _loop(loop_events)),
    ):
        async for _ev in service.handle_message(
            message="what is the meal cap?",
            session_id="sess-1",
            context={},
            user_context=UserContext(sid="sid-1"),
        ):
            pass
    payload = history.save_message.call_args_list[-1].kwargs["tool_result_json"]
    return json.loads(payload)["sources"], payload


async def test_an_answer_saves_the_file_and_the_place_not_the_passage():
    passage = {"file": "abc123", "seq": 3, "distance": 0.21, "content": EXCERPT}
    sources, payload = await _saved(
        [{"type": "sources", "items": [passage]}, {"type": "content", "text": "45 AUD."}]
    )

    assert sources == [{"file": "abc123", "seq": 3, "distance": None}]
    assert "Meals are capped" not in payload


async def test_the_same_passage_read_twice_is_saved_once():
    passage = {"file": "abc123", "seq": 3, "distance": 0.21, "content": EXCERPT}
    sources, _ = await _saved(
        [
            {"type": "sources", "items": [passage]},
            {"type": "sources", "items": [passage, {**passage, "seq": 4}]},
            {"type": "content", "text": "45 AUD."},
        ]
    )

    assert [s["seq"] for s in sources] == [3, 4]


async def test_a_chat_attachment_keeps_the_name_only_the_answer_has():
    passage = {
        "file": "att1",
        "seq": 0,
        "distance": 0.3,
        "content": EXCERPT,
        "file_name": "policy.pdf",
        "attachment": True,
    }
    sources, payload = await _saved(
        [{"type": "sources", "items": [passage]}, {"type": "content", "text": "45 AUD."}]
    )

    assert sources == [
        {"file": "att1", "seq": 0, "file_name": "policy.pdf", "attachment": True, "distance": None}
    ]
    assert "Meals are capped" not in payload


@respx.mock
async def test_a_chat_saved_in_the_old_shape_still_opens():
    """Rows written before the change still hold `content`; reading one back must not break."""
    old = json.dumps(
        {
            "sources": [{"file": "abc123", "seq": 3, "distance": 0.21, "content": EXCERPT}],
            "blocks": [{"type": "table", "columns": ["cap"], "rows": [["45"]]}],
        }
    )
    rows = [
        {"role": "assistant", "content": "45 AUD.", "tool_result_json": old},
        {"role": "user", "content": "what is the meal cap?", "tool_result_json": None},
    ]
    respx.get(f"{FRAPPE}/api/method/frappe.client.get_list").mock(
        return_value=Response(200, json={"message": rows})
    )

    history = await FrappeHistoryClient(base_url=FRAPPE).list_messages(sid="s", session="c")

    assert [h["role"] for h in history] == ["user", "assistant"]
    assert "45" in history[1]["content"]
