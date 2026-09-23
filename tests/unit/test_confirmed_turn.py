"""The turn a click starts: it runs the stored call once, with the token in a header only."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.sessions import StreamableHttpConnection
from test_agent_loop import _drain, _fake_llm_with_yields

from ai_agent.agent.loop import run_agent_loop
from ai_agent.agent.tool_registry import ToolRegistry
from ai_agent.config import Settings
from ai_agent.integrations.mcp import build_mcp_client_for_sid
from ai_agent.middleware.sid import UserContext
from ai_agent.services.chat import ANSWER_STOPPED, CONFIRMED_TURN, ChatService
from ai_agent.transport.sse import _require_sid, create_sse_router

# not hex, so a scanner reads it for what it is: the assertions need only a distinctive string
TOKEN = "not-a-real-confirmation-token-xyzzy"
STORED = {
    "tool": "create_document",
    "arguments": {"doctype": "Customer", "data": {"customer_name": "Acme"}},
    "token": TOKEN,
}


def _tool(calls: list[dict[str, Any]]) -> StructuredTool:
    async def _run(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "created CUST-002"

    return StructuredTool(
        name="create_document",
        description="create",
        args_schema={"type": "object", "properties": {}, "additionalProperties": True},
        coroutine=_run,
    )


def _llm(turns: list[list[BaseMessage]]) -> MagicMock:
    """A model that narrates whatever it is given, and records the messages it was given."""

    def _astream(messages, config=None):
        turns.append(list(messages))

        async def _gen():
            yield {
                "blocks": [{"type": "text", "payload": {"content": "Customer Acme is created."}}]
            }

        return _gen()

    structured = MagicMock()
    structured.astream = _astream
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def _service(history: MagicMock, turns: list[list[BaseMessage]]) -> ChatService:
    return ChatService(
        settings=Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp"),  # pyright: ignore[reportCallIssue]
        llm=_llm(turns),
        system_prompt_builder=lambda _ctx: "",
        history=history,
    )


def _history(saved: list[dict[str, Any]]) -> MagicMock:
    return MagicMock(
        save_message=AsyncMock(side_effect=lambda **kw: saved.append(kw)),
        ensure_session=AsyncMock(return_value="s-1"),
        list_messages=AsyncMock(return_value=[]),
    )


async def _run_confirmed(
    saved: list[dict[str, Any]],
    calls: list[dict[str, Any]],
    turns: list[list[BaseMessage]],
    built: list[tuple],
) -> list[dict[str, Any]]:
    mcp = MagicMock(get_tools=AsyncMock(return_value=[_tool(calls)]))

    def _build(settings, sid, confirmation_token=None):
        built.append((sid, confirmation_token))
        return mcp

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", side_effect=_build):
        return [
            e
            async for e in _service(_history(saved), turns).handle_message(
                session_id="s-1",
                context={},
                user_context=UserContext(sid="abc"),
                confirmation=dict(STORED),
            )
        ]


async def test_a_confirmed_turn_runs_exactly_the_stored_call():
    saved: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    turns: list[list[BaseMessage]] = []
    built: list[tuple] = []

    events = await _run_confirmed(saved, calls, turns, built)

    assert calls == [{"doctype": "Customer", "data": {"customer_name": "Acme"}}]
    assert [e["type"] for e in events] == ["session", "tool_call", "content", "done"]
    assert events[1] == {
        "type": "tool_call",
        "name": "create_document",
        "arguments": STORED["arguments"],
    }
    assert events[-1]["tools_called"] == ["create_document"]


async def test_the_token_travels_as_a_header_and_reaches_nothing_else():
    saved: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    turns: list[list[BaseMessage]] = []
    built: list[tuple] = []

    events = await _run_confirmed(saved, calls, turns, built)

    assert built == [("abc", TOKEN)]
    assert TOKEN not in str(calls), "the token must never be a tool argument"
    assert TOKEN not in str([m.content for turn in turns for m in turn]), (
        "the token must never reach the model"
    )
    assert TOKEN not in str(saved), "the token must never be written to history"
    assert TOKEN not in str(events), "the token must never go out on the stream"


async def test_stopping_a_confirmed_turn_keeps_the_text_the_write_already_earned():
    """Stop after the write has run cannot undo it, so the turn keeps what it said, as any does."""
    saved: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    mcp = MagicMock(get_tools=AsyncMock(return_value=[_tool(calls)]))
    turn = _service(_history(saved), []).handle_message(
        session_id="s-1",
        context={},
        user_context=UserContext(sid="abc"),
        confirmation=dict(STORED),
    )

    with patch("ai_agent.services.chat.build_mcp_client_for_sid", return_value=mcp):
        async for event in turn:
            if event["type"] == "content":
                await turn.aclose()

    assert calls == [STORED["arguments"]]
    assert [s["content"] for s in saved] == [
        f"Customer Acme is created.\n\n[incomplete] {ANSWER_STOPPED}"
    ]


async def test_a_confirmed_turn_saves_no_user_row():
    saved: list[dict[str, Any]] = []
    await _run_confirmed(saved, [], [], [])

    assert [s["role"] for s in saved] == ["assistant"]
    assert [s["content"] for s in saved] == ["Customer Acme is created."]


async def test_the_model_is_told_the_action_already_ran():
    turns: list[list[BaseMessage]] = []
    await _run_confirmed([], [], turns, [])

    replayed = "\n".join(str(m.content) for m in turns[0])
    assert CONFIRMED_TURN in replayed
    assert "created CUST-002" in replayed
    # the result is data, like any other tool result, not a second message from the user
    assert "<tool_results>" in replayed


def _headers(client) -> dict[str, str]:
    connection = cast(StreamableHttpConnection, client.connections["frappe"])
    return connection.get("headers") or {}


def test_the_header_is_absent_without_a_token():
    settings = Settings(_env_file=None, mcp_server_url="http://mcp:8080/mcp")  # pyright: ignore[reportCallIssue]

    assert _headers(build_mcp_client_for_sid(settings, sid="abc")) == {"Cookie": "sid=abc"}
    assert _headers(build_mcp_client_for_sid(settings, sid="abc", confirmation_token=TOKEN)) == {
        "Cookie": "sid=abc",
        "X-Frappe-Confirmation": TOKEN,
    }


async def test_a_second_write_the_model_asks_for_pauses_again():
    """The confirmed call burned the token, so the model's next write needs its own click."""
    calls: list[dict[str, Any]] = []
    llm = _fake_llm_with_yields(
        [
            [
                {
                    "blocks": [
                        {
                            "type": "tool_call",
                            "payload": {
                                "name": "create_document",
                                "arguments": {"doctype": "Lead"},
                            },
                        }
                    ]
                }
            ]
        ]
    )
    events = await _drain(
        run_agent_loop(
            llm=llm,
            tool_registry=ToolRegistry([_tool(calls)]),
            user_message=CONFIRMED_TURN,
            confirmed={"name": "create_document", "arguments": {"doctype": "Customer"}},
        )
    )

    assert calls == [{"doctype": "Customer"}]
    assert [e["type"] for e in events] == ["tool_call", "content", "tool_confirm"]
    assert events[-1]["arguments"] == {"doctype": "Lead"}


# --- the route -----------------------------------------------------------------


class _Recorder:
    def __init__(self) -> None:
        self.seen: list[dict[str, Any] | None] = []

    async def handle_message(
        self, *, message, session_id, context, user_context, confirmation=None
    ):
        self.seen.append(confirmation)
        yield {"type": "done", "tools_called": [], "data_quality": "high", "timestamp": "t"}


def _app(recorder: _Recorder) -> FastAPI:
    app = FastAPI()
    app.state.chat_service = recorder
    app.include_router(create_sse_router())
    app.dependency_overrides[_require_sid] = lambda: UserContext(sid="abc")
    return app


async def _post(body: dict[str, Any], recorder: _Recorder):
    async with AsyncClient(
        transport=ASGITransport(app=_app(recorder)), base_url="http://test"
    ) as ac:
        return await ac.post("/api/v1/chat", json=body)


async def test_the_route_passes_a_confirmation_with_no_message():
    recorder = _Recorder()
    response = await _post({"session_id": "s-1", "confirmation": STORED}, recorder)

    assert response.status_code == 200
    assert recorder.seen == [STORED]


async def test_the_route_refuses_a_body_with_both_or_neither():
    recorder = _Recorder()
    both = await _post({"message": "hi", "confirmation": STORED}, recorder)
    neither = await _post({"session_id": "s-1"}, recorder)

    assert (both.status_code, neither.status_code) == (422, 422)
    assert recorder.seen == []


async def test_the_route_refuses_a_confirmation_with_no_token():
    recorder = _Recorder()
    response = await _post(
        {"confirmation": {"tool": "create_document", "arguments": {}, "token": ""}}, recorder
    )

    assert response.status_code == 422
    assert recorder.seen == []


async def test_the_route_caps_the_arguments_it_replays():
    recorder = _Recorder()
    response = await _post(
        {
            "confirmation": {
                "tool": "create_document",
                "arguments": {"k": "x" * 9000},
                "token": TOKEN,
            }
        },
        recorder,
    )

    assert response.status_code == 422
    assert "confirmation.arguments" in response.text
