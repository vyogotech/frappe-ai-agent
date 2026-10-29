"""Model credentials from the usage check (integrations/usage.py) through to the model call.

Named to sort after test_narrowed_excepts: requests through create_app configure structlog with
first-use caching, which leaves later structlog.testing.capture_logs tests with nothing to capture.
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx
from httpx import ASGITransport, AsyncClient

from ai_agent.app import create_app
from ai_agent.config import Settings
from ai_agent.integrations import usage
from ai_agent.integrations.llm import create_llm
from ai_agent.integrations.usage import LLMCredentials
from ai_agent.services import chat
from ai_agent.services.chat import ChatService

FRAPPE = "http://frappe.test"
WHO = f"{FRAPPE}/api/method/frappe.auth.get_logged_user"
GATE = "http://usage.test/check"
SITE = "crm-acme.example.test"
LLM = {
    "base_url": "http://models.test/v1",
    "api_key": "sk-site",
    "user": "user-1",
}


@pytest.fixture(autouse=True)
def _nothing_kept():
    usage._kept.clear()
    yield
    usage._kept.clear()


class _Answers:
    def __init__(self) -> None:
        self.credentials: list[LLMCredentials | None] = []

    async def handle_message(
        self, *, message, session_id, context, user_context, confirmation=None, llm_credentials=None
    ):
        self.credentials.append(llm_credentials)
        yield {"type": "content", "text": "hi"}
        yield {"type": "done", "tools_called": [], "data_quality": "high", "timestamp": "t"}


async def _chat(body: dict | None = None, **settings) -> tuple[httpx.Response, _Answers]:
    app = create_app(Settings(_env_file=None, frappe_url=FRAPPE, usage_check_url=GATE, **settings))  # pyright: ignore[reportCallIssue]
    answers = _Answers()
    app.state.chat_service = answers
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        cookies={"sid": "alice-sid"},
        headers={"X-Frappe-Site-Name": SITE},
    ) as ac:
        response = await ac.post("/api/v1/chat", json=body or {"message": "hello"})
    return response, answers


def _events(response: httpx.Response) -> list[dict]:
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]


def _signed_in():
    respx.get(WHO).mock(return_value=httpx.Response(200, json={"message": "alice@example.test"}))


def _allowed(llm: dict | None = LLM):
    answer = {"allowed": True, **({"llm": llm} if llm else {})}
    return httpx.Response(200, json={"message": answer})


CONFIRM = {"session_id": "s", "confirmation": {"tool": "create_doc", "arguments": {}, "token": "t"}}


@respx.mock
async def test_an_allowed_answers_credentials_reach_the_turn():
    _signed_in()
    respx.post(GATE).mock(return_value=_allowed())
    _, answers = await _chat()
    assert answers.credentials == [LLMCredentials(**LLM)]


@respx.mock
async def test_an_answer_without_credentials_runs_as_before():
    _signed_in()
    respx.post(GATE).mock(return_value=_allowed(None))
    _, answers = await _chat()
    assert answers.credentials == [None]


@respx.mock
async def test_an_outage_uses_the_credentials_kept_from_the_last_answer():
    _signed_in()
    respx.post(GATE).mock(return_value=_allowed())
    await _chat(llm_credentials_required=True)
    respx.post(GATE).mock(side_effect=httpx.ConnectError("down"))
    _, answers = await _chat(llm_credentials_required=True)
    assert answers.credentials == [LLMCredentials(**LLM)]


@respx.mock
async def test_required_credentials_refuse_an_outage_with_nothing_kept():
    _signed_in()
    respx.post(GATE).mock(return_value=httpx.Response(503))
    response, answers = await _chat(llm_credentials_required=True)
    assert answers.credentials == []
    assert _events(response) == [{"type": "error", "message": usage.UNAVAILABLE}]


@respx.mock
async def test_required_credentials_refuse_an_answer_without_them():
    _signed_in()
    respx.post(GATE).mock(return_value=_allowed(None))
    response, answers = await _chat(llm_credentials_required=True)
    assert answers.credentials == []
    assert _events(response)[0]["message"] == usage.UNAVAILABLE


@respx.mock
async def test_kept_credentials_expire():
    _signed_in()
    respx.post(GATE).mock(return_value=_allowed())
    await _chat(usage_credentials_ttl_s=0)
    respx.post(GATE).mock(side_effect=httpx.ConnectError("down"))
    _, answers = await _chat(usage_credentials_ttl_s=0)
    assert answers.credentials == [None]


@respx.mock
async def test_a_confirmed_turn_reuses_the_kept_credentials_without_asking():
    _signed_in()
    gate = respx.post(GATE).mock(return_value=_allowed())
    await _chat()
    calls = gate.call_count
    _, answers = await _chat(CONFIRM)
    assert gate.call_count == calls
    assert answers.credentials == [LLMCredentials(**LLM)]


@respx.mock
async def test_a_confirmed_turn_with_nothing_kept_is_refused_when_credentials_are_required():
    _signed_in()
    respx.post(GATE)
    response, answers = await _chat(CONFIRM, llm_credentials_required=True)
    assert answers.credentials == []
    assert _events(response)[0]["message"] == usage.UNAVAILABLE


def _service() -> ChatService:
    settings = Settings(  # pyright: ignore[reportCallIssue]
        _env_file=None,
        llm_provider="openai",
        llm_base_url="http://agent-default.test/v1",
        llm_api_key="sk-agent",
        llm_model="gpt-4o-mini",
        frappe_url=FRAPPE,
    )
    return ChatService(settings=settings, llm=create_llm(settings))


async def test_a_turn_with_credentials_calls_the_model_with_them():
    service = _service()
    llm = await service._llm_for(None, LLMCredentials(**LLM))
    assert llm.openai_api_base == LLM["base_url"]  # type: ignore[attr-defined]
    assert llm.openai_api_key.get_secret_value() == "sk-site"  # type: ignore[attr-defined]
    assert llm.extra_body == {"user": LLM["user"]}  # type: ignore[attr-defined]
    assert await service._llm_for(None, LLMCredentials(**LLM)) is llm


async def test_each_user_and_key_gets_a_client_and_other_turns_keep_the_agents_key():
    service = _service()
    alice = await service._llm_for(None, LLMCredentials(**LLM))
    bob = await service._llm_for(None, LLMCredentials(**{**LLM, "user": "user-2"}))
    other = await service._llm_for(None, LLMCredentials(**{**LLM, "api_key": "sk-other"}))
    assert len({id(alice), id(bob), id(other)}) == 3
    default = await service._llm_for(None, None)
    assert default.openai_api_key.get_secret_value() == "sk-agent"  # type: ignore[attr-defined]


async def test_the_client_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(chat, "_SUPPLIED_LLMS_MAX", 2)
    service = _service()
    for n in range(4):
        await service._llm_for(None, LLMCredentials(**{**LLM, "user": f"u{n}"}))
    assert len(service._supplied_llms) == 2


class _StatusError(Exception):
    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status_code = status


@pytest.mark.parametrize(
    ("status", "text", "line"),
    [
        (402, "Insufficient credits", chat.MODEL_LIMIT),
        (429, "You exceeded your current quota", chat.MODEL_LIMIT),
        (400, "budget exceeded", chat.MODEL_LIMIT),
        (429, "Rate limit reached", chat.MODEL_RATE),
        (403, "model not allowed", chat.MODEL_REFUSED),
        (401, "invalid api key", chat.MODEL_REFUSED),
        (500, "upstream exploded", None),
        (None, "no status at all", None),
    ],
)
def test_model_refusals_get_a_plain_line(status, text, line):
    exc = _StatusError(status, text) if status is not None else RuntimeError(text)
    assert chat._model_refusal(exc) == line
