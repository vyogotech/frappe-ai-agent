"""Tests for `run_agent_loop` — the unified-schema agent loop."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_agent.agent.loop import _events_from_block, run_agent_loop


class _FakeRegistry:
    """In-test ToolRegistry stand-in. Records calls + returns scripted results."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._results = results or {}

    def schemas(self) -> str:
        return "(no tools)"

    def __len__(self) -> int:  # required by chat.py span attr
        return 0

    def names(self) -> set[str]:
        return set(self._results)

    async def ainvoke(self, name: str, args: dict[str, Any]) -> str:
        self.calls.append((name, args))
        return self._results.get(name, f"result for {name}")


def _fake_llm_with_yields(iterations: list[list[dict[str, Any]]]) -> MagicMock:
    """Build a fake `llm` whose `.with_structured_output().astream()` yields
    one iteration per call, where each iteration is a list of partial-dict
    snapshots ending with the iteration's final envelope.

    Each call to `astream` consumes the next iteration from `iterations`.
    """
    llm = MagicMock()
    iter_queue = list(iterations)

    def _astream(_messages):
        if not iter_queue:
            raise RuntimeError("test: ran out of scripted iterations")
        partials = iter_queue.pop(0)

        async def _gen():
            for p in partials:
                yield p

        return _gen()

    structured = MagicMock()
    structured.astream = _astream
    llm.with_structured_output.return_value = structured
    return llm


async def _drain(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


class TestZeroToolPath:
    @pytest.mark.asyncio
    async def test_text_only_envelope_emits_one_content_event(self):
        llm = _fake_llm_with_yields(
            [[{"blocks": [{"type": "text", "payload": {"content": "hello"}}]}]]
        )
        events = await _drain(
            run_agent_loop(
                llm=llm,
                tool_registry=_FakeRegistry(),
                user_message="hi",
            )
        )
        assert events == [{"type": "content", "text": "hello"}]

    @pytest.mark.asyncio
    async def test_structured_block_emits_content_block_event(self):
        llm = _fake_llm_with_yields(
            [
                [
                    {
                        "blocks": [
                            {
                                "type": "kpi",
                                "payload": {
                                    "metrics": [
                                        {"label": "Revenue", "value": 100, "format": "number"}
                                    ]
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
                tool_registry=_FakeRegistry(),
                user_message="show kpi",
            )
        )
        kinds = [e["type"] for e in events]
        assert kinds == ["content_block"]
        assert events[0]["block"]["type"] == "kpi"


class TestToolCallingPath:
    @pytest.mark.asyncio
    async def test_single_tool_call_then_answer(self):
        # Iteration 1: tool_call. Iteration 2: final text answer.
        llm = _fake_llm_with_yields(
            [
                [
                    {
                        "blocks": [
                            {
                                "type": "tool_call",
                                "payload": {"name": "get_count", "arguments": {"x": 1}},
                            }
                        ]
                    }
                ],
                [{"blocks": [{"type": "text", "payload": {"content": "got it"}}]}],
            ]
        )
        registry = _FakeRegistry(results={"get_count": "42"})
        events = await _drain(
            run_agent_loop(
                llm=llm,
                tool_registry=registry,
                user_message="count please",
            )
        )

        # SSE events: one tool_call from iteration 1, one content from iteration 2.
        kinds = [e["type"] for e in events]
        assert kinds == ["tool_call", "content"]
        assert events[0]["name"] == "get_count"
        assert events[0]["arguments"] == {"x": 1}
        # The registry was actually invoked with the right args.
        assert registry.calls == [("get_count", {"x": 1})]

    @pytest.mark.asyncio
    async def test_multi_tool_chain_emits_events_in_order(self):
        llm = _fake_llm_with_yields(
            [
                [{"blocks": [{"type": "tool_call", "payload": {"name": "a", "arguments": {}}}]}],
                [{"blocks": [{"type": "tool_call", "payload": {"name": "b", "arguments": {}}}]}],
                [{"blocks": [{"type": "text", "payload": {"content": "done"}}]}],
            ]
        )
        events = await _drain(
            run_agent_loop(
                llm=llm,
                tool_registry=_FakeRegistry(),
                user_message="hi",
            )
        )
        kinds = [(e["type"], e.get("name") or e.get("text")) for e in events]
        assert kinds == [("tool_call", "a"), ("tool_call", "b"), ("content", "done")]

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_in_one_iteration(self):
        # Some models emit multiple tool_call blocks in one envelope.
        llm = _fake_llm_with_yields(
            [
                [
                    {
                        "blocks": [
                            {"type": "tool_call", "payload": {"name": "a", "arguments": {}}},
                            {"type": "tool_call", "payload": {"name": "b", "arguments": {}}},
                        ]
                    }
                ],
                [{"blocks": [{"type": "text", "payload": {"content": "ok"}}]}],
            ]
        )
        registry = _FakeRegistry()
        events = await _drain(run_agent_loop(llm=llm, tool_registry=registry, user_message="hi"))
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert [t["name"] for t in tool_events] == ["a", "b"]
        assert [c[0] for c in registry.calls] == ["a", "b"]


class TestRepeatDetection:
    @pytest.mark.asyncio
    async def test_same_tool_same_args_three_times_bails_out(self):
        # 3 identical tool_call iterations → loop bails with text block.
        same_call = {
            "blocks": [{"type": "tool_call", "payload": {"name": "stuck", "arguments": {"q": 1}}}]
        }
        llm = _fake_llm_with_yields([[same_call], [same_call], [same_call]])
        events = await _drain(
            run_agent_loop(llm=llm, tool_registry=_FakeRegistry(), user_message="hi")
        )
        # We see 3 tool_call events then the bail-out text.
        kinds = [e["type"] for e in events]
        assert kinds.count("tool_call") == 3
        assert kinds[-1] == "content"
        assert "not making progress" in events[-1]["text"].lower()

    @pytest.mark.asyncio
    async def test_different_args_does_not_trip_repeat_guard(self):
        llm = _fake_llm_with_yields(
            [
                [
                    {
                        "blocks": [
                            {"type": "tool_call", "payload": {"name": "t", "arguments": {"q": 1}}}
                        ]
                    }
                ],
                [
                    {
                        "blocks": [
                            {"type": "tool_call", "payload": {"name": "t", "arguments": {"q": 2}}}
                        ]
                    }
                ],
                [{"blocks": [{"type": "text", "payload": {"content": "ok"}}]}],
            ]
        )
        events = await _drain(
            run_agent_loop(llm=llm, tool_registry=_FakeRegistry(), user_message="hi")
        )
        assert events[-1] == {"type": "content", "text": "ok"}


class TestMaxStepsCap:
    @pytest.mark.asyncio
    async def test_loop_exhausts_max_steps_emits_text_block(self):
        # 5 iterations, all tool_call, but max_steps=3 → bail.
        # Use DIFFERENT args each step so repeat detection doesn't trip first.
        iters = [
            [
                {
                    "blocks": [
                        {
                            "type": "tool_call",
                            "payload": {"name": "t", "arguments": {"i": i}},
                        }
                    ]
                }
            ]
            for i in range(5)
        ]
        llm = _fake_llm_with_yields(iters)
        events = await _drain(
            run_agent_loop(
                llm=llm,
                tool_registry=_FakeRegistry(),
                user_message="hi",
                max_steps=3,
            )
        )
        # 3 tool_call events then a bail-out text.
        tool_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_events) == 3
        assert events[-1]["type"] == "content"
        assert "couldn't converge" in events[-1]["text"].lower()


class TestLLMError:
    @pytest.mark.asyncio
    async def test_llm_exception_propagates(self):
        def _raising_astream(_messages):
            async def _gen():
                raise RuntimeError("provider down")
                yield  # unreachable

            return _gen()

        llm = MagicMock()
        structured = MagicMock()
        structured.astream = _raising_astream
        llm.with_structured_output.return_value = structured

        with pytest.raises(RuntimeError, match="provider down"):
            await _drain(run_agent_loop(llm=llm, tool_registry=_FakeRegistry(), user_message="hi"))


class TestEventsFromBlock:
    def test_tool_call_dropped(self):
        evs = _events_from_block({"type": "tool_call", "payload": {"name": "x", "arguments": {}}})
        assert evs == []

    def test_text_becomes_content(self):
        evs = _events_from_block({"type": "text", "payload": {"content": "hello"}})
        assert evs == [{"type": "content", "text": "hello"}]

    def test_structured_becomes_content_block(self):
        evs = _events_from_block(
            {
                "type": "kpi",
                "payload": {"metrics": [{"label": "X", "value": 1, "format": "number"}]},
            }
        )
        assert len(evs) == 1
        assert evs[0]["type"] == "content_block"
        assert evs[0]["block"]["type"] == "kpi"

    def test_malformed_dropped_silently(self):
        evs = _events_from_block({"not_a_real": "block"})
        assert evs == []
