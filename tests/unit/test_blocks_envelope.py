"""Tests for the JSON-envelope translator used with constrained-decoding LLMs."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ai_agent.blocks.envelope import (
    BLOCK_ENVELOPE_SCHEMA,
    ENVELOPE_FORMATTER_SYSTEM_PROMPT,
    build_formatter_messages,
    envelope_to_markup,
    iter_complete_blocks,
)
from ai_agent.blocks.parser import parse_blocks


def _envelope(blocks: list[dict]) -> str:
    return json.dumps({"blocks": blocks})


class TestEnvelopeToMarkup:
    def test_table_payload_wraps_as_ai_block(self):
        raw = _envelope(
            [
                {
                    "type": "table",
                    "payload": {
                        "title": "Customers",
                        "columns": [
                            {"key": "name", "label": "Name", "format": "text"},
                            {"key": "amount", "label": "Amount", "format": "currency"},
                        ],
                        "rows": [{"values": {"name": "Acme", "amount": 50000}}],
                    },
                }
            ]
        )
        markup = envelope_to_markup(raw)
        assert markup.startswith('<ai-block type="table">')
        assert markup.endswith("</ai-block>")
        assert '"title": "Customers"' in markup

    def test_text_payload_emits_as_bare_prose(self):
        raw = _envelope([{"type": "text", "payload": {"content": "Hello there"}}])
        assert envelope_to_markup(raw) == "Hello there"

    def test_multi_block_joined_with_blank_lines(self):
        raw = _envelope(
            [
                {"type": "text", "payload": {"content": "Q1 was strong."}},
                {
                    "type": "kpi",
                    "payload": {
                        "metrics": [{"label": "Revenue", "value": 145000, "format": "currency"}]
                    },
                },
            ]
        )
        markup = envelope_to_markup(raw)
        parts = markup.split("\n\n")
        assert parts[0] == "Q1 was strong."
        assert parts[1].startswith('<ai-block type="kpi">')

    def test_strips_json_fence(self):
        envelope = _envelope([{"type": "text", "payload": {"content": "fenced response"}}])
        fenced = f"```json\n{envelope}\n```"
        assert envelope_to_markup(fenced) == "fenced response"

    def test_extracts_object_from_surrounding_prose(self):
        # Gemma 3 4B is documented to add commentary around the JSON object even
        # under constrained decoding. The translator must still find the
        # envelope.
        envelope = _envelope([{"type": "text", "payload": {"content": "got it"}}])
        chatty = f"Sure, here's the response:\n{envelope}\nLet me know if you need more."
        assert envelope_to_markup(chatty) == "got it"

    def test_unknown_block_type_silently_dropped(self):
        # Better to drop than to emit a malformed <ai-block> the parser would
        # warn about. The schema's `oneOf` should prevent this at the source,
        # but be defensive in the translator.
        raw = json.dumps(
            {
                "blocks": [
                    {"type": "garbage", "payload": {"x": 1}},
                    {"type": "text", "payload": {"content": "ok"}},
                ]
            }
        )
        assert envelope_to_markup(raw) == "ok"

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            envelope_to_markup("")
        with pytest.raises(ValueError):
            envelope_to_markup("   \n  ")

    def test_no_json_object_raises(self):
        with pytest.raises(json.JSONDecodeError):
            envelope_to_markup("hello world")

    def test_missing_blocks_key_raises(self):
        with pytest.raises(ValueError):
            envelope_to_markup(json.dumps({"messages": []}))

    def test_chart_payload_round_trips_through_parser(self):
        # End-to-end: envelope JSON → markup → parse_blocks → ChartBlock pydantic.
        raw = _envelope(
            [
                {
                    "type": "chart",
                    "payload": {
                        "chart_type": "bar",
                        "title": "Monthly",
                        "data": {
                            "labels": ["Jan", "Feb"],
                            "datasets": [{"name": "Revenue", "values": [100, 200]}],
                        },
                        "options": {"format": "currency"},
                    },
                }
            ]
        )
        blocks = parse_blocks(envelope_to_markup(raw))
        assert len(blocks) == 1
        assert blocks[0].type == "chart"
        assert blocks[0].chart_type == "bar"
        assert blocks[0].options.format == "currency"

    def test_status_list_round_trips(self):
        raw = _envelope(
            [
                {
                    "type": "status_list",
                    "payload": {
                        "title": "Orders",
                        "items": [{"label": "SO-001", "status": "Paid", "color": "green"}],
                    },
                }
            ]
        )
        blocks = parse_blocks(envelope_to_markup(raw))
        assert len(blocks) == 1
        assert blocks[0].type == "status_list"
        assert blocks[0].items[0].color == "green"


class TestBlockEnvelopeSchema:
    def test_top_level_required_blocks(self):
        assert BLOCK_ENVELOPE_SCHEMA["required"] == ["blocks"]

    def test_blocks_capped(self):
        # 6-block ceiling prevents runaway generation on small models that
        # otherwise emit 100+ block envelopes when given an under-specified
        # prompt.
        assert BLOCK_ENVELOPE_SCHEMA["properties"]["blocks"]["maxItems"] == 6

    def test_one_of_covers_all_block_types(self):
        consts = {
            entry["properties"]["type"]["const"]
            for entry in BLOCK_ENVELOPE_SCHEMA["properties"]["blocks"]["items"]["oneOf"]
        }
        assert consts == {"text", "table", "chart", "kpi", "status_list"}


class TestBuildFormatterMessages:
    def test_returns_system_then_human(self):
        msgs = build_formatter_messages(
            user_message="how much did we make?",
            tool_results=[],
            draft="",
        )
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)

    def test_system_prompt_is_default(self):
        msgs = build_formatter_messages(user_message="anything", tool_results=[], draft="")
        assert msgs[0].content == ENVELOPE_FORMATTER_SYSTEM_PROMPT

    def test_system_prompt_overridable(self):
        msgs = build_formatter_messages(
            user_message="anything",
            tool_results=[],
            draft="",
            system_prompt="CUSTOM",
        )
        assert msgs[0].content == "CUSTOM"

    def test_user_message_includes_question_and_no_tools_marker(self):
        msgs = build_formatter_messages(
            user_message="give me totals",
            tool_results=[],
            draft="",
        )
        body = msgs[1].content
        assert isinstance(body, str)
        assert "USER ASKED:" in body
        assert "give me totals" in body
        assert "(no tools called)" in body
        assert "(empty)" in body

    def test_tool_results_serialised_into_body(self):
        msgs = build_formatter_messages(
            user_message="ok",
            tool_results=[
                {"name": "list_documents", "args": {"doctype": "Customer"}, "result": "[ok]"},
                {"name": "aggregate", "args": {}, "result": "{'total': 5}"},
            ],
            draft="agent says hi",
        )
        body = msgs[1].content
        assert isinstance(body, str)
        assert "list_documents" in body
        assert '"doctype": "Customer"' in body
        assert "aggregate" in body
        assert "agent says hi" in body

    def test_long_tool_result_is_truncated(self):
        long_result = "x" * 5000
        msgs = build_formatter_messages(
            user_message="ok",
            tool_results=[{"name": "t", "args": {}, "result": long_result}],
            draft="",
        )
        body = msgs[1].content
        assert isinstance(body, str)
        assert "…(truncated)" in body
        # Body should be substantially shorter than original 5000 chars
        # of result plus framing.
        assert len(body) < 4500


class TestIterCompleteBlocks:
    def test_emits_only_blocks_followed_by_a_next_block(self):
        # 3-block partial; only the first two are "definitely closed"
        # because the third is the latest and may still be growing.
        partial = {
            "blocks": [
                {"type": "text", "payload": {"content": "intro"}},
                {"type": "kpi", "payload": {"metrics": [{"label": "rev", "value": 100}]}},
                {"type": "table", "payload": {"title": "Tab", "columns": [], "rows": []}},
            ]
        }
        state: dict = {}
        emitted = list(iter_complete_blocks(partial, state, final=False))
        assert [b["type"] for b in emitted] == ["text", "kpi"]
        # 3rd block waits.
        emitted2 = list(iter_complete_blocks(partial, state, final=False))
        assert emitted2 == []

    def test_final_flag_flushes_last_block(self):
        partial = {
            "blocks": [
                {"type": "text", "payload": {"content": "hi"}},
                {"type": "kpi", "payload": {"metrics": [{"label": "x", "value": 1}]}},
            ]
        }
        state: dict = {}
        # First call with final=False emits only the first.
        first = list(iter_complete_blocks(partial, state, final=False))
        assert [b["type"] for b in first] == ["text"]
        # Then final=True flushes the second.
        second = list(iter_complete_blocks(partial, state, final=True))
        assert [b["type"] for b in second] == ["kpi"]

    def test_idempotent_across_redundant_yields(self):
        partial1 = {"blocks": [{"type": "text", "payload": {"content": "hi"}}]}
        partial2 = {
            "blocks": [
                {"type": "text", "payload": {"content": "hi"}},
                {"type": "kpi", "payload": {"metrics": [{"label": "x", "value": 1}]}},
            ]
        }
        state: dict = {}
        # First partial: 1 block, no next started, no final — emits nothing.
        assert list(iter_complete_blocks(partial1, state, final=False)) == []
        # Second partial: 2 blocks, first now has a next — emit first only.
        out = list(iter_complete_blocks(partial2, state, final=False))
        assert [b["type"] for b in out] == ["text"]
        # Third call with same partial2 — nothing new.
        assert list(iter_complete_blocks(partial2, state, final=False)) == []

    def test_skips_malformed_block_dicts(self):
        partial = {
            "blocks": [
                {"type": "text", "payload": {"content": "ok"}},
                {"invalid": "shape"},
                {"type": "kpi", "payload": {"metrics": [{"label": "x", "value": 1}]}},
            ]
        }
        state: dict = {}
        # final=True flushes everything.
        out = list(iter_complete_blocks(partial, state, final=True))
        assert [b["type"] for b in out] == ["text", "kpi"]

    def test_none_or_non_dict_input_yields_nothing(self):
        state: dict = {}
        assert list(iter_complete_blocks(None, state, final=True)) == []
        assert list(iter_complete_blocks({}, state, final=True)) == []
        assert list(iter_complete_blocks({"blocks": "not-a-list"}, state, final=True)) == []
