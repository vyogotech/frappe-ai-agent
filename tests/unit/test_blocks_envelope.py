"""Tests for the JSON-envelope translator used with constrained-decoding LLMs."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ai_agent.blocks.envelope import (
    BLOCK_ENVELOPE_SCHEMA,
    TOOL_CALL_TYPE,
    UNIFIED_AGENT_SYSTEM_PROMPT,
    block_envelope_schema,
    build_agent_messages,
    envelope_to_markup,
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

    def test_one_of_covers_all_block_types_including_tool_call(self):
        consts = {
            entry["properties"]["type"]["const"]
            for entry in BLOCK_ENVELOPE_SCHEMA["properties"]["blocks"]["items"]["oneOf"]
        }
        assert consts == {"text", "table", "chart", "kpi", "status_list", "tool_call"}

    def test_tool_call_type_constant(self):
        assert TOOL_CALL_TYPE == "tool_call"


class TestToolCallBlock:
    def test_envelope_to_markup_drops_tool_call_blocks(self):
        # tool_call is agent-loop machinery — must never become FE markup.
        raw = json.dumps(
            {
                "blocks": [
                    {"type": "tool_call", "payload": {"name": "x", "arguments": {}}},
                    {"type": "text", "payload": {"content": "visible answer"}},
                ]
            }
        )
        assert envelope_to_markup(raw) == "visible answer"

    def test_tool_call_payload_requires_name_and_arguments(self):
        tool_call_branch = next(
            entry
            for entry in BLOCK_ENVELOPE_SCHEMA["properties"]["blocks"]["items"]["oneOf"]
            if entry["properties"]["type"]["const"] == "tool_call"
        )
        payload_schema = tool_call_branch["properties"]["payload"]
        assert set(payload_schema["required"]) == {"name", "arguments"}
        assert payload_schema["additionalProperties"] is False


class TestBlockEnvelopeSchemaBuilder:
    """`block_envelope_schema` pins tool_call.name to the live tool set.

    Motivation: the base schema types `name` as a bare string, so the
    constrained-decoding grammar accepts `""`. granite4.2:8b emits
    `{"name": "", "arguments": {"doctype": "Role"}}` — schema-valid,
    unexecutable. The enum makes that unreachable at the token level.
    """

    NAMES: ClassVar[set[str]] = {"list_documents", "aggregate_documents", "get_document"}

    @staticmethod
    def _tool_call_branch(schema: dict) -> dict:
        return next(
            entry
            for entry in schema["properties"]["blocks"]["items"]["oneOf"]
            if entry["properties"]["type"]["const"] == "tool_call"
        )

    def test_pins_tool_call_name_to_enum(self):
        name_schema = self._tool_call_branch(block_envelope_schema(self.NAMES))["properties"][
            "payload"
        ]["properties"]["name"]
        assert name_schema == {"type": "string", "enum": sorted(self.NAMES)}

    def test_empty_name_is_not_in_the_enum(self):
        # The whole point: "" must be ungeneratable.
        name_schema = self._tool_call_branch(block_envelope_schema(self.NAMES))["properties"][
            "payload"
        ]["properties"]["name"]
        assert "" not in name_schema["enum"]

    def test_enum_is_sorted_regardless_of_set_iteration_order(self):
        # `names()` returns a set; PYTHONHASHSEED must not change the schema
        # (an unstable schema defeats any provider-side prompt/grammar cache).
        a = block_envelope_schema({"b_tool", "a_tool", "c_tool"})
        b = block_envelope_schema({"c_tool", "b_tool", "a_tool"})
        assert a == b
        assert self._tool_call_branch(a)["properties"]["payload"]["properties"]["name"]["enum"] == [
            "a_tool",
            "b_tool",
            "c_tool",
        ]

    def test_no_tools_returns_base_schema_unconstrained(self):
        # MCP-down soft-degrade (services.chat) hands us an empty registry.
        # An `enum: []` compiles to a grammar with no legal value and Ollama
        # then emits invalid JSON (`{"name": }`), breaking every degraded turn.
        for empty in (set(), None):
            schema = block_envelope_schema(empty)
            assert schema is BLOCK_ENVELOPE_SCHEMA
            assert self._tool_call_branch(schema)["properties"]["payload"]["properties"][
                "name"
            ] == {"type": "string"}

    def test_does_not_mutate_the_module_constant(self):
        before = json.dumps(BLOCK_ENVELOPE_SCHEMA, sort_keys=True)
        block_envelope_schema(self.NAMES)
        assert json.dumps(BLOCK_ENVELOPE_SCHEMA, sort_keys=True) == before

    def test_leaves_every_other_block_branch_untouched(self):
        patched = block_envelope_schema(self.NAMES)
        for btype in ("text", "table", "chart", "kpi", "status_list"):
            got = next(
                e
                for e in patched["properties"]["blocks"]["items"]["oneOf"]
                if e["properties"]["type"]["const"] == btype
            )
            want = next(
                e
                for e in BLOCK_ENVELOPE_SCHEMA["properties"]["blocks"]["items"]["oneOf"]
                if e["properties"]["type"]["const"] == btype
            )
            assert got == want

    def test_tool_call_branch_still_requires_name_and_arguments(self):
        payload = self._tool_call_branch(block_envelope_schema(self.NAMES))["properties"]["payload"]
        assert set(payload["required"]) == {"name", "arguments"}
        assert payload["additionalProperties"] is False


class TestBuildAgentMessages:
    def test_returns_system_then_human(self):
        msgs = build_agent_messages(user_message="how much did we make?")
        assert len(msgs) == 2
        assert isinstance(msgs[0], SystemMessage)
        assert isinstance(msgs[1], HumanMessage)
        assert msgs[1].content == "how much did we make?"

    def test_system_prompt_includes_unified_agent_text(self):
        msgs = build_agent_messages(user_message="x")
        assert UNIFIED_AGENT_SYSTEM_PROMPT in msgs[0].content

    def test_context_preamble_injected(self):
        msgs = build_agent_messages(
            user_message="x", context_preamble="Page: Dashboard\nCurrency: $ (USD)"
        )
        assert "Page: Dashboard" in msgs[0].content
        assert "Currency: $ (USD)" in msgs[0].content

    def test_tools_catalog_injected(self):
        msgs = build_agent_messages(
            user_message="x",
            tools_catalog="- list_documents: ...\n  args: {doctype: string}",
        )
        assert "list_documents" in msgs[0].content
        assert "Tools available this turn" in msgs[0].content

    def test_no_catalog_no_preamble_means_no_extra_sections(self):
        msgs = build_agent_messages(user_message="x")
        content = msgs[0].content
        assert "Tools available this turn" not in content
        assert "Request context" not in content

    def test_history_messages_inserted_between_system_and_user(self):
        from langchain_core.messages import AIMessage

        prior = [HumanMessage(content="prev question"), AIMessage(content="prev answer")]
        msgs = build_agent_messages(user_message="follow up", history=prior)
        assert isinstance(msgs[0], SystemMessage)
        assert msgs[1] is prior[0]
        assert msgs[2] is prior[1]
        assert msgs[3].content == "follow up"
