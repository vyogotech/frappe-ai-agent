"""An answer the model gave as blocks (a table, a chart) stays in the history it reads next turn."""

import json

import respx
from httpx import Response

from ai_agent.integrations.frappe_history import FrappeHistoryClient

FRAPPE = "http://frappe.test"


@respx.mock
async def test_a_block_only_answer_and_a_table_reach_the_next_turn():
    table = {"type": "table", "columns": ["name"], "rows": [["ACME"], ["Globex"]]}
    rows = [  # newest first, as Frappe returns them
        {"role": "user", "content": "sort that by name", "tool_result_json": None},
        {"role": "assistant", "content": "", "tool_result_json": json.dumps({"blocks": [table]})},
        {"role": "user", "content": "list my customers", "tool_result_json": None},
    ]
    respx.get(f"{FRAPPE}/api/method/frappe.client.get_list").mock(
        return_value=Response(200, json={"message": rows})
    )
    history = await FrappeHistoryClient(base_url=FRAPPE).list_messages(sid="s", session="c")
    assert [h["role"] for h in history] == ["user", "assistant", "user"]
    assert "Globex" in history[1]["content"]


@respx.mock
async def test_a_failed_turns_error_text_is_not_replayed():
    rows = [  # newest first
        {"role": "user", "content": "try again", "tool_result_json": None},
        {
            "role": "assistant",
            "content": "[error] ConnectError: mcp.internal refused",
            "tool_result_json": None,
        },
        {"role": "user", "content": "list my customers", "tool_result_json": None},
    ]
    respx.get(f"{FRAPPE}/api/method/frappe.client.get_list").mock(
        return_value=Response(200, json={"message": rows})
    )
    history = await FrappeHistoryClient(base_url=FRAPPE).list_messages(sid="s", session="c")
    assert [h["content"] for h in history] == ["list my customers", "try again"]
