from ai_agent.agent.block_parser import parse_blocks


def test_empty_returns_empty():
    assert parse_blocks("") == []
    assert parse_blocks("   \n  ") == []


def test_plain_text_only_returns_single_text_block():
    blocks = parse_blocks("Hello world!")
    assert blocks == [{"type": "text", "content": "Hello world!"}]


def test_single_table_block():
    text = (
        '<copilot-block type="table">'
        '{"title": "Users", "columns": [{"key": "name", "label": "Name"}], '
        '"rows": [{"values": {"name": "Admin"}}]}'
        "</copilot-block>"
    )
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0] == {
        "type": "table",
        "title": "Users",
        "columns": [{"key": "name", "label": "Name"}],
        "rows": [{"values": {"name": "Admin"}}],
    }


def test_text_then_block_then_text_preserves_order():
    text = (
        "Here are the users:\n"
        '<copilot-block type="table">{"columns": [], "rows": []}</copilot-block>\n'
        "That's 0 users."
    )
    blocks = parse_blocks(text)
    assert len(blocks) == 3
    assert blocks[0] == {"type": "text", "content": "Here are the users:"}
    assert blocks[1] == {"type": "table", "columns": [], "rows": []}
    assert blocks[2] == {"type": "text", "content": "That's 0 users."}


def test_multiple_blocks_interleaved_with_text():
    text = (
        "Summary:\n"
        '<copilot-block type="kpi">{"metrics": [{"label": "Total", "value": 42}]}</copilot-block>\n'
        "Breakdown:\n"
        '<copilot-block type="chart">{"chart_type": "bar", "data": {"labels": [], "datasets": []}}</copilot-block>\n'
        "Full details in the table below:\n"
        '<copilot-block type="table">{"columns": [], "rows": []}</copilot-block>'
    )
    blocks = parse_blocks(text)
    assert [b["type"] for b in blocks] == ["text", "kpi", "text", "chart", "text", "table"]
    assert blocks[0]["content"] == "Summary:"
    assert blocks[1]["metrics"] == [{"label": "Total", "value": 42}]
    assert blocks[2]["content"] == "Breakdown:"
    assert blocks[3]["chart_type"] == "bar"
    assert blocks[4]["content"] == "Full details in the table below:"


def test_malformed_json_falls_back_to_text_block():
    text = '<copilot-block type="table">not-json{{{</copilot-block>'
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0] == {"type": "text", "content": "not-json{{{"}


def test_block_payload_is_list_falls_back_to_text():
    text = '<copilot-block type="table">[1, 2, 3]</copilot-block>'
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0] == {"type": "text", "content": "[1, 2, 3]"}


def test_multiline_block_payload():
    text = (
        '<copilot-block type="kpi">\n'
        '{\n'
        '  "metrics": [\n'
        '    {"label": "Revenue", "value": 1000}\n'
        '  ]\n'
        '}\n'
        "</copilot-block>"
    )
    blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "kpi"
    assert blocks[0]["metrics"] == [{"label": "Revenue", "value": 1000}]


def test_adjacent_blocks_no_text_between():
    text = (
        '<copilot-block type="kpi">{"metrics": []}</copilot-block>'
        '<copilot-block type="chart">{"chart_type": "bar", "data": {"labels": [], "datasets": []}}</copilot-block>'
    )
    blocks = parse_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]["type"] == "kpi"
    assert blocks[1]["type"] == "chart"
