from ai_agent.blocks.models import (
    ChartBlock,
    ChartData,
    Column,
    Dataset,
    DocRoute,
    KPIBlock,
    KPIMetric,
    StatusItem,
    StatusListBlock,
    TableBlock,
    TableRow,
    TextBlock,
)
from ai_agent.blocks.parser import parse_blocks
from ai_agent.blocks.validators import validate_block


class TestTextBlock:
    def test_create(self):
        block = TextBlock(content="Hello")
        assert block.type == "text"
        assert block.content == "Hello"


class TestChartBlock:
    def test_create_bar(self):
        block = ChartBlock(
            chart_type="bar",
            title="Revenue",
            data=ChartData(
                labels=["Q1", "Q2"],
                datasets=[Dataset(name="Sales", values=[100, 200])],
            ),
        )
        assert block.type == "chart"
        assert block.chart_type == "bar"
        assert len(block.data.datasets) == 1

    def test_default_options(self):
        block = ChartBlock(
            chart_type="line",
            title="Test",
            data=ChartData(labels=["A"], datasets=[Dataset(name="S", values=[1])]),
        )
        assert block.options.format == "number"
        assert block.options.stacked is False


class TestTableBlock:
    def test_create_with_route(self):
        block = TableBlock(
            title="Customers",
            columns=[Column(key="name", label="Name")],
            rows=[
                TableRow(
                    values={"name": "Acme"},
                    route=DocRoute(doctype="Customer", name="Acme"),
                )
            ],
        )
        assert block.type == "table"
        assert block.rows[0].route.doctype == "Customer"


class TestKPIBlock:
    def test_create(self):
        block = KPIBlock(
            metrics=[
                KPIMetric(label="Revenue", value=50000, format="currency", trend="up"),
            ]
        )
        assert block.type == "kpi"
        assert block.metrics[0].trend == "up"


class TestStatusListBlock:
    def test_create(self):
        block = StatusListBlock(
            title="Orders",
            items=[StatusItem(label="SO-001", status="Completed", color="green")],
        )
        assert block.type == "status_list"
        assert block.items[0].color == "green"


class TestValidateBlock:
    def test_truncate_table_rows(self):
        block = TableBlock(
            title="Big Table",
            columns=[Column(key="id", label="ID")],
            rows=[TableRow(values={"id": i}) for i in range(200)],
        )
        validated = validate_block(block)
        assert len(validated.rows) == 100

    def test_truncate_kpi_metrics(self):
        block = KPIBlock(metrics=[KPIMetric(label=f"M{i}", value=i) for i in range(15)])
        validated = validate_block(block)
        assert len(validated.metrics) == 8

    def test_truncate_status_items(self):
        block = StatusListBlock(
            title="Big List",
            items=[StatusItem(label=f"Item {i}", status="OK", color="green") for i in range(80)],
        )
        validated = validate_block(block)
        assert len(validated.items) == 50

    def test_text_block_passthrough(self):
        block = TextBlock(content="Hello")
        validated = validate_block(block)
        assert validated.content == "Hello"

    def test_truncate_chart_datapoints(self):
        block = ChartBlock(
            chart_type="bar",
            title="Big Chart",
            data=ChartData(
                labels=["A"],
                datasets=[Dataset(name="S", values=list(range(600)))],
            ),
        )
        validated = validate_block(block)
        assert len(validated.data.datasets[0].values) == 500


class TestParseBlocks:
    def test_single_block(self):
        text = '<ai-block type="kpi">{"metrics": [{"label": "Rev", "value": 100}]}</ai-block>'
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "kpi"

    def test_plain_text_no_blocks(self):
        blocks = parse_blocks("Just some text")
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert blocks[0].content == "Just some text"

    def test_empty_string(self):
        blocks = parse_blocks("")
        assert len(blocks) == 0

    def test_unknown_block_type(self):
        text = '<ai-block type="unknown">{"data": 1}</ai-block>'
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert "Unknown block type" in blocks[0].content

    def test_chart_alias_pie_renders_as_chart(self):
        """Smaller models sometimes emit <ai-block type='pie'> instead of
        type='chart' with chart_type='pie'. Treat these top-level chart
        subtypes as aliases."""
        text = (
            '<ai-block type="pie">'
            '{"title": "T", "data": {"labels": ["A","B"], "datasets": [{"name": "S", "values": [1,2]}]}}'
            "</ai-block>"
        )
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "chart"
        assert blocks[0].chart_type == "pie"

    def test_chart_alias_bar_renders_as_chart(self):
        text = (
            '<ai-block type="bar">'
            '{"title": "T", "data": {"labels": ["A"], "datasets": [{"name": "S", "values": [1]}]}}'
            "</ai-block>"
        )
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "chart"
        assert blocks[0].chart_type == "bar"

    def test_chart_alias_respects_explicit_chart_type(self):
        """If the inner JSON does specify chart_type, that wins over the alias."""
        text = (
            '<ai-block type="pie">'
            '{"chart_type": "bar", "title": "T", "data": {"labels": ["A"], "datasets": [{"name": "S", "values": [1]}]}}'
            "</ai-block>"
        )
        blocks = parse_blocks(text)
        assert blocks[0].chart_type == "bar"

    def test_malformed_block_falls_back_to_raw_text(self):
        """Fix 2: malformed JSON should surface the raw content, not an opaque marker."""
        raw_payload = "{not valid json"
        text = f'<ai-block type="kpi">{raw_payload}</ai-block>'
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert "Could not render block" not in blocks[0].content
        assert raw_payload in blocks[0].content

    def test_malformed_schema_falls_back_to_raw_text(self):
        """A valid JSON payload that fails Pydantic validation also surfaces raw content."""
        raw_payload = '{"not_a_real_field": true}'
        text = f'<ai-block type="kpi">{raw_payload}</ai-block>'
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert "Could not render block" not in blocks[0].content
        assert raw_payload in blocks[0].content

    def test_malformed_block_long_content_is_truncated(self):
        """Raw fallback content exceeding 1000 chars is truncated with a marker."""
        raw_payload = "x" * 1200
        text = f'<ai-block type="kpi">{raw_payload}</ai-block>'
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "text"
        assert len(blocks[0].content) < 1200
        assert "truncated" in blocks[0].content

    def test_text_with_blocks(self):
        text = 'Before <ai-block type="text">{"content": "inside"}</ai-block> After'
        blocks = parse_blocks(text)
        assert len(blocks) == 3
        assert blocks[0].content == "Before"
        assert blocks[1].content == "inside"
        assert blocks[2].content == "After"

    def test_chart_truncation_in_parser(self):
        import json

        values = list(range(600))
        payload = json.dumps(
            {
                "chart_type": "bar",
                "title": "T",
                "data": {
                    "labels": ["A"],
                    "datasets": [{"name": "S", "values": values}],
                },
            }
        )
        text = f'<ai-block type="chart">{payload}</ai-block>'
        blocks = parse_blocks(text)
        assert len(blocks) == 1
        assert blocks[0].type == "chart"
        assert len(blocks[0].data.datasets[0].values) == 500
