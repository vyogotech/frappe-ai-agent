from ai_agent.agent.output import parse_agent_output
from ai_agent.agent.prompts import build_system_prompt
from ai_agent.agent.state import AgentState
from ai_agent.blocks.models import TableBlock, TextBlock


class TestAgentState:
    def test_state_has_messages_and_context(self):
        state: AgentState = {"messages": [], "context": {}}
        assert state["messages"] == []
        assert state["context"] == {}


class TestBuildSystemPrompt:
    def test_no_context(self):
        prompt = build_system_prompt({})
        assert "Frappe Copilot" in prompt
        assert "no specific page" in prompt

    def test_with_doctype_and_docname(self):
        prompt = build_system_prompt(
            {"route": "Form/Customer/CUST-001", "doctype": "Customer", "docname": "CUST-001"}
        )
        assert "Customer: CUST-001" in prompt

    def test_with_route_only(self):
        prompt = build_system_prompt({"route": "List/Sales Invoice"})
        assert "List/Sales Invoice" in prompt


class TestParseAgentOutput:
    def test_plain_text(self):
        blocks = parse_agent_output("Just a plain answer.")
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert blocks[0].content == "Just a plain answer."

    def test_copilot_block_extraction(self):
        text = (
            "Here are the results:\n"
            '<copilot-block type="table">'
            '{"title": "Test", "columns": [{"key": "name", "label": "Name"}], '
            '"rows": [{"values": {"name": "Acme"}}]}'
            "</copilot-block>\n"
            "That is all."
        )
        blocks = parse_agent_output(text)
        assert len(blocks) == 3  # text before, table, text after
        assert isinstance(blocks[0], TextBlock)
        assert isinstance(blocks[1], TableBlock)
        assert blocks[1].title == "Test"
        assert isinstance(blocks[2], TextBlock)

    def test_malformed_json_fallback(self):
        text = '<copilot-block type="table">{invalid json}</copilot-block>'
        blocks = parse_agent_output(text)
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)
        assert "Could not render" in blocks[0].content
