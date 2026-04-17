from ai_agent.agent.prompts import build_system_prompt


class TestBuildSystemPrompt:
    def test_no_context(self):
        prompt = build_system_prompt({})
        assert "Frappe AI" in prompt
        assert "no specific page" in prompt

    def test_with_doctype_and_docname(self):
        prompt = build_system_prompt(
            {"route": "Form/Customer/CUST-001", "doctype": "Customer", "docname": "CUST-001"}
        )
        assert "Customer: CUST-001" in prompt

    def test_with_route_only(self):
        prompt = build_system_prompt({"route": "List/Sales Invoice"})
        assert "List/Sales Invoice" in prompt
