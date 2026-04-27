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

    def test_currency_defaults_to_inr(self):
        prompt = build_system_prompt({})
        assert "Default currency: INR" in prompt
        assert "₹" in prompt

    def test_currency_usd(self):
        prompt = build_system_prompt({"currency": "USD"})
        assert "Default currency: USD" in prompt
        assert "$" in prompt

    def test_currency_eur(self):
        prompt = build_system_prompt({"currency": "EUR"})
        assert "Default currency: EUR" in prompt
        assert "€" in prompt

    def test_currency_unknown_falls_back_to_code_with_space(self):
        # Currencies not in the symbol map use the ISO code with a trailing
        # space, so "BHD" appears as "BHD " in prose to keep it unambiguous.
        prompt = build_system_prompt({"currency": "BHD"})
        assert "Default currency: BHD" in prompt
        assert "BHD " in prompt

    def test_currency_lowercase_normalized(self):
        prompt = build_system_prompt({"currency": "inr"})
        assert "Default currency: INR" in prompt
