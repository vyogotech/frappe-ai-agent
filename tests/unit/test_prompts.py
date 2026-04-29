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
        # The prompt was trimmed (11.5KB → 2.2KB) to fit qwen3.5:9b's context
        # after tool calls; currency now appears as "Currency: <symbol> (<code>)".
        assert "(INR)" in prompt
        assert "₹" in prompt

    def test_currency_usd(self):
        prompt = build_system_prompt({"currency": "USD"})
        assert "(USD)" in prompt
        assert "$" in prompt

    def test_currency_eur(self):
        prompt = build_system_prompt({"currency": "EUR"})
        assert "(EUR)" in prompt
        assert "€" in prompt

    def test_currency_unknown_falls_back_to_code_with_space(self):
        # Currencies not in the symbol map use the ISO code with a trailing
        # space, so "BHD" appears as "BHD " in prose to keep it unambiguous.
        prompt = build_system_prompt({"currency": "BHD"})
        assert "(BHD)" in prompt
        assert "BHD " in prompt

    def test_currency_lowercase_normalized(self):
        prompt = build_system_prompt({"currency": "inr"})
        assert "(INR)" in prompt

    # --- Anti-fabrication & tool-catalog coverage ----------------------------

    def test_explicit_no_fabrication_rule(self):
        prompt = build_system_prompt({})
        # Anti-fabrication is rule #1. Trimmed prompt phrases it as
        # "Never fabricate." (capitalised sentence start, not all-caps).
        assert "Never fabricate" in prompt

    def test_discovery_before_mutation_pattern_present(self):
        prompt = build_system_prompt({})
        # The blueprint-first pattern must be in the prompt so the LLM
        # doesn't invent fieldnames on create/update. After the trim the
        # pattern is expressed as a rule rather than a labelled section.
        assert "ff_get_doctype_blueprint" in prompt
        assert "create_document" in prompt
        assert "update_document" in prompt

    def test_advertises_aggregate_over_list_and_sum(self):
        # The trim kept this guidance because LLMs default to fetching lists
        # and summing in-context, which both wastes tokens and hits row caps.
        prompt = build_system_prompt({})
        assert "aggregate_documents" in prompt

    def test_does_not_advertise_hidden_pm_tools(self):
        prompt = build_system_prompt({})
        # These tools were unwired in the feature/mcp-go-sdk merge because
        # they returned hardcoded constants. The LLM must not learn they
        # exist or it will try to call them and the server will refuse.
        for hidden in (
            "calculate_project_metrics",
            "project_risk_assessment",
            "portfolio_dashboard",
        ):
            assert hidden not in prompt, (
                f"hidden tool {hidden!r} leaked into the prompt — agents will "
                f"call it and get 'tool not found'"
            )

    def test_does_not_reference_legacy_get_doctype_meta(self):
        # The previous prompt mentioned `get_doctype_meta` which is not in
        # the current MCP server's tool catalog. Replaced with ff_get_doctype_*.
        prompt = build_system_prompt({})
        assert "get_doctype_meta" not in prompt

    def test_warns_against_pie_bar_top_level_block_types(self):
        # qwen3.5:9b regularly emitted <ai-block type="pie"> instead of
        # <ai-block type="chart"> with chart_type:"pie". The parser now
        # aliases these, but the prompt should still discourage the
        # non-canonical form.
        prompt = build_system_prompt({})
        assert 'type="pie"' in prompt or 'type=\\"pie\\"' in prompt
