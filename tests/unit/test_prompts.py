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

    # --- Anti-fabrication & tool-catalog coverage ----------------------------

    def test_explicit_no_fabrication_rule(self):
        prompt = build_system_prompt({})
        # The number 1 absolute rule is the anti-fabrication directive.
        assert "NEVER fabricate" in prompt

    def test_discovery_before_mutation_pattern_present(self):
        prompt = build_system_prompt({})
        # The blueprint-first pattern for create/update must be in the prompt
        # so the LLM doesn't invent fieldnames.
        assert "ff_get_doctype_blueprint" in prompt
        assert "DISCOVERY-BEFORE-MUTATION" in prompt

    def test_lists_all_27_mcp_tools(self):
        prompt = build_system_prompt({})
        # Every tool the MCP server registers should appear in the catalog
        # so the LLM knows it exists. Hidden PM tool stubs (calculate_*,
        # project_risk_assessment, portfolio_dashboard) MUST NOT appear —
        # they're fabricated and were unwired in the merge.
        expected_tools = [
            # Core CRUD
            "get_document",
            "list_documents",
            "create_document",
            "update_document",
            "delete_document",
            "search_documents",
            # Aggregation & reporting
            "aggregate_documents",
            "run_report",
            # Cross-doctype search
            "global_search",
            # Generic analyzer
            "analyze_document",
            # FrappeForge (11)
            "ff_graph_stats",
            "ff_list_ingested_projects",
            "ff_search_doctype",
            "ff_get_doctype_detail",
            "ff_get_doctype_controllers",
            "ff_get_doctype_client_scripts",
            "ff_find_doctypes_with_field",
            "ff_get_doctype_links",
            "ff_search_methods",
            "ff_get_hooks",
            "ff_get_doctype_blueprint",
            # Project Management (6, post-merge)
            "get_project_status",
            "analyze_project_timeline",
            "get_resource_allocation",
            "generate_project_report",
            "resource_utilization_analysis",
            "budget_variance_analysis",
        ]
        for tool in expected_tools:
            assert tool in prompt, f"tool {tool!r} missing from system prompt"

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
                f"hidden tool {hidden!r} leaked into the prompt — agents will call "
                f"it and get 'tool not found'"
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
