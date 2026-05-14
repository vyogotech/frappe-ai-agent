"""Tests for `ToolRegistry` — the thin wrapper over MCP / LangChain tools."""

from __future__ import annotations

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ai_agent.agent.tool_registry import ToolRegistry


@tool
def _add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class _CustomerArgs(BaseModel):
    doctype: str = Field(description="The doctype to list")
    page_length: int = 20


@tool("list_customers", args_schema=_CustomerArgs)
def _list_customers(doctype: str, page_length: int) -> str:
    """List customers."""
    return f"{doctype}:{page_length}"


@tool
def _explodes(x: int) -> int:
    """A tool that raises."""
    raise RuntimeError(f"boom on {x}")


@tool
def _denies(x: int) -> int:
    """A tool that emits a permission-shaped error."""
    raise PermissionError("permission denied for this user")


class TestNamesAndContains:
    def test_names_returned_as_set(self):
        reg = ToolRegistry([_add, _list_customers])
        assert reg.names() == {"_add", "list_customers"}

    def test_contains_by_name(self):
        reg = ToolRegistry([_add])
        assert "_add" in reg
        assert "missing" not in reg

    def test_len(self):
        reg = ToolRegistry([_add, _list_customers])
        assert len(reg) == 2

    def test_dedup_by_name(self):
        # Two tools with the same name — last-write-wins.
        @tool
        def t() -> str:
            """First."""
            return "first"

        @tool
        def t2() -> str:
            """Second."""
            return "second"

        # Force same name to simulate the duplicate-name case.
        t2.name = "t"
        reg = ToolRegistry([t, t2])
        assert len(reg) == 1


class TestSchemas:
    def test_empty_registry_renders_placeholder(self):
        reg = ToolRegistry([])
        assert reg.schemas() == "(no tools available this turn)"

    def test_renders_name_description_and_args(self):
        reg = ToolRegistry([_list_customers])
        catalog = reg.schemas()
        assert "list_customers" in catalog
        assert "List customers" in catalog
        assert "doctype" in catalog
        assert "page_length" in catalog

    def test_required_args_marked_no_question_mark(self):
        # doctype is required (no default), page_length is optional (has default).
        reg = ToolRegistry([_list_customers])
        catalog = reg.schemas()
        # We render "doctype: string" (required) and "page_length?: integer".
        assert "page_length?" in catalog
        # Required field renders without "?" suffix
        assert "doctype:" in catalog and "doctype?:" not in catalog

    def test_tools_sorted_alphabetically(self):
        reg = ToolRegistry([_list_customers, _add])
        catalog = reg.schemas()
        # _add comes before list_customers alphabetically
        add_idx = catalog.index("_add")
        list_idx = catalog.index("list_customers")
        assert add_idx < list_idx


class TestAinvoke:
    @pytest.mark.asyncio
    async def test_invoke_returns_stringified_result(self):
        reg = ToolRegistry([_add])
        result = await reg.ainvoke("_add", {"a": 2, "b": 3})
        assert result == "5"

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_string(self):
        reg = ToolRegistry([_add])
        result = await reg.ainvoke("nonexistent", {})
        assert "error:" in result
        assert "nonexistent" in result

    @pytest.mark.asyncio
    async def test_tool_exception_becomes_result_string(self):
        reg = ToolRegistry([_explodes])
        result = await reg.ainvoke("_explodes", {"x": 7})
        assert "Tool call failed" in result
        assert "boom on 7" in result

    @pytest.mark.asyncio
    async def test_permission_error_returns_access_denied(self):
        reg = ToolRegistry([_denies])
        result = await reg.ainvoke("_denies", {"x": 1})
        assert "Access denied" in result
        assert "permission error" in result

    @pytest.mark.asyncio
    async def test_none_args_treated_as_empty(self):
        @tool
        def _no_args() -> str:
            """No args needed."""
            return "ok"

        reg = ToolRegistry([_no_args])
        result = await reg.ainvoke("_no_args", None)
        assert result == "ok"
