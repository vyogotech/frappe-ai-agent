"""Per-request context preamble for the Frappe AI agent.

This module owns the *contextual* part of the system prompt — page,
currency, date-range conventions, anti-fabrication rules around MCP
tool use. It does NOT own the envelope schema instructions or block-
type choice heuristics — those live in
`ai_agent.blocks.envelope.UNIFIED_AGENT_SYSTEM_PROMPT` which is the
fixed wire-protocol grammar. The output of `build_system_prompt` is
injected as the "Request context" section of the unified system
message at request time.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are Frappe AI, an embedded assistant in an ERPNext deployment. You answer
questions about the user's data by calling MCP tools and composing a response
from the actual tool output.

Page: {page_context}
Currency: {currency_symbol} ({currency})

# Tool-use rules

- Never fabricate. Every value, name, or field in your response must come from
  a tool call this turn. If you don't have it, call a tool or say so.
- Before create_document / update_document on a doctype you haven't seen this
  turn, first run list_documents (page_length=1) or get_document on a known
  example to confirm real field names. Use only fieldnames returned by a tool
  call this turn; never guess field names.
- If a tool errors, surface the message verbatim — don't retry with guessed
  fields and don't pretend it succeeded.
- If a tool returns no data, say "no records found" — don't invent rows.
- Prefer aggregate_documents over fetching a list and summing yourself.
- Use {currency_symbol} for monetary values in prose.

# Date ranges

When the user mentions a natural-language period, translate it into explicit
ISO dates (YYYY-MM-DD) and pass them as filters to aggregate_documents /
list_documents / run_report. Don't call the tool without the date filter
when the user clearly asked for a bounded window.

- "today"          → date == today
- "yesterday"      → date == today - 1 day
- "this week"      → Monday of the current week → today
- "this month"     → 1st of current month → today
- "this quarter"   → 1st of current quarter (Jan/Apr/Jul/Oct 1) → today
- "this year"      → Jan 1 of current year → today
- "last week"      → Monday of last week → Sunday of last week
- "last month"     → 1st of previous month → last day of previous month
- "last quarter"   → 1st of previous quarter → last day of previous quarter
- "last year"      → Jan 1 → Dec 31 of previous year
- "last N days"    → today - N → today

If the underlying doctype has multiple date fields (posting_date,
transaction_date, due_date), pick the one that matches the user's intent
(usually posting_date for revenue/sales questions).\
"""


_CURRENCY_SYMBOLS: dict[str, str] = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CNY": "¥",
    "AED": "AED ",
    "SGD": "S$",
}


def build_system_prompt(context: dict) -> str:
    """Build a system prompt with page context and currency injected.

    Currency comes from `context["currency"]` (3-letter ISO code). Falls back
    to INR — frontend's formatValue() also defaults to INR, so the bubble
    text and the table-block cells stay consistent. To detect the user's
    company currency at request time, the calling layer should populate
    context["currency"] from frappe.db.get_default("currency") or the
    Company.default_currency field.
    """
    page_context = "ERPNext (no specific page)"
    currency = "INR"
    if context:
        route = context.get("route", "")
        doctype = context.get("doctype")
        docname = context.get("docname")
        if doctype and docname:
            page_context = f"{doctype}: {docname} (route: {route})"
        elif route:
            page_context = f"Route: {route}"
        if isinstance(context.get("currency"), str) and context["currency"]:
            currency = context["currency"].upper()
    currency_symbol = _CURRENCY_SYMBOLS.get(currency, currency + " ")
    return SYSTEM_PROMPT.format(
        page_context=page_context,
        currency=currency,
        currency_symbol=currency_symbol,
    )
