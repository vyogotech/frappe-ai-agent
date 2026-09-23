"""The leak filter's cost grows with the answer, not with the answer squared."""

from __future__ import annotations

from ai_agent.agent import leak_filter as leak_filter_mod
from ai_agent.agent.leak_filter import StreamingLeakFilter

CHUNK = 8
FILLER = "Here are your open invoices for the month, with the totals by customer. "


def _stream(text: str) -> StreamingLeakFilter:
    leak_filter = StreamingLeakFilter()
    for i in range(0, len(text), CHUNK):
        leak_filter.observe(text[i : i + CHUNK])
    return leak_filter


def test_the_filter_reads_each_answer_a_bounded_number_of_times(monkeypatch):
    """Rescanning the buffer on every delta costs n²/2c; the probe counts either entry point."""
    scanned = {"chars": 0}

    def counting(original):
        def probe(text: str):
            scanned["chars"] += len(text)
            return original(text)

        return probe

    for name in ("detect_system_prompt_leak", "_fingerprint_in"):
        original = getattr(leak_filter_mod, name, None)
        if original is not None:
            monkeypatch.setattr(leak_filter_mod, name, counting(original))
    length = 40_000
    _stream((FILLER * (length // len(FILLER) + 1))[:length])

    # 6.8x the answer today (a 46-character overlap on every 8-character delta). The whole-buffer
    # scan this replaced read 2500x at this length, and twice that at twice the length.
    assert scanned["chars"] < 20 * length


def test_tool_names_far_apart_in_one_answer_still_trip_it():
    """The names are counted over the whole answer, so a window cannot be the only memory."""
    gap = FILLER * 40
    text = f"I can call list_documents{gap}and get_document{gap}and aggregate_documents for you."

    assert _stream(text).triggered is True


def test_two_tool_names_far_apart_do_not():
    gap = FILLER * 40
    text = f"I can call list_documents{gap}and get_document for you."

    assert _stream(text).triggered is False
