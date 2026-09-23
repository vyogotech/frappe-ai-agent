import json
from pathlib import Path

from ai_agent.transport.sse_events import contract_schema

CONTRACT = Path(__file__).parents[2] / "contract" / "sse-event.schema.json"


def test_the_streaming_envelope_matches_its_published_contract():
    assert contract_schema() == json.loads(CONTRACT.read_text()), (
        "the envelope changed: run `make contract`, then update frappe_ai's and Metis's own copies"
    )
