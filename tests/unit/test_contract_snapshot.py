import json
from pathlib import Path

from pydantic import TypeAdapter

from ai_agent.transport.sse_events import SSEEvent

SNAPSHOT = Path(__file__).parent / "snapshots" / "sse_event.schema.json"


def test_the_streaming_envelope_matches_its_snapshot():
    schema = TypeAdapter(SSEEvent).json_schema()
    assert schema == json.loads(SNAPSHOT.read_text()), (
        f"the envelope changed: regenerate {SNAPSHOT.name} and update frappe_ai and Metis with it"
    )
