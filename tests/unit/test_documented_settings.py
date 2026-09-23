"""The documents name exactly the settings the code defines, in both directions.

pydantic-settings derives each variable from `env_prefix` plus a field name, so no literal
`AI_AGENT_*` string exists in config.py for a grep to find; this compares the real thing.
The changelog is not read here: it may name a key a past release had.
"""

from __future__ import annotations

import re
from pathlib import Path

from ai_agent.config import Settings

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = ("README.md", ".env.example", "docker-compose.yml", "docker-compose.dev.yml.example")
# read by the container's start command, not by the application (ADR-023)
_EXTERNAL = {"AI_AGENT_WORKERS"}


def _documented() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for name in _DOCS:
        for key in re.findall(r"\bAI_AGENT_[A-Z_]+\b", (_ROOT / name).read_text()):
            found.setdefault(key, set()).add(name)
    return found


def _defined() -> set[str]:
    prefix = Settings.model_config.get("env_prefix", "")
    return {f"{prefix}{name.upper()}" for name in Settings.model_fields}


def test_every_setting_is_documented() -> None:
    assert not _defined() - set(_documented()), "settings with no line in README.md"


def test_no_document_names_a_setting_that_does_not_exist() -> None:
    documented = _documented()
    stale = {
        key: sorted(docs) for key, docs in documented.items() if key not in _defined() | _EXTERNAL
    }
    assert not stale, f"named in a document but no such setting: {stale}"
