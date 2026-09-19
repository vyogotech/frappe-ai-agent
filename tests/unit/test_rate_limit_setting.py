"""A rate limit the limiter cannot read stops the agent at startup, or slowapi stops limiting."""

import pytest
from pydantic import ValidationError

from ai_agent.config import Settings


def test_a_malformed_rate_limit_is_refused():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, agent_rate_limit="thirty a minute")  # pyright: ignore[reportCallIssue]


def test_the_documented_forms_are_accepted():
    for limit in ("30/minute", "30 per minute", "5/second;100/hour"):
        assert Settings(_env_file=None, agent_rate_limit=limit).agent_rate_limit == limit  # pyright: ignore[reportCallIssue]
