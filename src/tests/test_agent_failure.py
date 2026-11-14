import re
from typing import get_args

import pytest

from getwork.agent.failure import FAILURE_COPY, AgentFailure, failure_copy
from getwork.agent.models import FailureReason

pytestmark = pytest.mark.usefixtures("offline")

FORBIDDEN = {"error", "exception", "traceback", "http", "api", "token", "key", "llm", "json", "model"}
REASONS = ["timeout", "budget_exhausted", "llm_unavailable", "model_rejected", "invalid_reply"]


def test_failure_copy_covers_every_reason() -> None:
    assert set(FAILURE_COPY) == set(get_args(FailureReason)) == set(REASONS)
    for reason in REASONS:
        message, hint = failure_copy(reason)
        assert message and hint
    assert FAILURE_COPY["timeout"] == (
        "The assistant took too long to answer.",
        "Try again in a moment, or ask a shorter question.",
    )
    assert FAILURE_COPY["llm_unavailable"][1].endswith("to check the assistant's service.")


def test_copy_style_rules() -> None:
    for message, hint in FAILURE_COPY.values():
        assert len(message) <= 100
        for text in (message, hint):
            assert text.endswith(".")
            assert not re.search(r"\d", text)
            assert "://" not in text
            assert not FORBIDDEN & set(re.findall(r"[a-z]+", text.lower()))


def test_agent_failure_exception() -> None:
    for reason in REASONS:
        err = AgentFailure(reason)
        message, hint = FAILURE_COPY[reason]
        assert (err.reason, err.message, err.hint) == (reason, message, hint)
        assert str(err) == message
