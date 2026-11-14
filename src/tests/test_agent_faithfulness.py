import json

import pytest

from getwork.agent.models import AgentResult, Usage

pytestmark = pytest.mark.usefixtures("offline")


def make_result(ungrounded: list[str]) -> AgentResult:
    return AgentResult(
        status="ok",
        reply="Both documents list CSS and Terraform.",
        facts=[],
        trace=[],
        usage=Usage(llm_calls=1, tool_calls=0, elapsed_ms=3),
        model="m",
        backend="voyager",
        prompt_version="agent_v3",
        ungrounded_skills=ungrounded,
    )


def test_record_carries_ungrounded_skills() -> None:
    record = make_result(["Terraform"]).as_record()
    assert record["ungrounded_skills"] == ["Terraform"]
    assert record["reply"] == "Both documents list CSS and Terraform."
    assert json.loads(json.dumps(record)) == record
    assert make_result([]).as_record()["ungrounded_skills"] == []
