import json

import pytest
from agent_helpers import answer, make_runner, run

pytestmark = pytest.mark.usefixtures("offline")


def test_direct_answer() -> None:
    runner, fake = make_runner([answer("Both list CSS.")])
    result = run(runner)
    assert result.status == "ok" and result.reply == "Both list CSS."
    assert result.usage.llm_calls == 1 and len(fake.calls) == 1
    assert result.failure_reason is None and result.ungrounded_skills == []
    assert [f.id for f in result.facts] == ["f1", "f2", "f3", "f4"]
    assert (result.model, result.backend, result.prompt_version) == ("demo-model", "voyager", "agent_v3")
    assert json.dumps(result.as_record())


def test_answer_is_never_checked_or_retried() -> None:
    runner, fake = make_runner([answer("Both list CSS and Kubernetes.")])
    result = run(runner)
    assert result.status == "ok" and result.reply == "Both list CSS and Kubernetes."
    assert len(fake.calls) == 1 and "FEEDBACK" not in fake.calls[0].user
    assert result.ungrounded_skills == ["Kubernetes"]


def test_multiline_answer_keeps_line_breaks() -> None:
    text = "You could add:\n- Kubernetes\n- Spring"
    result = run(make_runner([text])[0])
    assert result.reply == text
