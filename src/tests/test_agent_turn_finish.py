import json

import pytest
from agent_helpers import answer, make_runner, run

pytestmark = pytest.mark.usefixtures("offline")


def test_off_topic_skips_verification() -> None:
    runner, fake = make_runner(['{"action": "answer", "off_topic": true}'])
    result = run(runner)
    assert result.status == "ok" and result.claims == [] and result.off_topic
    assert result.usage.llm_calls == 1 and len(fake.calls) == 1
    assert result.usage.verify_retries == 0 and result.failure_reason is None


def test_direct_answer() -> None:
    runner, fake = make_runner([answer("Both list CSS.")])
    result = run(runner)
    assert result.status == "ok" and not result.off_topic
    assert [c.text for c in result.claims] == ["Both list CSS."]
    assert result.usage.llm_calls == 1 and result.usage.verify_retries == 0
    assert [f.id for f in result.facts] == ["f1", "f2", "f3", "f4"]
    assert (result.model, result.backend, result.prompt_version) == ("demo-model", "voyager", "agent_v1")
    assert json.dumps(result.as_record())


def test_verification_retry_then_success() -> None:
    bad = answer("Both list CSS and Kubernetes.", skills=["CSS", "Kubernetes"])
    runner, fake = make_runner([bad, answer("Both list CSS.")])
    result = run(runner)
    assert result.status == "ok" and result.usage.verify_retries == 1 and result.usage.llm_calls == 2
    second = fake.calls[1].user
    assert "FEEDBACK\nYour last answer was rejected:" in second
    assert 'claim 1: skill "Kubernetes" is not in the skills of query, j-1.' in second
    assert "FEEDBACK" not in fake.calls[0].user


def test_feedback_cuts_violations_to_eight() -> None:
    many = answer("Both list CSS.", skills=[], facts=[], doc_ids=[])
    claims = json.loads(many)["claims"] * 8
    eight = json.dumps({"action": "answer", "claims": claims})
    runner, fake = make_runner([eight, answer()], max_verify_retries=1)
    run(runner)
    feedback = fake.calls[1].user.split("FEEDBACK\n")[1].split("\n\nQUESTION")[0].splitlines()
    assert feedback[0] == "Your last answer was rejected:"
    assert len(feedback) == 1 + 8


def test_unverified_after_two_retries() -> None:
    bad = answer("Both list CSS and Kubernetes.", skills=["CSS", "Kubernetes"])
    runner, fake = make_runner([bad] * 3)
    result = run(runner)
    assert result.status == "failed" and result.failure_reason == "unverified"
    assert result.claims == [] and len(fake.calls) == 3
    assert result.usage.verify_retries == 2 and result.usage.llm_calls == 3
