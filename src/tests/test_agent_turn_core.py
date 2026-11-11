import json

import pytest
from agent_helpers import SECRET, answer, make_ctx, make_runner, run

from getwork.agent import models, turn
from getwork.llm.client import FatalLLMError, PermanentLLMError, TransientLLMError

pytestmark = pytest.mark.usefixtures("offline")

def assert_failed(result: models.AgentResult, reason: str, calls: int) -> None:
    assert result.status == "failed" and result.failure_reason == reason
    assert result.claims == [] and result.usage.llm_calls == calls
    assert (result.model, result.backend, result.prompt_version) == ("demo-model", "voyager", "agent_v1")


def test_failure_reasons() -> None:
    runner, fake = make_runner([FatalLLMError("bad key sk-secret-123", status=401)])
    assert_failed(run(runner), "model_rejected", len(fake.calls))
    flaky = [TransientLLMError("down", status=503)] * 2
    runner, fake = make_runner(flaky)
    assert_failed(run(runner), "llm_unavailable", len(fake.calls))
    runner, fake = make_runner([PermanentLLMError("bad request", status=400)])
    assert_failed(run(runner), "llm_unavailable", len(fake.calls))
    runner, fake = make_runner(["nope"] * 3)
    assert_failed(run(runner), "invalid_reply", len(fake.calls))
    assert len(fake.calls) == 3


def test_attempts_capped_by_call_budget() -> None:
    runner, fake = make_runner(["nope"] * 5, max_llm_calls=2)
    assert_failed(run(runner), "invalid_reply", 2)
    assert len(fake.calls) == 2


def test_call_budget_checked_before_each_call() -> None:
    runner, fake = make_runner([])
    started = runner.new_turn(ctx=make_ctx(), question="Q", history=[])
    started.stats.calls = runner.cfg.max_llm_calls
    assert runner.decide(started).route == "fail"
    result = turn.drive(runner, started)
    assert_failed(result, "budget_exhausted", runner.cfg.max_llm_calls)
    assert fake.calls == []


def test_timeout_uses_injected_clock() -> None:
    ticks = iter([0.0, 61.0, 61.0, 61.0, 61.0])
    runner, fake = make_runner([answer()], clock=lambda: next(ticks))
    result = run(runner)
    assert_failed(result, "timeout", 0)
    assert fake.calls == []


def test_failed_result_has_no_claims() -> None:
    runner, _ = make_runner(["I think the answer is: Kubernetes rocks"] * 3)
    result = run(runner)
    record = result.as_record()
    assert record["claims"] == [] and record["status"] == "failed"
    assert "Kubernetes rocks" not in json.dumps(record)
    assert [f["id"] for f in record["facts"]] == ["f1", "f2", "f3", "f4"]


def test_secrets_not_in_result() -> None:
    runner, _ = make_runner([FatalLLMError(f"rejected {SECRET}", status=401)])
    assert SECRET not in json.dumps(run(runner).as_record())
    runner, _ = make_runner([f"{SECRET} not json"] * 3)
    assert SECRET not in json.dumps(run(runner).as_record())


def test_new_turn_defaults() -> None:
    runner, _ = make_runner([])
    assert runner.new_turn(ctx=make_ctx(), question="Q", history=[]).max_tool_steps == 4
    assert runner.new_turn(ctx=make_ctx(), question="Q", history=[], max_tool_steps=0).max_tool_steps == 0
    with pytest.raises(ValueError):
        runner.fail(runner.new_turn(ctx=make_ctx(), question="Q", history=[]))
