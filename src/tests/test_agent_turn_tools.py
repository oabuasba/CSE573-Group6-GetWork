import pytest
from agent_helpers import answer, make_runner, run, tool_call

pytestmark = pytest.mark.usefixtures("offline")


def test_tool_then_answer() -> None:
    runner, fake = make_runner([tool_call("compare_matches", id_a="j-1", id_b="j-2"), answer()])
    result = run(runner)
    assert result.status == "ok" and result.usage.llm_calls == 2 and result.usage.tool_calls == 1
    step = result.trace[0]
    assert (step.tool, step.ok, step.fact_ids) == ("compare_matches", True, ["f5", "f6", "f7"])
    assert step.args == {"id_a": "j-1", "id_b": "j-2"}
    assert "f7 [comparison] (j-1, j-2): j-1 vs j-2:" in fake.calls[1].user
    assert "f7 [comparison]" not in fake.calls[0].user


def test_tool_error_is_fed_back() -> None:
    runner, fake = make_runner([tool_call("compare_matches", id_a="j-1", id_b="j-9"), answer()])
    result = run(runner)
    assert result.status == "ok"
    assert result.trace[0].ok is False and result.trace[0].error == "j-9 is not in the result list."
    assert result.trace[0].fact_ids == []
    assert "compare_matches: j-9 is not in the result list." in fake.calls[1].user
    assert len(result.facts) == 4


def test_tool_step_limit() -> None:
    call = tool_call("cluster_lookup", name="CSS")
    runner, fake = make_runner([call, call, answer()], max_tool_steps=1)
    result = run(runner)
    assert result.status == "ok" and result.usage.tool_calls == 1 and result.usage.llm_calls == 3
    assert "FEEDBACK\nNo more tool calls are allowed; answer now." in fake.calls[2].user
    runner, fake = make_runner([call, answer()])
    result = run(runner, max_tool_steps=0)
    assert result.status == "ok" and result.trace == []
    assert "No more tool calls are allowed; answer now." in fake.calls[1].user
    runner, fake = make_runner([call] * 10, max_tool_steps=0)
    result = run(runner)
    assert result.status == "failed" and result.failure_reason == "budget_exhausted"
    assert len(fake.calls) == runner.cfg.max_llm_calls == result.usage.llm_calls
    runner, fake = make_runner([call] * 10)
    result = run(runner)
    assert result.failure_reason == "budget_exhausted" and len(fake.calls) == 6
    assert result.usage.tool_calls == 4
