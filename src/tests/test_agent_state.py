import pytest

from getwork.agent import models, state, tools
from getwork.agent.facts import FactBook
from getwork.llm.retry import CallStats, RetryConfig
from getwork.webapp import demo

pytestmark = pytest.mark.usefixtures("offline")


def test_agent_config_validation() -> None:
    cfg = state.AgentConfig(model="m", backend="voyager")
    assert (cfg.prompt_version, cfg.temperature) == ("agent_v3", 0.0)
    assert (cfg.max_tokens, cfg.reasoning_effort) == (1024, "none")
    assert (cfg.max_tool_steps, cfg.max_llm_calls, cfg.timeout_s) == (4, 6, 60.0)
    assert cfg.extra_body is None and cfg.retry == RetryConfig()
    state.AgentConfig(model="m", backend="b", max_tool_steps=0, max_llm_calls=1)
    bad = [
        {"model": ""},
        {"model": "  "},
        {"max_tool_steps": -1},
        {"max_llm_calls": 0},
        {"timeout_s": 0},
        {"timeout_s": -2.5},
    ]
    for overrides in bad:
        with pytest.raises(ValueError, match=next(iter(overrides))):
            state.AgentConfig(**{"model": "m", "backend": "b", **overrides})


def test_agent_config_rejects_verify_retries() -> None:
    with pytest.raises(TypeError):
        state.AgentConfig(model="m", backend="b", max_verify_retries=1)


def test_turn_defaults() -> None:
    eng = demo.DemoEngine()
    query = eng.get_document("r-1")
    ctx = tools.TurnContext(
        engine=eng,
        direction="resume_to_posting",
        system="Full",
        query=query,
        focus_id="j-1",
        refs={},
        docs={models.QUERY_DOC: query},
        book=FactBook(),
    )
    turn = state.Turn(ctx=ctx, question="Why?", history=[], max_tool_steps=4, started=1.5)
    assert turn.tool_notes == [] and turn.trace == []
    assert turn.stats == CallStats()
    assert turn.tool_steps == 0
    assert not hasattr(turn, "verify_retries") and not hasattr(turn, "dropped_claims")
    assert turn.feedback is None and turn.pending is None and turn.answer is None
    assert turn.route == "decide" and turn.failure_reason is None and turn.result is None
    other = state.Turn(ctx=ctx, question="Q", history=[], max_tool_steps=0, started=0.0)
    assert other.tool_notes is not turn.tool_notes
