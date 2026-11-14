import json

import pytest
from agent_helpers import answer, chat_request, good_answer, make_agent, tool_call

from getwork.agent import explainer, models
from getwork.agent.failure import AgentFailure
from getwork.webapp import demo
from getwork.webapp.engine import EngineError
from getwork.webapp.explain import build_evidence
from getwork.webapp.models import Evidence

pytestmark = pytest.mark.usefixtures("offline")

ENG = demo.DemoEngine()
RESUME = ENG.get_document("r-1")
POSTING = ENG.get_document("j-1")
RELATED = answer("The resume lists Vue where the posting asks for React.", skills=["Vue", "React"], facts=["f2"])
TWO_CLAIMS = json.dumps(
    {"action": "answer", "claims": [json.loads(good_answer())["claims"][0], json.loads(RELATED)["claims"][0]]}
)


def test_explain_agent_answer() -> None:
    agent, _, fake = make_agent([TWO_CLAIMS])
    explanation = agent.explain("resume_to_posting", RESUME, POSTING)
    assert explanation.summary == "Both documents list JavaScript and CSS."
    assert explanation.bullets == ["The resume lists Vue where the posting asks for React."]
    assert explanation.evidence == build_evidence("resume_to_posting", RESUME, POSTING)
    assert explanation.explainer.kind == "agent"
    assert (explanation.explainer.model, explanation.explainer.backend) == ("demo-model", "voyager")
    assert explanation.explainer.prompt_version == "agent_v1"
    for claim in explanation.claims:
        models.Claim.model_validate(claim)
    assert [f["id"] for f in explanation.facts] == ["f1", "f2", "f3", "f4"]
    for fact in explanation.facts:
        models.Fact.model_validate(fact)
    assert len(fake.calls) == 1
    assert explainer.EXPLAIN_QUESTION in fake.calls[0].user


def test_explain_never_uses_tools() -> None:
    agent, _, fake = make_agent([tool_call("cluster_lookup", name="CSS"), good_answer()])
    result = agent.explain_detailed("resume_to_posting", RESUME, POSTING)
    assert result.status == "ok" and result.trace == [] and result.usage.tool_calls == 0
    assert "No more tool calls are allowed; answer now." in fake.calls[1].user


def test_explain_failure_raises_agent_failure() -> None:
    agent, _, _ = make_agent(["nope"] * 3)
    with pytest.raises(AgentFailure) as failed:
        agent.explain("resume_to_posting", RESUME, POSTING)
    assert failed.value.reason == "invalid_reply"
    agent, _, _ = make_agent(['{"action": "answer", "off_topic": true}'])
    with pytest.raises(AgentFailure) as off_topic:
        agent.explain("resume_to_posting", RESUME, POSTING)
    assert off_topic.value.reason == "invalid_reply"
    bad = answer("Both list Kubernetes.", skills=["Kubernetes"])
    agent, _, _ = make_agent([bad] * 3)
    with pytest.raises(AgentFailure) as unverified:
        agent.explain("resume_to_posting", RESUME, POSTING)
    assert unverified.value.reason == "unverified"


def test_explain_detailed_returns_failed_result() -> None:
    agent, _, _ = make_agent(["nope"] * 3)
    result = agent.explain_detailed("resume_to_posting", RESUME, POSTING)
    assert result.status == "failed" and result.failure_reason == "invalid_reply" and result.claims == []


def test_chat_agent_answer() -> None:
    _, chat, fake = make_agent([good_answer()])
    response = chat.chat(chat_request())
    assert response.reply == "Both documents list JavaScript and CSS."
    assert [c.facts for c in response.claims] == [["f1"]]
    assert response.facts and response.trace == [] and response.usage.llm_calls == 1
    assert response.explainer.kind == "agent" and not response.off_topic
    assert response.suggested_questions == [
        "Why is j-1 ranked above j-2?",
        "Which skills count as related here?",
    ]
    assert "Why is j-1 ranked above j-2?" in fake.calls[0].user


def test_chat_compare_with_tools() -> None:
    compare = tool_call("compare_matches", id_a="j-1", id_b="j-2")
    final = answer("j-1 ranks above j-2 because of shared skills.", skills=[], facts=["f7"], doc_ids=["j-1", "j-2"])
    _, chat, fake = make_agent([compare, final])
    response = chat.chat(chat_request())
    assert [t.tool for t in response.trace] == ["compare_matches"] and response.trace[0].ok
    assert response.usage.tool_calls == 1 and response.usage.llm_calls == 2
    assert any(f.kind == "comparison" for f in response.facts)
    assert "f7 [comparison]" in fake.calls[1].user


def test_chat_request_errors() -> None:
    _, chat, _ = make_agent([])
    with pytest.raises(EngineError) as unknown_focus:
        chat.chat(chat_request(focus_id="j-404"))
    assert unknown_focus.value.code == "not_found"
    unknown_ref = [{"id": "j-1", "rank": 1, "score": 0.5}, {"id": "j-404", "rank": 2, "score": 0.1}]
    with pytest.raises(EngineError) as unknown_result:
        chat.chat(chat_request(matches=unknown_ref))
    assert unknown_result.value.code == "not_found"
    with pytest.raises(EngineError) as same_side:
        chat.chat(chat_request(focus_id="r-2"))
    assert same_side.value.code == "side_mismatch"


def test_chat_off_topic() -> None:
    _, chat, _ = make_agent(['{"action": "answer", "off_topic": true}'])
    response = chat.chat(chat_request(message="What is the weather?"))
    assert response.reply == explainer.OFF_TOPIC_REPLY
    assert response.off_topic and response.claims == []


def test_chat_failure_raises_agent_failure() -> None:
    _, chat, _ = make_agent(["nope"] * 3)
    with pytest.raises(AgentFailure) as failed:
        chat.chat(chat_request())
    assert failed.value.reason == "invalid_reply"


def test_chat_passes_trimmed_history() -> None:
    _, chat, fake = make_agent([good_answer()])
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(14)]
    chat.chat(chat_request(history=history))
    prompt = fake.calls[0].user
    assert "user: turn 4" in prompt and "assistant: turn 13" in prompt
    assert "turn 3" not in prompt


def evidence(*, gaps: bool, related: bool) -> Evidence:
    return Evidence.model_validate(
        {
            "shared": [],
            "related": [{"cluster": "C", "resume_skills": ["a"], "posting_skills": ["b"]}] if related else [],
            "gaps": [{"skill": "k", "cluster": "C"}] if gaps else [],
            "level": {"resume": "x", "posting": "y", "relation": "unknown"},
            "title_overlap": [],
        }
    )


def test_suggest_rules() -> None:
    refs = [models.ChatMatchRef(id=i, rank=r, score=0.1) for i, r in (("j-3", 3), ("j-1", 1), ("j-2", 2))]
    full = explainer.suggest("resume_to_posting", refs, "j-1", evidence(gaps=True, related=True))
    assert full == ["Which skills am I missing?", "Why is j-1 ranked above j-2?", "Which skills count as related here?"]
    posting = explainer.suggest("posting_to_resume", refs, "j-2", evidence(gaps=True, related=False))
    assert posting == ["Which required skills does this candidate lack?", "Why is j-2 ranked above j-3?"]
    assert explainer.suggest("resume_to_posting", refs, "j-3", evidence(gaps=False, related=False)) == []
    assert explainer.suggest("resume_to_posting", refs[:1], "j-3", evidence(gaps=False, related=True)) == [
        "Which skills count as related here?"
    ]
    assert explainer.suggest("resume_to_posting", [], "j-1", evidence(gaps=False, related=False)) == []
