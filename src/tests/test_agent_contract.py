import json
from pathlib import Path

import pytest

from getwork.agent.failure import FAILURE_COPY
from getwork.agent.models import ChatResponse, Claim, Fact
from getwork.webapp import models

pytestmark = pytest.mark.usefixtures("offline")

AGENT_CONTRACT = Path(__file__).resolve().parents[1] / "web" / "src" / "fixtures" / "contract_agent.json"


def load_contract() -> dict:
    return json.loads(AGENT_CONTRACT.read_text(encoding="utf-8"))


def test_chat_and_agent_sections_validate() -> None:
    data = load_contract()
    assert set(data) == {"explain_agent", "chat", "chat_error"}
    explanation = models.Explanation.model_validate(data["explain_agent"])
    assert explanation.explainer.kind == "agent"
    assert explanation.explainer.model == "demo-model"
    assert explanation.explainer.backend == "voyager"
    assert explanation.explainer.prompt_version == "agent_v1"
    assert len(explanation.claims) == 2 and len(explanation.facts) == 5
    for claim in explanation.claims:
        Claim.model_validate(claim)
    for fact in explanation.facts:
        Fact.model_validate(fact)
    chat = ChatResponse.model_validate(data["chat"])
    assert [f.id for f in chat.facts] == ["f1", "f2", "f3", "f4", "f5", "f6"]
    assert {"comparison", "rank_score"} <= {f.kind for f in chat.facts}
    assert [t.tool for t in chat.trace] == ["compare_matches"]
    assert len(chat.claims) == 2 and len(chat.suggested_questions) == 2


def test_fixture_claims_are_grounded() -> None:
    data = load_contract()
    for section in ("explain_agent", "chat"):
        facts = {f["id"]: f for f in data[section]["facts"]}
        for claim in data[section]["claims"]:
            cited = [facts[fact_id] for fact_id in claim["facts"]]
            cited_skills = {skill for fact in cited for skill in fact["skills"]}
            cited_docs = {doc for fact in cited for doc in fact["doc_ids"]}
            assert claim["facts"]
            assert set(claim["skills"]) <= cited_skills
            assert set(claim["doc_ids"]) <= cited_docs


def test_chat_error_matches_failure_copy() -> None:
    envelope = models.ErrorEnvelope.model_validate(load_contract()["chat_error"])
    message, hint = FAILURE_COPY["unverified"]
    assert envelope.error.code == "assistant_failed"
    assert (envelope.error.message, envelope.error.hint) == (message, hint)
    assert envelope.error.field is None
