import logging

import pytest
from agent_helpers import answer, chat_json, chat_request, good_answer, make_agent, make_client

from getwork.agent import grounding, vocab
from getwork.agent.failure import FAILURE_COPY
from getwork.webapp import demo

pytestmark = pytest.mark.usefixtures("offline")

INTRUDER = "Kubernetes"
HALLUCINATED = answer(f"Both documents list CSS and {INTRUDER}.", skills=["CSS", INTRUDER], facts=["f1"])
MISATTRIBUTED = answer("j-1 asks for Vue.", skills=["Vue"], facts=["f2"], doc_ids=["j-1"])
ENG = demo.DemoEngine()
DOCS = [ENG.get_document("r-1"), ENG.get_document("j-1")]


def test_hallucinated_skill_is_retried_then_corrected() -> None:
    client, fake = make_client([HALLUCINATED, good_answer()])
    response = client.post("/api/chat", json=chat_json())
    assert response.status_code == 200
    assert INTRUDER not in response.text
    assert len(fake.calls) == 2
    assert response.json()["usage"]["verify_retries"] == 1
    assert f'skill "{INTRUDER}" is not in the skills of query, j-1' in fake.calls[1].user


def test_persistent_hallucination_is_an_error_not_an_answer(caplog: pytest.LogCaptureFixture) -> None:
    client, _ = make_client([HALLUCINATED] * 3)
    with caplog.at_level(logging.DEBUG):
        response = client.post("/api/chat", json=chat_json())
    message, hint = FAILURE_COPY["unverified"]
    assert response.status_code == 502
    assert response.json() == {"error": {"code": "assistant_failed", "message": message, "hint": hint, "field": None}}
    assert INTRUDER not in response.text
    assert all(INTRUDER not in record.getMessage() for record in caplog.records)


def test_misattribution_rejected() -> None:
    client, fake = make_client([MISATTRIBUTED] * 3)
    response = client.post("/api/chat", json=chat_json())
    assert response.status_code == 502
    assert response.json()["error"]["message"] == FAILURE_COPY["unverified"][0]
    assert 'skill "Vue" is not in the skills of j-1' in fake.calls[1].user
    assert "Vue" not in response.text


def test_accepted_answers_have_no_ungrounded_skills() -> None:
    demo_vocab = vocab.demo_vocabulary()
    _, chat, _ = make_agent([HALLUCINATED, good_answer()])
    accepted = [chat.chat(chat_request())]
    _, chat, _ = make_agent([good_answer()])
    accepted.append(chat.chat(chat_request(message="Another question?")))
    assert len(accepted) == 2
    for response in accepted:
        assert grounding.ungrounded_skills(response.claims, DOCS, demo_vocab) == []
        assert INTRUDER not in response.reply


def test_declared_skills_checked_without_vocabulary() -> None:
    empty = grounding.make_vocabulary([], [])
    declared = answer("Both documents list CSS.", skills=["CSS", INTRUDER], facts=["f1"])
    client, fake = make_client([declared] * 3, vocab=empty)
    response = client.post("/api/chat", json=chat_json())
    assert response.status_code == 502
    assert f'skill "{INTRUDER}" is not in the skills of query, j-1' in fake.calls[1].user
