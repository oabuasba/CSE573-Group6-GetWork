import json

import pytest
from agent_helpers import SHARED_CLAIM, chat_json, make_client

from getwork.agent import grounding
from getwork.agent.failure import FAILURE_COPY
from getwork.agent.models import OMITTED_NOTE

pytestmark = pytest.mark.usefixtures("offline")

INTRUDER = "Kubernetes"


def claim(text: str, skills: list[str], facts: list[str]) -> dict:
    return {"text": text, "doc_ids": ["query", "j-1"], "skills": skills, "facts": facts}


def reply(*claims: dict) -> str:
    return json.dumps({"action": "answer", "claims": list(claims)})


GOOD = claim(SHARED_CLAIM[0], SHARED_CLAIM[1], SHARED_CLAIM[2])
BAD = claim(f"Adding {INTRUDER} would help.", [INTRUDER], ["f1"])
MIXED = reply(GOOD, BAD)


def test_mixed_answer_returns_grounded_claims_and_a_note() -> None:
    client, fake = make_client([MIXED] * 3)
    response = client.post("/api/chat", json=chat_json(message="what could i do to improve my resume?"))
    assert response.status_code == 200
    body = response.json()
    assert [c["text"] for c in body["claims"]] == [GOOD["text"]]
    assert body["note"] == OMITTED_NOTE
    assert INTRUDER not in response.text and len(fake.calls) == 3


def test_fully_grounded_answer_has_no_note() -> None:
    client, _ = make_client([reply(GOOD)])
    assert client.post("/api/chat", json=chat_json()).json()["note"] is None


def test_all_ungrounded_is_still_unverified() -> None:
    client, _ = make_client([reply(BAD)] * 3)
    response = client.post("/api/chat", json=chat_json())
    assert response.status_code == 502
    assert response.json()["error"]["message"] == FAILURE_COPY["unverified"][0]
    assert INTRUDER not in response.text


def test_retry_that_fixes_everything_has_no_note() -> None:
    client, _ = make_client([MIXED, reply(GOOD)])
    body = client.post("/api/chat", json=chat_json()).json()
    assert body["note"] is None and len(body["claims"]) == 1


LISTS_VOCAB = grounding.make_vocabulary(["Lists", "SQL"], [])


def test_plain_lowercase_words_are_not_skill_mentions() -> None:
    text = "Your resume lists skills but SQL is missing."
    assert grounding.skill_mentions(text, LISTS_VOCAB) == ["SQL"]


def test_common_word_forms_that_look_like_skills_are_still_detected() -> None:
    assert grounding.skill_mentions("Add Lists to your resume.", LISTS_VOCAB) == ["Lists"]
    assert grounding.skill_mentions("Add SQL.", LISTS_VOCAB) == ["SQL"]


def test_prose_lists_does_not_violate_a_grounded_claim() -> None:
    text = "Both documents list JavaScript and CSS, and the posting lists them first."
    vocab = grounding.make_vocabulary(["JavaScript", "CSS", "lists"], [])
    client, _ = make_client([reply(claim(text, ["JavaScript", "CSS"], ["f1"]))], vocab=vocab)
    assert client.post("/api/chat", json=chat_json()).status_code == 200


ADVICE_VOCAB = grounding.make_vocabulary(
    ["Cloud", "Leadership", "Security", "Communication", "Testing", "Python", "SQL"], []
)


@pytest.mark.parametrize("word", ["cloud", "leadership", "security", "communication", "testing"])
def test_lowercase_real_skills_undeclared_are_still_violations(word: str) -> None:
    assert grounding.skill_mentions(f"Add {word} experience.", ADVICE_VOCAB) != []


def test_lists_in_prose_is_not_a_skill_with_the_advice_vocab() -> None:
    vocab = grounding.make_vocabulary(["Lists", "Python"], [])
    assert grounding.skill_mentions("The posting lists Python.", vocab) == ["Python"]


def test_count_rule_violation_leaves_no_verified_claim() -> None:
    claims = [grounding.Claim(**GOOD) for _ in range(2)]
    count_rule = "claims: give between 1 and 8 claims, not 9."
    assert grounding.verified_claims(claims, [count_rule]) == []
    assert grounding.verified_claims(claims, [count_rule, "claim 2: bad."]) == []
    assert len(grounding.verified_claims(claims, ["claim 2: bad."])) == 1
