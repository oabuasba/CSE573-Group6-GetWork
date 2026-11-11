import pytest

from getwork.agent import grounding
from getwork.agent.models import Claim, Fact
from getwork.webapp.models import DocSkill, DocumentView

pytestmark = pytest.mark.usefixtures("offline")

VOCAB = grounding.make_vocabulary(
    ["React", "Vue", "Node.js", "Machine Learning", "Learning", "Go", "Kubernetes", "SQL", "Front-end"],
    ["Front-end"],
)


def make_doc(doc_id: str | None, side: str, skills: list[str]) -> DocumentView:
    return DocumentView(
        id=doc_id,
        side=side,
        title="T",
        level="Associate",
        level_source="listed",
        years_experience=None,
        skills=[DocSkill(skill=s, cluster="Front-end", via="taxonomy") for s in skills],
    )


SCOPE = {
    "query": make_doc(None, "resume", ["Vue", "SQL", "Node.js"]),
    "j-1": make_doc("j-1", "posting", ["React", "SQL", "Kubernetes"]),
    "j-2": make_doc("j-2", "posting", ["Machine Learning"]),
}
FACTS = {
    "f1": Fact(id="f1", kind="shared_skills", doc_ids=["query", "j-1"], skills=["SQL"], text="Shared skills: SQL."),
    "f2": Fact(
        id="f2",
        kind="rank_score",
        doc_ids=["j-1"],
        skills=["React", "Kubernetes"],
        text="j-1 is ranked 2 with similarity 0.750. Skills 10.",
    ),
    "f3": Fact(id="f3", kind="document_skills", doc_ids=["j-2"], skills=["Machine Learning"], text="x"),
}


def claim(**overrides: object) -> Claim:
    base = {"text": "Both list SQL.", "doc_ids": ["query", "j-1"], "skills": ["SQL"], "facts": ["f1"]}
    return Claim(**{**base, **overrides})


def verify(*claims: Claim) -> list[str]:
    return grounding.verify_claims(list(claims), facts=FACTS, scope=SCOPE, vocab=VOCAB)


def test_skill_mentions_rules() -> None:
    mentions = grounding.skill_mentions
    assert mentions("I know machine learning well", VOCAB) == ["Machine Learning"]
    assert mentions("learning is fun", VOCAB) == ["Learning"]
    assert mentions("Uses NODE.JS, and react.", VOCAB) == ["Node.js", "React"]
    assert mentions("react then React then Vue then REACT", VOCAB) == ["React", "Vue"]
    assert mentions("Front-end work", VOCAB) == []
    assert mentions("Go to the shop", VOCAB) == []
    assert mentions("nothing relevant here", VOCAB) == []
    assert mentions("", VOCAB) == []
    short = grounding.make_vocabulary(["Go", "C", "R"], [])
    assert grounding.skill_mentions("go c r", short) == []


def test_verify_accepts_valid_claims() -> None:
    assert verify(claim()) == []
    two = claim(text="j-1 asks for React.", doc_ids=["j-1"], skills=["React"], facts=["f2"])
    assert verify(claim(), two) == []
    assert verify(claim(text="j-1 is ranked 2.", doc_ids=["j-1"], skills=[], facts=["f2"])) == []


def test_verify_each_rule() -> None:
    assert verify() == ["claims: give between 1 and 8 claims, not 0."]
    assert verify(*[claim()] * 9) == ["claims: give between 1 and 8 claims, not 9."]
    assert verify(claim(text="See https://x.org SQL.")) == ["claim 1: remove links and code blocks from the text."]
    for bad_text in ("See http://x.org.", "Code ```SQL``` here."):
        assert verify(claim(text=bad_text)) == ["claim 1: remove links and code blocks from the text."]
    assert verify(claim(facts=[], skills=[], text="Hi.")) == ["claim 1: cite at least one fact id."]
    assert verify(claim(facts=["f1", "f9"])) == [
        'claim 1: fact "f9" does not exist; cite only fact ids you were given.'
    ]
    assert verify(claim(doc_ids=[], skills=[], text="Hi.")) == [
        "claim 1: name at least one document in doc_ids."
    ]
    no_docs = verify(claim(doc_ids=[], skills=["SQL"], text="Both list SQL."))
    assert no_docs == ["claim 1: name at least one document in doc_ids."]
    assert verify(claim(doc_ids=["query", "j-9"])) == ['claim 1: document "j-9" is not in this conversation.']
    assert verify(claim(doc_ids=["j-2"], skills=[], text="Hi.")) == [
        'claim 1: fact "f1" is about other documents than j-2.'
    ]
    assert verify(claim(skills=["Vue"], text="Hi.")) == ['claim 1: skill "Vue" is not in any fact you cited.']
    assert verify(claim(skills=["SQL", "Kubernetes"], text="Hi.")) == [
        'claim 1: skill "Kubernetes" is not in any fact you cited.'
    ]
    assert verify(claim(text="Both list SQL and Vue.", skills=["SQL"])) == [
        'claim 1: the text names skill "Vue"; add it to skills.'
    ]
    assert verify(claim(text="Both list SQL and React.", skills=["SQL"])) == [
        'claim 1: the text names skill "React"; add it to skills.'
    ]
    assert verify(claim(text="SQL appears 5 times.")) == [
        "claim 1: the number 5 does not appear in the facts you cited."
    ]
    assert verify(claim(text="j-1 is ranked 2 with 10 skills.", doc_ids=["j-1"], skills=[], facts=["f2"])) == []


def test_verify_catches_misattributed_skill() -> None:
    wrong = claim(text="j-1 asks for Vue.", doc_ids=["j-1"], skills=["Vue"], facts=["f2"])
    assert verify(wrong) == [
        'claim 1: skill "Vue" is not in the skills of j-1.',
        'claim 1: the text names skill "Vue", which is not in the skills of j-1.',
    ]
    quiet = claim(text="j-1 asks for it.", doc_ids=["j-1"], skills=["Vue"], facts=["f2"])
    assert verify(quiet) == ['claim 1: skill "Vue" is not in the skills of j-1.']


def test_named_skills_union() -> None:
    claims = [
        claim(text="Both list sql and React.", skills=["SQL"]),
        claim(text="Vue and Front-end.", skills=["vue", "Kubernetes", "Front-end"]),
    ]
    assert grounding.named_skills(claims, VOCAB) == ["SQL", "React", "vue", "Kubernetes", "Front-end"]
    assert grounding.named_skills(claims, VOCAB) == grounding.named_skills(claims, VOCAB)
    assert grounding.named_skills([], VOCAB) == []


def test_ungrounded_skills_definition() -> None:
    claims = [claim(text="Both list SQL and React.", skills=["SQL", "Kubernetes", "Front-end"])]
    docs = [SCOPE["query"]]
    assert grounding.ungrounded_skills(claims, docs, VOCAB) == ["Kubernetes", "React"]
    assert grounding.ungrounded_skills(claims, [SCOPE["query"], SCOPE["j-1"]], VOCAB) == []
    assert grounding.ungrounded_skills(claims, [], VOCAB) == ["SQL", "Kubernetes", "React"]
