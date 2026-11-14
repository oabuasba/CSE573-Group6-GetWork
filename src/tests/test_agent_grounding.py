import pytest

from getwork.agent import grounding
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
OUTSIDE_VOCAB = grounding.make_vocabulary(
    ["React", "SQL", "Terraform", "Cluster", "Lists", "Front-end"], ["Front-end"]
)


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


def test_ungrounded_skills_in_text() -> None:
    both = [SCOPE["query"], SCOPE["j-1"]]
    text = "Add Terraform and Cluster. React and SQL fit; Terraform again. Front-end too, and lists."
    assert grounding.ungrounded_skills(text, both, OUTSIDE_VOCAB) == ["Terraform", "Cluster"]
    assert grounding.ungrounded_skills("React, SQL and sql", both, OUTSIDE_VOCAB) == []
    assert grounding.ungrounded_skills("Vue is in the query", [SCOPE["query"]], VOCAB) == []
    assert grounding.ungrounded_skills("Front-end work", both, OUTSIDE_VOCAB) == []
    assert grounding.ungrounded_skills("Some lists here", both, OUTSIDE_VOCAB) == []
    assert grounding.ungrounded_skills("Lists matter", both, OUTSIDE_VOCAB) == ["Lists"]
    assert grounding.ungrounded_skills("React then React", [SCOPE["query"]], OUTSIDE_VOCAB) == ["React"]
    assert grounding.ungrounded_skills("", both, OUTSIDE_VOCAB) == []
    assert grounding.ungrounded_skills("React", [], OUTSIDE_VOCAB) == ["React"]
