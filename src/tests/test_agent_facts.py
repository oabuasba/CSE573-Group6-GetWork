import pytest

from getwork.agent import facts
from getwork.webapp.models import DocSkill, DocumentView, Evidence

pytestmark = pytest.mark.usefixtures("offline")


def make_doc(doc_id: str | None, side: str, skills: list[tuple[str, str]], title: str | None = "T") -> DocumentView:
    return DocumentView(
        id=doc_id,
        side=side,
        title=title,
        level="Associate",
        level_source="listed",
        years_experience=None,
        skills=[DocSkill(skill=s, cluster=c, via="taxonomy") for s, c in skills],
    )


QUERY = make_doc(None, "resume", [("Vue", "Front-end")])
MATCH = make_doc("j-1", "posting", [("React", "Front-end")])
EVIDENCE = Evidence.model_validate(
    {
        "shared": [{"skill": "JavaScript", "cluster": "Front-end"}, {"skill": "CSS", "cluster": "Front-end"}],
        "related": [{"cluster": "Front-end", "resume_skills": ["Vue"], "posting_skills": ["React"]}],
        "gaps": [{"skill": "Kubernetes", "cluster": "DevOps"}],
        "level": {"resume": "Associate", "posting": "Associate", "relation": "same"},
        "title_overlap": ["frontend"],
    }
)


def test_evidence_facts_exact_sentences() -> None:
    book = facts.FactBook()
    created = facts.evidence_facts(book, "resume_to_posting", QUERY, MATCH, EVIDENCE)
    assert [(f.id, f.kind) for f in created] == [
        ("f1", "shared_skills"),
        ("f2", "related_skills"),
        ("f3", "gap_skills"),
        ("f4", "level"),
        ("f5", "title_overlap"),
    ]
    assert [f.text for f in created] == [
        "Shared skills: JavaScript, CSS.",
        "Related through Front-end: the resume lists Vue; the posting asks for React.",
        "Skills the posting asks for that the resume does not show: Kubernetes.",
        "Experience levels: resume Associate, posting Associate (same).",
        "Shared title words: frontend.",
    ]
    assert created[0].doc_ids == ["query", "j-1"] and created[0].skills == ["JavaScript", "CSS"]
    assert created[1].cluster == "Front-end" and created[1].skills == ["Vue", "React"]
    assert list(book.facts) == ["f1", "f2", "f3", "f4", "f5"]


def test_evidence_facts_omits_empty_kinds() -> None:
    bare = Evidence.model_validate(
        {
            "shared": [],
            "related": [],
            "gaps": [],
            "level": {"resume": "Entry level", "posting": "Mid-Senior level", "relation": "different"},
            "title_overlap": [],
        }
    )
    created = facts.evidence_facts(facts.FactBook(), "resume_to_posting", QUERY, MATCH, bare)
    assert [f.text for f in created] == [
        "Experience levels: resume Entry level, posting Mid-Senior level (different)."
    ]


def test_document_and_rank_facts() -> None:
    doc = make_doc(
        "j-9",
        "posting",
        [("Excel", "Other"), ("React", "Front-end"), ("SQL", "Data"), ("CSS", "Front-end"), ("Word", "Other")],
        title=None,
    )
    book = facts.FactBook()
    fact = facts.document_fact(book, doc)
    assert fact.kind == "document_skills" and fact.doc_ids == ["j-9"]
    assert fact.text == (
        "Skills of j-9 (Untitled, Associate) by cluster: Front-end: React, CSS; Data: SQL; Other: Excel, Word."
    )
    assert facts.document_fact(book, QUERY).text.startswith("Skills of query (T, Associate)")
    assert facts.rank_fact(book, "j-1", rank=1, score=0.8333, system="Full").text == (
        "j-1 is ranked 1 with similarity 0.833."
    )
    assert facts.rank_fact(book, "j-1", rank=2, score=0.5, system="B2").text.endswith("similarity 0.500.")
    assert facts.rank_fact(book, "j-1", rank=3, score=0.25, system="B1+").text == (
        "j-1 is ranked 3 with keyword score 0.250."
    )


def test_fact_ids_sequential() -> None:
    book = facts.FactBook()
    book.add("level", doc_ids=["query"], skills=[], text="x")
    created = facts.evidence_facts(book, "resume_to_posting", QUERY, MATCH, EVIDENCE)
    assert [f.id for f in created] == ["f2", "f3", "f4", "f5", "f6"]
    again = facts.evidence_facts(facts.FactBook(), "resume_to_posting", QUERY, MATCH, EVIDENCE)
    other = facts.evidence_facts(facts.FactBook(), "resume_to_posting", QUERY, MATCH, EVIDENCE)
    assert again == other


def test_evidence_facts_label_query_as_query() -> None:
    anonymous = facts.evidence_facts(facts.FactBook(), "resume_to_posting", QUERY, MATCH, EVIDENCE)
    named_query = QUERY.model_copy(update={"id": "r-1"})
    named = facts.evidence_facts(facts.FactBook(), "resume_to_posting", named_query, MATCH, EVIDENCE)
    assert anonymous == named
    assert all(f.doc_ids == ["query", "j-1"] for f in named)
