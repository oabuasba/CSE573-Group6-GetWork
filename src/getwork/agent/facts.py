"""Code-generated facts: the only statements a claim may rest on (ADR-20)."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from getwork.agent.models import QUERY_DOC, Fact, FactKind
from getwork.taxonomy.model import OTHER
from getwork.webapp.models import Direction, DocumentView, Evidence

NO_ITEMS = "none"
UNTITLED = "Untitled"
KEYWORD_SYSTEM = "B1+"


def doc_ref(doc: DocumentView) -> str:
    """QUERY_DOC when doc.id is None, else doc.id."""
    return QUERY_DOC if doc.id is None else doc.id


def join_names(names: Sequence[str]) -> str:
    return ", ".join(names) if names else NO_ITEMS


@dataclass
class FactBook:
    """Creates facts with sequential ids and keeps them in creation order."""

    facts: dict[str, Fact] = field(default_factory=dict)

    def add(
        self,
        kind: FactKind,
        *,
        doc_ids: Sequence[str],
        skills: Sequence[str],
        text: str,
        cluster: str | None = None,
    ) -> Fact:
        fact = Fact(
            id=f"f{len(self.facts) + 1}",
            kind=kind,
            doc_ids=list(doc_ids),
            skills=list(skills),
            cluster=cluster,
            text=text,
        )
        self.facts[fact.id] = fact
        return fact


def evidence_facts(
    book: FactBook, direction: Direction, query: DocumentView, match: DocumentView, evidence: Evidence
) -> list[Fact]:
    """Seed facts for the focus pair, in this order, each only when non-empty: shared_skills, one related_skills per
    group, gap_skills, level (always), title_overlap. Sentences exactly as in "Fact sentences"."""
    # direction is part of the fixed signature; the evidence already carries the resume/posting roles.
    # The query side is always labelled "query", whatever query.id is.
    pair = [QUERY_DOC, doc_ref(match)]
    created: list[Fact] = []
    if evidence.shared:
        names = [s.skill for s in evidence.shared]
        text = f"Shared skills: {', '.join(names)}."
        created.append(book.add("shared_skills", doc_ids=pair, skills=names, text=text))
    for group in evidence.related:
        text = (
            f"Related through {group.cluster}: the resume lists {', '.join(group.resume_skills)}; "
            f"the posting asks for {', '.join(group.posting_skills)}."
        )
        skills = [*group.resume_skills, *group.posting_skills]
        created.append(book.add("related_skills", doc_ids=pair, skills=skills, text=text, cluster=group.cluster))
    if evidence.gaps:
        names = [s.skill for s in evidence.gaps]
        text = f"Skills the posting asks for that the resume does not show: {', '.join(names)}."
        created.append(book.add("gap_skills", doc_ids=pair, skills=names, text=text))
    level = evidence.level
    text = f"Experience levels: resume {level.resume}, posting {level.posting} ({level.relation})."
    created.append(book.add("level", doc_ids=pair, skills=[], text=text))
    if evidence.title_overlap:
        text = f"Shared title words: {', '.join(evidence.title_overlap)}."
        created.append(book.add("title_overlap", doc_ids=pair, skills=[], text=text))
    return created


def _cluster_groups(doc: DocumentView) -> list[tuple[str, list[str]]]:
    """Clusters in first-appearance order, with Other last."""
    groups: dict[str, list[str]] = {}
    for item in doc.skills:
        groups.setdefault(item.cluster, []).append(item.skill)
    named = [(name, skills) for name, skills in groups.items() if name != OTHER]
    return named + ([(OTHER, groups[OTHER])] if OTHER in groups else [])


def document_fact(book: FactBook, doc: DocumentView) -> Fact:
    ref = doc_ref(doc)
    groups = "; ".join(f"{name}: {', '.join(skills)}" for name, skills in _cluster_groups(doc))
    text = f"Skills of {ref} ({doc.title or UNTITLED}, {doc.level}) by cluster: {groups or NO_ITEMS}."
    return book.add("document_skills", doc_ids=[ref], skills=[s.skill for s in doc.skills], text=text)


def rank_fact(book: FactBook, doc_id: str, *, rank: int, score: float, system: str) -> Fact:
    label = "keyword score" if system == KEYWORD_SYSTEM else "similarity"
    text = f"{doc_id} is ranked {rank} with {label} {score:.3f}."
    return book.add("rank_score", doc_ids=[doc_id], skills=[], text=text)
