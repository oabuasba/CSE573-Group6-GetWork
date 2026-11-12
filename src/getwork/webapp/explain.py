"""Evidence from two final skill sets and the deterministic explainer (CC-18, ADR-16)."""

from typing import Protocol

from getwork.alias import normalize
from getwork.docembed import levels
from getwork.retrieval import b1plus
from getwork.taxonomy import model as taxonomy_model
from getwork.webapp import models

MAX_NAMES = 8
FALLBACK_BULLET = (
    "No shared or related skills were found; this match comes from overall document similarity."
)


def _index_by_key(skills: list[models.DocSkill]) -> dict[str, models.DocSkill]:
    """Skills by normalized key, first occurrence wins; empty keys are ignored."""
    indexed: dict[str, models.DocSkill] = {}
    for skill in skills:
        key = normalize.normalize_key(skill.skill)
        if key and key not in indexed:
            indexed[key] = skill
    return indexed


def _compare_levels(resume_level: str, posting_level: str) -> models.LevelComparison:
    resume_rank = levels.level_rank(resume_level)
    posting_rank = levels.level_rank(posting_level)
    if resume_rank is None or posting_rank is None:
        relation = "unknown"
    else:
        gap = abs(resume_rank - posting_rank)
        relation = "same" if gap == 0 else "adjacent" if gap == 1 else "different"
    return models.LevelComparison(resume=resume_level, posting=posting_level, relation=relation)


def _related_groups(
    posting: dict[str, models.DocSkill], resume: dict[str, models.DocSkill]
) -> list[models.RelatedGroup]:
    resume_open = [s for k, s in resume.items() if k not in posting]
    posting_open = [s for k, s in posting.items() if k not in resume]
    groups: list[models.RelatedGroup] = []
    seen: set[str] = set()
    for skill in posting_open:
        cluster = skill.cluster
        if cluster == taxonomy_model.OTHER or cluster in seen:
            continue
        seen.add(cluster)
        resume_names = [s.skill for s in resume_open if s.cluster == cluster]
        if resume_names:
            posting_names = [s.skill for s in posting_open if s.cluster == cluster]
            groups.append(
                models.RelatedGroup(
                    cluster=cluster, resume_skills=resume_names, posting_skills=posting_names
                )
            )
    return groups


def build_evidence(
    direction: models.Direction, query: models.DocumentView, match: models.DocumentView
) -> models.Evidence:
    """Shared, related and missing skills, level relation and title overlap of a pair.

    The result is the same for both directions when the two documents are swapped.
    """
    resume, posting = (query, match) if direction == "resume_to_posting" else (match, query)
    resume_by_key = _index_by_key(resume.skills)
    posting_by_key = _index_by_key(posting.skills)
    shared = [
        models.SharedSkill(skill=s.skill, cluster=s.cluster)
        for k, s in posting_by_key.items()
        if k in resume_by_key
    ]
    related = _related_groups(posting_by_key, resume_by_key)
    related_clusters = {g.cluster for g in related}
    gaps = [
        models.GapSkill(skill=s.skill, cluster=s.cluster)
        for k, s in posting_by_key.items()
        if k not in resume_by_key and s.cluster not in related_clusters
    ]
    overlap = b1plus.title_tokens(resume.title) & b1plus.title_tokens(posting.title)
    return models.Evidence(
        shared=shared,
        related=related,
        gaps=gaps,
        level=_compare_levels(resume.level, posting.level),
        title_overlap=sorted(overlap),
    )


class Explainer(Protocol):
    @property
    def info(self) -> models.ExplainerInfo: ...

    def explain(
        self, direction: models.Direction, query: models.DocumentView, match: models.DocumentView
    ) -> models.Explanation: ...


def _count(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _names(names: list[str]) -> str:
    shown = ", ".join(names[:MAX_NAMES])
    rest = len(names) - MAX_NAMES
    return f"{shown} and {rest} more" if rest > 0 else shown


def _level_bullet(level: models.LevelComparison) -> str | None:
    if level.relation == "same":
        return f"Experience level is the same ({level.resume})."
    if level.relation == "adjacent":
        return f"Experience levels are adjacent (resume: {level.resume}, posting: {level.posting})."
    if level.relation == "different":
        return f"Experience levels differ (resume: {level.resume}, posting: {level.posting})."
    return None


def _bullets(evidence: models.Evidence) -> list[str]:
    bullets: list[str] = []
    if evidence.shared:
        bullets.append(f"Shared skills: {_names([s.skill for s in evidence.shared])}.")
    for group in evidence.related:
        bullets.append(
            f"Related through {group.cluster}: the resume lists {', '.join(group.resume_skills)}, "
            f"the posting asks for {', '.join(group.posting_skills)}."
        )
    if evidence.gaps:
        gap_names = _names([g.skill for g in evidence.gaps])
        bullets.append(f"The posting asks for skills the resume does not show: {gap_names}.")
    level_bullet = _level_bullet(evidence.level)
    if level_bullet:
        bullets.append(level_bullet)
    if evidence.title_overlap:
        bullets.append(f"Shared title words: {', '.join(evidence.title_overlap)}.")
    return bullets or [FALLBACK_BULLET]


class TemplateExplainer:
    info: models.ExplainerInfo = models.ExplainerInfo(kind="template", model=None)

    def __init__(self, reason: str | None = None) -> None:
        if reason:
            self.info = models.ExplainerInfo(kind="template", model=None, reason=reason)

    def explain(
        self, direction: models.Direction, query: models.DocumentView, match: models.DocumentView
    ) -> models.Explanation:
        evidence = build_evidence(direction, query, match)
        summary = (
            f"{_count(len(evidence.shared), 'shared skill')}, "
            f"{_count(len(evidence.related), 'related group')}, "
            f"{_count(len(evidence.gaps), 'gap')}."
        )
        return models.Explanation(
            summary=summary, bullets=_bullets(evidence), evidence=evidence, explainer=self.info
        )
