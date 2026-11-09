"""Direct taxonomy lookup of document skills and the document completeness rules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from getwork.taxonomy.model import OTHER, Taxonomy

VIA_VALUES: tuple[str, ...] = ("taxonomy", "noise", "llm", "default")


@dataclass(frozen=True)
class SkillAssignment:
    skill: str
    cluster: str  # a cluster name or OTHER
    via: str  # one of VIA_VALUES


def split_known(
    skills: Sequence[str], tax: Taxonomy
) -> tuple[list[SkillAssignment], list[str]]:
    """Known skills become assignments in input order (via "taxonomy", or "noise" for OTHER).

    Returns (assignments, unseen skills in input order).
    """
    assignments: list[SkillAssignment] = []
    unseen: list[str] = []
    for skill in skills:
        cluster = tax.lookup(skill)
        if cluster is None:
            unseen.append(skill)
            continue
        via = "noise" if cluster == OTHER else "taxonomy"
        assignments.append(SkillAssignment(skill=skill, cluster=cluster, via=via))
    return assignments, unseen


def known_fraction(skills: Sequence[str], tax: Taxonomy) -> float | None:
    """Share of skills that tax.is_known (noise counts as known); None for an empty list."""
    if not skills:
        return None
    return sum(1 for s in skills if tax.is_known(s)) / len(skills)


def percentile_threshold(counts: Sequence[int], pct: float = 10.0) -> int:
    if len(counts) == 0:
        raise ValueError("percentile_threshold needs at least one count")
    return int(np.percentile(counts, pct, method="lower"))


def incomplete_reason(
    n_skills: int, known: float | None, p10: int, min_known: float = 0.5
) -> str | None:
    if n_skills == 0:
        return "no_skills"
    if n_skills < p10:
        return "few_skills"
    if known is not None and known < min_known:
        return "low_coverage"
    return None
