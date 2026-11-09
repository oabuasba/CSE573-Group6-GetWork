"""Full and B2 document texts (CC-4): the same fields, with and without cluster labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from getwork.docembed.levels import NOT_SPECIFIED
from getwork.taxonomy.model import OTHER

SERIALIZATION_VERSION: str = "ser_v1"
B2_ORDERS: tuple[str, ...] = ("grouped", "record")
NO_SKILLS = "none"
MAX_CLUSTER_CHARS = 60


def _collapse(text: str) -> str:
    return " ".join(text.split())


def clean_field(text: str | None) -> str:
    """Collapse whitespace runs to one space; None or blank gives NOT_SPECIFIED."""
    return _collapse(text or "") or NOT_SPECIFIED


def _clean_items(skill_clusters: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    cleaned = [(_collapse(item["skill"]), item["cluster"]) for item in skill_clusters]
    return [(skill, cluster) for skill, cluster in cleaned if skill]


def _group(
    items: Sequence[tuple[str, str]], cluster_order: Sequence[str]
) -> list[tuple[str, list[str]]]:
    names = [*cluster_order, OTHER]
    groups: dict[str, list[str]] = {name: [] for name in names}
    for skill, cluster in items:
        if cluster not in groups:
            raise ValueError(f"unknown cluster {cluster[:MAX_CLUSTER_CHARS]!r} for skill {skill!r}")
        groups[cluster].append(skill)
    return [(name, groups[name]) for name in names if groups[name]]


def group_skills(
    skill_clusters: Sequence[Mapping[str, str]], cluster_order: Sequence[str]
) -> list[tuple[str, list[str]]]:
    """Non-empty groups in cluster_order then OTHER; skills keep input order within a group."""
    return _group(items=_clean_items(skill_clusters), cluster_order=cluster_order)


def _header(title: str | None, level: str) -> str:
    return f"Title: {clean_field(title)}\nLevel: {clean_field(level)}\nSkills: "


def serialize_full(
    title: str | None,
    level: str,
    skill_clusters: Sequence[Mapping[str, str]],
    cluster_order: Sequence[str],
) -> str:
    groups = group_skills(skill_clusters, cluster_order)
    body = "; ".join(f"{name}: {', '.join(skills)}" for name, skills in groups)
    return _header(title, level) + (body or NO_SKILLS)


def serialize_b2(
    title: str | None,
    level: str,
    skill_clusters: Sequence[Mapping[str, str]],
    cluster_order: Sequence[str],
    order: str = "grouped",
) -> str:
    """Full's header and skills without labels; ValueError for an order not in B2_ORDERS."""
    if order not in B2_ORDERS:
        raise ValueError(f"b2 order must be one of {B2_ORDERS}, got {order!r}")
    items = _clean_items(skill_clusters)
    if order == "grouped":
        skills = [s for _, group in _group(items=items, cluster_order=cluster_order) for s in group]
    else:
        _group(items=items, cluster_order=cluster_order)  # same unknown-cluster check as Full
        skills = [skill for skill, _ in items]
    return _header(title, level) + (", ".join(skills) or NO_SKILLS)
