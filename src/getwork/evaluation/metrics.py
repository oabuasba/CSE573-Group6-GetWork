"""Per-query ranking metrics over the top 10 (label 2 is the only relevant label)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

METRICS: tuple[str, ...] = ("P@5", "P@10", "MRR", "nDCG@10")
TOP_K = 10
RELEVANT = 2
VALID_LABELS = (0, 1, 2)


def _check(labels: Sequence[int | None], *, name: str) -> None:
    if len(labels) > TOP_K:
        raise ValueError(f"{name} has {len(labels)} labels; at most {TOP_K} are allowed")
    for label in labels:
        if label is not None and (isinstance(label, bool) or label not in VALID_LABELS):
            raise ValueError(f"{name} has the label {label!r}; use 0, 1 or 2")


def _dcg(labels: Sequence[int | None]) -> float:
    return sum((label or 0) / math.log2(rank + 1) for rank, label in enumerate(labels, start=1))


def _precision(labels: Sequence[int | None], k: int) -> float:
    return sum(label == RELEVANT for label in labels[:k]) / k


def _reciprocal_rank(labels: Sequence[int | None]) -> float:
    for rank, label in enumerate(labels, start=1):
        if label == RELEVANT:
            return 1.0 / rank
    return 0.0


def query_metrics(labels: Sequence[int | None], ideal: Sequence[int]) -> dict[str, float]:
    """The METRICS of one ranking; labels are in rank order and None counts as 0."""
    _check(labels, name="labels")
    _check(ideal, name="ideal")
    ideal_dcg = _dcg(ideal)
    return {
        "P@5": _precision(labels, 5),
        "P@10": _precision(labels, 10),
        "MRR": _reciprocal_rank(labels),
        "nDCG@10": _dcg(labels) / ideal_dcg if ideal_dcg else 0.0,
    }


def ideal_labels(pool_labels: Iterable[int]) -> list[int]:
    """The 10 largest pooled labels, descending."""
    return sorted(pool_labels, reverse=True)[:TOP_K]
