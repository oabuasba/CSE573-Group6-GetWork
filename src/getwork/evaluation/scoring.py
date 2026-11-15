"""Per-query metrics from rankings and labels, and their aggregation into a result group."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from getwork.evaluation import metrics, pairs, stats
from getwork.retrieval.inputs import SYSTEM_IDS

TOP_K = metrics.TOP_K
GAPS: tuple[tuple[str, str], ...] = (("B2", "B1+"), ("Full", "B2"), ("Full", "B1+"))


class MissingLabelsError(ValueError):
    """Top-10 pairs without a label; count says how many."""

    def __init__(self, count: int) -> None:
        super().__init__(f"{count} top-10 pairs have no label")
        self.count = count


@dataclass(frozen=True)
class QueryScores:
    scores: dict[str, dict[str, dict[str, float]]]  # system -> query_id -> metric -> value
    n_missing: int
    n_no_relevant: int


def top_pair_ids(
    rankings: pa.Table, *, direction: str, query_ids: Sequence[str],
    pair_ids: Mapping[tuple[str, str], str],
) -> dict[str, dict[str, list[str]]]:
    """system -> query_id -> pair ids of the first 10 ranks, in rank order."""
    wanted = set(query_ids)
    columns = ("system", "direction", "query_id", "rank", "doc_id")
    rows = zip(*(rankings.column(name).to_pylist() for name in columns))
    ranked: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for system, row_direction, query, rank, doc in rows:
        if row_direction != direction or query not in wanted or rank > TOP_K:
            continue
        pair_id = pair_ids[pairs.pair_key(direction, query, doc)]
        ranked.setdefault(system, {}).setdefault(query, []).append((rank, pair_id))
    return {
        system: {q: [p for _, p in sorted(items)] for q, items in by_query.items()}
        for system, by_query in ranked.items()
    }


def _require_all(
    top: Mapping[str, Mapping[str, list[str]]], systems: Sequence[str], query_ids: Sequence[str]
) -> None:
    for system in systems:
        for query in query_ids:
            if query not in top.get(system, {}):
                raise ValueError(f"query {query!r} has no ranking for system {system!r}")


def _pooled(top: Mapping[str, Mapping[str, list[str]]], query: str) -> set[str]:
    return {p for by_query in top.values() for p in by_query.get(query, [])}


def _score_one_query(
    top: Mapping[str, Mapping[str, list[str]]], query: str, labels: Mapping[str, int]
) -> tuple[dict[str, dict[str, float]], bool]:
    ideal = metrics.ideal_labels(labels[p] for p in _pooled(top, query) if p in labels)
    scored = {
        system: metrics.query_metrics([labels.get(p) for p in by_query[query]], ideal)
        for system, by_query in top.items()
    }
    return scored, not any(ideal)


def score_queries(
    *, rankings: pa.Table, direction: str, query_ids: Sequence[str],
    pair_ids: Mapping[tuple[str, str], str], labels: Mapping[str, int],
    allow_missing: bool = False,
) -> QueryScores:
    top = top_pair_ids(rankings, direction=direction, query_ids=query_ids, pair_ids=pair_ids)
    _require_all(top, SYSTEM_IDS, query_ids)
    n_missing = sum(
        len({p for p in _pooled(top, query) if p not in labels}) for query in query_ids
    )
    if n_missing and not allow_missing:
        raise MissingLabelsError(n_missing)
    scores: dict[str, dict[str, dict[str, float]]] = {system: {} for system in SYSTEM_IDS}
    n_no_relevant = 0
    for query in query_ids:
        scored, no_relevant = _score_one_query(top, query, labels)
        n_no_relevant += no_relevant
        for system in SYSTEM_IDS:
            scores[system][query] = scored[system]
    return QueryScores(scores=scores, n_missing=n_missing, n_no_relevant=n_no_relevant)


def _vector(scores: QueryScores, system: str, metric: str) -> np.ndarray:
    by_query = scores.scores[system]
    return np.array([by_query[q][metric] for q in sorted(by_query)], dtype=float)


def _mean_ci(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float, list[float]]:
    mean, lower, upper = stats.bootstrap_ci(values, n_boot=n_boot, seed=seed)
    return mean, [lower, upper]


def _gap(first: np.ndarray, second: np.ndarray, *, n_boot: int, n_perm: int, seed: int) -> dict:
    diffs = first - second
    _, ci = _mean_ci(diffs, n_boot=n_boot, seed=seed)
    p = stats.randomization_test(diffs, n_perm=n_perm, seed=seed)
    return {"diff": float(diffs.mean()), "ci": ci, "p": p}


def _order(means: Mapping[str, float]) -> list[str]:
    return sorted(SYSTEM_IDS, key=lambda s: (-means[s], SYSTEM_IDS.index(s)))


def aggregate_group(scores: QueryScores, *, n_boot: int, n_perm: int, seed: int) -> dict:
    vectors = {s: {m: _vector(scores, s, m) for m in metrics.METRICS} for s in SYSTEM_IDS}
    systems: dict[str, dict] = {s: {} for s in SYSTEM_IDS}
    for system in SYSTEM_IDS:
        for metric in metrics.METRICS:
            mean, ci = _mean_ci(vectors[system][metric], n_boot=n_boot, seed=seed)
            systems[system][metric] = {"mean": mean, "ci": ci}
    gaps = {
        f"{a}-{b}": {
            m: _gap(vectors[a][m], vectors[b][m], n_boot=n_boot, n_perm=n_perm, seed=seed)
            for m in metrics.METRICS
        }
        for a, b in GAPS
    }
    order = {m: _order({s: systems[s][m]["mean"] for s in SYSTEM_IDS}) for m in metrics.METRICS}
    n_queries = len(scores.scores[SYSTEM_IDS[0]])
    return {"n_queries": n_queries, "systems": systems, "gaps": gaps, "order": order}
