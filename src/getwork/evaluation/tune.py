"""B1+ weight tuning: the candidate grid, its rankings and the pick of the best candidate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa

from getwork.evaluation import metrics, pooling, scoring
from getwork.retrieval import b1plus
from getwork.retrieval.b1plus import B1PlusWeights
from getwork.retrieval.inputs import DIRECTIONS, RetrievalInputs

K = pooling.K
TUNE_SYSTEM = "B1+"
TUNE_STAGE = "tune"
RESULT_KEYS: tuple[str, ...] = ("skills", "title", "level", "ndcg10")


def candidate_weights(step: float) -> list[B1PlusWeights]:
    visited: list[B1PlusWeights] = []

    def record(weights: B1PlusWeights) -> float:
        visited.append(weights)
        return 0.0

    b1plus.grid_search_weights(record, step)
    return visited


def candidate_rankings(
    data: RetrievalInputs, *, resume_queries: Sequence[str], posting_queries: Sequence[str],
    pool_ids: Sequence[str], weights: Sequence[B1PlusWeights], k: int = K,
) -> list[pa.Table]:
    return [
        pooling.stage_rankings(
            data, stage=TUNE_STAGE, resume_queries=resume_queries,
            posting_queries=posting_queries, pool_ids=pool_ids,
            systems=[TUNE_SYSTEM], k=k, weights=w,
        )
        for w in weights
    ]


def _queries_by_direction(
    resume_queries: Sequence[str], posting_queries: Sequence[str]
) -> dict[str, Sequence[str]]:
    return dict(zip(DIRECTIONS, (resume_queries, posting_queries)))


def _tops(
    table: pa.Table, *, queries: Mapping[str, Sequence[str]],
    pair_ids: Mapping[tuple[str, str], str],
) -> dict[tuple[str, str], list[str]]:
    """(direction, query_id) -> pair ids of the top 10 of the tuned system."""
    out: dict[tuple[str, str], list[str]] = {}
    for direction, query_ids in queries.items():
        top = scoring.top_pair_ids(
            table, direction=direction, query_ids=query_ids, pair_ids=pair_ids
        ).get(TUNE_SYSTEM, {})
        out.update({(direction, q): ids for q, ids in top.items()})
    return out


def _check_labeled(
    tops: Sequence[Mapping[tuple[str, str], list[str]]], labels: Mapping[str, int]
) -> None:
    missing = len({p for top in tops for ids in top.values() for p in ids if p not in labels})
    if missing:
        raise scoring.MissingLabelsError(missing)


def _ideals(
    tops: Sequence[Mapping[tuple[str, str], list[str]]], labels: Mapping[str, int]
) -> dict[tuple[str, str], list[int]]:
    pooled: dict[tuple[str, str], set[str]] = {}
    for top in tops:
        for key, ids in top.items():
            pooled.setdefault(key, set()).update(ids)
    return {key: metrics.ideal_labels(labels[p] for p in ids) for key, ids in pooled.items()}


def _objective(
    top: Mapping[tuple[str, str], list[str]], ideals: Mapping[tuple[str, str], list[int]],
    labels: Mapping[str, int],
) -> float:
    values = [
        metrics.query_metrics([labels[p] for p in ids], ideals[key])["nDCG@10"]
        for key, ids in top.items()
    ]
    return sum(values) / len(values)


def pick_weights(
    *, tables: Sequence[pa.Table], candidates: Sequence[B1PlusWeights],
    pair_ids: Mapping[tuple[str, str], str], labels: Mapping[str, int],
    resume_queries: Sequence[str], posting_queries: Sequence[str],
) -> tuple[B1PlusWeights, list[dict]]:
    if not candidates or len(tables) != len(candidates):
        raise ValueError(f"{len(tables)} tables for {len(candidates)} candidates")
    if not resume_queries and not posting_queries:
        raise ValueError("pick_weights needs at least one query")
    queries = _queries_by_direction(resume_queries, posting_queries)
    tops = [_tops(table, queries=queries, pair_ids=pair_ids) for table in tables]
    _check_labeled(tops, labels)
    ideals = _ideals(tops, labels)
    objectives = [_objective(top, ideals, labels) for top in tops]
    best = max(range(len(candidates)), key=lambda i: (objectives[i], -i))
    rows = [
        {"skills": w.skills, "title": w.title, "level": w.level, "ndcg10": value}
        for w, value in zip(candidates, objectives)
    ]
    return candidates[best], rows
