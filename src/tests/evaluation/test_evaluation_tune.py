from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest
from evaluation_helpers import make_eval_tree

from getwork.evaluation import loading, pooling, scoring, tune
from getwork.evaluation.pooling import RANKINGS_SCHEMA
from getwork.retrieval import b1plus
from getwork.retrieval.b1plus import B1PlusWeights

pytestmark = pytest.mark.usefixtures("offline")

RESUME_QUERIES = ["r1", "r2"]
POSTING_QUERIES = ["q1"]
PAIR_IDS = {
    **{(r, f"d{i}"): f"{r}-d{i}" for r in RESUME_QUERIES for i in range(1, 5)},
    **{(f"d{i}", q): f"{q}-d{i}" for q in POSTING_QUERIES for i in range(1, 5)},
}


def test_candidate_weights() -> None:
    grid = tune.candidate_weights(0.1)
    assert len(grid) == 66 and len(set(grid)) == 66
    assert grid[0] == B1PlusWeights(1.0, 0.0, 0.0)
    assert all(abs(w.skills + w.title + w.level - 1) < 1e-9 for w in grid)
    seen: list[B1PlusWeights] = []
    b1plus.grid_search_weights(lambda w: seen.append(w) or 0.0, 0.1)
    assert grid == seen
    with pytest.raises(ValueError):
        tune.candidate_weights(0.3)


def test_candidate_rankings_shape(tmp_path: Path) -> None:
    root, processed = make_eval_tree(tmp_path, n_resumes=14, n_postings=12)
    data = loading.load_eval_inputs(root, processed).data
    resumes, postings = ["r-1", "r-2", "r-3"], ["p-1", "p-2"]
    pool = ["p-3", "p-4", "p-5", "p-6"]
    weights = [B1PlusWeights(1.0, 0.0, 0.0), B1PlusWeights(0.2, 0.4, 0.4)]
    tables = tune.candidate_rankings(
        data, resume_queries=resumes, posting_queries=postings, pool_ids=pool, weights=weights
    )
    assert len(tables) == 2
    for table, w in zip(tables, weights):
        assert table.schema == RANKINGS_SCHEMA
        assert set(table.column("system").to_pylist()) == {"B1+"}
        assert set(table.column("stage").to_pylist()) == {"tune"}
        rows = table.to_pylist()
        r2p = [r for r in rows if r["direction"] == "resume_to_posting"]
        p2r = [r for r in rows if r["direction"] == "posting_to_resume"]
        assert {r["query_id"] for r in r2p} == set(resumes)
        assert {r["doc_id"] for r in r2p} <= set(pool)
        assert {r["query_id"] for r in p2r} == set(postings)
        expected = pooling.stage_rankings(
            data, stage="tune", resume_queries=resumes, posting_queries=postings,
            pool_ids=pool, systems=["B1+"], weights=w,
        )
        assert table.equals(expected)


def table_from(top: dict[tuple[str, str], list[str]]) -> pa.Table:
    rows = [
        ("tune", "B1+", direction, query, rank, doc, 1.0)
        for (direction, query), docs in top.items()
        for rank, doc in enumerate(docs, start=1)
    ]
    cols = list(zip(*rows))
    names = [f.name for f in RANKINGS_SCHEMA]
    return pa.table(dict(zip(names, cols))).cast(RANKINGS_SCHEMA)


def make_top(order: list[str]) -> dict[tuple[str, str], list[str]]:
    top = {("resume_to_posting", r): list(order) for r in RESUME_QUERIES}
    top[("posting_to_resume", "q1")] = list(order)
    return top


LABELS = {
    **{f"{r}-d{i}": label for r in RESUME_QUERIES for i, label in zip(range(1, 5), (2, 1, 0, 2))},
    **{f"q1-d{i}": label for i, label in zip(range(1, 5), (2, 1, 0, 2))},
}
GOOD = ["d1", "d4", "d2", "d3"]
BAD = ["d3", "d2", "d4", "d1"]
MID = ["d1", "d2", "d3", "d4"]


def pick(tables, candidates, labels=LABELS):
    return tune.pick_weights(
        tables=tables, candidates=candidates, pair_ids=PAIR_IDS, labels=labels,
        resume_queries=RESUME_QUERIES, posting_queries=POSTING_QUERIES,
    )


def independent_ndcg(order: list[str]) -> float:
    import math

    gains = {"d1": 2, "d2": 1, "d3": 0, "d4": 2}
    dcg = sum(gains[d] / math.log2(i + 2) for i, d in enumerate(order))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gains.values(), reverse=True)))
    return dcg / ideal


def test_pick_weights_best_and_ties() -> None:
    grid = tune.candidate_weights(0.1)
    orders = {i: BAD for i in range(len(grid))}
    orders[40], orders[10] = GOOD, MID
    tables = [table_from(make_top(orders[i])) for i in range(len(grid))]
    best, rows = pick(tables, grid)
    assert best == grid[40] and len(rows) == 66
    assert rows[40] == {
        "skills": grid[40].skills, "title": grid[40].title, "level": grid[40].level,
        "ndcg10": pytest.approx(independent_ndcg(GOOD)),
    }
    assert rows[10]["ndcg10"] == pytest.approx(independent_ndcg(MID))
    assert rows[0]["ndcg10"] == pytest.approx(independent_ndcg(BAD))
    assert [(r["skills"], r["title"], r["level"]) for r in rows] == [
        (w.skills, w.title, w.level) for w in grid
    ]
    orders[5] = GOOD
    tied, _ = pick([table_from(make_top(orders[i])) for i in range(len(grid))], grid)
    assert tied == grid[5]


def test_pick_weights_missing_labels() -> None:
    candidates = [B1PlusWeights(1.0, 0.0, 0.0), B1PlusWeights(0.0, 1.0, 0.0)]
    tables = [table_from(make_top(GOOD)), table_from(make_top(BAD))]
    labels = {k: v for k, v in LABELS.items() if k not in ("r1-d3", "q1-d3", "r2-d1")}
    with pytest.raises(scoring.MissingLabelsError) as info:
        pick(tables, candidates, labels)
    assert info.value.count == 3
