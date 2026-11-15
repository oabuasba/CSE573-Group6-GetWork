from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from getwork.evaluation import metrics, scoring, stats
from getwork.evaluation.pooling import RANKINGS_SCHEMA
from getwork.retrieval.inputs import SYSTEM_IDS

pytestmark = pytest.mark.usefixtures("offline")

DIRECTION = "resume_to_posting"
LABELS = {"p1": 2, "p2": 0, "p3": 1, "p4": 2, "p5": 0, "p6": 1}
PAIR_IDS = {(f"r{q}", f"d{i}"): f"p{i}" for q in (1, 2) for i in range(1, 7)}
TOP = {
    "B1+": {"r1": ["d1", "d2", "d3"], "r2": ["d2", "d1"]},
    "B2": {"r1": ["d3", "d1"], "r2": ["d4", "d5", "d6"]},
    "Full": {"r1": ["d1", "d2", "d3"], "r2": ["d6", "d4"]},
}


def make_rankings(top: dict = TOP, extra: list[tuple] | None = None) -> pa.Table:
    rows = [
        (system, DIRECTION, query, rank, doc, 1.0 / rank)
        for system, by_query in top.items()
        for query, docs in by_query.items()
        for rank, doc in enumerate(docs, start=1)
    ]
    rows += extra or []
    cols = list(zip(*rows))
    names = [f.name for f in RANKINGS_SCHEMA]
    data = {"stage": ["test"] * len(rows), **dict(zip(names[1:], cols))}
    return pa.table(data).cast(RANKINGS_SCHEMA)


def score(**kwargs):
    args = {
        "rankings": make_rankings(), "direction": DIRECTION, "query_ids": ["r1", "r2"],
        "pair_ids": PAIR_IDS, "labels": LABELS,
    }
    return scoring.score_queries(**{**args, **kwargs})


def test_score_queries_matches_query_metrics() -> None:
    result = score()
    ideal_r1 = metrics.ideal_labels([2, 0, 1])
    ideal_r2 = metrics.ideal_labels([0, 2, 0, 1, 2])
    for system in SYSTEM_IDS:
        for query, ideal in (("r1", ideal_r1), ("r2", ideal_r2)):
            docs = TOP[system][query]
            expected = metrics.query_metrics([LABELS["p" + d[1:]] for d in docs], ideal)
            assert result.scores[system][query] == expected
    assert result.n_missing == 0 and result.n_no_relevant == 0


def test_rank_order_and_first_ten_only() -> None:
    top = {s: {"r1": [f"d{i}" for i in range(1, 7)]} for s in SYSTEM_IDS}
    shuffled = make_rankings(top).take([5, 4, 3, 2, 1, 0] + list(range(6, 18)))
    result = score(rankings=shuffled, query_ids=["r1"])
    assert result.scores["B1+"]["r1"] == metrics.query_metrics(
        [LABELS[f"p{i}"] for i in range(1, 7)], metrics.ideal_labels(LABELS.values())
    )
    deep = [("B1+", DIRECTION, "r1", 11, "d99", 0.0)]
    pair_ids = {**PAIR_IDS, ("r1", "d99"): "p99"}
    again = score(rankings=make_rankings(top, deep), query_ids=["r1"], pair_ids=pair_ids)
    assert again.scores == result.scores


def test_missing_labels_error_and_allow_missing() -> None:
    labels = {k: v for k, v in LABELS.items() if k not in ("p3", "p6")}
    with pytest.raises(scoring.MissingLabelsError) as info:
        score(labels=labels)
    assert info.value.count == 2 and isinstance(info.value, ValueError)
    result = score(labels=labels, allow_missing=True)
    assert result.n_missing == 2
    ideal = metrics.ideal_labels([2, 0, 0, 2])
    expected_r1 = metrics.query_metrics([None, 2], metrics.ideal_labels([2, 0]))
    assert result.scores["B2"]["r1"] == expected_r1
    assert result.scores["B2"]["r2"] == metrics.query_metrics([2, 0, None], ideal)


def test_n_no_relevant_and_errors() -> None:
    zero = {k: 0 for k in LABELS}
    assert score(labels=zero).n_no_relevant == 2
    assert score(labels={**zero, "p4": 1}).n_no_relevant == 1
    with pytest.raises(ValueError, match="r3"):
        score(query_ids=["r1", "r3"])
    with pytest.raises(KeyError):
        score(pair_ids={})
    partial = {s: by for s, by in TOP.items() if s != "Full"}
    with pytest.raises(ValueError, match="Full"):
        score(rankings=make_rankings(partial))


def make_scores(values: dict[str, list[float]]) -> scoring.QueryScores:
    scores = {
        system: {
            f"q{i:03d}": {m: v for m in metrics.METRICS} for i, v in enumerate(vector)
        }
        for system, vector in values.items()
    }
    return scoring.QueryScores(scores=scores, n_missing=0, n_no_relevant=0)


def test_aggregate_group_values() -> None:
    rng = np.random.default_rng(3)
    values = {s: list(rng.random(12)) for s in SYSTEM_IDS}
    group = scoring.aggregate_group(make_scores(values), n_boot=300, n_perm=300, seed=7)
    assert group["n_queries"] == 12
    for system in SYSTEM_IDS:
        mean, lo, hi = stats.bootstrap_ci(values[system], n_boot=300, seed=7)
        cell = group["systems"][system]["nDCG@10"]
        assert cell["mean"] == mean and cell["ci"] == [lo, hi]
    assert list(group["gaps"]) == ["B2-B1+", "Full-B2", "Full-B1+"]
    for name, (a, b) in {"B2-B1+": ("B2", "B1+"), "Full-B2": ("Full", "B2"),
                         "Full-B1+": ("Full", "B1+")}.items():  # fmt: skip
        diffs = np.array(values[a]) - np.array(values[b])
        gap = group["gaps"][name]["MRR"]
        _, lo, hi = stats.bootstrap_ci(diffs, n_boot=300, seed=7)
        assert gap["diff"] == pytest.approx(np.mean(values[a]) - np.mean(values[b]))
        assert gap["ci"] == [lo, hi]
        assert gap["p"] == stats.randomization_test(diffs, n_perm=300, seed=7)
        assert list(group["gaps"][name]) == list(metrics.METRICS)
    means = {s: group["systems"][s]["P@5"]["mean"] for s in SYSTEM_IDS}
    assert group["order"]["P@5"] == sorted(SYSTEM_IDS, key=lambda s: -means[s])


def test_aggregate_group_ties() -> None:
    same = {s: [0.2, 0.5, 0.9] for s in SYSTEM_IDS}
    group = scoring.aggregate_group(make_scores(same), n_boot=100, n_perm=100, seed=1)
    for metric in metrics.METRICS:
        assert group["order"][metric] == ["B1+", "B2", "Full"]
        assert all(group["gaps"][g][metric]["p"] == 1.0 for g in group["gaps"])
    partial = {"B1+": [0.1, 0.1], "B2": [0.5, 0.5], "Full": [0.5, 0.5]}
    group = scoring.aggregate_group(make_scores(partial), n_boot=50, n_perm=50, seed=1)
    assert group["order"]["MRR"] == ["B2", "Full", "B1+"]
