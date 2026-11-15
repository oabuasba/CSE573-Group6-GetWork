from __future__ import annotations

import math

import pytest

from getwork.evaluation import metrics

pytestmark = pytest.mark.usefixtures("offline")

LABELS = [2, 0, 1, 2, 0, 0, 0, 0, 0, 0]
IDEAL = [2, 2, 1, 0, 0, 0, 0, 0, 0, 0]


def test_query_metrics_worked_example() -> None:
    got = metrics.query_metrics(LABELS, IDEAL)
    assert tuple(got) == metrics.METRICS
    assert got["P@5"] == pytest.approx(0.4)
    assert got["P@10"] == pytest.approx(0.2)
    assert got["MRR"] == 1.0
    assert got["nDCG@10"] == pytest.approx(0.8935, abs=1e-4)
    dcg = 2 / math.log2(2) + 1 / math.log2(4) + 2 / math.log2(5)
    idcg = 2 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert got["nDCG@10"] == pytest.approx(dcg / idcg, abs=1e-12)


def test_mrr_and_precision_edges() -> None:
    third = metrics.query_metrics([0, 1, 2], IDEAL)
    assert third["MRR"] == pytest.approx(1 / 3)
    assert third["P@10"] == pytest.approx(0.1)
    none = metrics.query_metrics([0, 1, 1, 0, 0], IDEAL)
    assert (none["MRR"], none["P@5"], none["P@10"]) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        metrics.query_metrics([0] * 11, IDEAL)
    with pytest.raises(ValueError):
        metrics.query_metrics([3], IDEAL)


def test_partial_labels_count_only_in_ndcg() -> None:
    with_none = metrics.query_metrics([None, 2], IDEAL)
    with_zero = metrics.query_metrics([0, 2], IDEAL)
    assert with_none == with_zero
    ones = metrics.query_metrics([1, 1, 1], IDEAL)
    assert (ones["P@5"], ones["MRR"]) == (0.0, 0.0)
    assert ones["nDCG@10"] > 0


def test_ideal_labels() -> None:
    assert metrics.ideal_labels([0, 2, 1, 2, 0] * 4) == [2] * 8 + [1] * 2
    assert metrics.ideal_labels([1, 2, 0]) == [2, 1, 0]
    assert metrics.ideal_labels([]) == []


def test_ndcg_zero_ideal_and_perfect() -> None:
    assert metrics.query_metrics([1, 2], [0] * 10)["nDCG@10"] == 0.0
    pool = [0, 2, 1, 1, 2, 0, 0, 1, 2, 0, 1, 0]
    ideal = metrics.ideal_labels(pool)
    assert metrics.query_metrics(ideal, ideal)["nDCG@10"] == pytest.approx(1.0)
