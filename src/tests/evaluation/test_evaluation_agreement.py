from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from getwork.evaluation import agreement, stats
from getwork.retrieval.inputs import DIRECTIONS

pytestmark = pytest.mark.usefixtures("offline")


def test_agreement_stats() -> None:
    a = {"x1": 0, "x2": 1, "x3": 2, "x4": 2, "only_a": 1}
    b = {"x1": 0, "x2": 2, "x3": 2, "x4": 1, "only_b": 0}
    result = agreement.agreement_stats(a, b)
    assert result["n"] == 4 and result["pct_agree"] == 0.5
    shared = ["x1", "x2", "x3", "x4"]
    xs, ys = [a[i] for i in shared], [b[i] for i in shared]
    assert result["kappa"] == stats.cohen_kappa(xs, ys)
    assert result["kappa_quadratic"] == stats.cohen_kappa(xs, ys, weights="quadratic")
    assert result["confusion"] == [[1, 0, 0], [0, 0, 1], [0, 1, 1]]
    assert [sum(row) for row in result["confusion"]] == [1, 1, 2]
    empty = agreement.agreement_stats({"a": 1}, {"b": 1})
    assert empty["n"] == 0
    assert empty["kappa"] is None and empty["kappa_quadratic"] is None
    assert empty["pct_agree"] is None
    numpy_ids = agreement.agreement_stats({"a": np.int64(2)}, {"a": np.int64(2)})
    assert numpy_ids["n"] == 1


def test_fallback_decision() -> None:
    decide = agreement.fallback_decision
    assert decide(0.5, 0.6) == "judges_reliable"
    assert decide(0.9, 0.8) == "judges_reliable"
    assert decide(0.7, 0.59) == "revise_once"
    assert decide(0.5, 0.4) == "subjective"
    assert decide(None, 0.9) == "undetermined" and decide(0.9, None) == "undetermined"
    assert decide(0.5, 0.45, threshold=0.4) == "judges_reliable"
    assert decide(0.35, 0.3, threshold=0.4) == "subjective"


def sets_for(n_resumes: int, n_postings: int) -> dict:
    return {"stages": {"human": {
        "resumes": [f"r{i:02d}" for i in reversed(range(n_resumes))],
        "postings": [f"p{i:02d}" for i in range(n_postings)],
    }}}


def test_split_halves() -> None:
    for n, sizes in ((10, (5, 5)), (5, (3, 2)), (1, (1, 0))):
        halves = agreement.split_halves(sets_for(n, n))
        for side in ("resumes", "postings"):
            assert (len(halves["first"][side]), len(halves["second"][side])) == sizes
        assert halves["first"]["resumes"] + halves["second"]["resumes"] == sorted(
            sets_for(n, n)["stages"]["human"]["resumes"]
        )


def test_half_pair_ids_second_is_untouched() -> None:
    halves = {"first": {"resumes": ["r1"], "postings": ["q1"]},
              "second": {"resumes": ["r2", "r3"], "postings": ["q2"]}}  # fmt: skip
    rows = [
        ("resume_to_posting", "r1", "q2"), ("resume_to_posting", "r2", "q3"),
        ("resume_to_posting", "r3", "q2"), ("resume_to_posting", "other", "q9"),
        ("posting_to_resume", "q2", "r1"), ("posting_to_resume", "q2", "r2"),
        ("posting_to_resume", "q1", "r2"), ("posting_to_resume", "nobody", "r3"),
    ]
    cols = list(zip(*rows))
    table = pa.table({
        "direction": cols[0], "query_id": cols[1], "doc_id": cols[2],
    })
    pair_ids = {}
    for direction, query, doc in rows:
        key = (query, doc) if direction == "resume_to_posting" else (doc, query)
        pair_ids[key] = f"{key[0]}|{key[1]}"
    result = agreement.half_pair_ids(table, halves, pair_ids)
    assert result["first"] == {"r1|q2", "r2|q1"}
    assert result["second"] == {"r2|q3", "r3|q2", "r2|q2"}
    assert not result["first"] & result["second"]


def _group(means: dict[str, tuple[float, float, float]], gap: float) -> dict:
    def direction(values: tuple[float, float, float]) -> dict:
        return {
            "systems": {s: {"nDCG@10": {"mean": v}, "MRR": {"mean": 0.0}}
                        for s, v in zip(("B1+", "B2", "Full"), values)},
            "gaps": {name: {"nDCG@10": {"diff": gap}, "MRR": {"diff": 9.0}}
                     for name in ("B2-B1+", "Full-B2", "Full-B1+")},
        }

    return {"directions": {d: direction(means[d]) for d in DIRECTIONS}}


def test_tau_blocks() -> None:
    full_b2_b1 = (0.1, 0.2, 0.3)
    full_b1_b2 = (0.2, 0.1, 0.3)
    groups = {
        "judge@human": _group({d: full_b2_b1 for d in DIRECTIONS}, 0.1),
        "a@human": _group({d: full_b2_b1 for d in DIRECTIONS}, 0.2),
        "b@human": _group({d: full_b1_b2 for d in DIRECTIONS}, 0.3),
    }
    result = agreement.tau_blocks({"groups": groups}, ["a", "b"])
    for direction in DIRECTIONS:
        taus = result["nDCG@10"][direction]
        assert taus["judge-vs-a"] == 1.0
        assert taus["judge-vs-b"] == pytest.approx(1 / 3)
        assert taus["a-vs-b"] == pytest.approx(1 / 3)
        assert result["gaps"]["b@human"][direction]["Full-B2"] == 0.3
    assert set(result["gaps"]) == {"judge@human", "a@human", "b@human"}
    one = agreement.tau_blocks({"groups": groups}, ["a"])
    assert set(one["nDCG@10"]["resume_to_posting"]) == {"judge-vs-a"}
    assert agreement.tau_blocks({"groups": groups}, ["a", "zed"]) is None
    assert agreement.tau_blocks({"groups": {"a@human": groups["a@human"]}}, ["a"]) is None
    assert agreement.tau_blocks({"groups": {}}, []) is None
