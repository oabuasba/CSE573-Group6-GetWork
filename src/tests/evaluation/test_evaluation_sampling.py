from __future__ import annotations

import random

import pytest

from getwork.evaluation import sampling

pytestmark = pytest.mark.usefixtures("offline")

LABELS_20_20_5 = (
    {f"a-{i}": "A" for i in range(20)}
    | {f"b-{i}": "B" for i in range(20)}
    | {f"c-{i}": "C" for i in range(5)}
)


def _per_label(ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for id_ in ids:
        counts[id_[0]] = counts.get(id_[0], 0) + 1
    return counts


def test_sample_pool_under_cap() -> None:
    assert sampling.sample_pool(["p-3", "p-1", "p-2"], cap=3, seed=0) == ["p-1", "p-2", "p-3"]
    assert sampling.sample_pool(["p-2", "p-1"], cap=10, seed=5) == ["p-1", "p-2"]


def test_sample_pool_over_cap_is_seeded() -> None:
    ids = [f"p-{i}" for i in range(100)]
    pool = sampling.sample_pool(ids, cap=30, seed=1)
    assert len(pool) == len(set(pool)) == 30 and set(pool) <= set(ids) and pool == sorted(pool)
    shuffled = ids[:]
    random.Random(9).shuffle(shuffled)
    assert sampling.sample_pool(shuffled, cap=30, seed=1) == pool
    assert sampling.sample_pool(ids, cap=30, seed=2) != pool


def test_stratified_quotas() -> None:
    nine = sampling.stratified_sample(LABELS_20_20_5, 9, seed=0, purpose="x")
    assert _per_label(nine) == {"a": 3, "b": 3, "c": 3}
    ten = sampling.stratified_sample(LABELS_20_20_5, 10, seed=0, purpose="x")
    assert _per_label(ten) == {"a": 4, "b": 3, "c": 3}


def test_stratified_shortfall_redistributed() -> None:
    fifteen = sampling.stratified_sample(LABELS_20_20_5, 15, seed=0, purpose="x")
    assert _per_label(fifteen) == {"a": 5, "b": 5, "c": 5}
    twenty_five = sampling.stratified_sample(LABELS_20_20_5, 25, seed=0, purpose="x")
    assert _per_label(twenty_five) == {"a": 11, "b": 9, "c": 5}


def test_stratified_ignores_unlabeled_and_is_deterministic() -> None:
    labels = LABELS_20_20_5 | {"u-1": None, "u-2": "", "u-3": None}
    first = sampling.stratified_sample(labels, 12, seed=3, purpose="x")
    assert not any(id_.startswith("u-") for id_ in first)
    assert first == sampling.stratified_sample(labels, 12, seed=3, purpose="x")
    assert first != sampling.stratified_sample(labels, 12, seed=4, purpose="x")
    assert first == sorted(first)
    with pytest.raises(ValueError, match="46"):
        sampling.stratified_sample(labels, 46, seed=3, purpose="x")


def test_sample_ids_too_many() -> None:
    with pytest.raises(ValueError, match="5"):
        sampling.sample_ids(["p-1", "p-2"], 5, seed=0, purpose="x")
    assert sampling.sample_ids(["p-2", "p-1"], 2, seed=0, purpose="x") == ["p-1", "p-2"]


def _inputs(n_resumes: int = 60, n_postings: int = 50) -> dict:
    names = ("Data", "Cloud", "Web")
    labels = {
        f"r-{i}": (None if i % 10 == 9 else names[i % 3]) for i in range(n_resumes)
    }
    return {"resume_labels": labels, "posting_ids": [f"p-{i}" for i in range(n_postings)]}


def test_build_sets_invariants() -> None:
    params = sampling.SampleParams(
        seed=2, pool_cap=30, dev_resumes=6, dev_postings=5, test_resumes=12,
        test_postings=10, human_resumes=3, human_postings=4,
    )
    inputs = _inputs()
    sets = sampling.build_sets(**inputs, params=params)
    stages, pool = sets["stages"], set(sets["pool"])
    assert sets["n_pool"] == len(pool) == 30
    assert sets["n_resumes_total"] == 60 and sets["n_resumes_unlabeled"] == 6
    assert sets["n_resumes_labeled"] == 54 and sets["n_postings_total"] == 50
    for side in ("resumes", "postings"):
        assert not set(stages["dev"][side]) & set(stages["test"][side])
        assert set(stages["human"][side]) <= set(stages["test"][side])
    for stage in stages.values():
        assert set(stage["postings"]) <= pool
        assert all(inputs["resume_labels"][id_] for id_ in stage["resumes"])
    sizes = {s: {k: len(v) for k, v in sides.items()} for s, sides in stages.items()}
    assert sizes == {
        "dev": {"resumes": 6, "postings": 5},
        "test": {"resumes": 12, "postings": 10},
        "human": {"resumes": 3, "postings": 4},
    }
    assert sets["seed"] == 2 and sets["schema"] == 1


def test_dev_unchanged_by_test_size() -> None:
    small = sampling.SampleParams(pool_cap=30, test_resumes=10, test_postings=8,
                                    human_postings=5)
    large = sampling.SampleParams(pool_cap=30, test_resumes=20, test_postings=15,
                                    human_postings=5)
    a = sampling.build_sets(**_inputs(), params=small)["stages"]["dev"]
    b = sampling.build_sets(**_inputs(), params=large)["stages"]["dev"]
    assert a == b


def test_build_sets_too_small() -> None:
    with pytest.raises(ValueError, match=r"54.*10.*100"):
        sampling.build_sets(**_inputs(), params=sampling.SampleParams(pool_cap=30))
    with pytest.raises(ValueError, match=r"30.*10.*100"):
        sampling.build_sets(
            **_inputs(), params=sampling.SampleParams(pool_cap=30, test_resumes=10)
        )
