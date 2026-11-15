from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from getwork.evaluation import ari

pytestmark = pytest.mark.usefixtures("offline")


def _doc(*pairs: tuple[str, str]) -> dict:
    return {
        "skills": [s for s, _ in pairs],
        "skill_clusters": [{"skill": s, "cluster": c} for s, c in pairs],
    }


def test_cluster_profiles() -> None:
    docs = {
        "r1": _doc(("a", "X"), ("b", "X"), ("c", "Y"), ("z", "Other")),
        "r2": _doc(("d", "Y")),
    }
    got = ari.cluster_profiles(docs, ["Y", "X"])
    assert got.dtype == np.float32
    assert got.tolist() == [[1, 2], [1, 0]]
    with pytest.raises(ValueError, match="Nope"):
        ari.cluster_profiles({"r": _doc(("a", "Nope"))}, ["X"])


def test_skill_profiles() -> None:
    docs = {"r1": _doc(("a", "X"), ("b", "X")), "r2": _doc(("c", "Y"))}
    got = ari.skill_profiles(docs, ["c", "a", "A"])
    assert got.dtype == np.float32
    assert got.tolist() == [[0, 1, 0], [1, 0, 0]]


def test_ari_over_seeds() -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    groups = np.repeat(np.arange(3), 20)
    X = centers[groups] + rng.normal(size=(60, 2))
    labels = [f"g{g}" for g in groups]
    result = ari.ari_over_seeds(X, labels, k=3, seeds=[0, 1, 2])
    assert set(result) == {"dims", "ari_by_seed", "ari_mean", "ari_sd"}
    assert result["dims"] == 2
    assert result["ari_by_seed"] == [1.0, 1.0, 1.0]
    assert result == ari.ari_over_seeds(X, labels, k=3, seeds=[0, 1, 2])

    rng = np.random.default_rng(0)
    noise = rng.normal(size=(300, 4))
    random_labels = [str(v) for v in rng.integers(0, 3, size=300)]
    got = ari.ari_over_seeds(noise, random_labels, k=3, seeds=[0, 1])
    assert abs(got["ari_mean"]) < 0.1
    assert got["ari_sd"] == pytest.approx(float(np.std(got["ari_by_seed"])))


def test_vocabulary_gap() -> None:
    table = pa.table(
        {
            "skill": ["a", "b", "c", "d", "e"],
            "df_resumes": [3, 0, 2, 5, 0],
            "df_postings": [4, 2, 0, 1, 0],
            "selected": [True, True, True, True, False],
        }
    )
    assert ari.vocabulary_gap(table) == {
        "n_retained": 4,
        "n_one_corpus": 2,
        "fraction_one_corpus": 0.5,
    }
