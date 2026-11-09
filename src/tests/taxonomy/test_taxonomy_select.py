import pytest

from getwork.taxonomy import grid, select

pytestmark = pytest.mark.usefixtures("offline")


def _result(
    config_id: str,
    silhouette: float | None = 0.5,
    noise: float = 0.1,
    n_clusters: int = 4,
    skipped: str | None = None,
) -> grid.ConfigResult:
    return grid.ConfigResult(
        config_id=config_id,
        method="kmeans",
        params={"k": n_clusters},
        n_clusters=n_clusters,
        noise_fraction=noise,
        silhouette=silhouette,
        largest_cluster_fraction=0.4,
        skipped=skipped,
    )


def test_eligibility_rules() -> None:
    results = [
        _result("ok", 0.3),
        _result("noisy", 0.9, noise=0.31),
        _result("undefined", None),
        _result("skipped", 0.9, skipped="needs fewer clusters than skills"),
        _result("one-cluster", 0.9, n_clusters=1),
        _result("at-cap", 0.2, noise=0.30),
        _result("exact-min", 0.1, n_clusters=2),
    ]
    kept = select.eligible(results, max_noise=0.30, min_clusters=2)
    assert [r.config_id for r in kept] == ["ok", "at-cap", "exact-min"]
    chosen = select.choose(results, max_noise=0.30, min_clusters=2)
    assert chosen is not None and chosen.config_id == "ok"


def test_choose_tie_breaks_and_none() -> None:
    tie = [
        _result("c", 0.5, noise=0.2),
        _result("b", 0.5 + 1e-12, noise=0.1),
        _result("a", 0.5, noise=0.1),
        _result("low", 0.4, noise=0.0),
    ]
    assert select.choose(tie, 0.3, 2).config_id == "a"
    noise_decides = [_result("a", 0.5, noise=0.2), _result("z", 0.5, noise=0.05)]
    assert select.choose(noise_decides, 0.3, 2).config_id == "z"
    id_decides = [_result("b", 0.5, noise=0.1), _result("a", 0.5, noise=0.1)]
    assert select.choose(id_decides, 0.3, 2).config_id == "a"
    assert select.choose([_result("x", None), _result("y", 0.5, noise=0.9)], 0.3, 2) is None
    assert select.choose([], 0.3, 2) is None


def test_selection_record_keys() -> None:
    record = select.selection_record(
        _result("kmeans-01", 0.25), source="auto", context={"seed": 0}, n_eligible=3
    )
    assert record == {
        "config_id": "kmeans-01",
        "method": "kmeans",
        "params": {"k": 4},
        "n_clusters": 4,
        "noise_fraction": 0.1,
        "silhouette": 0.25,
        "largest_cluster_fraction": 0.4,
        "source": "auto",
        "n_eligible": 3,
        "context": {"seed": 0},
    }
    manual = select.selection_record(
        _result("kmeans-01"), source="manual", context={}, n_eligible=None
    )
    assert manual["source"] == "manual" and manual["n_eligible"] is None
