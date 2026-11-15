from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from getwork.evaluation import stats

pytestmark = pytest.mark.usefixtures("offline")

A = [0, 0, 1, 1, 2, 2, 0, 1, 2, 2]
B = [0, 1, 1, 1, 2, 0, 0, 1, 2, 2]


def _confusion(a: list[int], b: list[int]) -> np.ndarray:
    m = np.zeros((3, 3))
    for x, y in zip(a, b):
        m[x, y] += 1
    return m / len(a)


def test_kappa_known_values() -> None:
    assert stats.cohen_kappa(A, A) == pytest.approx(1.0)
    m = _confusion(A, B)
    po = np.trace(m)
    pe = float(m.sum(axis=1) @ m.sum(axis=0))
    assert stats.cohen_kappa(A, B) == pytest.approx((po - pe) / (1 - pe), abs=1e-9)
    w = np.array([[((i - j) / 2) ** 2 for j in range(3)] for i in range(3)])
    e = np.outer(m.sum(axis=1), m.sum(axis=0))
    expected = 1 - (w * m).sum() / (w * e).sum()
    assert stats.cohen_kappa(A, B, weights="quadratic") == pytest.approx(expected, abs=1e-9)


def test_kappa_quadratic_penalizes_distance() -> None:
    base = [0, 0, 1, 1, 2, 2, 0, 1]
    near = [1, 0, 1, 1, 2, 2, 0, 1]  # one 0-vs-1 disagreement
    far = [2, 0, 1, 1, 2, 2, 0, 1]  # one 0-vs-2 disagreement
    assert stats.cohen_kappa(base, far, weights="quadratic") < stats.cohen_kappa(
        base, near, weights="quadratic"
    )


def test_kappa_undefined() -> None:
    assert stats.cohen_kappa([1, 1, 1], [1, 1, 1]) is None
    assert stats.cohen_kappa([], []) is None
    with pytest.raises(ValueError):
        stats.cohen_kappa([0, 1], [0])
    with pytest.raises(ValueError):
        stats.cohen_kappa([0, 3], [0, 1])
    with pytest.raises(ValueError):
        stats.cohen_kappa([0, 1], [0, 1], weights="linear")


def test_tau_three_systems() -> None:
    x = [1, 2, 3]
    for y, want in (([1, 2, 3], 1.0), ([3, 2, 1], -1.0), ([1, 3, 2], 1 / 3), ([2, 3, 1], -1 / 3)):
        assert stats.kendall_tau(x, y) == pytest.approx(want, abs=1e-12)


def test_tau_ties_and_constant() -> None:
    assert stats.kendall_tau([1, 1, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(2 * 3))
    assert stats.kendall_tau([1, 1, 1], [1, 2, 3]) is None
    assert stats.kendall_tau([1, 2, 3], [4, 4, 4]) is None


def test_bootstrap_ci() -> None:
    rng = np.random.default_rng(5)
    values = list(rng.random(50))
    mean, lo, hi = stats.bootstrap_ci(values, n_boot=500, seed=1)
    assert lo <= mean <= hi
    assert stats.bootstrap_ci(values, n_boot=500, seed=1) == (mean, lo, hi)
    assert stats.bootstrap_ci(values, n_boot=500, seed=2) != (mean, lo, hi)
    assert stats.bootstrap_ci([0.5] * 7, n_boot=100) == pytest.approx((0.5, 0.5, 0.5))
    small = stats.bootstrap_ci(list(rng.normal(size=25)), n_boot=500)
    large = stats.bootstrap_ci(list(rng.normal(size=400)), n_boot=500)
    assert large[2] - large[1] < small[2] - small[1]


def test_randomization_test_limits() -> None:
    assert stats.randomization_test([0.0] * 5) == 1.0
    assert stats.randomization_test([1.0] * 30, n_perm=2000) < 0.01
    d = [0.3, -0.1, 0.5, 0.2, -0.4, 0.7]
    assert stats.randomization_test(d, n_perm=500, seed=3) == stats.randomization_test(
        [-v for v in d], n_perm=500, seed=3
    )
    assert stats.randomization_test(d, n_perm=500, seed=3) == stats.randomization_test(
        d, n_perm=500, seed=3
    )
    assert stats.randomization_test([1.0] * 30, n_perm=100) >= 1 / 101
    with pytest.raises(ValueError):
        stats.randomization_test([])


def test_randomization_test_matches_exact() -> None:
    d = np.array([0.9, -0.2, 0.4, 0.7, -0.1, 0.5, 0.3, 0.8])
    observed = abs(d.mean())
    means = [abs((np.array(s) * d).mean()) for s in itertools.product([-1, 1], repeat=8)]
    exact = sum(m >= observed - 1e-12 for m in means) / len(means)
    assert stats.randomization_test(list(d), n_perm=20000, seed=0) == pytest.approx(exact, abs=0.02)


def test_tau_accepts_numpy_inputs() -> None:
    cases = [
        ([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1]), ([1, 2, 3], [1, 3, 2]),
        ([1, 2, 3], [2, 3, 1]), ([1, 1, 2], [1, 2, 3]),
    ]
    for x, y in cases:
        want = stats.kendall_tau(x, y)
        assert stats.kendall_tau(np.array(x, dtype=float), np.array(y, dtype=float)) == want
        assert stats.kendall_tau([np.float64(v) for v in x], y) == want
        assert stats.kendall_tau(x, np.array(y)) == want


def test_stats_edge_errors() -> None:
    assert stats.cohen_kappa([0, 0, 0], [1, 1, 1]) == 0.0
    with pytest.raises(ValueError):
        stats.kendall_tau([1, 2], [1])
    with pytest.raises(ValueError):
        stats.bootstrap_ci([])
