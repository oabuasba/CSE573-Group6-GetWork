"""Agreement and significance statistics: kappa, Kendall tau-b, bootstrap CI, sign-flip test."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

CHUNK_ROWS = 1000
CHUNK_THRESHOLD = 5_000_000
WEIGHT_CHOICES: tuple[str | None, ...] = (None, "quadratic")
TIE_TOLERANCE = 1e-12


def cohen_kappa(
    a: Sequence[int],
    b: Sequence[int],
    *,
    weights: str | None = None,
    categories: Sequence[int] = (0, 1, 2),
) -> float | None:
    """Cohen's kappa; None when empty or when chance agreement leaves nothing to explain."""
    if weights not in WEIGHT_CHOICES:
        raise ValueError(f"unknown weights {weights!r}; choose from {list(WEIGHT_CHOICES)}")
    if len(a) != len(b):
        raise ValueError(f"raters have different lengths: {len(a)} and {len(b)}")
    if not a:
        return None
    index = {c: i for i, c in enumerate(categories)}
    k = len(index)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        if x not in index or y not in index:
            raise ValueError(f"label pair ({x}, {y}) is outside the categories {list(categories)}")
        observed[index[x], index[y]] += 1
    observed /= len(a)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    return _weighted_kappa(observed, expected, weights)


def _weighted_kappa(
    observed: np.ndarray, expected: np.ndarray, weights: str | None
) -> float | None:
    k = observed.shape[0]
    if weights is None:
        loss = 1.0 - np.eye(k)
    else:
        grid = np.arange(k)
        loss = ((grid[:, None] - grid[None, :]) / max(k - 1, 1)) ** 2
    denominator = float((loss * expected).sum())
    if denominator < TIE_TOLERANCE:
        return None
    return 1.0 - float((loss * observed).sum()) / denominator


def kendall_tau(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Tau-b by explicit pair counting; None when either side is constant."""
    if len(x) != len(y):
        raise ValueError(f"sequences have different lengths: {len(x)} and {len(y)}")
    net = tied_x = tied_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = x[i] - x[j], y[i] - y[j]
            tied_x += int(dx == 0)
            tied_y += int(dy == 0)
            net += (int(dx > 0) - int(dx < 0)) * (int(dy > 0) - int(dy < 0))
    pairs = n * (n - 1) // 2
    denominator = (pairs - tied_x) * (pairs - tied_y)
    if denominator <= 0:
        return None
    return net / math.sqrt(denominator)


def _chunk_sizes(n_draws: int, width: int) -> list[int]:
    """Row counts per batch, so n_draws * width stays bounded."""
    step = CHUNK_ROWS if n_draws * width > CHUNK_THRESHOLD else max(n_draws, 1)
    return [min(step, n_draws - start) for start in range(0, n_draws, step)]


def bootstrap_ci(
    values: Sequence[float], *, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(mean, lower, upper): percentile interval of resampled means."""
    if not len(values):
        raise ValueError("bootstrap_ci needs at least one value")
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.concatenate([
        data[rng.integers(0, len(data), size=(rows, len(data)))].mean(axis=1)
        for rows in _chunk_sizes(n_boot, len(data))
    ])
    lower, upper = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(data.mean()), float(lower), float(upper)


def randomization_test(diffs: Sequence[float], *, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip p-value, never below 1 / (n_perm + 1)."""
    if not len(diffs):
        raise ValueError("randomization_test needs at least one difference")
    data = np.asarray(diffs, dtype=float)
    if not data.any():
        return 1.0
    rng = np.random.default_rng(seed)
    signs = np.array([-1.0, 1.0])
    observed = abs(float(data.mean()))
    count = 0
    for rows in _chunk_sizes(n_perm, len(data)):
        flips = rng.choice(signs, size=(rows, len(data)))
        count += int((np.abs((flips * data).mean(axis=1)) >= observed - TIE_TOLERANCE).sum())
    return (1 + count) / (1 + n_perm)
