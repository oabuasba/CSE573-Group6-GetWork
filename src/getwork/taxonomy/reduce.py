"""UMAP reduction of the skill vectors (TX-3); the reducer is injectable (TX-8)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

MIN_ROWS = 20


@dataclass(frozen=True)
class ReduceConfig:
    n_components: int = 5
    n_neighbors: int = 15
    min_dist: float = 0.0
    seed: int = 0


ReduceFn = Callable[[np.ndarray, ReduceConfig], np.ndarray]


def umap_reduce(vectors: np.ndarray, cfg: ReduceConfig) -> np.ndarray:
    n = len(vectors)
    if n < MIN_ROWS:
        raise ValueError(f"UMAP needs at least {MIN_ROWS} skills, got {n}")
    import umap  # deferred: importing umap compiles with numba and is slow

    reducer = umap.UMAP(
        n_components=cfg.n_components,
        n_neighbors=min(cfg.n_neighbors, n - 1),
        min_dist=cfg.min_dist,
        metric="cosine",
        random_state=cfg.seed,
    )
    return np.asarray(reducer.fit_transform(vectors), dtype=np.float32)
