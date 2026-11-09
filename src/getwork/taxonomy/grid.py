"""The clustering grid: configurations, running them and scoring the labels (TX-3, TX-4)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields

import numpy as np

DBSCAN = "dbscan"
HDBSCAN = "hdbscan"
KMEANS = "kmeans"
KMEANS_N_INIT = 10
SKIPPED_LABEL = -2
SKIP_REASON = "needs fewer clusters than skills"
INT_FIELDS = frozenset({"dbscan_min_samples", "hdbscan_min_cluster_size", "kmeans_k"})


@dataclass(frozen=True)
class GridConfig:
    dbscan_eps: tuple[float, ...] = (0.2, 0.3, 0.5, 0.75, 1.0, 1.5)
    dbscan_min_samples: tuple[int, ...] = (3, 5, 10)
    hdbscan_min_cluster_size: tuple[int, ...] = (5, 10, 15, 25, 40)
    kmeans_k: tuple[int, ...] = (10, 15, 20, 30, 40, 60)

    @staticmethod
    def from_mapping(raw: Mapping[str, object]) -> "GridConfig":
        names = {f.name for f in fields(GridConfig)}
        for key in raw:
            if key not in names:
                raise ValueError(f"unknown grid key {key!r}; expected one of {sorted(names)}")
        return GridConfig(**{key: _check_values(key, value) for key, value in raw.items()})


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_values(key: str, value: object) -> tuple[float | int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"grid key {key!r} must be a non-empty list of positive numbers")
    if not all(_is_number(v) and v > 0 for v in value):
        raise ValueError(f"grid key {key!r} must contain only positive numbers")
    if key in INT_FIELDS:
        if not all(isinstance(v, int) for v in value):
            raise ValueError(f"grid key {key!r} must contain only integers")
        return tuple(int(v) for v in value)
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class ClusterConfig:
    config_id: str
    method: str
    params: Mapping[str, float | int]


def enumerate_configs(grid: GridConfig) -> list[ClusterConfig]:
    dbscan = [
        {"eps": eps, "min_samples": min_samples}
        for eps in grid.dbscan_eps
        for min_samples in grid.dbscan_min_samples
    ]
    hdbscan = [{"min_cluster_size": size} for size in grid.hdbscan_min_cluster_size]
    kmeans = [{"k": k} for k in grid.kmeans_k]
    configs: list[ClusterConfig] = []
    for method, all_params in ((DBSCAN, dbscan), (HDBSCAN, hdbscan), (KMEANS, kmeans)):
        for i, params in enumerate(all_params, start=1):
            configs.append(
                ClusterConfig(config_id=f"{method}-{i:02d}", method=method, params=params)
            )
    return configs


def run_config(cfg: ClusterConfig, reduced: np.ndarray, seed: int) -> np.ndarray | None:
    """Labels (int32, -1 = noise) or None when the configuration cannot run on this many rows."""
    from sklearn import cluster  # deferred to keep importing this module cheap

    n = len(reduced)
    if cfg.method == DBSCAN:
        model = cluster.DBSCAN(eps=cfg.params["eps"], min_samples=cfg.params["min_samples"])
    elif cfg.method == HDBSCAN:
        if cfg.params["min_cluster_size"] > n:
            return None
        model = cluster.HDBSCAN(
            min_cluster_size=cfg.params["min_cluster_size"], copy=True
        )
    elif cfg.method == KMEANS:
        if cfg.params["k"] >= n:
            return None
        model = cluster.KMeans(
            n_clusters=cfg.params["k"], n_init=KMEANS_N_INIT, random_state=seed
        )
    else:
        raise ValueError(f"unknown clustering method {cfg.method!r}")
    return model.fit_predict(reduced).astype(np.int32)


@dataclass(frozen=True)
class ConfigResult:
    config_id: str
    method: str
    params: Mapping[str, float | int]
    n_clusters: int
    noise_fraction: float
    silhouette: float | None
    largest_cluster_fraction: float
    skipped: str | None


def score_labels(
    labels: np.ndarray, vectors: np.ndarray
) -> tuple[int, float, float | None, float]:
    """Silhouette is cosine, on the original vectors, for clustered rows only (TX-4)."""
    from sklearn import metrics

    clustered = labels >= 0
    sizes = np.bincount(labels[clustered]) if clustered.any() else np.array([], dtype=int)
    sizes = sizes[sizes > 0]
    n_clusters = int(len(sizes))
    noise_fraction = float((labels == -1).sum() / len(labels))
    largest = float(sizes.max() / len(labels)) if n_clusters else 0.0
    silhouette = None
    if n_clusters >= 2 and int(clustered.sum()) > n_clusters:
        silhouette = float(
            metrics.silhouette_score(
                vectors[clustered], labels[clustered], metric="cosine"
            )
        )
    return n_clusters, noise_fraction, silhouette, largest


def _skipped_result(cfg: ClusterConfig) -> ConfigResult:
    return ConfigResult(
        config_id=cfg.config_id,
        method=cfg.method,
        params=cfg.params,
        n_clusters=0,
        noise_fraction=1.0,
        silhouette=None,
        largest_cluster_fraction=0.0,
        skipped=SKIP_REASON,
    )


def _scored_result(cfg: ClusterConfig, labels: np.ndarray, vectors: np.ndarray) -> ConfigResult:
    n_clusters, noise, silhouette, largest = score_labels(labels, vectors)
    return ConfigResult(
        config_id=cfg.config_id,
        method=cfg.method,
        params=cfg.params,
        n_clusters=n_clusters,
        noise_fraction=noise,
        silhouette=silhouette,
        largest_cluster_fraction=largest,
        skipped=None,
    )


def run_grid(
    vectors: np.ndarray,
    reduced: np.ndarray,
    configs: Sequence[ClusterConfig],
    seed: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[ConfigResult], np.ndarray]:
    results: list[ConfigResult] = []
    matrix = np.full((len(configs), len(vectors)), SKIPPED_LABEL, dtype=np.int32)
    for i, cfg in enumerate(configs):
        labels = run_config(cfg, reduced, seed)
        if labels is None:
            results.append(_skipped_result(cfg))
        else:
            matrix[i] = labels
            results.append(_scored_result(cfg, labels, vectors))
        if progress is not None:
            progress(i + 1, len(configs))
    return results, matrix
