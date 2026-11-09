"""Automatic configuration choice under the ADR-6 noise cap (TX-5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from getwork.taxonomy.grid import ConfigResult

SILHOUETTE_DECIMALS = 9


def eligible(
    results: Sequence[ConfigResult], max_noise: float, min_clusters: int
) -> list[ConfigResult]:
    return [
        r
        for r in results
        if r.skipped is None
        and r.silhouette is not None
        and r.noise_fraction <= max_noise
        and r.n_clusters >= min_clusters
    ]


def _rank_key(result: ConfigResult) -> tuple[float, float, str]:
    if result.silhouette is None:
        raise ValueError(f"{result.config_id} has no silhouette and cannot be ranked")
    return (
        -round(result.silhouette, SILHOUETTE_DECIMALS),
        result.noise_fraction,
        result.config_id,
    )


def choose(
    results: Sequence[ConfigResult], max_noise: float, min_clusters: int
) -> ConfigResult | None:
    candidates = eligible(results, max_noise, min_clusters)
    return min(candidates, key=_rank_key) if candidates else None


def selection_record(
    result: ConfigResult,
    *,
    source: str,
    context: Mapping[str, object],
    n_eligible: int | None,
) -> dict:
    return {
        "config_id": result.config_id,
        "method": result.method,
        "params": dict(result.params),
        "n_clusters": result.n_clusters,
        "noise_fraction": result.noise_fraction,
        "silhouette": result.silhouette,
        "largest_cluster_fraction": result.largest_cluster_fraction,
        "source": source,
        "n_eligible": n_eligible,
        "context": dict(context),
    }
