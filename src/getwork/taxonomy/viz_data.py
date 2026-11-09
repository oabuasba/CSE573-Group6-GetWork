"""Inputs of the taxonomy plots: validated loading, 2-D projection and per-cluster tables.

Pure numpy/pandas on purpose: nothing here imports a plotting library.
"""

from __future__ import annotations

import colorsys
import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from getwork import paths
from getwork.taxonomy import cli_common, cli_name, counts, grid, model, report
from getwork.taxonomy.build import NOISE_LABEL
from getwork.taxonomy.grid import ConfigResult
from getwork.taxonomy.reduce import ReduceConfig, ReduceFn, umap_reduce

NOISE_NAME = "Noise"
NOISE_COLOR = "#b3b3b3"
PROJECTION_NEIGHBORS = 15
PROJECTION_MIN_DIST = 0.1
GOLDEN_RATIO_STEP = 0.618033988749895
LABEL_CHAR_WIDTH = 0.0036  # fraction of the x span per character at font size 6
LABEL_HEIGHT = 0.0125  # fraction of the y span
TABLE_COLUMNS: tuple[str, ...] = (
    "cluster_id", "name", "size", "df_resumes", "df_postings",
    "n_both", "n_resumes_only", "n_postings_only",
)


@dataclass(frozen=True)
class PlotData:
    skills: list[str]
    vectors: np.ndarray
    labels: np.ndarray  # the chosen configuration's labels, aligned with skills
    names: dict[int, str]
    df_resumes: np.ndarray
    df_postings: np.ndarray
    results: list[ConfigResult]
    chosen_id: str


def _read_df_columns(root: Path, skills: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    path = paths.skills_dir(root) / counts.COUNTS_FILE
    table = pq.read_table(path, columns=["skill", "df_resumes", "df_postings"])
    rows = {r["skill"]: r for r in table.to_pylist()}
    return (
        np.array([rows[s]["df_resumes"] for s in skills], dtype=np.int64),
        np.array([rows[s]["df_postings"] for s in skills], dtype=np.int64),
    )


def _read_names(root: Path, labels: np.ndarray) -> dict[int, str]:
    path = paths.taxonomy_dir(root) / cli_name.NAMES_FILE
    if not path.exists():
        raise FileNotFoundError(f"names file not found: {path} (run taxonomy name)")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    names = {int(r["cluster_id"]): model.check_name(r["name"]) for r in rows}
    sizes = {
        int(k): int(v)
        for k, v in zip(*np.unique(labels[labels >= 0], return_counts=True))
    }
    if len(names) != len(rows) or set(names) != set(sizes):
        raise ValueError(
            f"{path} does not match the selected clustering (cluster ids differ); "
            "run taxonomy name --force"
        )
    stale = [
        r["cluster_id"]
        for r in rows
        if r.get("size") and sizes[int(r["cluster_id"])] != int(r["size"])
    ]
    if stale:
        raise ValueError(f"{path} has cluster sizes that differ from the selection: {stale}")
    return names


def _check_vectors(root: Path, skills: list[str]) -> np.ndarray:
    vectors = cli_common.read_skill_vectors(root)
    current = cli_common.read_counts_summary(root)["selected_skills_version"]
    if vectors.meta.get("selected_skills_version") != current or list(vectors.ids) != skills:
        raise ValueError(
            "skill vectors do not match the selected skills; run taxonomy embed again"
        )
    return vectors.vectors


def load_plot_data(root: Path) -> PlotData:
    """Load and cross-check every file the plots use; raises before anything is drawn."""
    state = cli_common.load_selection_state(root)
    vectors = _check_vectors(root, state.skills)
    labels = np.asarray(state.labels, dtype=np.int64)
    if (labels == grid.SKIPPED_LABEL).any():
        raise ValueError("the selected configuration was skipped and has no clusters")
    df_resumes, df_postings = _read_df_columns(root, state.skills)
    results, _ = report.read_grid(paths.taxonomy_dir(root))
    return PlotData(
        skills=state.skills,
        vectors=vectors,
        labels=labels,
        names=_read_names(root, labels),
        df_resumes=df_resumes,
        df_postings=df_postings,
        results=results,
        chosen_id=state.selection["config_id"],
    )


def project_2d(vectors: np.ndarray, *, seed: int, reduce_fn: ReduceFn = umap_reduce) -> np.ndarray:
    """Separate 2-component cosine UMAP; its distances differ from the clustering space."""
    config = ReduceConfig(
        n_components=2, n_neighbors=PROJECTION_NEIGHBORS, min_dist=PROJECTION_MIN_DIST, seed=seed
    )
    return np.asarray(reduce_fn(vectors, config), dtype=np.float64)


def cluster_ids(labels: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.unique(labels) if c >= 0)


def cluster_colors(ids: Sequence[int]) -> dict[int, str]:
    """Distinct, deterministic hex colors per cluster id; noise is always grey."""
    colors = {NOISE_LABEL: NOISE_COLOR}
    for position, cluster_id in enumerate(ids):
        hue = (position * GOLDEN_RATIO_STEP) % 1.0
        saturation, value = (0.75, 0.85) if position % 2 == 0 else (0.9, 0.65)
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors[cluster_id] = f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"
    return colors


def short_label(*, name: str, max_chars: int) -> str:
    return name if len(name) <= max_chars else name[: max(max_chars - 1, 1)].rstrip() + "…"


def centroids(*, coords: np.ndarray, labels: np.ndarray) -> dict[int, tuple[float, float]]:
    """Median position of each cluster's skills (robust to a stray member)."""
    return {
        c: (float(np.median(coords[labels == c, 0])), float(np.median(coords[labels == c, 1])))
        for c in cluster_ids(labels)
    }


def _overlap(*, a: np.ndarray, b: np.ndarray, half_w: float, height: float) -> tuple[float, float]:
    return half_w - abs(a[0] - b[0]), height - abs(a[1] - b[1])


def spread_labels(
    *, anchors: np.ndarray, widths: np.ndarray, height: float, max_rounds: int = 300
) -> np.ndarray:
    """Move label centers apart until no two boxes (widths[i] x height) overlap."""
    pos = np.array(anchors, dtype=np.float64)
    for _ in range(max_rounds):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                pen_x, pen_y = _overlap(
                    a=pos[i], b=pos[j], half_w=(widths[i] + widths[j]) / 2, height=height
                )
                if pen_x <= 0 or pen_y <= 0:
                    continue
                axis = 0 if pen_x < pen_y else 1
                sign = 1.0 if pos[j][axis] >= pos[i][axis] else -1.0
                push = (pen_x if axis == 0 else pen_y) / 2 + 1e-9
                pos[i][axis] -= sign * push
                pos[j][axis] += sign * push
                moved = True
        if not moved:
            break
    return pos


def label_layout(
    *, coords: np.ndarray, labels: np.ndarray, texts: dict[int, str]
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Cluster ids, their centroids and de-overlapped label positions (largest cluster first)."""
    centers = centroids(coords=coords, labels=labels)
    sizes = {c: int((labels == c).sum()) for c in centers}
    ids = sorted(centers, key=lambda c: (-sizes[c], c))
    anchors = np.array([centers[c] for c in ids]).reshape(-1, 2)
    x_span = float(np.ptp(coords[:, 0])) or 1.0
    y_span = float(np.ptp(coords[:, 1])) or 1.0
    widths = np.array([len(texts[c]) * LABEL_CHAR_WIDTH * x_span for c in ids])
    scaled = anchors / [x_span, y_span]
    placed = spread_labels(anchors=scaled, widths=widths / x_span, height=LABEL_HEIGHT)
    placed = placed * [x_span, y_span]
    return ids, anchors, placed


def cluster_table(data: PlotData) -> pd.DataFrame:
    """One row per cluster (noise last): size, summed document frequencies, corpus split."""
    rows = []
    for cluster_id in [*cluster_ids(data.labels), NOISE_LABEL]:
        mask = data.labels == cluster_id
        if not mask.any():
            continue
        res, post = data.df_resumes[mask], data.df_postings[mask]
        rows.append({
            "cluster_id": cluster_id,
            "name": NOISE_NAME if cluster_id == NOISE_LABEL else data.names[cluster_id],
            "size": int(mask.sum()),
            "df_resumes": int(res.sum()),
            "df_postings": int(post.sum()),
            "n_both": int(((res > 0) & (post > 0)).sum()),
            "n_resumes_only": int(((res > 0) & (post == 0)).sum()),
            "n_postings_only": int(((res == 0) & (post > 0)).sum()),
        })
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def one_corpus_fraction(data: PlotData) -> float:
    one = (data.df_resumes == 0) | (data.df_postings == 0)
    return float(one.mean()) if len(one) else 0.0


def grid_table(results: Sequence[ConfigResult]) -> pd.DataFrame:
    """Plottable grid rows; skipped configurations have no clustering and are left out."""
    rows = [
        {
            "config_id": r.config_id, "method": r.method, "n_clusters": r.n_clusters,
            "noise_fraction": r.noise_fraction, "silhouette": r.silhouette,
        }
        for r in results
        if r.skipped is None
    ]
    columns = ["config_id", "method", "n_clusters", "noise_fraction", "silhouette"]
    return pd.DataFrame(rows, columns=columns)
