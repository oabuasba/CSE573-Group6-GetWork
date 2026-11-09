"""Static matplotlib figures of the taxonomy (built without pyplot, so no window ever opens)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from getwork import fileio
from getwork.taxonomy import viz_axes, viz_data
from getwork.taxonomy.build import NOISE_LABEL
from getwork.taxonomy.viz_data import PlotData

DPI = 130
LABEL_FONT = 6
HIGHLIGHT_COLOR = "#d62728"
BOTH_COLOR, RESUMES_COLOR, POSTINGS_COLOR = "#7f7f7f", "#1f77b4", "#ff7f0e"
METHOD_MARKERS = {"dbscan": "o", "hdbscan": "s", "kmeans": "^"}
EXTREME_CLUSTER_LABELS = 10
TOP_SKILL_LABELS = 12


MAP_CAPTION = (
    "2-D UMAP of the same skill vectors (cosine, own fixed seed). Distances here are not the "
    "5-D space the clustering used, so neighbours can be split and gaps can mislead."
)


def save_png(fig: Figure, path: Path) -> None:
    fileio.atomic_write(path, lambda tmp: fig.savefig(tmp, format="png", dpi=DPI))


def _scatter_cluster(
    ax: Axes, *, coords: np.ndarray, color: str, label: str, size: float
) -> None:
    ax.scatter(coords[:, 0], coords[:, 1], s=size, c=color, label=label, linewidths=0)


def cluster_map_figure(
    data: PlotData, coords: np.ndarray, *, max_label_chars: int
) -> Figure:
    """One collection per cluster (noise first, grey), names spread out from the centroids."""
    fig = Figure(figsize=(16, 12))
    ax = fig.subplots()
    ids = viz_data.cluster_ids(data.labels)
    colors = viz_data.cluster_colors(ids)
    noise = data.labels == NOISE_LABEL
    if noise.any():
        _scatter_cluster(
            ax, coords=coords[noise], color=colors[NOISE_LABEL], label=viz_data.NOISE_NAME, size=14
        )
    for cluster_id in ids:
        mask = data.labels == cluster_id
        _scatter_cluster(
            ax, coords=coords[mask], color=colors[cluster_id], label=data.names[cluster_id],
            size=26,
        )
    _label_clusters(ax, data, coords, max_label_chars=max_label_chars)
    ax.set_title(
        f"Skill clusters ({data.chosen_id}): {len(ids)} clusters, {int(noise.sum())} noise skills"
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    fig.text(0.5, 0.005, MAP_CAPTION, ha="center", fontsize=8, color="#555555")
    return fig


def _label_clusters(
    ax: Axes, data: PlotData, coords: np.ndarray, *, max_label_chars: int
) -> None:
    texts = {
        c: viz_data.short_label(name=data.names[c], max_chars=max_label_chars)
        for c in viz_data.cluster_ids(data.labels)
    }
    if not texts:
        return
    ids, anchors, placed = viz_data.label_layout(coords=coords, labels=data.labels, texts=texts)
    for cluster_id, anchor, target in zip(ids, anchors, placed):
        ax.annotate(
            texts[cluster_id], xy=tuple(anchor), xytext=tuple(target), fontsize=LABEL_FONT,
            ha="center", va="center", weight="bold",
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
            arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.4},
        )


def cluster_sizes_figure(data: PlotData) -> Figure:
    """Horizontal bars of skills per cluster, largest first, with the noise bar on top."""
    table = viz_data.cluster_table(data)
    noise = table[table["cluster_id"] == NOISE_LABEL]
    clusters = table[table["cluster_id"] != NOISE_LABEL].sort_values(
        by=["size", "cluster_id"], ascending=[False, True]
    )
    rows = pd.concat([noise, clusters], ignore_index=True)
    colors = viz_data.cluster_colors(viz_data.cluster_ids(data.labels))
    fig = Figure(figsize=(9, max(4.0, 0.24 * len(rows) + 1.5)))
    ax = fig.subplots()
    positions = np.arange(len(rows))
    bar_colors = [colors[int(c)] for c in rows["cluster_id"]]
    bars = ax.barh(positions, rows["size"], color=bar_colors)
    ax.set_yticks(positions, labels=list(rows["name"]), fontsize=7)
    ax.invert_yaxis()
    ax.bar_label(bars, fontsize=7, padding=2)
    ax.set_xlabel("skills in cluster")
    ax.set_title("Skills per cluster (grey = noise, not in any cluster)")
    fig.tight_layout()
    return fig


def corpus_balance_figure(data: PlotData) -> Figure:
    """One point per cluster: summed resume vs posting mentions of its skills."""
    table = viz_data.cluster_table(data)
    colors = viz_data.cluster_colors(viz_data.cluster_ids(data.labels))
    x, y = viz_axes.log_position(table["df_resumes"]), viz_axes.log_position(table["df_postings"])
    fig = Figure(figsize=(9, 8))
    ax = fig.subplots()
    ax.scatter(
        x, y, s=20 + 6 * table["size"], c=[colors[int(c)] for c in table["cluster_id"]],
        edgecolors="black", linewidths=0.4,
    )
    share = float(table["df_postings"].sum()) / max(float(table["df_resumes"].sum()), 1.0)
    reference = np.array([x.min(), x.max()])
    ax.plot(reference, reference * share, color="#999999", linestyle="--", linewidth=0.8)
    viz_axes.set_log_axes(ax, x=x, y=y)
    _annotate_extremes(ax, table)
    ax.set_xlabel("resume mentions (sum of df_resumes over the cluster's skills)")
    ax.set_ylabel("posting mentions (sum of df_postings); 0 is drawn on the lowest line")
    ax.set_title(
        "Cluster mentions by corpus (point size = skills; grey = noise)\n"
        f"dashed line: overall postings/resumes ratio {share:.3f}"
    )
    fig.tight_layout()
    return fig


def _annotate_extremes(ax: Axes, table: pd.DataFrame) -> None:
    ratio = np.abs(np.log((table["df_resumes"] + 1) / (table["df_postings"] + 1)))
    rows = table.loc[ratio.sort_values(ascending=False).index[:EXTREME_CLUSTER_LABELS]]
    points = np.stack(
        [viz_axes.log_position(rows["df_resumes"]), viz_axes.log_position(rows["df_postings"])],
        axis=1,
    )
    texts = [viz_data.short_label(name=str(n), max_chars=24) for n in rows["name"]]
    viz_axes.annotate_spread(ax, points=points, texts=texts, fontsize=7)


def corpus_only_figure(data: PlotData) -> Figure:
    """Stacked bars per cluster: skills in both corpora, resumes only, postings only."""
    table = viz_data.cluster_table(data)
    one_side = (table["n_resumes_only"] + table["n_postings_only"]) / table["size"]
    table = table.assign(one_side=one_side).sort_values(by="one_side", ascending=True)
    fig = Figure(figsize=(9, max(4.0, 0.24 * len(table) + 1.5)))
    ax = fig.subplots()
    positions = np.arange(len(table))
    left = np.zeros(len(table))
    for column, color, label in (
        ("n_both", BOTH_COLOR, "both corpora"),
        ("n_resumes_only", RESUMES_COLOR, "resumes only"),
        ("n_postings_only", POSTINGS_COLOR, "postings only"),
    ):
        if table[column].sum() == 0:
            continue
        ax.barh(positions, table[column], left=left, color=color, label=label)
        left = left + table[column].to_numpy()
    ax.set_yticks(positions, labels=list(table["name"]), fontsize=7)
    ax.set_xlabel("skills")
    ax.set_xlim(right=float(left.max()) * 1.12)
    ax.set_title(
        f"Skills seen in one corpus only: {viz_data.one_corpus_fraction(data):.1%} of all skills"
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def skill_scatter_figure(data: PlotData) -> Figure:
    """Each skill at (df_resumes, df_postings); noise is a separate grey collection."""
    colors = viz_data.cluster_colors(viz_data.cluster_ids(data.labels))
    x, y = viz_axes.log_position(data.df_resumes), viz_axes.log_position(data.df_postings)
    fig = Figure(figsize=(9, 8))
    ax = fig.subplots()
    noise = data.labels == NOISE_LABEL
    if noise.any():
        ax.scatter(x[noise], y[noise], s=12, c=colors[NOISE_LABEL], label=viz_data.NOISE_NAME)
    member = ~noise
    ax.scatter(
        x[member], y[member], s=18, alpha=0.8, c=[colors[int(c)] for c in data.labels[member]],
    )
    viz_axes.set_log_axes(ax, x=x, y=y)
    _label_top_skills(ax, data, x=x, y=y)
    ax.set_xlabel("df_resumes (resumes mentioning the skill)")
    ax.set_ylabel("df_postings (postings mentioning the skill); 0 is drawn on the lowest line")
    ax.set_title(
        "Skills by document frequency (grey = noise; cluster colors as in cluster_map.png)"
    )
    fig.tight_layout()
    return fig


def _label_top_skills(ax: Axes, data: PlotData, *, x: np.ndarray, y: np.ndarray) -> None:
    total = data.df_resumes + data.df_postings
    top = np.argsort(-total, kind="stable")[:TOP_SKILL_LABELS]
    points = np.stack([x[top], y[top]], axis=1)
    viz_axes.annotate_spread(ax, points=points, texts=[data.skills[i] for i in top], fontsize=7)


def grid_overview_figure(data: PlotData) -> Figure:
    """Clusters vs noise for every grid configuration, silhouette as color, chosen ringed."""
    table = viz_data.grid_table(data.results)
    n_skipped = len(data.results) - len(table)
    silhouette = table["silhouette"].astype(float)
    norm = Normalize(vmin=float(silhouette.min()), vmax=float(silhouette.max()))
    fig = Figure(figsize=(9, 7))
    ax = fig.subplots()
    mappable = None
    for method, marker in METHOD_MARKERS.items():
        part = table[table["method"] == method]
        if part.empty:
            continue
        mappable = ax.scatter(
            part["n_clusters"], part["noise_fraction"], c=part["silhouette"].astype(float),
            cmap="viridis", norm=norm, marker=marker, s=60, label=method,
        )
    _highlight_chosen(ax, table, data.chosen_id)
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label="silhouette")
    ax.set_xlabel("n_clusters")
    ax.set_ylabel("noise_fraction")
    ax.set_title(f"Grid overview ({len(table)} configurations, {n_skipped} skipped)")
    ax.legend(loc="upper right")
    return fig


def _highlight_chosen(ax: Axes, table: object, chosen_id: str) -> None:
    row = table[table["config_id"] == chosen_id]
    if row.empty:
        return
    x, y = float(row["n_clusters"].iloc[0]), float(row["noise_fraction"].iloc[0])
    ax.scatter(
        [x], [y], s=320, facecolors="none", edgecolors=HIGHLIGHT_COLOR, linewidths=2.2,
        label=f"chosen: {chosen_id}",
    )
    ax.annotate(
        chosen_id, xy=(x, y), xytext=(10, 12), textcoords="offset points",
        color=HIGHLIGHT_COLOR, fontsize=9, weight="bold",
    )
