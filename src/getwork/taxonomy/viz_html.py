"""Interactive plotly view of the cluster map: one standalone HTML file, hover per skill."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from getwork.taxonomy import viz_data
from getwork.taxonomy.build import NOISE_LABEL
from getwork.taxonomy.viz_data import PlotData

HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b><br>cluster: %{customdata[1]}<br>"
    "df_resumes: %{customdata[2]}<br>df_postings: %{customdata[3]}<extra></extra>"
)
LABELS_TRACE = "cluster names"
LEADERS_TRACE = "label leader lines"


def _skill_trace(
    data: PlotData, coords: np.ndarray, *, mask: np.ndarray, name: str, color: str
) -> go.Scatter:
    rows = np.flatnonzero(mask)
    customdata = [
        [data.skills[i], name, int(data.df_resumes[i]), int(data.df_postings[i])] for i in rows
    ]
    return go.Scatter(
        x=coords[rows, 0], y=coords[rows, 1], mode="markers", name=name,
        marker={"color": color, "size": 8, "opacity": 0.85}, customdata=customdata,
        hovertemplate=HOVER_TEMPLATE,
    )


def _label_traces(
    data: PlotData, coords: np.ndarray, *, max_label_chars: int
) -> tuple[go.Scatter, go.Scatter]:
    """Leader lines from each centroid to its label, and the label texts themselves."""
    texts = {
        c: viz_data.short_label(name=data.names[c], max_chars=max_label_chars)
        for c in viz_data.cluster_ids(data.labels)
    }
    ids, anchors, placed = viz_data.label_layout(coords=coords, labels=data.labels, texts=texts)
    line_x: list[float | None] = []
    line_y: list[float | None] = []
    for anchor, target in zip(anchors, placed):
        line_x += [float(anchor[0]), float(target[0]), None]
        line_y += [float(anchor[1]), float(target[1]), None]
    leaders = go.Scatter(
        x=line_x, y=line_y, mode="lines", line={"color": "#888888", "width": 0.7},
        name=LEADERS_TRACE, hoverinfo="skip", showlegend=False,
    )
    labels = go.Scatter(
        x=placed[:, 0], y=placed[:, 1], mode="text", text=[texts[c] for c in ids],
        textfont={"size": 10, "color": "#222222"}, name=LABELS_TRACE, hoverinfo="skip",
    )
    return leaders, labels


def cluster_map_figure(data: PlotData, coords: np.ndarray, *, max_label_chars: int) -> go.Figure:
    """Traces: noise (grey), one per cluster, label leader lines, then the cluster names."""
    ids = viz_data.cluster_ids(data.labels)
    colors = viz_data.cluster_colors(ids)
    fig = go.Figure()
    noise = data.labels == NOISE_LABEL
    if noise.any():
        fig.add_trace(_skill_trace(
            data, coords, mask=noise, name=viz_data.NOISE_NAME, color=colors[NOISE_LABEL]
        ))
    for cluster_id in ids:
        fig.add_trace(_skill_trace(
            data, coords, mask=data.labels == cluster_id, name=data.names[cluster_id],
            color=colors[cluster_id],
        ))
    if ids:
        for trace in _label_traces(data, coords, max_label_chars=max_label_chars):
            fig.add_trace(trace)
    fig.update_layout(
        title=f"Skill clusters ({data.chosen_id}); 2-D UMAP, not the 5-D clustering space",
        xaxis={"title": "UMAP 1"}, yaxis={"title": "UMAP 2"}, height=850,
        template="plotly_white",
    )
    return fig


def figure_html(fig: go.Figure) -> str:
    """A self-contained page: plotly.js is embedded, so it opens offline."""
    return fig.to_html(include_plotlyjs=True, full_html=True)
