from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest
import viz_helpers as vh
from matplotlib.colors import to_hex

from getwork.taxonomy import viz_axes, viz_data, viz_html, viz_plots

pytestmark = pytest.mark.usefixtures("offline")

NOISE = viz_data.NOISE_COLOR


@pytest.fixture
def data(tmp_path: Path) -> viz_data.PlotData:
    return viz_data.load_plot_data(vh.make_tree(tmp_path))


@pytest.fixture
def coords() -> np.ndarray:
    return vh.make_coords()


def _hex(collection: object) -> str:
    return to_hex(collection.get_facecolor()[0])


def test_map_has_a_grey_noise_collection_and_one_colored_collection_per_cluster(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    ax = viz_plots.cluster_map_figure(data, coords, max_label_chars=30).axes[0]
    colors = viz_data.cluster_colors([0, 1, 3])
    labels = [c.get_label() for c in ax.collections]
    assert labels == ["Noise", *vh.NAMES.values()]
    assert [_hex(c) for c in ax.collections] == [NOISE, colors[0], colors[1], colors[3]]
    sizes = [len(c.get_offsets()) for c in ax.collections]
    assert sizes == [5, 5, 8, 12]


def test_map_points_belong_to_the_right_cluster(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    ax = viz_plots.cluster_map_figure(data, coords, max_label_chars=30).axes[0]
    noise, first, third = ax.collections[0], ax.collections[1], ax.collections[3]
    assert np.allclose(noise.get_offsets(), coords[vh.LABELS == -1])
    assert np.allclose(first.get_offsets(), coords[vh.LABELS == 0])
    assert np.allclose(third.get_offsets(), coords[vh.LABELS == 3])


def test_map_labels_sit_on_their_own_hand_written_centroid(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    ax = viz_plots.cluster_map_figure(data, coords, max_label_chars=30).axes[0]
    expected = {
        "Alpha tools": (1.0, 50.0),
        "Beta / Gamma": (200.0, 7.0),
        "A very long cluster name here": (-30.0, 90.0),
    }
    assert {t.get_text() for t in ax.texts} == set(expected)
    for note in ax.texts:
        assert note.xy == pytest.approx(expected[note.get_text()])
        assert note.xyann == pytest.approx(expected[note.get_text()])


def test_map_labels_are_shortened_to_the_limit(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    ax = viz_plots.cluster_map_figure(data, coords, max_label_chars=10).axes[0]
    assert all(len(t.get_text()) <= 10 for t in ax.texts)
    assert len(ax.texts) == 3


def test_map_without_noise_has_no_noise_collection(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    labels = np.where(data.labels < 0, 0, data.labels)
    clean = dataclasses.replace(data, labels=labels)
    ax = viz_plots.cluster_map_figure(clean, coords, max_label_chars=30).axes[0]
    assert "Noise" not in [c.get_label() for c in ax.collections]


def test_sizes_bars_show_noise_first_then_clusters_by_size(data: viz_data.PlotData) -> None:
    ax = viz_plots.cluster_sizes_figure(data).axes[0]
    names = [t.get_text() for t in ax.get_yticklabels()]
    widths = [p.get_width() for p in ax.patches]
    assert names == ["Noise", vh.NAMES[3], vh.NAMES[1], vh.NAMES[0]]
    assert widths == [5, 12, 8, 5]
    assert to_hex(ax.patches[0].get_facecolor()) == NOISE


def test_balance_points_are_cluster_sums_with_resumes_on_x(data: viz_data.PlotData) -> None:
    ax = viz_plots.corpus_balance_figure(data).axes[0]
    offsets = ax.collections[0].get_offsets()
    first = [sum(vh.df_resumes(i) for i in range(5)), sum(vh.df_postings(i) for i in range(5))]
    third = [
        sum(vh.df_resumes(i) for i in range(13, 25)),
        sum(vh.df_postings(i) for i in range(13, 25)),
    ]
    assert offsets.shape == (4, 2)
    assert offsets[0].tolist() == first
    assert offsets[2].tolist() == third
    assert to_hex(ax.collections[0].get_facecolor()[3]) == NOISE
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")
    assert ax.get_xlim()[0] > 0.5 * offsets[:, 0].min() / 2
    assert ax.get_xlim()[0] <= offsets[:, 0].min()
    assert ax.get_xlim()[1] > offsets[:, 0].max()
    assert ax.get_ylim()[1] > offsets[:, 1].max()
    assert ax.get_legend() is None


def test_one_corpus_bars_stack_only_the_non_empty_segments(data: viz_data.PlotData) -> None:
    ax = viz_plots.corpus_only_figure(data).axes[0]
    assert len(ax.patches) == 2 * 4
    assert "20.0%" in ax.get_title()
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["both corpora", "resumes only"]
    # rows run noise, cluster 3, cluster 1, cluster 0 (least to most one-sided)
    assert [p.get_width() for p in ax.patches[0:4]] == [5, 11, 6, 2]
    assert [p.get_width() for p in ax.patches[4:8]] == [0, 1, 2, 3]
    assert ax.get_xlim()[1] > 13  # room right of the longest bar (12 skills)


def test_one_corpus_bars_show_postings_only_when_it_exists(data: viz_data.PlotData) -> None:
    swapped = dataclasses.replace(data, df_resumes=data.df_postings, df_postings=data.df_resumes)
    ax = viz_plots.corpus_only_figure(swapped).axes[0]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend == ["both corpora", "postings only"]
    assert [p.get_width() for p in ax.patches[4:8]] == [0, 1, 2, 3]


def test_skill_scatter_plots_resume_df_on_x_and_colors_noise_grey(
    data: viz_data.PlotData,
) -> None:
    ax = viz_plots.skill_scatter_figure(data).axes[0]
    noise, members = ax.collections
    resumes = viz_axes.log_position(data.df_resumes)
    postings = viz_axes.log_position(data.df_postings)
    assert noise.get_offsets().tolist() == np.stack([resumes[25:], postings[25:]], 1).tolist()
    assert _hex(noise) == NOISE
    assert members.get_offsets().tolist() == np.stack([resumes[:25], postings[:25]], 1).tolist()
    assert members.get_offsets()[10].tolist() == [vh.df_resumes(10), vh.df_postings(10)]
    assert ax.get_legend() is None
    first, last = viz_data.cluster_colors([0, 1, 3])[0], viz_data.cluster_colors([0, 1, 3])[3]
    assert to_hex(members.get_facecolor()[0]) == first
    assert to_hex(members.get_facecolor()[24]) == last


def test_skill_scatter_draws_zero_counts_on_a_labelled_lane(data: viz_data.PlotData) -> None:
    ax = viz_plots.skill_scatter_figure(data).axes[0]
    members = ax.collections[1]
    assert members.get_offsets()[0].tolist() == [vh.df_resumes(0), viz_axes.ZERO_LANE]
    assert "0" in [t.get_text() for t in ax.get_yticklabels()]
    assert (ax.get_xscale(), ax.get_yscale()) == ("log", "log")


def _chosen_collection(ax: object) -> object:
    return next(c for c in ax.collections if str(c.get_label()).startswith("chosen"))


def test_grid_overview_highlights_the_chosen_configuration(data: viz_data.PlotData) -> None:
    ax = viz_plots.grid_overview_figure(data).axes[0]
    chosen = _chosen_collection(ax)
    assert chosen.get_label() == "chosen: dbscan-02"
    assert chosen.get_offsets().tolist() == [[3.0, 5 / 30]]
    assert to_hex(chosen.get_edgecolor()[0]) == viz_plots.HIGHLIGHT_COLOR
    plotted = sum(len(c.get_offsets()) for c in ax.collections if c is not chosen)
    assert plotted == 2
    assert "1 skipped" in ax.get_title()


def test_grid_overview_follows_the_selected_config(data: viz_data.PlotData) -> None:
    other = dataclasses.replace(data, chosen_id="dbscan-01")
    ax = viz_plots.grid_overview_figure(other).axes[0]
    assert _chosen_collection(ax).get_offsets().tolist() == [[7.0, 0.1]]


def test_save_png_writes_a_png(data: viz_data.PlotData, tmp_path: Path) -> None:
    path = tmp_path / "out" / "sizes.png"
    viz_plots.save_png(viz_plots.cluster_sizes_figure(data), path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_html_map_has_noise_cluster_leader_and_label_traces(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    fig = viz_html.cluster_map_figure(data, coords, max_label_chars=30)
    names = [t.name for t in fig.data]
    assert names == ["Noise", *vh.NAMES.values(), "label leader lines", "cluster names"]
    colors = viz_data.cluster_colors([0, 1, 3])
    assert fig.data[0].marker.color == NOISE
    assert [t.marker.color for t in fig.data[1:4]] == [colors[0], colors[1], colors[3]]
    assert fig.data[5].mode == "text"
    assert sorted(fig.data[5].text) == sorted(vh.NAMES.values())


def test_html_labels_sit_at_their_own_centroid_with_leader_lines(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    fig = viz_html.cluster_map_figure(data, coords, max_label_chars=30)
    leaders, labels = fig.data[4], fig.data[5]
    expected = {
        "Alpha tools": (1.0, 50.0),
        "Beta / Gamma": (200.0, 7.0),
        "A very long cluster name here": (-30.0, 90.0),
    }
    placed = {t: (x, y) for t, x, y in zip(labels.text, labels.x, labels.y)}
    assert set(placed) == set(expected)
    for text, position in expected.items():
        assert placed[text] == pytest.approx(position)
    starts = {(x, y) for x, y in zip(leaders.x[0::3], leaders.y[0::3])}
    assert len(starts) == 3
    for position in expected.values():
        assert any(s == pytest.approx(position) for s in starts)


@pytest.mark.parametrize(
    ("shift", "center", "moved_axis"),
    [
        ((-198.5, 43.2), (1.5, 50.2), 1),  # nearly on top of cluster 0: pushed apart in y
        ((-197.6, 43.0), (2.4, 50.0), 0),  # side by side, boxes just touching: pushed in x
    ],
)
def test_html_leader_lines_run_from_the_centroid_to_a_moved_label(
    data: viz_data.PlotData,
    coords: np.ndarray,
    shift: tuple[float, float],
    center: tuple[float, float],
    moved_axis: int,
) -> None:
    close = coords.copy()
    close[vh.LABELS == 1] += shift  # cluster 1 now centers at `center`, next to cluster 0
    fig = viz_html.cluster_map_figure(data, close, max_label_chars=30)
    leaders, labels = fig.data[4], fig.data[5]
    starts = list(zip(leaders.x[0::3], leaders.y[0::3]))
    ends = list(zip(leaders.x[1::3], leaders.y[1::3]))
    assert ends == list(zip(labels.x, labels.y))
    centers = {(-30.0, 90.0), center, (1.0, 50.0)}
    assert all(any(s == pytest.approx(c) for c in centers) for s in starts)
    assert len(starts) == 3
    moved = [(s, e) for s, e in zip(starts, ends) if s != pytest.approx(e)]
    assert moved
    assert all(abs(e[moved_axis] - s[moved_axis]) > 1e-9 for s, e in moved)
    assert all(abs(e[1 - moved_axis] - s[1 - moved_axis]) < 1e-6 for s, e in moved)


def test_html_hover_rows_carry_skill_cluster_and_both_df(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    fig = viz_html.cluster_map_figure(data, coords, max_label_chars=30)
    beta = fig.data[2]
    assert len(beta.customdata) == 8
    assert beta.customdata[0] == ["skill05", "Beta / Gamma", vh.df_resumes(5), vh.df_postings(5)]
    noise_row = fig.data[0].customdata[-1]
    assert noise_row == ["skill29", "Noise", vh.df_resumes(29), vh.df_postings(29)]
    assert np.allclose(beta.x, coords[5:13, 0])
    assert np.allclose(beta.y, coords[5:13, 1])
    assert "customdata[2]" in beta.hovertemplate and "df_resumes" in beta.hovertemplate


def test_html_page_is_standalone_with_plotly_embedded(
    data: viz_data.PlotData, coords: np.ndarray
) -> None:
    page = viz_html.figure_html(viz_html.cluster_map_figure(data, coords, max_label_chars=30))
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert "skill07" in page and "Alpha tools" in page
    assert not re.search(r"<script[^>]*\ssrc=", page)  # cdn mode loads plotly.js this way
    assert len(page) > 1_000_000
    assert "plotly.js" in page


def _axes_fraction(ax: object, point: tuple[float, float]) -> np.ndarray:
    display = ax.transData.transform(point)
    return ax.transAxes.inverted().transform(display)


def _overlaps(a: np.ndarray, b: np.ndarray, widths: tuple[float, float]) -> bool:
    half = (widths[0] + widths[1]) / 2
    height = viz_axes.LABEL_HEIGHT_FRACTION
    return abs(a[0] - b[0]) < half - 1e-9 and abs(a[1] - b[1]) < height - 1e-9


def _check_spread_labels(ax: object) -> None:
    notes = ax.texts
    for note in notes:
        target = np.array(note.xyann)
        assert ((target >= 0) & (target <= 1)).all()
        assert not np.allclose(target, _axes_fraction(ax, tuple(note.xy)))
    for i, first in enumerate(notes):
        for second in notes[i + 1 :]:
            char = viz_axes.LABEL_CHAR_FRACTION
            widths = tuple(len(n.get_text()) * char for n in (first, second))
            assert not _overlaps(np.array(first.xyann), np.array(second.xyann), widths)


# Skills ranked by df_resumes + df_postings (not by index, name or df_resumes alone).
TOP_SKILLS_BY_TOTAL = [27, 26, 29, 25, 28, 24, 23, 19, 22, 18, 21, 17]


def test_skill_scatter_labels_the_top_skills_by_total_at_their_own_points(
    data: viz_data.PlotData,
) -> None:
    ax = viz_plots.skill_scatter_figure(data).axes[0]
    expected = {
        f"skill{i:02d}": (float(vh.df_resumes(i)), float(max(vh.df_postings(i), 0.5)))
        for i in TOP_SKILLS_BY_TOTAL
    }
    assert len(ax.texts) == viz_plots.TOP_SKILL_LABELS == len(TOP_SKILLS_BY_TOTAL)
    assert {t.get_text(): tuple(t.xy) for t in ax.texts} == expected
    _check_spread_labels(ax)


def test_skill_scatter_labels_every_skill_when_there_are_fewer_than_the_limit(
    data: viz_data.PlotData,
) -> None:
    few = dataclasses.replace(
        data, skills=data.skills[:5], vectors=data.vectors[:5], labels=data.labels[:5],
        df_resumes=data.df_resumes[:5], df_postings=data.df_postings[:5],
    )
    ax = viz_plots.skill_scatter_figure(few).axes[0]
    assert sorted(t.get_text() for t in ax.texts) == [f"skill{i:02d}" for i in range(5)]


# (resumes, postings) per cluster 0..13; the first 10 names by skew are listed below.
CLUSTER_DF = [
    (100, 100), (900, 3), (50, 40), (800, 100), (20, 300), (400, 200), (1000, 0),
    (10, 10), (300, 30), (5, 60), (600, 550), (2000, 40), (70, 5), (250, 250),
]
NOISE_DF = [(7000, 1), (7000, 1)]
MOST_SKEWED = ["Noise", 6, 1, 11, 4, 12, 9, 8, 3, 5]


def _many_clusters(data: viz_data.PlotData) -> viz_data.PlotData:
    pairs = [*CLUSTER_DF, *NOISE_DF]
    n = len(pairs)
    return dataclasses.replace(
        data,
        skills=[f"s{i}" for i in range(n)],
        vectors=data.vectors[:n],
        labels=np.array([*range(len(CLUSTER_DF)), -1, -1]),
        names={k: f"Cluster {k:02d}" for k in range(len(CLUSTER_DF))},
        df_resumes=np.array([r for r, _ in pairs]),
        df_postings=np.array([p for _, p in pairs]),
    )


def test_balance_labels_the_most_skewed_clusters_at_their_own_points(
    data: viz_data.PlotData,
) -> None:
    ax = viz_plots.corpus_balance_figure(_many_clusters(data)).axes[0]
    expected = {}
    for key in MOST_SKEWED:
        if key == "Noise":
            expected["Noise"] = (14000.0, 2.0)
        else:
            res, post = CLUSTER_DF[key]
            expected[f"Cluster {key:02d}"] = (float(res), float(max(post, 0.5)))
    assert len(ax.texts) == viz_plots.EXTREME_CLUSTER_LABELS == len(MOST_SKEWED)
    assert {t.get_text(): tuple(t.xy) for t in ax.texts} == expected
    _check_spread_labels(ax)


def test_balance_labels_every_cluster_when_there_are_fewer_than_the_limit(
    data: viz_data.PlotData,
) -> None:
    ax = viz_plots.corpus_balance_figure(data).axes[0]
    groups = {
        "Alpha tools": range(0, 5),
        "Beta / Gamma": range(5, 13),
        "A very long cluster nam…": range(13, 25),  # shortened to 24 characters
        "Noise": range(25, 30),
    }
    expected = {
        name: (
            float(sum(vh.df_resumes(i) for i in rows)),
            float(sum(vh.df_postings(i) for i in rows)),
        )
        for name, rows in groups.items()
    }
    assert len(ax.texts) == 4
    assert {t.get_text(): tuple(t.xy) for t in ax.texts} == expected
