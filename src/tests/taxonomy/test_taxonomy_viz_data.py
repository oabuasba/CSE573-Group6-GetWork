from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
import viz_helpers as vh

from getwork import fileio, paths
from getwork.taxonomy import counts, report, viz_data
from getwork.taxonomy.reduce import ReduceConfig

pytestmark = pytest.mark.usefixtures("offline")


@pytest.fixture
def data(tmp_path: Path) -> viz_data.PlotData:
    return viz_data.load_plot_data(vh.make_tree(tmp_path))


def test_load_aligns_skills_labels_names_and_df(data: viz_data.PlotData) -> None:
    assert data.skills == vh.SKILLS
    assert data.labels.tolist() == vh.LABELS.tolist()
    assert data.names == vh.NAMES
    assert data.chosen_id == vh.CHOSEN
    assert data.vectors.shape == (30, 8)
    assert data.df_resumes.tolist() == [vh.df_resumes(i) for i in range(30)]
    assert data.df_postings.tolist() == [vh.df_postings(i) for i in range(30)]
    assert [r.config_id for r in data.results] == ["dbscan-01", "dbscan-02", "kmeans-01"]


def test_load_refuses_selection_from_another_grid(tmp_path: Path) -> None:
    root = vh.make_tree(tmp_path)
    out = paths.taxonomy_dir(root)
    report.write_context(out, {"selected_skills_version": vh.VERSION, "seed": 99})
    with pytest.raises(ValueError, match="grid"):
        viz_data.load_plot_data(root)


def test_load_refuses_counts_changed_after_the_grid(tmp_path: Path) -> None:
    root = vh.make_tree(tmp_path)
    summary_path = paths.skills_dir(root) / counts.SUMMARY_FILE
    summary = fileio.read_json(summary_path)
    fileio.write_json_atomic(summary_path, {**summary, "selected_skills_version": "sha256:zz"})
    with pytest.raises(ValueError, match="different selected skill list"):
        viz_data.load_plot_data(root)


def test_load_refuses_vectors_of_another_skill_list(tmp_path: Path) -> None:
    root = vh.make_tree(tmp_path, vectors_version="sha256:other")
    with pytest.raises(ValueError, match="taxonomy embed"):
        viz_data.load_plot_data(root)


def test_load_refuses_vectors_in_another_row_order_with_the_same_version(
    tmp_path: Path,
) -> None:
    root = vh.make_tree(tmp_path, vector_ids=list(reversed(vh.SKILLS)))
    with pytest.raises(ValueError, match="taxonomy embed"):
        viz_data.load_plot_data(root)


def test_load_refuses_a_skipped_selected_configuration(tmp_path: Path) -> None:
    root = vh.make_tree(tmp_path, chosen=2)
    with pytest.raises(ValueError, match="configuration was skipped"):
        viz_data.load_plot_data(root)


def test_load_refuses_names_for_other_cluster_ids(tmp_path: Path) -> None:
    root = vh.make_tree(tmp_path)
    names_path = paths.taxonomy_dir(root) / "cluster_names.csv"
    text = names_path.read_text(encoding="utf-8").replace("\n3,", "\n4,")
    names_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy name"):
        viz_data.load_plot_data(root)


def test_load_refuses_names_with_other_sizes(tmp_path: Path) -> None:
    root = vh.make_tree(tmp_path)
    vh.write_names(root, sizes={0: 10, 1: 9, 3: 5})
    with pytest.raises(ValueError, match="sizes"):
        viz_data.load_plot_data(root)


@pytest.mark.parametrize(
    ("relative", "hint"),
    [
        ("taxonomy/cluster_names.csv", "taxonomy name"),
        ("skills/skill_vectors.npy", "taxonomy embed"),
        ("skills/skill_counts.parquet", "taxonomy counts"),
        ("taxonomy/selection.json", "taxonomy grid"),
    ],
)
def test_load_missing_file_names_it_and_the_command(
    tmp_path: Path, relative: str, hint: str
) -> None:
    root = vh.make_tree(tmp_path)
    (root / relative).unlink()
    with pytest.raises(FileNotFoundError) as info:
        viz_data.load_plot_data(root)
    assert hint in str(info.value)
    assert Path(relative).name in str(info.value)


def test_project_2d_asks_for_two_cosine_components_with_the_seed() -> None:
    seen: list[ReduceConfig] = []

    def reducer(vectors: np.ndarray, config: ReduceConfig) -> np.ndarray:
        seen.append(config)
        return vectors[:, :2]

    vectors = np.arange(40, dtype=np.float32).reshape(20, 2)
    coords = viz_data.project_2d(vectors, seed=7, reduce_fn=reducer)
    assert (seen[0].n_components, seen[0].seed) == (2, 7)
    assert (seen[0].n_neighbors, seen[0].min_dist) == (15, 0.1)
    assert coords.shape == (20, 2)


def test_project_2d_real_umap_is_deterministic_per_seed() -> None:
    vectors = np.random.default_rng(1).normal(size=(40, 8)).astype(np.float32)
    first = viz_data.project_2d(vectors, seed=3)
    assert first.shape == (40, 2)
    assert np.array_equal(first, viz_data.project_2d(vectors, seed=3))
    assert not np.array_equal(first, viz_data.project_2d(vectors, seed=4))


def test_cluster_colors_are_distinct_stable_and_noise_is_grey() -> None:
    ids = list(range(63))
    colors = viz_data.cluster_colors(ids)
    assert colors == viz_data.cluster_colors(ids)
    assert colors[-1] == viz_data.NOISE_COLOR
    red, green, blue = (int(viz_data.NOISE_COLOR[i : i + 2], 16) for i in (1, 3, 5))
    assert red == green == blue
    assert len({colors[i] for i in ids}) == 63
    assert viz_data.NOISE_COLOR not in {colors[i] for i in ids}
    assert set(viz_data.cluster_colors([0, 1, 3])) == {-1, 0, 1, 3}


def test_short_label_truncates_with_an_ellipsis() -> None:
    assert viz_data.short_label(name="abc", max_chars=5) == "abc"
    assert viz_data.short_label(name="abcdefgh", max_chars=5) == "abcd…"
    assert len(viz_data.short_label(name="abcdefgh", max_chars=5)) == 5


def test_centroids_use_the_median_of_each_cluster_only() -> None:
    coords = np.array([[0.0, 1.0], [2.0, 9.0], [100.0, 500.0], [10.0, 30.0], [5.0, 7.0]])
    labels = np.array([0, 0, 0, 2, -1])
    result = viz_data.centroids(coords=coords, labels=labels)
    assert result == {0: (2.0, 9.0), 2: (10.0, 30.0)}


def test_centroids_of_the_fixture_are_its_hand_written_centers() -> None:
    result = viz_data.centroids(coords=vh.make_coords(), labels=vh.LABELS)
    assert set(result) == {0, 1, 3}
    for cluster_id, center in result.items():
        assert center == pytest.approx(vh.CENTERS[cluster_id])


def test_spread_labels_removes_overlap_and_keeps_far_labels() -> None:
    far = np.array([[0.0, 0.0], [10.0, 10.0]])
    widths = np.array([1.0, 1.0])
    assert np.array_equal(viz_data.spread_labels(anchors=far, widths=widths, height=0.5), far)
    stacked = np.zeros((6, 2))
    full = np.full(6, 1.0)
    placed = viz_data.spread_labels(anchors=stacked, widths=full, height=0.5)
    for i in range(6):
        for j in range(i + 1, 6):
            dx, dy = np.abs(placed[i] - placed[j])
            assert dx >= 1.0 - 1e-6 or dy >= 0.5 - 1e-6
    again = viz_data.spread_labels(anchors=stacked, widths=full, height=0.5)
    assert np.array_equal(placed, again)


def test_label_layout_orders_by_size_and_keeps_each_anchor_with_its_id() -> None:
    texts = {0: "a", 1: "b", 3: "c"}
    ids, anchors, _ = viz_data.label_layout(
        coords=vh.make_coords(), labels=vh.LABELS, texts=texts
    )
    assert ids == [3, 1, 0]
    assert np.allclose(anchors, [[-30.0, 90.0], [200.0, 7.0], [1.0, 50.0]])


def test_label_layout_separates_close_anchors_by_their_own_widths() -> None:
    coords = np.array([[0.0, 0.0], [7.0, 0.0], [100.0, 0.0]])
    labels = np.array([0, 1, -1])
    texts = {0: "x" * 40, 1: "ab"}
    ids, anchors, placed = viz_data.label_layout(coords=coords, labels=labels, texts=texts)
    half_sum = (40 + 2) / 2 * viz_data.LABEL_CHAR_WIDTH * 100.0
    gap = abs(placed[ids.index(1)][0] - placed[ids.index(0)][0])
    assert half_sum <= gap <= half_sum + 0.05
    assert placed[:, 1].tolist() == [0.0, 0.0]
    assert placed[ids.index(0)][0] < placed[ids.index(1)][0]


@pytest.mark.parametrize("bad_name", ["", "A, B"])
def test_load_refuses_invalid_cluster_names(tmp_path: Path, bad_name: str) -> None:
    root = vh.make_tree(tmp_path)
    path = paths.taxonomy_dir(root) / "cluster_names.csv"
    rows = [[0, bad_name, 5], [1, "Beta / Gamma", 8], [3, "Third", 12]]
    fileio.write_csv_atomic(path, ("cluster_id", "name", "size"), rows)
    with pytest.raises(ValueError, match="cluster name"):
        viz_data.load_plot_data(root)


def test_cluster_table_sums_df_columns_in_the_right_places(data: viz_data.PlotData) -> None:
    table = viz_data.cluster_table(data)
    assert table["cluster_id"].tolist() == [0, 1, 3, -1]
    assert table["name"].tolist() == [*vh.NAMES.values(), "Noise"]
    assert table["size"].tolist() == [5, 8, 12, 5]
    first = table.iloc[0]
    assert first["df_resumes"] == sum(vh.df_resumes(i) for i in range(5))
    assert first["df_postings"] == sum(vh.df_postings(i) for i in range(5))
    assert (first["n_resumes_only"], first["n_postings_only"], first["n_both"]) == (3, 0, 2)


def test_cluster_table_counts_postings_only_skills(data: viz_data.PlotData) -> None:
    swapped = dataclasses.replace(data, df_resumes=data.df_postings, df_postings=data.df_resumes)
    first = viz_data.cluster_table(swapped).iloc[0]
    assert (first["n_resumes_only"], first["n_postings_only"]) == (0, 3)


def test_one_corpus_fraction(data: viz_data.PlotData) -> None:
    assert viz_data.one_corpus_fraction(data) == 6 / 30


def test_grid_table_leaves_out_skipped_configurations(data: viz_data.PlotData) -> None:
    table = viz_data.grid_table(data.results)
    assert table["config_id"].tolist() == ["dbscan-01", "dbscan-02"]
