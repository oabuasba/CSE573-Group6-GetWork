import numpy as np
import pytest
from sklearn import metrics

from getwork.taxonomy import grid, reduce

pytestmark = pytest.mark.usefixtures("offline")


def _blobs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    points = np.vstack([c + rng.normal(scale=0.3, size=(20, 3)) for c in centers])
    return points, np.repeat([0, 1, 2], 20)


def _same_partition(labels: np.ndarray, truth: np.ndarray) -> bool:
    pairs = {(int(a), int(b)) for a, b in zip(labels, truth)}
    return len(pairs) == len(set(truth.tolist())) == len({a for a, _ in pairs})


def test_enumerate_configs_order_and_ids() -> None:
    configs = grid.enumerate_configs(grid.GridConfig())
    assert [c.method for c in configs].count("dbscan") == 18
    assert [c.method for c in configs].count("hdbscan") == 5
    assert [c.method for c in configs].count("kmeans") == 6
    assert [c.method for c in configs] == ["dbscan"] * 18 + ["hdbscan"] * 5 + ["kmeans"] * 6
    assert configs[0].config_id == "dbscan-01"
    assert configs[17].config_id == "dbscan-18"
    assert configs[18].config_id == "hdbscan-01"
    assert configs[-1].config_id == "kmeans-06"
    assert configs[0].params == {"eps": 0.2, "min_samples": 3}
    assert configs[1].params == {"eps": 0.2, "min_samples": 5}
    assert configs[3].params == {"eps": 0.3, "min_samples": 3}
    assert configs[18].params == {"min_cluster_size": 5}
    assert configs[-1].params == {"k": 60}


def test_grid_config_from_mapping() -> None:
    cfg = grid.GridConfig.from_mapping({"kmeans_k": [2, 3], "dbscan_eps": [1, 0.5]})
    assert cfg.kmeans_k == (2, 3)
    assert cfg.dbscan_eps == (1.0, 0.5)
    assert cfg.hdbscan_min_cluster_size == grid.GridConfig().hdbscan_min_cluster_size
    assert grid.GridConfig.from_mapping({}) == grid.GridConfig()
    with pytest.raises(ValueError, match="bogus"):
        grid.GridConfig.from_mapping({"bogus": [1]})
    with pytest.raises(ValueError, match="kmeans_k"):
        grid.GridConfig.from_mapping({"kmeans_k": []})
    with pytest.raises(ValueError, match="kmeans_k"):
        grid.GridConfig.from_mapping({"kmeans_k": [3, 0]})
    with pytest.raises(ValueError, match="dbscan_eps"):
        grid.GridConfig.from_mapping({"dbscan_eps": [-1.0]})
    with pytest.raises(ValueError, match="dbscan_min_samples"):
        grid.GridConfig.from_mapping({"dbscan_min_samples": [2.5]})
    with pytest.raises(ValueError, match="hdbscan_min_cluster_size"):
        grid.GridConfig.from_mapping({"hdbscan_min_cluster_size": 5})


def test_run_config_methods_on_blobs() -> None:
    points, truth = _blobs()
    kmeans = grid.ClusterConfig(config_id="kmeans-01", method="kmeans", params={"k": 3})
    labels = grid.run_config(kmeans, points, 0)
    assert labels is not None and labels.dtype == np.int32
    assert _same_partition(labels, truth)
    assert np.array_equal(labels, grid.run_config(kmeans, points, 0))

    dbscan = grid.ClusterConfig(
        config_id="dbscan-01", method="dbscan", params={"eps": 1.0, "min_samples": 3}
    )
    with_outlier = np.vstack([points, [[50.0, 50.0, 50.0]]])
    labels = grid.run_config(dbscan, with_outlier, 0)
    assert labels is not None
    assert labels[-1] == -1
    assert _same_partition(labels[:-1], truth)
    assert np.array_equal(labels, grid.run_config(dbscan, with_outlier, 0))

    hdbscan = grid.ClusterConfig(
        config_id="hdbscan-01", method="hdbscan", params={"min_cluster_size": 5}
    )
    labels = grid.run_config(hdbscan, points, 0)
    assert labels is not None and len(set(labels.tolist()) - {-1}) >= 2
    assert np.array_equal(labels, grid.run_config(hdbscan, points, 0))


def test_run_config_infeasible_returns_none() -> None:
    points, _ = _blobs()
    n = len(points)
    too_many_k = grid.ClusterConfig(config_id="kmeans-01", method="kmeans", params={"k": n})
    too_big = grid.ClusterConfig(
        config_id="hdbscan-01", method="hdbscan", params={"min_cluster_size": n + 1}
    )
    at_limit = grid.ClusterConfig(
        config_id="hdbscan-02", method="hdbscan", params={"min_cluster_size": n}
    )
    assert grid.run_config(at_limit, points, 0) is not None
    assert grid.run_config(too_many_k, points, 0) is None
    assert grid.run_config(too_big, points, 0) is None


def test_umap_reduce_shape_determinism_and_small_input() -> None:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16)) * 5
    vectors = np.vstack([c + rng.normal(scale=0.3, size=(20, 16)) for c in centers])
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    cfg = reduce.ReduceConfig()
    first = reduce.umap_reduce(vectors, cfg)
    assert first.dtype == np.float32 and first.shape == (60, 5)
    assert np.array_equal(first, reduce.umap_reduce(vectors, cfg))
    big_neighbors = reduce.ReduceConfig(n_neighbors=500)
    assert reduce.umap_reduce(vectors, big_neighbors).shape == (60, 5)
    with pytest.raises(ValueError):
        reduce.umap_reduce(vectors[:19], cfg)


def test_score_labels_values() -> None:
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(10, 4)) + 2.0
    labels = np.array([0, 0, 0, 1, 1, -1, -1, 2, 2, 2])
    n_clusters, noise, silhouette, largest = grid.score_labels(labels, vectors)
    assert (n_clusters, noise, largest) == (3, 0.2, 0.3)
    mask = labels >= 0
    expected = metrics.silhouette_score(vectors[mask], labels[mask], metric="cosine")
    assert silhouette == pytest.approx(expected)
    assert isinstance(silhouette, float)
    assert silhouette != pytest.approx(
        metrics.silhouette_score(vectors, np.where(labels < 0, 3, labels), metric="cosine")
    )
    one_cluster = grid.score_labels(np.array([0, 0, 0, -1]), vectors[:4])
    assert one_cluster == (1, 0.25, None, 0.75)
    no_room = grid.score_labels(np.array([0, 1, 2, 3]), vectors[:4])
    assert no_room[0] == 4 and no_room[2] is None
    all_noise = grid.score_labels(np.array([-1, -1, -1]), vectors[:3])
    assert all_noise == (0, 1.0, None, 0.0)


def test_run_grid_shapes_skipped_and_progress() -> None:
    points, _ = _blobs()
    configs = [
        grid.ClusterConfig(config_id="kmeans-01", method="kmeans", params={"k": 3}),
        grid.ClusterConfig(config_id="kmeans-02", method="kmeans", params={"k": 500}),
        grid.ClusterConfig(
            config_id="dbscan-01", method="dbscan", params={"eps": 0.001, "min_samples": 5}
        ),
    ]
    calls: list[tuple[int, int]] = []
    results, matrix = grid.run_grid(
        points, points, configs, 0, progress=lambda done, total: calls.append((done, total))
    )
    assert [r.config_id for r in results] == ["kmeans-01", "kmeans-02", "dbscan-01"]
    assert matrix.shape == (3, 60) and matrix.dtype == np.int32
    assert results[0].n_clusters == 3 and results[0].skipped is None
    assert results[0].silhouette is not None and results[0].silhouette > 0.5
    assert results[1].skipped and results[1].silhouette is None
    assert results[1].n_clusters == 0 and results[1].noise_fraction == 1.0
    assert (matrix[1] == -2).all()
    assert results[2].noise_fraction == 1.0 and results[2].skipped is None
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_umap_reduce_accepts_exactly_min_rows() -> None:
    vectors = np.random.default_rng(1).normal(size=(20, 8)).astype(np.float32)
    assert reduce.umap_reduce(vectors, reduce.ReduceConfig()).shape == (20, 5)
