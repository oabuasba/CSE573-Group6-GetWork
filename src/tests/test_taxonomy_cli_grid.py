from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from getwork.embed import store
from getwork.taxonomy import cli, cli_common, counts, grid, report
from getwork.taxonomy.reduce import ReduceConfig

pytestmark = pytest.mark.usefixtures("offline")

VERSION = "sha256:aaaaaaaaaaaa"

N_BLOB = 10
GRID_FILE = {
    "dbscan_eps": [0.5], "dbscan_min_samples": [3],
    "hdbscan_min_cluster_size": [5], "kmeans_k": [3, 100],
}
ALL_NOISE_GRID = {
    "dbscan_eps": [0.001], "dbscan_min_samples": [5],
    "hdbscan_min_cluster_size": [100], "kmeans_k": [100],
}


def _reducer(vectors: np.ndarray, config: object) -> np.ndarray:
    return vectors[:, :3]


def _blob_vectors(n_blobs: int, per_blob: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(0)
    rows = []
    for blob in range(n_blobs):
        center = np.zeros(8)
        center[blob % 3] = 1.0
        rows.extend(center + 0.05 * rng.standard_normal(8) for _ in range(per_blob))
    vectors = np.asarray(rows, dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return [f"skill-{i:02d}" for i in range(len(rows))], vectors


def _grid_root(
    tmp_path: Path, *, n_per_blob: int = N_BLOB, vectors_version: str | None = None
) -> Path:
    root = tmp_path / "art"
    skills, vectors = _blob_vectors(3, n_per_blob)
    items = [
        counts.SkillCount(skill=s, df_total=100 - i, df_resumes=50, df_postings=50 - i)
        for i, s in enumerate(skills)
    ]
    summary = counts.counts_summary(items, items, {"resumes": 1, "postings": 1}, VERSION, 1000, 2)
    counts.write_counts(root / "skills", items, items, summary)
    meta = {
        "embedding": {"model": "m", "backend": "fake", "dim": 8, "max_seq_length": None,
                      "dtype": None},
        "kind": "skills",
        "selected_skills_version": vectors_version or summary["selected_skills_version"],
        "alias_table_version": VERSION,
    }
    store.write_vectors(root / "skills", "skill_vectors", skills, vectors, meta)
    return root


def _run_grid(root: Path, grid: dict | None, *extra: str, reducer: object = _reducer) -> int:
    args = ["grid", "--artifacts-dir", str(root), *extra]
    if grid is not None:
        grid_file = root.parent / "grid.json"
        grid_file.write_text(json.dumps(grid))
        args += ["--grid-file", str(grid_file)]
    return cli.main(args, reduce_fn=reducer)  # type: ignore[arg-type]


def test_grid_writes_selection_and_review(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _grid_root(tmp_path)
    assert _run_grid(root, GRID_FILE) == 0
    out = root / "taxonomy"
    results, labels = report.read_grid(out)
    assert [r.config_id for r in results] == ["dbscan-01", "hdbscan-01", "kmeans-01", "kmeans-02"]
    assert labels.shape == (4, 3 * N_BLOB) and results[3].skipped is not None
    selection = report.read_selection(out)
    assert selection["source"] == "auto" and selection["n_clusters"] == 3
    assert selection["n_eligible"] == 3 and selection["config_id"] in {"dbscan-01", "kmeans-01"}
    assert json.loads(capsys.readouterr().out) == selection
    context = report.read_context(out)
    assert context["reduce"] == {"n_components": 5, "n_neighbors": 15, "min_dist": 0.0, "seed": 0}
    assert context["grid"]["kmeans_k"] == [3, 100] and context["grid"]["dbscan_eps"] == [0.5]
    assert (context["seed"], context["max_noise"], context["min_clusters"]) == (0, 0.3, 2)
    assert context["alias_table_version"] == VERSION and context["embedding"]["model"] == "m"
    assert context["selected_skills_version"] == selection["context"]["selected_skills_version"]
    review = (out / f"review_{selection['config_id']}.csv").read_text().splitlines()
    assert review[0] == "cluster_id,size,df_sum,members"
    assert selection["config_id"] == "dbscan-01"
    expected_df = [955, 855, 755]  # sum of 100 - i over the ten skills of each blob
    assert review[1:] == [
        f"{c},{N_BLOB},{expected_df[c]},"
        + "; ".join(f"skill-{i:02d}" for i in range(c * N_BLOB, (c + 1) * N_BLOB))
        for c in range(3)
    ]


def test_grid_option_values_reach_reducer_and_context(tmp_path: Path) -> None:
    root = _grid_root(tmp_path)
    seen: list[ReduceConfig] = []

    def spy(vectors: np.ndarray, config: ReduceConfig) -> np.ndarray:
        seen.append(config)
        return vectors[:, :3]

    code = _run_grid(
        root, GRID_FILE, "--seed", "7", "--umap-components", "3", "--umap-neighbors", "9",
        "--min-clusters", "2", "--max-noise", "0.5", reducer=spy,
    )
    assert code == 0 and len(seen) == 1
    assert (seen[0].n_components, seen[0].n_neighbors, seen[0].seed) == (3, 9, 7)
    context = report.read_context(root / "taxonomy")
    assert (context["seed"], context["max_noise"]) == (7, 0.5)


def test_grid_no_eligible_config_exit_2_keeps_grid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _grid_root(tmp_path)
    assert _run_grid(root, ALL_NOISE_GRID, "--max-noise", "0") == 2
    err = capsys.readouterr().err
    assert "--max-noise" in err and "--grid-file" in err
    out = root / "taxonomy"
    results, _ = report.read_grid(out)
    assert results[0].noise_fraction == 1.0
    assert (out / "grid_context.json").exists()
    assert not (out / "selection.json").exists() and not list(out.glob("review_*.csv"))


def test_grid_rejects_stale_vectors_and_tiny_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(vectors: np.ndarray, config: object) -> np.ndarray:
        raise AssertionError("reducer must not run")

    stale = _grid_root(tmp_path / "a", vectors_version="sha256:000000000000")
    assert _run_grid(stale, GRID_FILE, reducer=boom) == 2
    assert "taxonomy embed" in capsys.readouterr().err
    assert not (stale / "taxonomy").exists()
    tiny = _grid_root(tmp_path / "b", n_per_blob=6)
    assert _run_grid(tiny, GRID_FILE, reducer=boom) == 2
    err = capsys.readouterr().err
    assert "--min-df" in err and "18" in err
    assert not (tiny / "taxonomy").exists()


def test_grid_missing_inputs_and_bad_options_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    assert _run_grid(empty, None) == 2
    assert "counts_summary.json" in capsys.readouterr().err and not empty.exists()
    root = _grid_root(tmp_path / "x")
    for bad in ("1.5", "-0.1", "abc"):
        assert _run_grid(root, GRID_FILE, "--max-noise", bad) == 2
    assert _run_grid(root, {"bogus": [1]}) == 2
    assert "bogus" in capsys.readouterr().err
    assert _run_grid(root, GRID_FILE, "--min-clusters", "0") == 2
    assert not (root / "taxonomy").exists()
    vectors_only = _grid_root(tmp_path / "y")
    (vectors_only / "skills" / "skill_vectors.npy").unlink()
    assert _run_grid(vectors_only, GRID_FILE) == 2
    assert "taxonomy embed" in capsys.readouterr().err


def test_select_manual_override(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _grid_root(tmp_path)
    assert _run_grid(root, GRID_FILE) == 0
    out = root / "taxonomy"
    auto = report.read_selection(out)
    other = "hdbscan-01" if auto["config_id"] != "hdbscan-01" else "kmeans-01"
    capsys.readouterr()
    assert cli.main(["select", "--artifacts-dir", str(root), "--config-id", other]) == 0
    manual = report.read_selection(out)
    assert manual["config_id"] == other and manual["source"] == "manual"
    assert manual["context"] == report.read_context(out) and manual["n_eligible"] is None
    assert json.loads(capsys.readouterr().out) == manual
    assert (out / f"review_{other}.csv").read_text().startswith("cluster_id,size,df_sum,members")
    before = (out / "selection.json").read_bytes()
    for bad in ("nope-99", "kmeans-02"):
        assert cli.main(["select", "--artifacts-dir", str(root), "--config-id", bad]) == 2
        assert (out / "selection.json").read_bytes() == before
    assert "skipped" in capsys.readouterr().err


def test_select_requires_config_id_and_grid_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _grid_root(tmp_path)
    assert cli.main(["select", "--artifacts-dir", str(root)]) == 2
    assert cli.main(["select", "--artifacts-dir", str(root), "--config-id", "kmeans-01"]) == 2
    assert "run taxonomy grid" in capsys.readouterr().err
    assert not (root / "taxonomy").exists()


def test_grid_defaults_are_pinned() -> None:
    args = cli.build_parser().parse_args(["grid"])
    assert (args.max_noise, args.min_clusters, args.seed) == (0.30, 2, 0)
    assert (args.umap_components, args.umap_neighbors, args.grid_file) == (5, 15, None)
    assert cli.build_parser().parse_args(["embed"]).chunk_size == 256
    assert cli.build_parser().parse_args(["select", "--config-id", "x"]).artifacts_dir == Path(
        "artifacts_trial"
    )

def _noisy_reducer(vectors: np.ndarray, config: object) -> np.ndarray:
    reduced = vectors[:, :3].copy()
    reduced[-3:] = [[5.0, 5.0, 5.0], [-5.0, 5.0, -5.0], [5.0, -5.0, -5.0]]
    return reduced


NOISY_GRID = {
    "dbscan_eps": [0.5], "dbscan_min_samples": [3],
    "hdbscan_min_cluster_size": [100], "kmeans_k": [100],
}


def test_grid_max_noise_alone_decides_eligibility(tmp_path: Path) -> None:
    root = _grid_root(tmp_path)
    out = root / "taxonomy"
    assert _run_grid(root, NOISY_GRID, reducer=_noisy_reducer) == 0
    results, _ = report.read_grid(out)
    assert results[0].n_clusters == 3 and results[0].noise_fraction == pytest.approx(0.1)
    assert report.read_selection(out)["config_id"] == "dbscan-01"
    (out / "selection.json").unlink()
    assert _run_grid(root, NOISY_GRID, "--max-noise", "0.2", reducer=_noisy_reducer) == 0
    assert _run_grid(root, NOISY_GRID, "--max-noise", "0.05", reducer=_noisy_reducer) == 2
    assert (out / "grid_results.csv").exists() and (out / "grid_labels.npy").exists()
    assert report.read_context(out)["max_noise"] == 0.05


def test_grid_min_clusters_and_seed_take_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _grid_root(tmp_path)
    assert _run_grid(root, GRID_FILE, "--min-clusters", "4") == 2
    assert report.read_context(root / "taxonomy")["min_clusters"] == 4
    seeds: list[int] = []
    real_run_grid = grid.run_grid

    def spy(*positional: object, **keyword: object) -> object:
        seeds.append(keyword["seed"] if "seed" in keyword else positional[3])  # type: ignore
        return real_run_grid(*positional, **keyword)  # type: ignore[arg-type]

    monkeypatch.setattr(grid, "run_grid", spy)
    assert _run_grid(root, GRID_FILE, "--seed", "11") == 0
    assert seeds == [11]


MULTI_GRID = {
    "dbscan_eps": [0.001], "dbscan_min_samples": [3],
    "hdbscan_min_cluster_size": [100], "kmeans_k": [2, 3],
}


def _review_lines(out: Path, config_id: str) -> list[str]:
    return (out / f"review_{config_id}.csv").read_text().splitlines()


def _expected_review(root: Path, config_id: str) -> list[str]:
    out = root / "taxonomy"
    results, labels = report.read_grid(out)
    skills, df = cli_common.selected_with_df(root)
    row = [r.config_id for r in results].index(config_id)
    rows = report.review_rows(labels[row], skills, df)
    return [",".join(str(r[c]) for c in report.REVIEW_COLUMNS) for r in rows]


def test_review_file_uses_the_chosen_configurations_own_labels(tmp_path: Path) -> None:
    root = _grid_root(tmp_path)
    out = root / "taxonomy"
    assert _run_grid(root, MULTI_GRID) == 0
    chosen = report.read_selection(out)["config_id"]
    assert chosen == "kmeans-02"
    lines = _review_lines(out, chosen)
    assert lines[1:] == _expected_review(root, chosen)
    assert sorted(int(line.split(",")[2]) for line in lines[1:]) == [755, 855, 955]
    assert cli.main(["select", "--artifacts-dir", str(root), "--config-id", "kmeans-01"]) == 0
    manual = _review_lines(out, "kmeans-01")
    assert manual[1:] == _expected_review(root, "kmeans-01")
    assert sorted(int(line.split(",")[1]) for line in manual[1:]) == [10, 20]
    assert len(manual) == 3 and manual[1:] != lines[1:]


def test_select_rejects_labels_from_a_different_skill_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _grid_root(tmp_path)
    assert _run_grid(root, GRID_FILE) == 0
    out = root / "taxonomy"
    before = (out / "selection.json").read_bytes()
    np.save(out / "grid_labels.npy", np.zeros((4, 5), dtype=np.int32))
    capsys.readouterr()
    assert cli.main(["select", "--artifacts-dir", str(root), "--config-id", "kmeans-01"]) == 2
    assert "taxonomy grid again" in capsys.readouterr().err
    assert (out / "selection.json").read_bytes() == before


def test_grid_file_errors_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _grid_root(tmp_path)
    missing = tmp_path / "missing.json"
    assert cli.main(["grid", "--artifacts-dir", str(root), "--grid-file", str(missing)]) == 2
    assert str(missing) in capsys.readouterr().err
    for name, text, message in (("bad.json", "{oops", "not valid JSON"),
                                ("list.json", "[1, 2]", "JSON object")):
        path = tmp_path / name
        path.write_text(text)
        assert cli.main(["grid", "--artifacts-dir", str(root), "--grid-file", str(path)]) == 2
        assert message in capsys.readouterr().err
    assert not (root / "taxonomy").exists()

def test_grid_rejects_vectors_whose_ids_differ_from_selected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _grid_root(tmp_path)
    current = store.read_vectors(root / "skills", "skill_vectors")
    store.write_vectors(
        root / "skills", "skill_vectors", current.ids[::-1], current.vectors, current.meta
    )
    assert _run_grid(root, GRID_FILE) == 2
    assert "rank order" in capsys.readouterr().err
    assert not (root / "taxonomy").exists()


def test_selection_n_eligible_follows_max_noise_and_min_clusters(tmp_path: Path) -> None:
    root = _grid_root(tmp_path)
    grid_file = {
        "dbscan_eps": [0.5], "dbscan_min_samples": [3],
        "hdbscan_min_cluster_size": [100], "kmeans_k": [2, 3],
    }

    def n_eligible(*extra: str) -> int | None:
        code = _run_grid(root, grid_file, *extra, reducer=_noisy_reducer)
        return report.read_selection(root / "taxonomy")["n_eligible"] if code == 0 else None

    assert n_eligible() == 3
    assert n_eligible("--min-clusters", "3") == 2
    assert n_eligible("--max-noise", "0.05") == 2
    assert n_eligible("--max-noise", "0.05", "--min-clusters", "3") == 1
    assert n_eligible("--min-clusters", "4") is None
