from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import viz_helpers as vh

from getwork import fileio, paths
from getwork.taxonomy import cli, cli_plot, counts
from getwork.taxonomy.reduce import ReduceConfig

pytestmark = pytest.mark.usefixtures("offline")

PACKAGE = Path(__file__).resolve().parents[2] / "getwork" / "taxonomy"
ALL_FILES = {
    "cluster_map.png", "cluster_map.html", "cluster_sizes.png", "cluster_corpus_balance.png",
    "cluster_one_corpus.png", "skill_document_frequency.png", "grid_overview.png",
    "cluster_summary.csv", "index.html",
}


def _reducer(vectors: np.ndarray, config: ReduceConfig) -> np.ndarray:
    _reducer.seen.append(config)
    return np.stack([np.arange(len(vectors), dtype=float), vectors[:, 0]], axis=1)


_reducer.seen = []  # type: ignore[attr-defined]


def _plot(root: Path, *extra: str) -> int:
    return cli.main(["plot", "--artifacts-dir", str(root), *extra], reduce_fn=_reducer)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return vh.make_tree(tmp_path / "art")


def _written(root: Path) -> set[str]:
    plots = cli_plot.default_out_dir(root)
    return {p.name for p in plots.iterdir()}


def test_plot_writes_every_output_to_the_default_dir(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _plot(root) == 0
    assert _written(root) == ALL_FILES
    out = json.loads(capsys.readouterr().out)
    assert set(out["files"]) == ALL_FILES
    plots = root / "taxonomy" / "plots"
    assert (plots / "cluster_map.png").read_bytes()[:4] == b"\x89PNG"
    assert "skill07" in (plots / "cluster_map.html").read_text(encoding="utf-8")


def test_index_links_every_plot_and_embeds_the_interactive_map(root: Path) -> None:
    assert _plot(root) == 0
    index = (root / "taxonomy" / "plots" / "index.html").read_text(encoding="utf-8")
    for name in ALL_FILES - {"index.html", "cluster_summary.csv"}:
        assert name in index, name
    assert "<iframe" in index
    assert "3 clusters" in index and "5 of 30 skills are noise" in index
    assert "20.0%" in index


def test_summary_csv_rows_match_the_clusters(root: Path) -> None:
    assert _plot(root) == 0
    rows = vh.read_csv(root / "taxonomy" / "plots" / "cluster_summary.csv")
    assert [r["cluster_id"] for r in rows] == ["0", "1", "3", "-1"]
    assert rows[0]["name"] == "Alpha tools"
    assert rows[0]["df_resumes"] == str(sum(vh.df_resumes(i) for i in range(5)))
    assert rows[0]["n_resumes_only"] == "3"


def test_no_interactive_skips_the_html_map(root: Path) -> None:
    assert _plot(root, "--no-interactive") == 0
    assert _written(root) == ALL_FILES - {"cluster_map.html"}
    index = (root / "taxonomy" / "plots" / "index.html").read_text(encoding="utf-8")
    assert "<iframe" not in index and "cluster_map.html" not in index


def test_out_dir_and_seed_are_used(root: Path, tmp_path: Path) -> None:
    _reducer.seen.clear()
    out = tmp_path / "elsewhere"
    assert _plot(root, "--out-dir", str(out), "--seed", "11", "--no-interactive") == 0
    assert {p.name for p in out.iterdir()} == ALL_FILES - {"cluster_map.html"}
    assert not (root / "taxonomy" / "plots").exists()
    assert (_reducer.seen[0].n_components, _reducer.seen[0].seed) == (2, 11)


def _break_selection_context(root: Path) -> None:
    path = paths.taxonomy_dir(root) / "selection.json"
    record = fileio.read_json(path)
    record["context"]["seed"] = 5
    fileio.write_json_atomic(path, record)


def _break_counts_version(root: Path) -> None:
    path = paths.skills_dir(root) / counts.SUMMARY_FILE
    fileio.write_json_atomic(path, {"selected_skills_version": "sha256:zz"})


def _drop_config_id(root: Path) -> None:
    path = paths.taxonomy_dir(root) / "selection.json"
    record = fileio.read_json(path)
    del record["config_id"]
    fileio.write_json_atomic(path, record)


@pytest.mark.parametrize(
    ("damage", "needle"),
    [
        (lambda r: (r / "taxonomy" / "cluster_names.csv").unlink(), "taxonomy name"),
        (lambda r: (r / "skills" / "skill_vectors.npy").unlink(), "taxonomy embed"),
        (lambda r: (r / "taxonomy" / "grid_labels.npy").unlink(), "grid_labels.npy"),
        (lambda r: (r / "taxonomy" / "selection.json").unlink(), "selection.json"),
        (_break_selection_context, "grid_context.json"),
        (_break_counts_version, "different selected skill list"),
        (_drop_config_id, "config_id"),
    ],
)
def test_input_errors_exit_2_and_write_nothing(
    root: Path, capsys: pytest.CaptureFixture[str], damage: object, needle: str
) -> None:
    damage(root)
    before = vh.tree_listing(root)
    assert _plot(root) == 2
    assert needle in capsys.readouterr().err
    assert vh.tree_listing(root) == before


def test_error_with_explicit_out_dir_creates_it_not(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "plots-out"
    assert _plot(tmp_path / "empty", "--out-dir", str(out)) == 2
    assert "error:" in capsys.readouterr().err
    assert not out.exists()


def test_unwritable_out_dir_exits_2(root: Path, tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    assert _plot(root, "--out-dir", str(blocker / "sub")) == 2


def test_bad_max_label_chars_is_a_usage_error(root: Path) -> None:
    assert _plot(root, "--max-label-chars", "0") == 2
    assert not (root / "taxonomy" / "plots").exists()


def test_importing_the_cli_does_not_load_plotting_or_heavy_libraries() -> None:
    code = (
        "import sys; import getwork.taxonomy.cli;"
        "bad = [m for m in ('matplotlib', 'plotly', 'torch', 'sentence_transformers')"
        " if m in sys.modules]; assert not bad, bad"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_readme_documents_the_plots() -> None:
    text = (PACKAGE / "README.md").read_text(encoding="utf-8")
    assert "## Plots" in text
    for needle in ("taxonomy plot", "index.html", "--no-interactive", "grid_overview.png"):
        assert needle in text, needle
