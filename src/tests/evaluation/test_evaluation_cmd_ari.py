from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest
from evaluation_helpers import SELECTION, make_eval_tree

from getwork import fileio, paths
from getwork.evaluation import cli

pytestmark = pytest.mark.usefixtures("offline")


def _args(tmp_path: Path, out: str = "out") -> list[str]:
    return [
        "ari", "--seeds", "3", "--k", "3",
        "--artifacts-dir", str(tmp_path / "tree" / "art"),
        "--processed-dir", str(tmp_path / "tree" / "processed"),
        "--out-dir", str(tmp_path / out),
    ]


def test_ari_command_output(tmp_path: Path) -> None:
    root, processed = make_eval_tree(tmp_path / "tree")
    assert cli.main(_args(tmp_path)) == 0
    path = tmp_path / "out" / "ari.json"
    result = fileio.read_json(path)
    first_labels = pq.read_table(paths.processed_parquet(processed, "resumes")).to_pydict()
    n_unlabeled = sum(1 for label in first_labels["first_label"] if not label)
    cluster_names = fileio.read_json(paths.taxonomy_dir(root) / "taxonomy.json")["clusters"]
    n_selected = pq.read_table(paths.skills_dir(root) / "skill_counts.parquet").num_rows
    assert set(result) == {
        "schema", "k", "seeds", "n_resumes", "n_unlabeled_excluded", "labels", "clusters",
        "raw_skills", "taxonomy", "vocabulary_gap",
    }
    assert result["seeds"] == [0, 1, 2] and result["k"] == 3
    assert n_unlabeled > 0 and result["n_unlabeled_excluded"] == n_unlabeled
    assert result["n_resumes"] == len(first_labels["id"]) - n_unlabeled
    assert result["clusters"]["dims"] == len(cluster_names)
    assert result["raw_skills"]["dims"] == n_selected
    assert result["taxonomy"] == SELECTION
    assert len(result["clusters"]["ari_by_seed"]) == 3
    first = path.read_bytes()
    assert cli.main(_args(tmp_path)) == 0
    assert path.read_bytes() == first


@pytest.mark.parametrize(
    "missing", [("taxonomy", "selection.json"), ("skills", "skill_counts.parquet")]
)
def test_ari_command_missing_inputs(
    tmp_path: Path, missing: tuple[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = make_eval_tree(tmp_path / "tree")
    (root / missing[0] / missing[1]).unlink()
    assert cli.main(_args(tmp_path)) == 2
    assert missing[1] in capsys.readouterr().err
    assert not (tmp_path / "out" / "ari.json").exists()
