from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from evaluation_helpers import FIRST_LABELS, PIPELINE_MODEL, make_eval_tree

from getwork import paths
from getwork.evaluation import loading

pytestmark = pytest.mark.usefixtures("offline")


def _write_resumes(processed_dir: Path, rows: dict[str, list]) -> None:
    path = paths.processed_parquet(processed_dir, "resumes")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows), path)


def test_load_eval_inputs(tmp_path: Path) -> None:
    root, processed = make_eval_tree(tmp_path, n_resumes=20, n_postings=6)
    loaded = loading.load_eval_inputs(root, processed)
    assert loaded.root == root and len(loaded.data.resumes) == 20 and len(loaded.data.postings) == 6
    assert list(loaded.first_labels) == list(loaded.data.resumes)
    unlabeled = [id_ for id_, label in loaded.first_labels.items() if label is None]
    assert unlabeled == ["r-10", "r-20"]
    assert loaded.first_labels["r-1"] == FIRST_LABELS[0]
    parent = tmp_path / "runs"
    parent.mkdir()
    (parent / "v1").symlink_to(root)
    assert loading.load_eval_inputs(parent, processed).root == parent / "v1"


def test_missing_processed_resume(tmp_path: Path) -> None:
    root, processed = make_eval_tree(tmp_path, n_resumes=5, n_postings=4)
    _write_resumes(processed, {"id": ["r-1", "r-2"], "first_label": ["a", "b"]})
    with pytest.raises(ValueError, match="r-3"):
        loading.load_eval_inputs(root, processed)


def test_read_first_labels(tmp_path: Path) -> None:
    _write_resumes(tmp_path, {"id": ["a", "b", "c"], "first_label": ["x", "", None]})
    assert loading.read_first_labels(tmp_path, ["c", "a", "b"]) == {"c": None, "a": "x", "b": None}
    with pytest.raises(ValueError, match="zzz"):
        loading.read_first_labels(tmp_path, ["a", "zzz"])


def test_read_texts(tmp_path: Path) -> None:
    _, processed = make_eval_tree(tmp_path, n_resumes=3, n_postings=2)
    assert loading.read_texts(processed, "postings", ["p-2", "p-1"]) == {
        "p-2": "posting text p-2 python sql", "p-1": "posting text p-1 python sql",
    }
    with pytest.raises(KeyError, match="p-9"):
        loading.read_texts(processed, "postings", ["p-1", "p-9"])
    with pytest.raises(KeyError) as err:
        loading.read_texts(processed, "resumes", [f"x{i}" for i in range(8)])
    assert "x4" in str(err.value) and "x5" not in str(err.value)


def test_pipeline_models(tmp_path: Path) -> None:
    root, _ = make_eval_tree(tmp_path, n_resumes=3, n_postings=2)
    assert loading.pipeline_models(root) == ["m1", PIPELINE_MODEL]
    assert loading.pipeline_models(root, extra=["zz", "m1", ""]) == ["m1", PIPELINE_MODEL, "zz"]
    (paths.canonical_dir(root, "postings") / "postings_canonical.jsonl").unlink()
    assert loading.pipeline_models(root) == ["m1", PIPELINE_MODEL]
