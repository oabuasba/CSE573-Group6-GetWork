from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from evaluation_helpers import make_eval_tree

from getwork.evaluation import cli

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "2", "--dev-postings", "2", "--test-resumes", "4", "--test-postings", "4",
    "--human-resumes", "2", "--human-postings", "2", "--pool-cap", "10", "--seed", "7",
]
OUTPUTS = ("run.json", "sets.json", "pool_postings.json")


def _sample(tmp_path: Path, *extra: str) -> tuple[int, Path]:
    root, processed = make_eval_tree(tmp_path / "tree") if not (tmp_path / "tree").exists() \
        else (tmp_path / "tree" / "art", tmp_path / "tree" / "processed")
    out = tmp_path / "out"
    code = cli.main([
        "sample", "--artifacts-dir", str(root), "--processed-dir", str(processed),
        "--out-dir", str(out), *SMALL, *extra,
    ])
    return code, out


def _read(out: Path) -> dict[str, bytes]:
    return {name: (out / name).read_bytes() for name in OUTPUTS}


def _hash_tree(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*")) if p.is_file()
    }


def test_sample_writes_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = _sample(tmp_path)
    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["stages"]["test"] == {"resumes": 4, "postings": 4}
    assert summary["n_pool"] == 10 and summary["n_resumes_unlabeled"] == 4
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["seed"] == 7 and manifest["params"]["pool_cap"] == 10
    assert {"documents", "embedding", "n_resumes", "n_postings"} <= set(manifest["stamps"])
    assert manifest["artifacts_dir"] == str(tmp_path / "tree" / "art")
    sets = json.loads((out / "sets.json").read_text())
    assert "pool" not in sets and set(sets["stages"]) == {"dev", "test", "human"}
    pool = json.loads((out / "pool_postings.json").read_text())["ids"]
    assert len(pool) == 10 and set(sets["stages"]["test"]["postings"]) <= set(pool)


def test_sample_refuses_overwrite_and_is_reproducible(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _sample(tmp_path)[0] == 0
    out = tmp_path / "out"
    before = _read(out)
    capsys.readouterr()
    code, _ = _sample(tmp_path)
    err = capsys.readouterr().err
    assert code == 2 and "run.json" in err and "--force" in err
    assert _read(out) == before
    assert _sample(tmp_path, "--force")[0] == 0
    assert _read(out) == before


def test_sample_does_not_touch_inputs(tmp_path: Path) -> None:
    make_eval_tree(tmp_path / "tree")
    before = _hash_tree(tmp_path / "tree")
    assert _sample(tmp_path)[0] == 0
    assert _hash_tree(tmp_path / "tree") == before
