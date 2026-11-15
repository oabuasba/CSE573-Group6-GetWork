from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from evaluation_helpers import make_eval_tree

from getwork.evaluation import cli, pairs, pooling

pytestmark = pytest.mark.usefixtures("offline")

SMALL = [
    "--dev-resumes", "3", "--dev-postings", "3", "--test-resumes", "6", "--test-postings", "6",
    "--human-resumes", "2", "--human-postings", "2", "--pool-cap", "20",
]


def _common(tmp_path: Path, tree: str = "tree") -> list[str]:
    return [
        "--artifacts-dir", str(tmp_path / tree / "art"),
        "--processed-dir", str(tmp_path / tree / "processed"),
        "--out-dir", str(tmp_path / "out"),
    ]


def _sampled(tmp_path: Path) -> list[str]:
    make_eval_tree(tmp_path / "tree")
    common = _common(tmp_path)
    assert cli.main(["sample", *common, *SMALL]) == 0
    return common


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_pool_dev_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    common = _sampled(tmp_path)
    out = tmp_path / "out"
    capsys.readouterr()
    assert cli.main(["pool", "--stage", "dev", *common]) == 0
    summary = json.loads(capsys.readouterr().out)
    rankings = pq.read_table(out / "pool_rankings_dev.parquet")
    assert rankings.schema == pooling.RANKINGS_SCHEMA
    assert set(rankings.column("stage").to_pylist()) == {"dev"}
    assert set(rankings.column("system").to_pylist()) == {"B1+", "B2", "Full"}
    assert set(rankings.column("direction").to_pylist()) == set(pairs.DIRECTIONS)
    sets = json.loads((out / "sets.json").read_text())["stages"]["dev"]
    queries = set(rankings.column("query_id").to_pylist())
    assert queries == set(sets["resumes"]) | set(sets["postings"])
    pair_table = pq.read_table(out / "pairs_dev.parquet")
    blind = _read_jsonl(out / "pairs_blind_dev.jsonl")
    assert len(blind) == pair_table.num_rows == summary["pairs"]
    assert summary["rows"] == rankings.num_rows
    weights = json.loads((out / "run.json").read_text())["weights"]["dev"]
    assert weights["source"] == "default" and weights["skills"] > 0
    assert not (out / "pairs_blind_human.jsonl").exists()


def test_pool_test_writes_human_blind_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = _sampled(tmp_path)
    out = tmp_path / "out"
    capsys.readouterr()
    assert cli.main(["pool", "--stage", "test", *common]) == 0
    captured = capsys.readouterr()
    assert "defaults (not tuned)" in captured.err
    summary = json.loads(captured.out)
    rankings = pq.read_table(out / "pool_rankings_test.parquet")
    sets = json.loads((out / "sets.json").read_text())["stages"]["human"]
    seed = json.loads((out / "run.json").read_text())["seed"]
    expected = pairs.human_pair_ids(
        rankings, human_resumes=sets["resumes"], human_postings=sets["postings"], seed=seed
    )
    human = _read_jsonl(out / "pairs_blind_human.jsonl")
    assert {row["pair_id"] for row in human} == expected and expected
    assert summary["human_pairs"] == len(expected)


def test_pool_weights_file_and_source(tmp_path: Path) -> None:
    common = _sampled(tmp_path)
    out = tmp_path / "out"
    good = tmp_path / "w.json"
    good.write_text(json.dumps({"skills": 0.5, "title": 0.3, "level": 0.2}))
    assert cli.main(["pool", "--stage", "dev", "--weights-file", str(good), *common]) == 0
    weights = json.loads((out / "run.json").read_text())["weights"]["dev"]
    assert weights == {"skills": 0.5, "title": 0.3, "level": 0.2, "source": f"file:{good}"}
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert cli.main(["pool", "--stage", "test", "--weights-file", str(bad), *common]) == 2
    assert not (out / "pool_rankings_test.parquet").exists()
    assert cli.main(["pool", "--stage", "dev", "--weights", "1,0,0", "--force", *common]) == 0
    assert json.loads((out / "run.json").read_text())["weights"]["dev"]["source"] == "cli"
    assert cli.main(["pool", "--stage", "dev", *common]) == 2


def test_pool_requires_sample(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_eval_tree(tmp_path / "tree")
    assert cli.main(["pool", "--stage", "dev", *_common(tmp_path)]) == 2
    assert "evaluate sample" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_pool_refuses_stale_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    common = _sampled(tmp_path)
    make_eval_tree(tmp_path / "other", n_resumes=41)
    capsys.readouterr()
    stale = _common(tmp_path, "other")
    assert cli.main(["pool", "--stage", "dev", *stale]) == 2
    assert "changed since sampling" in capsys.readouterr().err
    assert not (tmp_path / "out" / "pool_rankings_dev.parquet").exists()
