from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from getwork import fileio
from getwork.evaluation import cli, judge_run, judges, layout
from getwork.evaluation.fake import FakeJudge
from getwork.evaluation.judges import JudgeInfo, PairText, Verdict

pytestmark = pytest.mark.usefixtures("offline")


def _rubric_file(tmp_path: Path, text: str = "rubric text") -> Path:
    path = tmp_path / "rubric_x.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_judges(out: Path, rubric: judges.Rubric, *, skip: str | None = None) -> None:
    pairs = [PairText(pair_id=f"p-{i}", resume_text="r", posting_text="t") for i in range(3)]
    for judge_id, label in (("judge-a", 2), ("judge-b", 1)):
        if judge_id == skip:
            continue
        info = JudgeInfo(id=judge_id, model=f"m-{judge_id}", family=judge_id, backend="fake")
        judge = FakeJudge(info, lambda pair, label=label: Verdict(label, False, "why"))
        judge_run.run_judge(
            judge, pairs, path=layout.judge_path(out, judge_id), rubric=rubric
        )


def test_freeze_rubric(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out, rubric_file = tmp_path / "out", _rubric_file(tmp_path)
    rubric = judges.load_rubric(rubric_file)
    code = cli.main(["freeze-rubric", "--out-dir", str(out), "--rubric-file", str(rubric_file)])
    assert code == 0
    assert capsys.readouterr().out.strip() == rubric.hash
    frozen = json.loads((out / "rubric_frozen.json").read_text())
    assert set(frozen) == {"rubric_version", "rubric_hash", "frozen_at"}
    assert frozen["rubric_version"] == "rubric_x" and frozen["rubric_hash"] == rubric.hash
    assert len(frozen["frozen_at"]) == 20 and frozen["frozen_at"].endswith("Z")
    assert json.loads((out / "run.json").read_text())["rubric_hash"] == rubric.hash
    assert cli.main(["freeze-rubric", "--out-dir", str(out), "--rubric-file",
                     str(tmp_path / "missing.md")]) == 2


def test_freeze_refuses_overwrite(tmp_path: Path) -> None:
    out, first = tmp_path / "out", _rubric_file(tmp_path)
    argv = ["freeze-rubric", "--out-dir", str(out), "--rubric-file"]
    assert cli.main([*argv, str(first)]) == 0
    before = (out / "rubric_frozen.json").read_bytes()
    second = _rubric_file(tmp_path, "other text")
    assert cli.main([*argv, str(second)]) == 2
    assert (out / "rubric_frozen.json").read_bytes() == before
    assert cli.main([*argv, str(second), "--force"]) == 0
    new_hash = judges.load_rubric(second).hash
    assert json.loads((out / "rubric_frozen.json").read_text())["rubric_hash"] == new_hash


def test_labels_rebuild_is_idempotent(tmp_path: Path) -> None:
    out, rubric_file = tmp_path / "out", _rubric_file(tmp_path)
    rubric = judges.load_rubric(rubric_file)
    argv = ["labels", "--out-dir", str(out), "--rubric-file", str(rubric_file)]
    _write_judges(out, rubric, skip="judge-b")
    assert cli.main(argv) == 2
    assert not (out / "labels_judge.parquet").exists()
    _write_judges(out, rubric)
    # rows of another rubric hash must be ignored
    other = judges.Rubric(version="old", text="old", hash="sha256:000000000000")
    stale = layout.judge_path(out, "judge-a")
    fileio.write_jsonl_atomic(stale, [
        *[json.loads(x) for x in stale.read_text().splitlines()],
        {**json.loads(stale.read_text().splitlines()[0]), "rubric_hash": other.hash,
         "pair_id": "p-9"},
    ])
    assert cli.main(argv) == 0
    first = (out / "labels_judge.parquet").read_bytes()
    table = pq.read_table(out / "labels_judge.parquet").to_pydict()
    assert table["pair_id"] == ["p-0", "p-1", "p-2"]
    assert table["label"] == [1, 1, 1] and table["disagree"] == [True] * 3
    assert cli.main(argv) == 0
    assert (out / "labels_judge.parquet").read_bytes() == first
