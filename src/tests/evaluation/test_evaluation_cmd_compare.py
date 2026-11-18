from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from getwork.evaluation import cli, pairs

pytestmark = pytest.mark.usefixtures("offline")

V1, V2 = "sha256:aaaaaaaaaaaa", "sha256:bbbbbbbbbbbb"


def _row(judge: str, pair_id: str, rubric_hash: str, label: int, status: str = "ok") -> dict:
    return {
        "pair_id": pair_id, "judge": judge, "rubric_version": "v1" if rubric_hash == V1 else "v2",
        "rubric_hash": rubric_hash, "status": status,
        "label": label if status == "ok" else None,
        "effective_label": label if status == "ok" else None,
    }


def _write(out: Path, rows: dict[str, list[tuple[str, str, int]]]) -> None:
    """rows: judge id -> [(pair_id, rubric_hash, label)]."""
    (out / "judges").mkdir(parents=True, exist_ok=True)
    for judge, items in rows.items():
        lines = [json.dumps(_row(judge, p, h, label)) for p, h, label in items]
        (out / "judges" / f"{judge}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pairs(out: Path, directions: dict[str, list[str]]) -> None:
    table = pa.table(
        {"pair_id": list(directions), "resume_id": ["r"] * len(directions),
         "posting_id": ["p"] * len(directions), "directions": list(directions.values())},
        schema=pairs.PAIRS_SCHEMA,
    )
    pq.write_table(table, out / "pairs_dev.parquet")


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> dict:
    assert cli.main(["compare-rubrics", *argv]) == 0
    return json.loads(capsys.readouterr().out)


def test_compare_splits_one_hash_over_dirs_and_uses_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    smoke, new = tmp_path / "smoke", tmp_path / "new"
    _write(smoke, {"judge-a": [("p1", V1, 1), ("p2", V1, 2)],
                   "judge-b": [("p1", V1, 2), ("p2", V1, 2)]})
    _write(new, {
        "judge-a": [("p3", V1, 1), ("p4", V1, 0), ("p1", V2, 1), ("p2", V2, 2), ("p3", V2, 2),
                    ("p5", V2, 0)],
        "judge-b": [("p3", V1, 2), ("p4", V1, 0), ("p1", V2, 1), ("p2", V2, 2), ("p3", V2, 2)],
    })
    both = ["resume_to_posting", "posting_to_resume"]
    _write_pairs(new, {"p1": ["resume_to_posting"], "p3": both})
    out = _run(capsys, "--out-dir", str(smoke), "--out-dir", str(new))
    assert out["overlap_pairs"] == 3
    assert out["pairs_judged_by_both_judges"] == {V1: 4, V2: 3}
    v1, v2 = out["rubrics"][V1], out["rubrics"][V2]
    assert v1["n"] == v2["n"] == 3
    assert v1["pct_agree"] == pytest.approx(1 / 3) and v2["pct_agree"] == 1.0
    assert v1["disagreements"]["total"] == {"a1_b2": 2, "a2_b1": 0, "other": 0}
    assert v1["disagreements"]["by_direction"]["resume_to_posting"]["a1_b2"] == 2
    assert v1["disagreements"]["by_direction"]["posting_to_resume"]["a1_b2"] == 1
    assert v1["distribution"]["judge-a"] == {"0": 0, "1": 2, "2": 1}
    assert v1["distribution"]["judge-b"] == {"0": 0, "1": 0, "2": 3}
    assert v1["kappa_ge1_vs_0"] is None
    assert v2["kappa"] == 1.0 and v2["kappa_label2_vs_not"] == 1.0


def test_compare_kappas_and_hash_selection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = [0, 0, 1, 1, 2, 2, 2, 1]
    b = [0, 1, 1, 2, 2, 2, 1, 1]
    _write(tmp_path, {
        "judge-a": [(f"p{i}", V1, x) for i, x in enumerate(a)],
        "judge-b": [(f"p{i}", V1, x) for i, x in enumerate(b)],
    })
    out = _run(capsys, "--out-dir", str(tmp_path), "--rubric-hash", V1)
    block = out["rubrics"][V1]
    assert block["n"] == 8 and block["pct_agree"] == 0.625
    assert 0 < block["kappa"] < 1 and block["kappa_quadratic"] > block["kappa"]
    assert block["kappa_ge1_vs_0"] is not None and block["kappa_label2_vs_not"] is not None
    assert block["confusion_rows_a_cols_b"][1][2] == 1
    assert block["disagreements"]["total"] == {"a1_b2": 1, "a2_b1": 1, "other": 1}


def test_compare_degenerate_single_label_gives_no_kappa(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, {"judge-a": [("p1", V1, 2), ("p2", V1, 2)],
                      "judge-b": [("p1", V1, 2), ("p2", V1, 2)]})
    block = _run(capsys, "--out-dir", str(tmp_path))["rubrics"][V1]
    assert block["pct_agree"] == 1.0 and block["kappa"] is None
    assert block["kappa_quadratic"] is None and block["kappa_ge1_vs_0"] is None


def test_compare_empty_overlap_does_not_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, {"judge-a": [("p1", V1, 1), ("p2", V2, 1)],
                      "judge-b": [("p1", V1, 1), ("p2", V2, 1)]})
    out = _run(capsys, "--out-dir", str(tmp_path))
    assert out["overlap_pairs"] == 0
    assert out["rubrics"][V1]["kappa"] is None and out["rubrics"][V1]["pct_agree"] is None


def test_compare_ignores_failed_rows_and_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(tmp_path, {"judge-a": [("p1", V1, 1)], "judge-b": [("p1", V1, 1)]})
    failed = json.dumps(_row("judge-b", "p2", V1, 0, status="failed"))
    with (tmp_path / "judges" / "judge-b.jsonl").open("a", encoding="utf-8") as f:
        f.write(failed + "\n")
    assert _run(capsys, "--out-dir", str(tmp_path))["pairs_judged_by_both_judges"] == {V1: 1}
    assert cli.main(["compare-rubrics", "--out-dir", str(tmp_path), "--rubric-hash", "x"]) == 2
    assert "no judge rows" in capsys.readouterr().err
    assert cli.main(["compare-rubrics", "--out-dir", str(tmp_path / "missing")]) == 2
    assert "judge file not found" in capsys.readouterr().err


V3 = "sha256:cccccccccccc"


def test_compare_three_rubrics_json_and_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pair_ids = ["p1", "p2", "p3", "p4"]
    plans = {V1: ([1, 1, 2, 0], [2, 2, 2, 0]), V2: ([1, 2, 2, 0], [2, 2, 1, 0]),
             V3: ([1, 2, 2, 0], [1, 2, 2, 0])}
    _write(tmp_path, {
        judge: [(p, h, plans[h][i][n]) for h in plans for n, p in enumerate(pair_ids)]
        for i, judge in enumerate(("judge-a", "judge-b"))
    })
    _write_pairs(tmp_path, {"p1": ["resume_to_posting"], "p2": ["posting_to_resume"],
                            "p3": ["resume_to_posting"], "p4": ["resume_to_posting"]})
    out = _run(capsys, "--out-dir", str(tmp_path))
    assert out["overlap_pairs"] == 4 and set(out["rubrics"]) == {V1, V2, V3}
    assert out["rubrics"][V1]["disagreements"]["total"]["a1_b2"] == 2
    assert out["rubrics"][V3]["pct_agree"] == 1.0
    assert cli.main(["compare-rubrics", "--out-dir", str(tmp_path), "--format", "table"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "overlap pairs: 4" and len(lines) == 4
    first = lines[1]
    for needle in ("v1", V1, "n=4", "agree=50%", "a1_b2=2 (posting_to_resume=1, "
                   "resume_to_posting=1)", "a(0/1/2)=1/2/1", "b(0/1/2)=1/0/3"):
        assert needle in first, needle
    assert "kappa=" in first and "wkappa=" in first and "k2=" in first
    assert "agree=100%" in lines[3]
