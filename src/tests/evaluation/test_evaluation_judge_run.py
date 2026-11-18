from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from getwork.evaluation import judge_run
from getwork.evaluation.fake import FakeJudge
from getwork.evaluation.judges import (
    JudgeError,
    JudgeFatalError,
    JudgeInfo,
    PairText,
    Rubric,
    Verdict,
)

pytestmark = pytest.mark.usefixtures("offline")

INFO = JudgeInfo(id="judge-a", model="m-a", family="fam-a", backend="fake")
RUBRIC = Rubric(version="judge_v1", text="rubric", hash="sha256:aaaaaaaaaaaa")
OTHER_RUBRIC = Rubric(version="judge_v1", text="rubric 2", hash="sha256:bbbbbbbbbbbb")


def _pairs(n: int = 6) -> list[PairText]:
    return [PairText(pair_id=f"p-{i}", resume_text="r", posting_text="t") for i in range(n)]


def _verdict(pair: PairText) -> Verdict:
    return Verdict(label=2, seniority_mismatch=pair.pair_id == "p-0", rationale="because")


def _lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_rows_and_cap(tmp_path: Path) -> None:
    path = tmp_path / "judges" / "judge-a.jsonl"
    summary = judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(3), path=path, rubric=RUBRIC)
    rows = _lines(path)
    assert (summary.n_pairs, summary.n_ok, summary.n_failed, summary.fatal) == (3, 3, 0, None)
    assert len(rows) == 3
    for row in rows:
        assert tuple(row) == judge_run.JUDGE_ROW_KEYS
        assert (row["judge"], row["model"], row["family"]) == ("judge-a", "m-a", "fam-a")
        assert (row["rubric_version"], row["rubric_hash"]) == ("judge_v1", RUBRIC.hash)
    capped = next(r for r in rows if r["pair_id"] == "p-0")
    assert (capped["label"], capped["effective_label"]) == (2, 1)
    assert next(r for r in rows if r["pair_id"] == "p-1")["effective_label"] == 2


def test_failed_pair_does_not_stop_run_and_is_retried(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    judge = FakeJudge(INFO, _verdict, fail={"p-2": JudgeError("invalid_reply")})
    summary = judge_run.run_judge(judge, _pairs(), path=path, rubric=RUBRIC)
    assert (summary.n_ok, summary.n_failed) == (5, 1)
    failed = judge_run.read_judge_rows(path, RUBRIC.hash)["p-2"]
    assert failed["status"] == "failed" and failed["failure_reason"] == "invalid_reply"
    assert failed["label"] is None and failed["effective_label"] is None
    retry = FakeJudge(INFO, _verdict)
    again = judge_run.run_judge(retry, _pairs(), path=path, rubric=RUBRIC)
    assert retry.calls == ["p-2"]
    assert (again.n_skipped, again.n_ok, again.n_failed) == (5, 1, 0)
    assert judge_run.read_judge_rows(path, RUBRIC.hash)["p-2"]["status"] == "ok"


def test_resume_and_rubric_change(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(4), path=path, rubric=RUBRIC)
    same = FakeJudge(INFO, _verdict)
    summary = judge_run.run_judge(same, _pairs(4), path=path, rubric=RUBRIC)
    assert same.calls == [] and summary.n_skipped == 4 and summary.n_ok == 0
    other = FakeJudge(INFO, _verdict)
    judge_run.run_judge(other, _pairs(4), path=path, rubric=OTHER_RUBRIC)
    assert sorted(other.calls) == ["p-0", "p-1", "p-2", "p-3"]
    rows = judge_run.read_judge_rows(path, OTHER_RUBRIC.hash)
    assert len(rows) == 4
    assert {r["rubric_hash"] for r in rows.values()} == {OTHER_RUBRIC.hash}


def test_second_rubric_keeps_first_rubric_rows_and_resumes_either(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(4), path=path, rubric=RUBRIC)
    first = path.read_bytes()
    judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(6), path=path, rubric=OTHER_RUBRIC)
    assert path.read_bytes().startswith(first)
    assert len(judge_run.read_judge_rows(path, RUBRIC.hash)) == 4
    assert len(judge_run.read_judge_rows(path, OTHER_RUBRIC.hash)) == 6
    again = FakeJudge(INFO, _verdict)
    judge_run.run_judge(again, _pairs(6), path=path, rubric=RUBRIC)
    assert sorted(again.calls) == ["p-4", "p-5"]
    assert len(judge_run.read_judge_rows(path, OTHER_RUBRIC.hash)) == 6


def test_workers_do_not_change_rows(tmp_path: Path) -> None:
    pairs = _pairs(12)
    results = []
    for workers in (1, 4):
        path = tmp_path / f"w{workers}.jsonl"
        judge_run.run_judge(
            FakeJudge(INFO, _verdict, fail={"p-5": JudgeError("rejected")}),
            pairs, path=path, rubric=RUBRIC, workers=workers,
        )
        results.append({r["pair_id"]: r for r in _lines(path)})
    assert results[0] == results[1] and len(results[0]) == 12


def test_fatal_error_keeps_rows(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"

    def verdict(pair: PairText) -> Verdict:
        if pair.pair_id == "p-2":
            raise JudgeFatalError("bad key")
        return _verdict(pair)

    summary = judge_run.run_judge(
        FakeJudge(INFO, verdict), _pairs(10), path=path, rubric=RUBRIC, workers=1
    )
    assert summary.fatal == "bad key"
    assert summary.n_ok == 2 and summary.n_pairs == 10
    assert [r["pair_id"] for r in _lines(path)] == ["p-0", "p-1"]


def _row(pair_id: str, rubric_hash: str = RUBRIC.hash) -> str:
    row = {k: None for k in judge_run.JUDGE_ROW_KEYS}
    row.update(pair_id=pair_id, rubric_hash=rubric_hash, status="ok")
    return json.dumps(row)


def test_read_rows_tolerates_torn_last_line(tmp_path: Path) -> None:
    torn = tmp_path / "torn.jsonl"
    torn.write_text(_row("p-1") + "\n" + _row("p-2")[:30], encoding="utf-8")
    assert list(judge_run.read_judge_rows(torn, RUBRIC.hash)) == ["p-1"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n" + _row("p-1") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl.*line 1"):
        judge_run.read_judge_rows(bad, RUBRIC.hash)


def test_read_ignores_invalid_last_line_even_with_newline(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_row("p-1") + "\n{half\n", encoding="utf-8")
    assert list(judge_run.read_judge_rows(path, RUBRIC.hash)) == ["p-1"]


def test_row_without_pair_id_names_file_and_line(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text('{"x": 1}\n' + _row("p-1") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"a\.jsonl.*line 1"):
        judge_run.read_judge_rows(path, RUBRIC.hash)


def test_resume_after_torn_line_keeps_file_readable(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(
        _row("p-0") + "\n" + _row("p-1") + '\n{"pair_id": "p-9", "rub', encoding="utf-8"
    )
    judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(4), path=path, rubric=RUBRIC)
    rows = judge_run.read_judge_rows(path, RUBRIC.hash)
    assert sorted(rows) == ["p-0", "p-1", "p-2", "p-3"]


def test_missing_file_reads_as_empty(tmp_path: Path) -> None:
    assert judge_run.read_judge_rows(tmp_path / "none.jsonl", RUBRIC.hash) == {}


def test_ok_row_beats_failed_and_later_beats_earlier(tmp_path: Path) -> None:
    def row(pair_id: str, status: str, rationale: str) -> str:
        data = json.loads(_row(pair_id))
        data.update(status=status, rationale=rationale)
        return json.dumps(data)

    path = tmp_path / "a.jsonl"
    path.write_text("\n".join([
        row("ok-fail", "ok", "1"), row("ok-fail", "failed", "2"),
        row("fail-ok", "failed", "1"), row("fail-ok", "ok", "2"),
        row("fail-fail", "failed", "1"), row("fail-fail", "failed", "2"),
        row("ok-ok", "ok", "1"), row("ok-ok", "ok", "2"),
    ]) + "\n", encoding="utf-8")
    rows = judge_run.read_judge_rows(path, RUBRIC.hash)
    assert {k: (v["status"], v["rationale"]) for k, v in rows.items()} == {
        "ok-fail": ("ok", "1"), "fail-ok": ("ok", "2"),
        "fail-fail": ("failed", "2"), "ok-ok": ("ok", "2"),
    }


def test_summary_counts_truncated_and_skipped(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(2), path=path, rubric=RUBRIC)
    summary = judge_run.run_judge(
        FakeJudge(INFO, _verdict, truncated=True), _pairs(5), path=path, rubric=RUBRIC
    )
    assert (summary.n_pairs, summary.n_skipped, summary.n_ok, summary.n_truncated) == (5, 2, 3, 3)


def _resume(path: Path, n: int = 4) -> FakeJudge:
    judge = FakeJudge(INFO, _verdict)
    judge_run.run_judge(judge, _pairs(n), path=path, rubric=RUBRIC)
    return judge


def test_complete_last_row_without_newline_survives(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_row("p-0") + "\n" + _row("p-1"), encoding="utf-8")
    judge = _resume(path)
    assert sorted(judge.calls) == ["p-2", "p-3"]
    assert sorted(judge_run.read_judge_rows(path, RUBRIC.hash)) == ["p-0", "p-1", "p-2", "p-3"]


def test_single_complete_row_without_newline_survives(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_row("p-0"), encoding="utf-8")
    judge = _resume(path, n=2)
    assert judge.calls == ["p-1"]
    assert sorted(judge_run.read_judge_rows(path, RUBRIC.hash)) == ["p-0", "p-1"]


def test_invalid_last_line_with_newline_is_removed_on_resume(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_row("p-0") + "\n{half\n", encoding="utf-8")
    _resume(path)
    assert sorted(judge_run.read_judge_rows(path, RUBRIC.hash)) == ["p-0", "p-1", "p-2", "p-3"]


@pytest.mark.parametrize("content", ["", "\n  \n", '{"pair_id": "p-', "{half\n", '{"x": 1}'])
def test_resume_from_degenerate_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(content, encoding="utf-8")
    _resume(path, n=3)
    assert sorted(judge_run.read_judge_rows(path, RUBRIC.hash)) == ["p-0", "p-1", "p-2"]


def test_whitespace_trailing_line_after_valid_row(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_row("p-0") + "\n   ", encoding="utf-8")
    judge = _resume(path, n=2)
    assert judge.calls == ["p-1"]
    assert sorted(judge_run.read_judge_rows(path, RUBRIC.hash)) == ["p-0", "p-1"]


def test_failed_repair_leaves_original_and_no_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "a.jsonl"
    path.write_text(_row("p-0") + "\n{half\n", encoding="utf-8")
    before = path.read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr("os.replace", boom)
    with pytest.raises(OSError, match="disk gone"):
        judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(2), path=path, rubric=RUBRIC)
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["a.jsonl"]


def _progress_lines(stream: io.StringIO) -> list[str]:
    return stream.getvalue().splitlines()


def test_progress_every_and_final_line(tmp_path: Path) -> None:
    stream = io.StringIO()
    judge_run.run_judge(
        FakeJudge(INFO, _verdict, fail={"p-1": JudgeError("invalid_reply")}), _pairs(5),
        path=tmp_path / "a.jsonl", rubric=RUBRIC, workers=1, progress_every=2,
        progress_stream=stream,
    )
    lines = _progress_lines(stream)
    assert [ln.split(") ")[0] for ln in lines] == [
        "[judge-a] judged 2/5 pairs (1 ok, 1 failed, 0 skipped as already done",
        "[judge-a] judged 4/5 pairs (3 ok, 1 failed, 0 skipped as already done",
        "[judge-a] judged 5/5 pairs (4 ok, 1 failed, 0 skipped as already done",
    ]
    assert all("elapsed" in ln for ln in lines) and "ETA" not in lines[-1]
    assert "because" not in stream.getvalue()


def test_progress_counts_resumed_pairs(tmp_path: Path) -> None:
    path = tmp_path / "a.jsonl"
    judge_run.run_judge(FakeJudge(INFO, _verdict), _pairs(3), path=path, rubric=RUBRIC)
    stream = io.StringIO()
    judge_run.run_judge(
        FakeJudge(INFO, _verdict), _pairs(5), path=path, rubric=RUBRIC, progress_every=1,
        progress_stream=stream,
    )
    lines = _progress_lines(stream)
    assert lines[0].startswith("[judge-a] judged 4/5 pairs (1 ok, 0 failed, 3 skipped")
    assert lines[-1].startswith("[judge-a] judged 5/5 pairs (2 ok, 0 failed, 3 skipped")


def test_progress_zero_is_silent(tmp_path: Path) -> None:
    stream = io.StringIO()
    judge_run.run_judge(
        FakeJudge(INFO, _verdict), _pairs(), path=tmp_path / "a.jsonl", rubric=RUBRIC,
        progress_every=0, progress_stream=stream,
    )
    assert stream.getvalue() == ""


def test_progress_with_workers_has_consistent_counts(tmp_path: Path) -> None:
    stream = io.StringIO()
    summary = judge_run.run_judge(
        FakeJudge(INFO, _verdict), _pairs(30), path=tmp_path / "a.jsonl", rubric=RUBRIC,
        workers=8, progress_every=7, progress_stream=stream,
    )
    lines = _progress_lines(stream)
    assert summary.n_ok == 30 and len(lines) == 5
    assert lines[-1].startswith("[judge-a] judged 30/30 pairs (30 ok, 0 failed, 0 skipped")
    done = [int(ln.split("judged ")[1].split("/")[0]) for ln in lines]
    assert done == sorted(done)
