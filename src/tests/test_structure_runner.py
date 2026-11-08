import csv
import json
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from getwork.llm.client import LLMRequest
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.structure import runner, runstate, schema, store
from getwork.structure.extract import ExtractConfig
from getwork.structure.runstate import RunConfig

pytestmark = pytest.mark.usefixtures("offline")

MODEL = "m1"
WAIT_SECONDS = 5


def _id_of(request: LLMRequest) -> str:
    match = re.search(r"doc (r-\d+)", request.user)
    assert match, request.user
    return match.group(1)


def _ids(count: int) -> list[str]:
    return [f"r-{n:05d}" for n in range(1, count + 1)]


def _rows(count: int, long_text: dict[str, str] | None = None) -> list[dict]:
    extra = long_text or {}
    return [{"id": i, "text_clean": extra.get(i, f"doc {i}")} for i in _ids(count)]


def _ok(request: LLMRequest) -> str:
    return make_output_json(title=f"T {_id_of(request)}")


def _cfg(in_dir: Path, out_dir: Path, **kw) -> RunConfig:
    extract = kw.pop("extract", ExtractConfig(model=MODEL))
    return RunConfig(
        source="resumes", in_dir=in_dir, out_dir=out_dir, extract=extract, **kw
    )


def _go(cfg: RunConfig, handler: Callable[[LLMRequest], str]):
    client = FakeLLMClient(handler=handler)
    result = runner.run(cfg, client, sleep=lambda s: None)
    return result, client


def _jsonl_ids(path: Path) -> list[str]:
    return [json.loads(line)["id"] for line in path.read_text().splitlines()]


def _failure_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _paths(out_dir: Path) -> store.OutputPaths:
    return store.output_paths(source="resumes", out_dir=out_dir)


def _record(id_: str) -> dict:
    output = schema.parse_model_output(make_output_json(title=f"T {id_}"))
    return schema.to_record(output, id_, "resumes", MODEL, "extract_v1")


def _seed(out_dir: Path, ids: Sequence[str], tail: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(_record(i)) + "\n" for i in ids)
    _paths(out_dir).jsonl.write_text(lines + tail)


def test_outputs_sorted_regardless_of_completion_order(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(4))
    ids = _ids(4)
    finished = {i: threading.Event() for i in ids}
    order: list[str] = []

    def handler(request: LLMRequest) -> str:
        id_ = _id_of(request)
        index = ids.index(id_)
        if index + 1 < len(ids):
            assert finished[ids[index + 1]].wait(WAIT_SECONDS)
        if id_ not in order:  # a failing document is called again for its retries
            order.append(id_)
        finished[id_].set()
        return make_output_json() if id_ != ids[1] else "not json"

    cfg = _cfg(in_dir, tmp_path / "out", workers=4)
    _go(cfg, handler)
    assert order == list(reversed(ids))
    paths = _paths(cfg.out_dir)
    assert _jsonl_ids(paths.jsonl) == [ids[0], ids[2], ids[3]]
    assert pq.read_table(paths.parquet).column("id").to_pylist() == [ids[0], ids[2], ids[3]]
    assert [r["id"] for r in _failure_rows(paths.failures)] == [ids[1]]


def test_runs_are_byte_identical_across_worker_counts(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(6))

    def handler(request: LLMRequest) -> str:
        return "bad" if _id_of(request) == "r-00003" else _ok(request)

    blobs = []
    for workers in (1, 4):
        cfg = _cfg(in_dir, tmp_path / f"out{workers}", workers=workers)
        _go(cfg, handler)
        paths = _paths(cfg.out_dir)
        blobs.append([p.read_bytes() for p in (paths.jsonl, paths.parquet, paths.failures, paths.summary)])
    assert blobs[0] == blobs[1]
    assert b"r-00003" in blobs[0][2]


def test_resume_skips_done_ids(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(5))
    out_dir = tmp_path / "out"
    _seed(out_dir, _ids(5)[:2])
    result, client = _go(_cfg(in_dir, out_dir), _ok)
    assert sorted(_id_of(c) for c in client.calls) == _ids(5)[2:]
    assert result.summary["already_done"] == 2
    assert result.summary["done_total"] == 5


def test_larger_limit_only_processes_new_ids(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(8))
    out_dir = tmp_path / "out"
    _, first = _go(_cfg(in_dir, out_dir, limit=3), _ok)
    _, second = _go(_cfg(in_dir, out_dir, limit=5), _ok)
    assert len(first.calls) == 3
    assert len(second.calls) == 2


def test_resume_after_partial_line(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(3))
    out_dir = tmp_path / "out"
    _seed(out_dir, _ids(3)[:1], tail='{"id": "r-00002", "sour')
    result, client = _go(_cfg(in_dir, out_dir), _ok)
    assert result.exit_code == 0
    assert len(client.calls) == 2
    assert _jsonl_ids(_paths(out_dir).jsonl) == _ids(3)


def _fail_second(request: LLMRequest) -> str:
    return "not json" if _id_of(request) == "r-00002" else _ok(request)


def test_failed_record_logged_and_run_continues(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(3))
    cfg = _cfg(in_dir, tmp_path / "out")
    result, _ = _go(cfg, _fail_second)
    assert result.exit_code == 1
    assert _jsonl_ids(_paths(cfg.out_dir).jsonl) == ["r-00001", "r-00003"]
    (row,) = _failure_rows(_paths(cfg.out_dir).failures)
    assert row["id"] == "r-00002"
    assert row["stage"] == "validation"
    assert row["error_type"] == "OutputValidationError"
    assert row["message"]
    assert int(row["attempts"]) == 3


def test_failed_record_retried_on_next_run(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(3))
    cfg = _cfg(in_dir, tmp_path / "out")
    _go(cfg, _fail_second)
    result, client = _go(cfg, _ok)
    assert result.exit_code == 0
    assert [_id_of(c) for c in client.calls] == ["r-00002"]
    assert _jsonl_ids(_paths(cfg.out_dir).jsonl) == _ids(3)
    assert _failure_rows(_paths(cfg.out_dir).failures) == []


def test_failures_for_unselected_ids_preserved(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(4))
    out_dir = tmp_path / "out"

    def fail_fourth(request: LLMRequest) -> str:
        return "bad" if _id_of(request) == "r-00004" else _ok(request)

    _go(_cfg(in_dir, out_dir), fail_fourth)
    result, _ = _go(_cfg(in_dir, out_dir, limit=2), _ok)
    assert [r["id"] for r in _failure_rows(_paths(out_dir).failures)] == ["r-00004"]
    assert result.summary["failures_total"] == 1


def test_summary_counts_for_a_real_run(write_clean_parquet, tmp_path) -> None:
    long_id = "r-00003"
    in_dir = write_clean_parquet("resumes", _rows(4, {long_id: f"doc {long_id} " + "x" * 100}))
    extract = ExtractConfig(model=MODEL, max_input_chars=60)
    cfg = _cfg(in_dir, tmp_path / "out", extract=extract, workers=2)

    def handler(request: LLMRequest) -> str:
        return "bad" if _id_of(request) == "r-00002" else _ok(request)

    result, client = _go(cfg, handler)
    summary = result.summary
    assert json.loads(_paths(cfg.out_dir).summary.read_text()) == summary
    assert summary == runstate.build_summary(
        cfg=cfg,
        counters=_expected_counters(len(client.calls)),
        facts=runstate.RunFacts(
            total_in_source=4, selected=4, already_done=0, todo=4, done_total=3, failures_total=1
        ),
        status="complete",
        abort_reason=None,
    )
    assert (summary["attempted"], summary["succeeded"], summary["failed"]) == (4, 3, 1)
    assert summary["not_attempted"] == 0
    assert summary["truncated_inputs"] == 1
    assert summary["calls"] == 6
    assert summary["failures_by_stage"]["validation"] == 1
    assert result.exit_code == 1


def _expected_counters(calls: int) -> runstate.Counters:
    counters = runstate.Counters(attempted=4, succeeded=3, failed=1, calls=calls, truncated_inputs=1)
    counters.failures_by_stage["validation"] = 1
    return counters


def test_exit_code_zero_when_all_succeed(write_clean_parquet, tmp_path) -> None:
    in_dir = write_clean_parquet("resumes", _rows(3))
    result, _ = _go(_cfg(in_dir, tmp_path / "out"), _ok)
    assert result.exit_code == 0
    assert result.summary["status"] == "complete"
    assert result.summary["abort_reason"] is None
    assert result.summary["failed"] == 0
