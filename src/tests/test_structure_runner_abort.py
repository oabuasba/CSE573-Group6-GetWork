import json
import re
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from getwork.llm.client import FatalLLMError, LLMRequest
from getwork.llm.fake import FakeLLMClient, make_output_json
from getwork.structure import runner, store
from getwork.structure.extract import ExtractConfig, Outcome
from getwork.structure.runstate import RunConfig, RunResult

pytestmark = pytest.mark.usefixtures("offline")

MODEL = "m1"


def _id_of(request: LLMRequest) -> str:
    match = re.search(r"doc (r-\d+)", request.user)
    assert match, request.user
    return match.group(1)


def _rows(count: int) -> list[dict]:
    return [{"id": f"r-{n:05d}", "text_clean": f"doc r-{n:05d}"} for n in range(1, count + 1)]


def _cfg(in_dir: Path, out_dir: Path, **kw: Any) -> RunConfig:
    return RunConfig(
        source="resumes", in_dir=in_dir, out_dir=out_dir, extract=ExtractConfig(model=MODEL),
        workers=kw.pop("workers", 1), **kw,
    )


def _paths(out_dir: Path) -> store.OutputPaths:
    return store.output_paths(source="resumes", out_dir=out_dir)


def _ok(request: LLMRequest) -> str:
    return make_output_json(title="T")


def _run(cfg: RunConfig, handler: Callable[[LLMRequest], str], **kw: Any) -> RunResult:
    return runner.run(cfg=cfg, client=FakeLLMClient(handler=handler), sleep=lambda s: None, **kw)


def _jsonl_ids(out_dir: Path) -> list[str]:
    lines = _paths(out_dir).jsonl.read_text().splitlines()
    return [json.loads(line)["id"] for line in lines]


def test_fatal_error_aborts_and_keeps_checkpoint(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(5))
    out_dir = tmp_path / "out"
    calls: list[str] = []

    def handler(request: LLMRequest) -> str:
        calls.append(_id_of(request))
        if len(calls) == 4:
            raise FatalLLMError("denied", status=401)
        return _ok(request)

    result = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), handler=handler)
    assert result.exit_code == 3
    assert result.summary["status"] == "aborted"
    assert "401" in result.summary["abort_reason"]
    assert result.summary["attempted"] == 3
    assert result.summary["not_attempted"] == 2
    paths = _paths(out_dir)
    assert len(_jsonl_ids(out_dir)) == 3
    assert pq.read_table(paths.parquet).num_rows == 3
    assert paths.failures.exists()


def test_consecutive_failures_trip_circuit_breaker(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(6))
    cfg = _cfg(in_dir=in_dir, out_dir=tmp_path / "out", max_consecutive_failures=3)
    result = _run(cfg=cfg, handler=lambda request: "not json")
    assert result.exit_code == 3
    assert result.summary["status"] == "aborted"
    assert result.summary["abort_reason"] == "3 consecutive record failures"
    assert result.summary["attempted"] == 3


def test_success_resets_consecutive_failures(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(6))
    cfg = _cfg(in_dir=in_dir, out_dir=tmp_path / "out", max_consecutive_failures=2)

    def handler(request: LLMRequest) -> str:
        number = int(_id_of(request).split("-")[1])
        return "not json" if number % 2 else _ok(request)

    result = _run(cfg=cfg, handler=handler)
    assert result.summary["status"] == "complete"
    assert result.exit_code == 1
    assert result.summary["attempted"] == 6
    assert result.summary["failed"] == 3


def test_stop_event_finalizes_and_is_resumable(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(6))
    out_dir = tmp_path / "out"
    stop = threading.Event()
    count = 0

    def handler(request: LLMRequest) -> str:
        nonlocal count
        count += 1
        if count == 3:
            stop.set()
        return _ok(request)

    result = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), handler=handler, stop=stop)
    assert result.exit_code == 130
    assert result.summary["status"] == "interrupted"
    assert len(_jsonl_ids(out_dir)) == 3
    paths = _paths(out_dir)
    assert paths.parquet.exists()
    assert paths.failures.exists()
    assert paths.summary.exists()
    second = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), handler=_ok)
    assert second.exit_code == 0
    assert second.summary["done_total"] == 6


def test_keyboard_interrupt_in_main_thread_finalizes(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(6))
    out_dir = tmp_path / "out"
    logged = 0

    def log(message: str) -> None:
        nonlocal logged
        logged += 1
        if logged == 3:
            raise KeyboardInterrupt

    result = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir, progress_every=1), handler=_ok, log=log)
    assert result.exit_code == 130
    assert result.summary["status"] == "interrupted"
    paths = _paths(out_dir)
    ids = _jsonl_ids(out_dir)
    assert len(ids) >= 3
    assert pq.read_table(paths.parquet).column("id").to_pylist() == ids
    assert result.summary["done_total"] == len(ids)
    assert paths.failures.exists()


def _raising_structure_one(monkeypatch: pytest.MonkeyPatch, bad_ids: set[str] | None) -> None:
    real = runner.structure_one

    def wrapper(**kwargs: Any) -> Outcome:
        if bad_ids is None or kwargs["doc"].id in bad_ids:
            raise RuntimeError("bug")
        return real(**kwargs)

    monkeypatch.setattr(runner, "structure_one", wrapper)


def test_worker_exception_becomes_permanent_failure(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(4))
    out_dir = tmp_path / "out"
    _raising_structure_one(monkeypatch, {"r-00002"})
    result = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), handler=_ok)
    assert result.exit_code == 1
    assert result.summary["succeeded"] == 3
    assert result.summary["failed"] == 1
    rows = store.read_failures(_paths(out_dir).failures)
    assert [(r["id"], r["stage"], r["error_type"]) for r in rows] == [
        ("r-00002", "permanent", "RuntimeError")
    ]


def test_worker_exceptions_count_toward_breaker(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(5))
    _raising_structure_one(monkeypatch, None)
    cfg = _cfg(in_dir=in_dir, out_dir=tmp_path / "out", max_consecutive_failures=2)
    result = _run(cfg=cfg, handler=_ok)
    assert result.exit_code == 3
    assert result.summary["status"] == "aborted"
    assert result.summary["abort_reason"] == "2 consecutive record failures"


def test_connection_style_fatal_error_aborts_and_resumes(
    write_clean_parquet: Callable[[str, Sequence[dict]], Path],
    tmp_path: Path,
) -> None:
    in_dir = write_clean_parquet("resumes", _rows(5))
    out_dir = tmp_path / "out"
    calls: list[str] = []

    def down(request: LLMRequest) -> str:
        calls.append(_id_of(request))
        if len(calls) == 3:
            raise FatalLLMError("cannot reach the LLM server at http://localhost:8000/v1")
        return _ok(request)

    result = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), handler=down)
    assert result.exit_code == 3
    assert "http://localhost:8000/v1" in result.summary["abort_reason"]
    assert len(_jsonl_ids(out_dir)) == 2
    resumed = _run(cfg=_cfg(in_dir=in_dir, out_dir=out_dir), handler=_ok)
    assert resumed.exit_code == 0
    assert len(_jsonl_ids(out_dir)) == 5
