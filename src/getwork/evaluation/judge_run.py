"""The resumable concurrent judge runner and the reader of its jsonl rows (S-2)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from getwork import fileio
from getwork.evaluation import judges
from getwork.evaluation.judges import (
    Judge,
    JudgeError,
    JudgeFatalError,
    PairText,
    Rubric,
    Verdict,
)

JUDGE_ROW_KEYS: tuple[str, ...] = (
    "pair_id", "judge", "model", "family", "rubric_version", "rubric_hash",
    "status", "label", "seniority_mismatch", "effective_label", "rationale", "truncated",
    "failure_reason",
)
OK_STATUS = "ok"
FAILED_STATUS = "failed"


@dataclass(frozen=True)
class RunSummary:
    n_pairs: int
    n_skipped: int
    n_ok: int
    n_failed: int
    n_truncated: int
    fatal: str | None


def _parse_row(path: Path, number: int, line: bytes) -> dict:
    try:
        row = json.loads(line.decode("utf-8"))
    except ValueError as e:
        raise ValueError(f"{path}: line {number} is not valid JSON: {e}") from e
    if not isinstance(row, dict) or "pair_id" not in row:
        raise ValueError(f"{path}: line {number} is not a judge row with a pair_id")
    return row


def _scan(path: Path) -> tuple[list[dict], bytes]:
    """(valid rows in file order, the repaired file content); one rule for reader and writer.

    An invalid last line (including valid JSON without a pair_id) is dropped as a torn write;
    an invalid line anywhere else raises ValueError naming file and line.
    """
    data = path.read_bytes() if path.exists() else b""
    lines, offset = [], 0
    for number, raw in enumerate(data.split(b"\n"), start=1):
        if raw.strip():
            lines.append((number, raw, offset + len(raw)))
        offset += len(raw) + 1
    rows: list[dict] = []
    valid_end = 0
    for position, (number, raw, end) in enumerate(lines):
        try:
            rows.append(_parse_row(path, number, raw))
        except ValueError:
            if position != len(lines) - 1:
                raise
            break
        valid_end = end
    return rows, (data[:valid_end] + b"\n" if valid_end else b"")


def _best_rows(rows: Sequence[dict], rubric_hash: str) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        if row.get("rubric_hash") != rubric_hash:
            continue
        kept = best.get(row["pair_id"])
        if kept is not None and kept["status"] == OK_STATUS and row["status"] != OK_STATUS:
            continue
        best[row["pair_id"]] = row
    return best


def read_judge_rows(path: Path, rubric_hash: str) -> dict[str, dict]:
    """Best row per pair for this hash: ok beats failed, later beats earlier of one status.

    A missing file gives {}; a torn (invalid) last line is ignored.
    """
    return _best_rows(_scan(Path(path))[0], rubric_hash)


def _write_durably(tmp: Path, content: bytes) -> None:
    with tmp.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def _repair_file(path: Path, repaired: bytes) -> None:
    """Replace the file by its repaired content (torn last line dropped, last row terminated).

    Goes through a temp file and os.replace, so a crash never loses judged rows.
    """
    if not path.exists() or repaired == path.read_bytes():
        return
    fileio.atomic_write(path, lambda tmp: _write_durably(tmp, repaired))


def _base_row(judge: Judge, rubric: Rubric, pair_id: str) -> dict:
    info = judge.info
    return {
        "pair_id": pair_id, "judge": info.id, "model": info.model, "family": info.family,
        "rubric_version": rubric.version, "rubric_hash": rubric.hash,
    }


def _ok_row(base: dict, verdict: Verdict, truncated: bool) -> dict:
    return {
        **base, "status": OK_STATUS, "label": verdict.label,
        "seniority_mismatch": verdict.seniority_mismatch,
        "effective_label": judges.effective_label(verdict), "rationale": verdict.rationale,
        "truncated": truncated, "failure_reason": None,
    }


def _failed_row(base: dict, reason: str) -> dict:
    return {
        **base, "status": FAILED_STATUS, "label": None, "seniority_mismatch": None,
        "effective_label": None, "rationale": None, "truncated": False,
        "failure_reason": reason,
    }


def _judge_pair(judge: Judge, rubric: Rubric, pair: PairText) -> dict:
    """Returns the row; JudgeFatalError propagates."""
    base = _base_row(judge=judge, rubric=rubric, pair_id=pair.pair_id)
    try:
        verdict, truncated = judge.judge(pair)
    except JudgeError as e:
        return _failed_row(base=base, reason=e.reason)
    return _ok_row(base=base, verdict=verdict, truncated=truncated)


class _Collector:
    """Owns the pending futures; only its caller's thread writes rows."""

    def __init__(self, handle: TextIO) -> None:
        self.handle = handle
        self.pending: set[Future] = set()
        self.n_ok = self.n_failed = self.n_truncated = 0
        self.fatal: str | None = None

    def drain_some(self) -> None:
        finished, self.pending = wait(self.pending, return_when=FIRST_COMPLETED)
        for future in finished:
            try:
                row = future.result()
            except JudgeFatalError as e:
                self.fatal = self.fatal or str(e) or "fatal judge error"
                continue
            self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.handle.flush()
            if row["status"] == OK_STATUS:
                self.n_ok += 1
            else:
                self.n_failed += 1
            self.n_truncated += bool(row["truncated"])


def run_judge(judge: Judge, pairs: Sequence[PairText], *, path: Path, rubric: Rubric,
              workers: int = 4) -> RunSummary:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, repaired = _scan(path)
    done = _best_rows(rows, rubric.hash)
    todo = [p for p in pairs if done.get(p.pair_id, {}).get("status") != OK_STATUS]
    width = max(1, workers)
    _repair_file(path, repaired)
    with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=width
    ) as pool:
        run = _Collector(handle)
        for pair in todo:
            run.pending.add(pool.submit(_judge_pair, judge, rubric, pair))
            while len(run.pending) >= width or (run.fatal and run.pending):
                run.drain_some()
            if run.fatal:
                break
        while run.pending:
            run.drain_some()
    return RunSummary(
        n_pairs=len(pairs), n_skipped=len(pairs) - len(todo), n_ok=run.n_ok,
        n_failed=run.n_failed, n_truncated=run.n_truncated, fatal=run.fatal,
    )
