"""Output paths and the append-only checkpoint for the structurer."""

import csv
import json
import os
import tempfile
import threading
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO

import pyarrow as pa
import pyarrow.parquet as pq

from getwork.structure import inputs, schema
from getwork.structure.schema import Failure

try:
    import fcntl
except ImportError:  # non-POSIX platforms are unsupported; skip locking
    fcntl = None

FSYNC_EVERY = 50


@dataclass(frozen=True)
class OutputPaths:
    jsonl: Path
    parquet: Path
    failures: Path
    summary: Path


def output_paths(source: str, out_dir: Path) -> OutputPaths:
    return OutputPaths(
        jsonl=out_dir / f"{source}_structured.jsonl",
        parquet=out_dir / f"{source}_structured.parquet",
        failures=out_dir / f"{source}_structured_failures.csv",
        summary=out_dir / f"{source}_structured_summary.json",
    )


class AlreadyRunningError(RuntimeError):
    pass


class CheckpointWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._appends = 0
        self._file: IO[str] | None = open(path, "a", encoding="utf-8", newline="\n")
        if fcntl is not None:
            try:
                fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self._file.close()
                self._file = None
                raise AlreadyRunningError(f"another run holds the checkpoint: {path}")
            except OSError:
                self._file.close()
                self._file = None
                raise

    def append(self, record: dict) -> None:
        with self._lock:
            if self._file is None:
                raise ValueError(f"checkpoint is closed: {self._path}")
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
            self._appends += 1
            if self._appends % FSYNC_EVERY == 0:
                os.fsync(self._file.fileno())

    def close(self) -> None:
        with self._lock:
            if self._file is None:
                return
            self._file.flush()
            os.fsync(self._file.fileno())
            if fcntl is not None:
                fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _parse_record(raw: bytes) -> dict | None:
    try:
        obj = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("id"), str):
        return obj
    return None


def read_checkpoint(path: Path, repair: bool = True) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = path.read_bytes()
    *complete, tail = data.split(b"\n")
    records: dict[str, dict] = {}
    for number, raw in enumerate(complete, start=1):
        obj = _parse_record(raw)
        if obj is None:
            raise ValueError(f"{path}: line {number} is not a JSON object with a string id")
        records.setdefault(obj["id"], obj)
    if tail:
        obj = _parse_record(tail)
        if obj is not None:
            records.setdefault(obj["id"], obj)
            if repair:
                with open(path, "ab") as f:
                    f.write(b"\n")
        elif repair:
            with open(path, "r+b") as f:
                f.truncate(len(data) - len(tail))
    return records


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_final(paths: OutputPaths, records: Mapping[str, dict]) -> None:
    rows = [
        {key: records[id_][key] for key in schema.RECORD_KEYS}
        for id_ in inputs.sort_ids(records)
    ]

    def write_jsonl(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def write_parquet(tmp: Path) -> None:
        table = pa.Table.from_pylist(rows, schema=schema.PARQUET_SCHEMA)
        pq.write_table(table, tmp, compression="zstd")

    _atomic_write(paths.jsonl, write_jsonl)
    _atomic_write(paths.parquet, write_parquet)


def read_failures(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [{**row, "attempts": int(row["attempts"])} for row in rows]


def merge_failures(
    previous: Sequence[dict],
    current: Sequence[Failure],
    attempted: Collection[str],
    done: Collection[str],
) -> list[dict]:
    merged = {
        row["id"]: row for row in previous if row["id"] not in attempted and row["id"] not in done
    }
    for failure in current:
        merged[failure.id] = asdict(failure)
    for id_ in done:
        merged.pop(id_, None)
    return [merged[id_] for id_ in inputs.sort_ids(merged)]


def write_failures(path: Path, rows: Sequence[dict]) -> None:
    def write_csv(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(schema.FAILURE_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

    _atomic_write(path, write_csv)


def write_summary(path: Path, summary: dict) -> None:
    _atomic_write(
        path, lambda tmp: tmp.write_text(json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8")
    )
