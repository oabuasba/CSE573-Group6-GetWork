"""Atomic writers with normal file permissions, shared by all pipeline stages."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a temp file in the target directory, then replace the target.

    Reads the process umask, which is not thread-safe: call from the main thread only.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, obj: object) -> None:
    text = json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)
    atomic_write(path, lambda tmp: tmp.write_bytes(text.encode("utf-8")))


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    def write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    atomic_write(path, write)


def write_csv_atomic(
    path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]
) -> None:
    def write(tmp: Path) -> None:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    atomic_write(path, write)


def write_parquet_atomic(path: Path, table: pa.Table) -> None:
    atomic_write(path, lambda tmp: pq.write_table(table, tmp, compression="zstd"))


def read_json(path: Path) -> object:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"file not found: {path}") from e
