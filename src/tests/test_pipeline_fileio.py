import csv
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from getwork import fileio

pytestmark = pytest.mark.usefixtures("offline")


def _leftovers(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


def test_atomic_write_permissions_and_cleanup(tmp_path: Path) -> None:
    old = os.umask(0o022)
    try:
        target = tmp_path / "sub" / "out.txt"
        fileio.atomic_write(target, lambda tmp: tmp.write_text("hi"))
    finally:
        os.umask(old)
    assert target.read_text() == "hi"
    assert target.stat().st_mode & 0o777 == 0o644
    assert _leftovers(target.parent) == []


def test_failed_write_keeps_old_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("old")

    def failing(tmp: Path) -> None:
        tmp.write_text("partial")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        fileio.atomic_write(target, failing)
    assert target.read_text() == "old"
    assert _leftovers(tmp_path) == []


def test_json_jsonl_formats(tmp_path: Path) -> None:
    fileio.write_json_atomic(tmp_path / "a.json", {"b": 1, "a": "é"})
    assert (tmp_path / "a.json").read_bytes() == '{\n  "a": "é",\n  "b": 1\n}'.encode()

    fileio.write_jsonl_atomic(tmp_path / "a.jsonl", [{"z": 1, "a": "é"}, {"k": 2}])
    assert (tmp_path / "a.jsonl").read_bytes() == '{"z": 1, "a": "é"}\n{"k": 2}\n'.encode()


def test_csv_roundtrip_and_empty_header(tmp_path: Path) -> None:
    rows = [["a,b", 'say "hi"', "line1\nline2"], ["x", "y", "z"]]
    fileio.write_csv_atomic(tmp_path / "t.csv", ["c1", "c2", "c3"], rows)
    with open(tmp_path / "t.csv", encoding="utf-8", newline="") as f:
        assert list(csv.reader(f)) == [["c1", "c2", "c3"], *rows]

    fileio.write_csv_atomic(tmp_path / "e.csv", ["c1", "c2"], [])
    assert (tmp_path / "e.csv").read_bytes() == b"c1,c2\r\n"


def test_parquet_roundtrip_zstd(tmp_path: Path) -> None:
    table = pa.table({"id": ["r-1", "r-2"], "n": [1, 2]})
    path = tmp_path / "t.parquet"
    fileio.write_parquet_atomic(path, table)
    assert pq.read_table(path).equals(table)
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


def test_read_json_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="nope.json"):
        fileio.read_json(missing)
    (tmp_path / "ok.json").write_text(json.dumps({"a": 1}))
    assert fileio.read_json(tmp_path / "ok.json") == {"a": 1}
