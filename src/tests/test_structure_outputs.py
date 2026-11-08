import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from getwork.structure import schema, store
from getwork.structure.schema import Failure, ModelOutput

pytestmark = pytest.mark.usefixtures("offline")


def _record(id_: str, **overrides: object) -> dict:
    fields = {
        "title": "Engineer", "skills": ["Python"], "experience": [], "education": [],
        "years_experience": None,
    }
    fields.update(overrides)
    output = ModelOutput.model_validate(fields)
    return schema.to_record(output, id_, "resumes", "m", "v1")


def _failure(id_: str, message: str = "boom", attempts: int = 1) -> Failure:
    return Failure(id=id_, stage="transient", error_type="Err", message=message, attempts=attempts)


def test_write_final_sorts_numeric_and_is_atomic(tmp_path: Path) -> None:
    paths = store.output_paths("resumes", tmp_path)
    records = {i: _record(i) for i in ("r-2", "r-10", "r-1")}
    store.write_final(paths, records)
    lines = paths.jsonl.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == ["r-1", "r-2", "r-10"]
    assert list(json.loads(lines[0])) == list(schema.RECORD_KEYS)
    assert pq.read_table(paths.parquet).column("id").to_pylist() == ["r-1", "r-2", "r-10"]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([paths.jsonl.name, paths.parquet.name])


def test_parquet_roundtrip_with_nulls_and_empty_lists(tmp_path: Path) -> None:
    paths = store.output_paths("resumes", tmp_path)
    full = _record(
        "r-1",
        experience=[{"role": None, "company": "Acme", "duration_months": 12}],
        education=[{"degree": "BS", "field": None, "institution": None}],
        years_experience=2.5,
    )
    store.write_final(paths, {"r-1": full, "r-2": _record("r-2")})
    table = pq.read_table(paths.parquet)
    assert table.schema == schema.PARQUET_SCHEMA
    assert table.to_pylist()[0]["experience"] == full["experience"]
    assert table.to_pylist()[1]["years_experience"] is None

    empty_paths = store.output_paths("postings", tmp_path)
    store.write_final(empty_paths, {"r-1": _record("r-1"), "r-2": _record("r-2")})
    assert pq.read_table(empty_paths.parquet).schema == schema.PARQUET_SCHEMA

    store.write_final(empty_paths, {})
    assert empty_paths.jsonl.read_text(encoding="utf-8") == ""
    assert pq.read_table(empty_paths.parquet).num_rows == 0


def test_merge_failures_rules() -> None:
    previous = [
        {"id": "r-10", "stage": "transient", "error_type": "E", "message": "old", "attempts": 1},
        {"id": "r-3", "stage": "transient", "error_type": "E", "message": "old", "attempts": 1},
        {"id": "r-4", "stage": "transient", "error_type": "E", "message": "old", "attempts": 1},
        {"id": "r-5", "stage": "transient", "error_type": "E", "message": "old", "attempts": 1},
    ]
    merged = store.merge_failures(
        previous, [_failure("r-3", "new", 2), _failure("r-2")], attempted={"r-3", "r-4", "r-2"}, done={"r-5", "r-4"},
    )
    assert [row["id"] for row in merged] == ["r-2", "r-3", "r-10"]
    assert merged[1]["message"] == "new" and merged[1]["attempts"] == 2
    assert merged[2]["message"] == "old"


def test_failures_csv_roundtrip_with_awkward_messages(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    rows = [
        {"id": "r-1", "stage": "validation", "error_type": "V", "message": 'a, "b"\nline2 é', "attempts": 3},
        {"id": "r-2", "stage": "permanent", "error_type": "P", "message": "", "attempts": 1},
    ]
    store.write_failures(path, rows)
    assert store.read_failures(path) == rows
    assert store.read_failures(tmp_path / "missing.csv") == []
    assert [p.name for p in tmp_path.iterdir()] == ["f.csv"]


def test_write_failures_header_when_empty(tmp_path: Path) -> None:
    path = tmp_path / "f.csv"
    store.write_failures(path, [])
    assert path.read_text(encoding="utf-8").strip() == ",".join(schema.FAILURE_COLUMNS)
    assert store.read_failures(path) == []


def test_write_summary_format(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    store.write_summary(path, {"b": 1, "a": "é"})
    assert path.read_text(encoding="utf-8") == json.dumps({"a": "é", "b": 1}, indent=2)
