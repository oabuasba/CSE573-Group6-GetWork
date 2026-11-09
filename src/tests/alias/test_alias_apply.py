import copy
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from getwork.alias import apply
from getwork.alias.table import make_table
from getwork.structure import schema

pytestmark = pytest.mark.usefixtures("offline")

TABLE = make_table({"js": "JavaScript", "javascript": "JavaScript", "java": "Java"})


def _record(id_: str, skills: list[str], source: str = "resumes") -> dict:
    output = schema.ModelOutput(
        title="Dev", skills=skills, experience=[], education=[], years_experience=None
    )
    record = schema.to_record(
        output, record_id=id_, source=source, model="m", prompt_version="p"
    )
    record["skills"] = list(skills)  # the validator would already drop case duplicates
    return record


def test_canonicalize_record_keys_and_raw() -> None:
    record = _record("r-1", ["JS", "Java", "JAVA", "Rust"])
    out = apply.canonicalize_record(record, TABLE)
    assert tuple(out) == apply.CANONICAL_KEYS
    assert out["skills"] == ["JavaScript", "Java", "Rust"]
    assert out["skills_raw"] == ["JS", "Java", "JAVA", "Rust"]
    assert out["alias_table_version"] == TABLE.version
    assert out["title"] == "Dev"


def test_canonicalize_record_upgrades_legacy_and_does_not_mutate() -> None:
    record = _record("r-1", ["JS"])
    del record["backend"]
    before = copy.deepcopy(record)
    out = apply.canonicalize_record(record, TABLE)
    assert out["backend"] == "voyager"
    assert record == before


def _records() -> dict[str, dict]:
    return {
        "r-10": _record("r-10", ["JS", "Go"]),
        "r-2": _record("r-2", ["Java", "java"]),
        "r-3": _record("r-3", []),
    }


def test_canonicalize_source_outputs_sorted(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    apply.canonicalize_source("resumes", _records(), TABLE, out_dir)
    lines = (out_dir / "resumes_canonical.jsonl").read_text(encoding="utf-8").splitlines()
    ids = [json.loads(line)["id"] for line in lines]
    assert ids == ["r-2", "r-3", "r-10"]
    assert list(json.loads(lines[0])) == list(apply.CANONICAL_KEYS)
    table = pq.read_table(out_dir / "resumes_canonical.parquet")
    assert table.schema.equals(apply.CANONICAL_PARQUET_SCHEMA)
    assert table.column("id").to_pylist() == ids
    assert (out_dir / "resumes_canonical_summary.json").is_file()


def test_canonicalize_source_empty_skills_keep_schema(tmp_path: Path) -> None:
    records = {"r-1": _record("r-1", [])}
    apply.canonicalize_source("resumes", records, TABLE, tmp_path)
    table = pq.read_table(tmp_path / "resumes_canonical.parquet")
    assert table.schema.equals(apply.CANONICAL_PARQUET_SCHEMA)


def test_canonicalize_source_summary_counts(tmp_path: Path) -> None:
    summary = apply.canonicalize_source("resumes", _records(), TABLE, tmp_path)
    assert summary == {
        "source": "resumes",
        "alias_table_version": TABLE.version,
        "n_records": 3,
        "skills_before": 4,
        "skills_after": 3,
        "distinct_before": 4,
        "distinct_after": 3,
        "n_records_changed": 2,
        "n_records_without_skills": 1,
    }
    on_disk = json.loads((tmp_path / "resumes_canonical_summary.json").read_text("utf-8"))
    assert on_disk == summary


def test_canonicalize_source_is_deterministic(tmp_path: Path) -> None:
    apply.canonicalize_source("resumes", _records(), TABLE, tmp_path / "a")
    apply.canonicalize_source("resumes", _records(), TABLE, tmp_path / "b")
    for name in ("jsonl", "parquet"):
        file = f"resumes_canonical.{name}"
        assert (tmp_path / "a" / file).read_bytes() == (tmp_path / "b" / file).read_bytes()
    summary = "resumes_canonical_summary.json"
    assert (tmp_path / "a" / summary).read_bytes() == (tmp_path / "b" / summary).read_bytes()


def test_read_canonical_order_and_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="alias-table apply") as info:
        apply.read_canonical(tmp_path, "resumes")
    assert str(apply.canonical_path(tmp_path, "resumes")) in str(info.value)

    out_dir = tmp_path / "canonical" / "resumes"
    apply.canonicalize_source("resumes", _records(), TABLE, out_dir)
    assert apply.canonical_path(tmp_path, "resumes") == out_dir / "resumes_canonical.jsonl"
    assert list(apply.read_canonical(tmp_path, "resumes")) == ["r-2", "r-3", "r-10"]

    path = apply.canonical_path(tmp_path, "resumes")
    lines = path.read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[1])
    del broken["skills_raw"]
    lines[1] = json.dumps(broken)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="r-3"):
        apply.read_canonical(tmp_path, "resumes")


def test_single_alias_version() -> None:
    a = apply.canonicalize_record(_record("r-1", ["JS"]), TABLE)
    b = apply.canonicalize_record(_record("j-1", ["JS"], "postings"), TABLE)
    both = {"resumes": {"r-1": a}, "postings": {"j-1": b}}
    assert apply.single_alias_version(both) == TABLE.version

    other = make_table({"x": "X"})
    c = apply.canonicalize_record(_record("j-1", ["JS"], "postings"), other)
    with pytest.raises(ValueError) as info:
        apply.single_alias_version({"resumes": {"r-1": a}, "postings": {"j-1": c}})
    assert TABLE.version in str(info.value) and other.version in str(info.value)

    with pytest.raises(ValueError):
        apply.single_alias_version({"resumes": {}, "postings": {}})
