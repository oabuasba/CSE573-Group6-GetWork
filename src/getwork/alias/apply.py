"""Rewrite structured records through the alias table and read the canonical files."""

from collections.abc import Mapping
from pathlib import Path

import pyarrow as pa

from getwork import fileio, paths
from getwork.alias.table import AliasTable
from getwork.structure import inputs, schema, store

CANONICAL_KEYS: tuple[str, ...] = (*schema.RECORD_KEYS, "skills_raw", "alias_table_version")

CANONICAL_PARQUET_SCHEMA: pa.Schema = pa.schema([
    *schema.PARQUET_SCHEMA,
    ("skills_raw", pa.list_(pa.string())),
    ("alias_table_version", pa.string()),
])


def canonicalize_record(record: Mapping[str, object], table: AliasTable) -> dict:
    upgraded = schema.upgrade_record(record)
    row = {key: upgraded[key] for key in schema.RECORD_KEYS}
    row["skills"] = table.canonicalize_list(upgraded["skills"])
    row["skills_raw"] = list(upgraded["skills"])
    row["alias_table_version"] = table.version
    return {key: row[key] for key in CANONICAL_KEYS}


def _summarize(source: str, rows: list[dict], table: AliasTable) -> dict:
    return {
        "source": source,
        "alias_table_version": table.version,
        "n_records": len(rows),
        "skills_before": sum(len(r["skills_raw"]) for r in rows),
        "skills_after": sum(len(r["skills"]) for r in rows),
        "distinct_before": len({s for r in rows for s in r["skills_raw"]}),
        "distinct_after": len({s for r in rows for s in r["skills"]}),
        "n_records_changed": sum(1 for r in rows if r["skills"] != r["skills_raw"]),
        "n_records_without_skills": sum(1 for r in rows if not r["skills"]),
    }


def canonicalize_source(
    source: str, records: Mapping[str, dict], table: AliasTable, out_dir: Path
) -> dict:
    paths.check_source(source)
    rows = [canonicalize_record(records[id_], table) for id_ in inputs.sort_ids(records)]
    summary = _summarize(source, rows, table)
    fileio.write_jsonl_atomic(out_dir / f"{source}_canonical.jsonl", rows)
    fileio.write_parquet_atomic(
        out_dir / f"{source}_canonical.parquet",
        pa.Table.from_pylist(rows, schema=CANONICAL_PARQUET_SCHEMA),
    )
    fileio.write_json_atomic(out_dir / f"{source}_canonical_summary.json", summary)
    return summary


def canonical_path(root: Path, source: str) -> Path:
    return paths.canonical_dir(root, source) / f"{source}_canonical.jsonl"


def read_canonical(root: Path, source: str) -> dict[str, dict]:
    path = canonical_path(root, source)
    if not path.exists():
        raise FileNotFoundError(f"canonical file not found: {path} (run alias-table apply)")
    records = store.read_checkpoint(path, repair=False)
    for id_, record in records.items():
        missing = [key for key in CANONICAL_KEYS if key not in record]
        if missing:
            raise ValueError(f"{path}: record {id_!r} is missing keys {missing}")
    return {id_: records[id_] for id_ in inputs.sort_ids(records)}


def single_alias_version(records_by_source: Mapping[str, Mapping[str, dict]]) -> str:
    versions = {
        record["alias_table_version"]
        for records in records_by_source.values()
        for record in records.values()
    }
    if not versions:
        raise ValueError("no canonical records to read an alias_table_version from")
    if len(versions) > 1:
        raise ValueError(f"records carry different alias_table_versions: {sorted(versions)}")
    return versions.pop()
