"""The documents table (one row per relabeled record) and its provenance checks."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa

from getwork import fileio, paths
from getwork.docembed import levels, serialize
from getwork.structure import inputs
from getwork.taxonomy.model import OTHER, Taxonomy

DOCUMENT_KEYS: tuple[str, ...] = (
    "id", "source", "title", "level", "level_source", "years_experience", "skills",
    "skill_clusters", "text_full", "text_b2", "alias_table_version", "taxonomy_version",
    "relabel_mode", "relabel_model", "relabel_backend", "serialization_version",
)

_STR = pa.string()
DOCUMENT_PARQUET_SCHEMA: pa.Schema = pa.schema([
    ("id", _STR), ("source", _STR), ("title", _STR), ("level", _STR), ("level_source", _STR),
    ("years_experience", pa.float64()),
    ("skills", pa.list_(_STR)),
    ("skill_clusters", pa.list_(pa.struct([("skill", _STR), ("cluster", _STR), ("via", _STR)]))),
    ("text_full", _STR), ("text_b2", _STR), ("alias_table_version", _STR),
    ("taxonomy_version", _STR), ("relabel_mode", _STR), ("relabel_model", _STR),
    ("relabel_backend", _STR), ("serialization_version", _STR),
])

PROVENANCE_KEYS: tuple[str, ...] = (
    "alias_table_version", "taxonomy_version", "relabel_mode", "relabel_model", "relabel_backend",
)
COMPARE_KEYS: tuple[str, ...] = (
    *PROVENANCE_KEYS, "serialization_version", "level_bands", "b2_order",
)
N_EXAMPLES = 3
BUILD_HINT = "run doc-embed build"


class DocEmbedMismatchError(RuntimeError):
    """Documents or vectors made under different settings must not be compared."""


def build_document(
    record: Mapping[str, object],
    listed_level: object,
    tax: Taxonomy,
    bands: tuple[float, float] = levels.DEFAULT_BANDS,
    b2_order: str = "grouped",
) -> dict:
    level, level_source = levels.resolve_level(
        source=record["source"],
        listed_raw=listed_level,
        years=record["years_experience"],
        bands=bands,
    )
    clusters = copy.deepcopy(record["skill_clusters"])
    order = tax.cluster_names()
    doc = {
        "id": record["id"],
        "source": record["source"],
        "title": record["title"],
        "level": level,
        "level_source": level_source,
        "years_experience": record["years_experience"],
        "skills": list(record["skills"]),
        "skill_clusters": clusters,
        "text_full": serialize.serialize_full(
            title=record["title"], level=level, skill_clusters=clusters, cluster_order=order
        ),
        "text_b2": serialize.serialize_b2(
            title=record["title"], level=level, skill_clusters=clusters,
            cluster_order=order, order=b2_order,
        ),
        **{key: record[key] for key in PROVENANCE_KEYS},
        "serialization_version": serialize.SERIALIZATION_VERSION,
    }
    return {key: doc[key] for key in DOCUMENT_KEYS}


def provenance_of(records: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    """The PROVENANCE_KEYS values shared by all records; ValueError if empty or mixed."""
    if not records:
        raise ValueError("no records to take provenance from")
    first_id, first = next(iter(records.items()))
    found = {key: first[key] for key in PROVENANCE_KEYS}
    for id_, record in records.items():
        for key in PROVENANCE_KEYS:
            if record[key] != found[key]:
                raise ValueError(
                    f"records disagree on {key}: {first_id!r} has {found[key]!r} "
                    f"but {id_!r} has {record[key]!r}"
                )
    return found


def check_comparable(
    a: Mapping[str, object], b: Mapping[str, object], keys: Sequence[str] = COMPARE_KEYS
) -> None:
    """Raise DocEmbedMismatchError naming every key whose values differ (absent counts as None)."""
    diffs = [f"{k}: {a.get(k)!r} vs {b.get(k)!r}" for k in keys if a.get(k) != b.get(k)]
    if diffs:
        raise DocEmbedMismatchError("documents are not comparable: " + "; ".join(diffs))


def _level_counts(docs: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {level: 0 for level in (*levels.LEVELS, levels.NOT_SPECIFIED)}
    for doc in docs:
        counts[doc["level"]] += 1
    return counts


def _summarize(
    *,
    source: str,
    docs: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
    bands: tuple[float, float],
    b2_order: str,
) -> dict:
    n_skills = sum(len(d["skills"]) for d in docs)
    n_other = sum(c["cluster"] == OTHER for d in docs for c in d["skill_clusters"])
    source_counts = {name: 0 for name in levels.LEVEL_SOURCES}
    for doc in docs:
        source_counts[doc["level_source"]] += 1
    return {
        "source": source,
        "n_documents": len(docs),
        **provenance,
        "serialization_version": serialize.SERIALIZATION_VERSION,
        "level_bands": list(bands),
        "b2_order": b2_order,
        "level_counts": _level_counts(docs),
        "level_source_counts": source_counts,
        "n_empty_skills": sum(not d["skills"] for d in docs),
        "mean_skills": n_skills / len(docs) if docs else 0.0,
        "other_share": n_other / n_skills if n_skills else 0.0,
        "examples": [
            {"id": d["id"], "text_full": d["text_full"], "text_b2": d["text_b2"]}
            for d in docs[:N_EXAMPLES]
        ],
    }


def build_documents(
    source: str,
    relabeled: Mapping[str, dict],
    listed_levels: Mapping[str, object],
    tax: Taxonomy,
    out_dir: Path,
    *,
    bands: tuple[float, float] = levels.DEFAULT_BANDS,
    b2_order: str = "grouped",
) -> dict:
    """Write the jsonl, parquet and summary files for one source and return the summary."""
    provenance = provenance_of(relabeled)
    docs = [
        build_document(
            record=relabeled[id_],
            listed_level=listed_levels.get(id_),
            tax=tax,
            bands=bands,
            b2_order=b2_order,
        )
        for id_ in inputs.sort_ids(relabeled)
    ]
    summary = _summarize(
        source=source, docs=docs, provenance=provenance, bands=bands, b2_order=b2_order
    )
    fileio.write_jsonl_atomic(out_dir / f"{source}_documents.jsonl", docs)
    table = pa.Table.from_pylist(docs, schema=DOCUMENT_PARQUET_SCHEMA)
    fileio.write_parquet_atomic(out_dir / f"{source}_documents.parquet", table)
    fileio.write_json_atomic(out_dir / f"{source}_documents_summary.json", summary)
    return summary


def documents_path(root: Path, source: str) -> Path:
    return paths.documents_dir(root, source) / f"{source}_documents.jsonl"


def _summary_path(root: Path, source: str) -> Path:
    return paths.documents_dir(root, source) / f"{source}_documents_summary.json"


def read_documents(root: Path, source: str) -> dict[str, dict]:
    """Documents in numeric id order; FileNotFoundError or ValueError name the problem."""
    path = documents_path(root, source)
    if not path.exists():
        raise FileNotFoundError(f"documents file not found: {path} ({BUILD_HINT})")
    docs: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {line_no} is not valid JSON ({e.msg})") from e
            missing = [key for key in DOCUMENT_KEYS if key not in doc]
            if missing:
                raise ValueError(f"{path}: document {doc.get('id')!r} is missing keys {missing}")
            docs[doc["id"]] = doc
    return {id_: docs[id_] for id_ in inputs.sort_ids(docs)}


def read_documents_summary(root: Path, source: str) -> dict:
    path = _summary_path(root, source)
    if not path.exists():
        raise FileNotFoundError(f"documents summary not found: {path} ({BUILD_HINT})")
    return fileio.read_json(path)
