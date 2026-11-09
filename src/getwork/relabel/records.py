"""Relabeled records (CC-15): per-skill clusters, final skill sets, outputs and their reader."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa

from getwork import fileio, paths
from getwork.alias.apply import CANONICAL_KEYS, CANONICAL_PARQUET_SCHEMA
from getwork.alias.normalize import normalize_key
from getwork.alias.table import AliasTable
from getwork.relabel import lookup
from getwork.relabel.llm_relabeler import Relabeler
from getwork.relabel.lookup import SkillAssignment
from getwork.structure import inputs, store
from getwork.taxonomy.model import OTHER, Taxonomy

RELABEL_KEYS: tuple[str, ...] = (
    *CANONICAL_KEYS, "added_skills", "skill_clusters", "incomplete_reason", "reextracted",
    "relabel_mode", "taxonomy_version", "relabel_model", "relabel_backend",
    "relabel_prompt_version",
)

RELABEL_PARQUET_SCHEMA: pa.Schema = pa.schema([
    *CANONICAL_PARQUET_SCHEMA,
    ("added_skills", pa.list_(pa.string())),
    ("skill_clusters", pa.list_(pa.struct([
        ("skill", pa.string()), ("cluster", pa.string()), ("via", pa.string()),
    ]))),
    ("incomplete_reason", pa.string()),
    ("reextracted", pa.bool_()),
    ("relabel_mode", pa.string()),
    ("taxonomy_version", pa.string()),
    ("relabel_model", pa.string()),
    ("relabel_backend", pa.string()),
    ("relabel_prompt_version", pa.string()),
])

FAILURE_COLUMNS: tuple[str, ...] = ("id", "stage", "message")
INCOMPLETE_REASONS: tuple[str, ...] = ("no_skills", "few_skills", "low_coverage")
NO_TEXT_MESSAGE = "no cleaned text found for the document"
REEXTRACT_FAILED_MESSAGE = "re-extraction failed (validation or permanent error)"


def build_relabel_record(
    record: Mapping[str, object],
    assignments: Sequence[SkillAssignment],
    added_skills: Sequence[str],
    incomplete: str | None,
    reextracted: bool,
    tax: Taxonomy,
    relabeler: Relabeler | None,
) -> dict:
    values: dict[str, object] = {key: copy.deepcopy(record[key]) for key in CANONICAL_KEYS}
    values["skills"] = [a.skill for a in assignments]
    values["added_skills"] = list(added_skills)
    values["skill_clusters"] = [
        {"skill": a.skill, "cluster": a.cluster, "via": a.via} for a in assignments
    ]
    values["incomplete_reason"] = incomplete
    values["reextracted"] = reextracted
    values["relabel_mode"] = "llm" if relabeler else "lookup_only"
    values["taxonomy_version"] = tax.version
    values["relabel_model"] = relabeler.model if relabeler else None
    values["relabel_backend"] = relabeler.backend if relabeler else None
    values["relabel_prompt_version"] = relabeler.prompt_version if relabeler else None
    return {key: values[key] for key in RELABEL_KEYS}


@dataclass
class _Doc:
    id: str
    record: Mapping[str, object]
    unseen: list[str]
    known: float | None
    reason: str | None
    assignments: list[SkillAssignment] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    reextracted: bool = False


def check_versions(records: Mapping[str, dict], *, tax: Taxonomy, table: AliasTable) -> None:
    if table.version != tax.alias_table_version:
        raise ValueError(
            f"alias table version {table.version} differs from the taxonomy's "
            f"alias_table_version {tax.alias_table_version}"
        )
    for id_, record in records.items():
        if record["alias_table_version"] != table.version:
            raise ValueError(
                f"record {id_!r} has alias_table_version {record['alias_table_version']} "
                f"but the alias table is {table.version}"
            )


def _first_pass(
    records: Mapping[str, dict], tax: Taxonomy, min_known: float
) -> tuple[list[_Doc], int]:
    ids = inputs.sort_ids(records)
    counts = [len(records[i]["skills"]) for i in ids]
    p10 = lookup.percentile_threshold(counts=counts) if counts else 0
    docs = []
    for id_ in ids:
        skills = records[id_]["skills"]
        _, unseen = lookup.split_known(skills=skills, tax=tax)
        known = lookup.known_fraction(skills=skills, tax=tax)
        reason = lookup.incomplete_reason(
            n_skills=len(skills), known=known, p10=p10, min_known=min_known
        )
        docs.append(_Doc(id=id_, record=records[id_], unseen=unseen, known=known, reason=reason))
    return docs, p10


def _resolve_unseen(docs: Sequence[_Doc], relabeler: Relabeler | None) -> dict[str, str]:
    """normalize_key -> cluster for every unseen skill the relabeler resolved."""
    first: dict[str, str] = {}
    for doc in docs:
        for skill in doc.unseen:
            first.setdefault(normalize_key(skill), skill)
    if relabeler is None or not first:
        return {}
    answers = relabeler.assign_unseen([first[key] for key in sorted(first)])
    return {normalize_key(skill): cluster for skill, cluster in answers.items()}


def _assign_pass_one(doc: _Doc, tax: Taxonomy, resolved: Mapping[str, str]) -> None:
    skills = doc.record["skills"]
    known, _ = lookup.split_known(skills=skills, tax=tax)
    known_iter = iter(known)
    for skill in skills:
        if tax.is_known(skill):
            doc.assignments.append(next(known_iter))
        elif (cluster := resolved.get(normalize_key(skill))) is not None:
            doc.assignments.append(SkillAssignment(skill=skill, cluster=cluster, via="llm"))
        else:
            doc.assignments.append(SkillAssignment(skill=skill, cluster=OTHER, via="default"))


def _added_assignment(*, skill: str, cluster: str, tax: Taxonomy) -> SkillAssignment:
    known, _ = lookup.split_known(skills=[skill], tax=tax)
    return known[0] if known else SkillAssignment(skill=skill, cluster=cluster, via="llm")


def _merge_added(
    doc: _Doc, pairs: Sequence[tuple[str, str]], *, tax: Taxonomy, table: AliasTable
) -> None:
    present = {normalize_key(a.skill) for a in doc.assignments}
    for skill, cluster in pairs:
        canonical = table.canonical(skill)
        if canonical is None or normalize_key(canonical) in present:
            continue
        present.add(normalize_key(canonical))
        doc.assignments.append(_added_assignment(skill=canonical, cluster=cluster, tax=tax))
        doc.added.append(canonical)


def _reextract_doc(
    doc: _Doc,
    relabeler: Relabeler,
    *,
    get_text: Callable[[str], str | None],
    tax: Taxonomy,
    table: AliasTable,
) -> tuple[str, str] | None:
    """Re-extract one incomplete document; returns (stage, message) when it failed."""
    text = get_text(doc.id)
    if text is None:
        return "no_text", NO_TEXT_MESSAGE
    pairs = relabeler.reextract(
        doc_id=doc.id, text=text, skills=[a.skill for a in doc.assignments]
    )
    if pairs is None:
        return "reextract", REEXTRACT_FAILED_MESSAGE
    doc.reextracted = True
    _merge_added(doc, pairs, tax=tax, table=table)
    return None


def _summary(
    *,
    source: str,
    docs: Sequence[_Doc],
    p10: int,
    failures: Sequence[tuple[str, str, str]],
    resolved: Mapping[str, str],
    tax: Taxonomy,
    table: AliasTable,
    relabeler: Relabeler | None,
) -> dict:
    via_counts = dict.fromkeys(lookup.VIA_VALUES, 0)
    for doc in docs:
        for a in doc.assignments:
            via_counts[a.via] += 1
    unseen_keys = {normalize_key(s) for doc in docs for s in doc.unseen}
    fractions = [doc.known for doc in docs if doc.known is not None]
    return {
        "source": source,
        "relabel_mode": "llm" if relabeler else "lookup_only",
        "taxonomy_version": tax.version,
        "alias_table_version": table.version,
        "n_records": len(docs),
        "p10_threshold": p10,
        "n_incomplete": sum(1 for d in docs if d.reason is not None),
        "incomplete_by_reason": {r: sum(1 for d in docs if d.reason == r)
                                 for r in INCOMPLETE_REASONS},
        "n_reextracted": sum(1 for d in docs if d.reextracted),
        "n_reextract_failed": len(failures),
        "n_skills_final": sum(len(d.assignments) for d in docs),
        "via_counts": via_counts,
        "n_unseen_distinct": len(unseen_keys),
        "n_unseen_unresolved": len(unseen_keys - set(resolved)) if relabeler else 0,
        "mean_known_fraction": sum(fractions) / len(fractions) if fractions else 0.0,
        "relabel_model": relabeler.model if relabeler else None,
        "relabel_backend": relabeler.backend if relabeler else None,
        "relabel_prompt_version": relabeler.prompt_version if relabeler else None,
    }


def _write_outputs(
    *,
    source: str,
    rows: list[dict],
    summary: dict,
    failures: Sequence[tuple[str, str, str]],
    out_dir: Path,
) -> None:
    fileio.write_jsonl_atomic(out_dir / f"{source}_relabeled.jsonl", rows)
    fileio.write_parquet_atomic(
        out_dir / f"{source}_relabeled.parquet",
        pa.Table.from_pylist(rows, schema=RELABEL_PARQUET_SCHEMA),
    )
    fileio.write_json_atomic(out_dir / f"{source}_relabeled_summary.json", summary)
    fileio.write_csv_atomic(out_dir / f"{source}_relabel_failures.csv", FAILURE_COLUMNS, failures)


def relabel_source(
    source: str,
    records: Mapping[str, dict],
    tax: Taxonomy,
    table: AliasTable,
    get_text: Callable[[str], str | None],
    relabeler: Relabeler | None,
    out_dir: Path,
    *,
    min_known: float = 0.5,
) -> dict:
    paths.check_source(source)
    check_versions(records=records, tax=tax, table=table)
    docs, p10 = _first_pass(records, tax, min_known)
    resolved = _resolve_unseen(docs, relabeler)
    failures: list[tuple[str, str, str]] = []
    for doc in docs:
        _assign_pass_one(doc, tax, resolved)
        if relabeler is None or doc.reason is None:
            continue
        failed = _reextract_doc(
            doc, relabeler, get_text=get_text, tax=tax, table=table
        )
        if failed is not None:
            failures.append((doc.id, *failed))
    rows = [
        build_relabel_record(
            doc.record, doc.assignments, doc.added, doc.reason, doc.reextracted, tax, relabeler
        )
        for doc in docs
    ]
    summary = _summary(
        source=source, docs=docs, p10=p10, failures=failures, resolved=resolved, tax=tax,
        table=table, relabeler=relabeler,
    )
    _write_outputs(
        source=source, rows=rows, summary=summary, failures=failures, out_dir=out_dir
    )
    return summary


def relabeled_path(root: Path, source: str) -> Path:
    return paths.relabeled_dir(root, source) / f"{source}_relabeled.jsonl"


def read_relabeled(root: Path, source: str) -> dict[str, dict]:
    path = relabeled_path(root, source)
    if not path.exists():
        raise FileNotFoundError(f"relabeled file not found: {path} (run relabel)")
    records = store.read_checkpoint(path, repair=False)
    for id_, record in records.items():
        missing = [key for key in RELABEL_KEYS if key not in record]
        if missing:
            raise ValueError(f"{path}: record {id_!r} is missing keys {missing}")
    return {id_: records[id_] for id_ in inputs.sort_ids(records)}
