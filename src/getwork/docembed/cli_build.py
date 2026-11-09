"""The doc-embed build subcommand: relabeled records and listed levels to documents tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from getwork import paths
from getwork.docembed import documents, levels, serialize
from getwork.relabel import records
from getwork.taxonomy import model as taxonomy_model

TAXONOMY_FILE = "taxonomy.json"
LISTED_LEVEL_COLUMNS = ["id", "experience_level"]


def add_build_parser(sub: argparse._SubParsersAction, *, source_choices: tuple[str, ...]) -> None:
    parser = sub.add_parser("build", help="build the documents tables (texts only, offline)")
    parser.add_argument("--source", choices=source_choices, default="both")
    parser.add_argument("--processed-dir", type=Path, default=paths.DEFAULT_PROCESSED_DIR)
    parser.add_argument("--level-bands", default=_format_bands(levels.DEFAULT_BANDS))
    parser.add_argument("--b2-order", choices=serialize.B2_ORDERS, default="grouped")


def _format_bands(bands: tuple[float, float]) -> str:
    return ",".join(f"{b:g}" for b in bands)


def _load_taxonomy(root: Path) -> taxonomy_model.Taxonomy:
    path = paths.taxonomy_dir(root) / TAXONOMY_FILE
    if not path.exists():
        raise FileNotFoundError(f"taxonomy not found: {path} (run taxonomy build)")
    return taxonomy_model.load_taxonomy(path)


def _check_against_taxonomy(
    *, source: str, source_records: dict[str, dict], tax: taxonomy_model.Taxonomy
) -> None:
    for id_, record in source_records.items():
        for key, expected in (
            ("alias_table_version", tax.alias_table_version),
            ("taxonomy_version", tax.version),
        ):
            if record[key] != expected:
                raise ValueError(
                    f"{source} record {id_!r} has {key} {record[key]} but taxonomy.json "
                    f"has {expected}; run relabel again"
                )


def _read_listed_levels(processed_dir: Path, source: str) -> dict[str, object]:
    if source != "postings":
        return {}
    path = paths.processed_parquet(processed_dir, source)
    if not path.exists():
        print(
            f"note: {path} not found; posting levels are derived from years of experience",
            file=sys.stderr,
        )
        return {}
    try:
        frame = pd.read_parquet(path, columns=LISTED_LEVEL_COLUMNS)
    except (KeyError, ValueError, OSError) as exc:
        raise ValueError(
            f"cannot read columns {LISTED_LEVEL_COLUMNS} from {path}: {exc}"
        ) from exc
    return dict(zip(frame["id"], frame["experience_level"]))


def _dry_run(
    *,
    source_records: dict[str, dict],
    listed: dict[str, object],
    tax: taxonomy_model.Taxonomy,
    bands: tuple[float, float],
    b2_order: str,
) -> None:
    """Serialize every record without writing, so a bad record fails before any output exists."""
    for id_, record in source_records.items():
        documents.build_document(
            record=record, listed_level=listed.get(id_), tax=tax, bands=bands, b2_order=b2_order
        )


def run_build(args: argparse.Namespace) -> int:
    """Validate every input first; ValueError or FileNotFoundError mean exit 2, nothing written."""
    bands = levels.parse_bands(args.level_bands)
    root = args.artifacts_dir
    tax = _load_taxonomy(root)
    sources = paths.SOURCES if args.source == "both" else (args.source,)
    relabeled = {source: records.read_relabeled(root=root, source=source) for source in sources}
    provenance = {}
    for source, source_records in relabeled.items():
        _check_against_taxonomy(source=source, source_records=source_records, tax=tax)
        provenance[source] = documents.provenance_of(source_records)
    if len(provenance) == 2:
        documents.check_comparable(
            a=provenance["resumes"], b=provenance["postings"], keys=documents.PROVENANCE_KEYS
        )
    listed = {source: _read_listed_levels(args.processed_dir, source) for source in sources}
    for source, source_records in relabeled.items():
        _dry_run(
            source_records=source_records, listed=listed[source], tax=tax, bands=bands,
            b2_order=args.b2_order,
        )
    for source, source_records in relabeled.items():
        summary = documents.build_documents(
            source=source,
            relabeled=source_records,
            listed_levels=listed[source],
            tax=tax,
            out_dir=paths.documents_dir(root, source),
            bands=bands,
            b2_order=args.b2_order,
        )
        print(json.dumps(summary))
    return 0
