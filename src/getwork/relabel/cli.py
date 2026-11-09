"""Command line entry point for step 5 (relabel): lookup-only by default, LLM with --use-llm."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from getwork import paths
from getwork.alias import apply
from getwork.alias import cli as alias_cli
from getwork.alias import table as alias_table
from getwork.llm import factory
from getwork.llm.client import FatalLLMError, LLMClient, describe_exception
from getwork.llm.settings import ConfigError
from getwork.relabel import records
from getwork.relabel.cache import RelabelProvenanceError
from getwork.relabel.llm_relabeler import LLMRelabeler
from getwork.taxonomy import model as taxonomy_model

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_USAGE = 2
EXIT_FATAL_LLM = 3
TAXONOMY_FILE = "taxonomy.json"
SOURCE_CHOICES = ("resumes", "postings", "both")
DEFAULT_MIN_KNOWN = 0.5
DEFAULT_MAX_INPUT_CHARS = 48000
DEFAULT_UNSEEN_BATCH_SIZE = 50


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return number


def _fraction(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}") from None
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1: {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relabel", description="Give every document skill a cluster (pipeline step 5)."
    )
    parser.add_argument("--source", choices=SOURCE_CHOICES, default="both")
    parser.add_argument("--artifacts-dir", type=Path, default=paths.DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--processed-dir", type=Path, default=paths.DEFAULT_PROCESSED_DIR)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--min-known", type=_fraction, default=DEFAULT_MIN_KNOWN)
    parser.add_argument("--max-input-chars", type=_positive_int, default=DEFAULT_MAX_INPUT_CHARS)
    parser.add_argument(
        "--unseen-batch-size", type=_positive_int, default=DEFAULT_UNSEEN_BATCH_SIZE
    )
    parser.add_argument("--allow-mixed", action="store_true")
    factory.add_llm_args(parser)
    return parser


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


class CleanTextReader:
    """Reads text_clean of one source on first use; a missing file yields no text."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._texts: dict[str, str] | None = None

    def get(self, doc_id: str) -> str | None:
        if self._texts is None:
            self._texts = self._load()
        return self._texts.get(doc_id)

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        frame = pd.read_parquet(self._path, columns=["id", "text_clean"])
        return {i: t for i, t in zip(frame["id"], frame["text_clean"]) if isinstance(t, str)}


@dataclass(frozen=True)
class Inputs:
    table: alias_table.AliasTable
    tax: taxonomy_model.Taxonomy
    records: dict[str, dict[str, dict]]


def load_inputs(root: Path, sources: tuple[str, ...]) -> Inputs:
    """Read and cross-check every input; raises FileNotFoundError or ValueError."""
    table_path = paths.alias_dir(root) / alias_cli.TABLE_FILE
    if not table_path.exists():
        raise FileNotFoundError(f"alias table not found: {table_path} (run alias-table build)")
    taxonomy_path = paths.taxonomy_dir(root) / TAXONOMY_FILE
    if not taxonomy_path.exists():
        raise FileNotFoundError(f"taxonomy not found: {taxonomy_path} (run taxonomy build)")
    table = alias_table.load_table(table_path)
    tax = taxonomy_model.load_taxonomy(taxonomy_path)
    by_source = {source: apply.read_canonical(root, source) for source in sources}
    for source_records in by_source.values():
        records.check_versions(records=source_records, tax=tax, table=table)
    return Inputs(table=table, tax=tax, records=by_source)


def _make_relabeler(
    args: argparse.Namespace, bundle: factory.ClientBundle, tax: taxonomy_model.Taxonomy
) -> LLMRelabeler:
    return LLMRelabeler(
        bundle.client,
        tax,
        model=bundle.model,
        backend=bundle.backend,
        cache_dir=paths.relabel_cache_dir(args.artifacts_dir),
        batch_size=args.unseen_batch_size,
        max_input_chars=args.max_input_chars,
        extra_body=factory.parse_extra_body(args.extra_body),
        secrets=bundle.secrets,
        allow_mixed=args.allow_mixed,
    )


def _has_failures(summary: Mapping[str, object]) -> bool:
    return bool(summary["n_reextract_failed"] or summary["n_unseen_unresolved"])


def _relabel_sources(
    args: argparse.Namespace, inputs: Inputs, relabeler: LLMRelabeler | None
) -> int:
    failed = False
    for source, source_records in inputs.records.items():
        reader = CleanTextReader(paths.processed_parquet(args.processed_dir, source))
        summary = records.relabel_source(
            source,
            source_records,
            tax=inputs.tax,
            table=inputs.table,
            get_text=reader.get,
            relabeler=relabeler,
            out_dir=paths.relabeled_dir(args.artifacts_dir, source),
            min_known=args.min_known,
        )
        print(json.dumps(summary))
        failed = failed or _has_failures(summary)
    return EXIT_FAILURES if failed else EXIT_OK


def _run(
    args: argparse.Namespace, client: LLMClient | None, environ: Mapping[str, str] | None
) -> int:
    bundle = None
    if args.use_llm:
        bundle = factory.make_client(
            backend=args.backend,
            model=args.model,
            env_file=Path(args.env_file),
            environ=os.environ if environ is None else environ,
            client=client,
        )
    sources = paths.SOURCES if args.source == "both" else (args.source,)
    inputs = load_inputs(args.artifacts_dir, sources)
    relabeler = _make_relabeler(args, bundle, inputs.tax) if bundle else None
    try:
        return _relabel_sources(args, inputs, relabeler)
    except FatalLLMError as exc:
        print(f"error: {describe_exception(exc, bundle.secrets if bundle else ())}",
              file=sys.stderr)
        return EXIT_FATAL_LLM


def main(
    argv: list[str] | None = None,
    *,
    client: LLMClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Exit codes: 0 ok, 1 finished with failures (unresolved unseen skills or failed
    re-extractions), 2 usage, config or input error, 3 fatal LLM error."""
    args = build_parser().parse_args(argv)
    try:
        return _run(args, client, environ)
    except (ConfigError, RelabelProvenanceError, FileNotFoundError, ValueError) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
