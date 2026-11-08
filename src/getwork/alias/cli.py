"""Command line entry point for the alias table (build, apply, propose)."""

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path

from getwork import fileio, paths
from getwork.alias import apply, build, propose
from getwork.alias import table as alias_table
from getwork.llm import factory
from getwork.llm.client import FatalLLMError, LLMClient, describe_exception
from getwork.llm.settings import ConfigError
from getwork.structure import schema, store

EXIT_OK = 0
EXIT_FAILED_BATCHES = 1
EXIT_USAGE = 2
EXIT_FATAL_LLM = 3
TABLE_FILE = "alias_table.json"
REPORT_FILE = "alias_build_report.json"
DRAFT_FILE = "alias_draft.csv"
DRAFT_SUMMARY_FILE = "alias_draft_summary.json"
DRAFT_COLUMNS = (
    "alias", "canonical", "alias_count", "canonical_count", "batch",
    "status", "model", "backend", "prompt_version",
)  # fmt: skip


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer: {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alias-table", description="Build and apply the canonical skill alias table."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("build", "build the alias table from the structured records"),
        ("apply", "rewrite every record's skills through the table"),
        ("propose", "draft more alias pairs with an LLM (user-run, writes a draft CSV only)"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--structured-dir", type=Path, default=paths.DEFAULT_STRUCTURED_DIR)
        p.add_argument("--artifacts-dir", type=Path, default=paths.DEFAULT_ARTIFACTS_DIR)
        if name == "build":
            p.add_argument("--curated", type=Path, default=build.CURATED_PATH)
        if name == "propose":
            factory.add_llm_args(p)
            p.add_argument("--batch-size", type=_positive_int, default=100)
            p.add_argument("--reference-size", type=_positive_int, default=150)
            p.add_argument("--max-skills", type=_positive_int, default=600)
            p.add_argument("--max-tokens", type=_positive_int, default=4096)
    return parser


def _error(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


def _check_record(record: Mapping[str, object], path: Path) -> None:
    upgraded = schema.upgrade_record(record)
    missing = [k for k in schema.RECORD_KEYS if k not in upgraded]
    if missing or not isinstance(upgraded.get("skills"), list):
        raise ValueError(f"{path}: record {upgraded.get('id')!r} does not match the record keys")


def _read_structured(structured_dir: Path) -> dict[str, dict[str, dict]]:
    """Read both sources without modifying them; every file must exist and be valid."""
    result = {}
    for source in paths.SOURCES:
        path = paths.structured_jsonl(structured_dir, source)
        if not path.exists():
            raise FileNotFoundError(f"structured file not found: {path}")
        records = store.read_checkpoint(path, repair=False)
        for record in records.values():
            _check_record(record, path)
        result[source] = records
    return result


def _surface_forms(records_by_source: Mapping[str, Mapping[str, dict]]) -> dict[str, int]:
    skill_lists = [
        r["skills"] for records in records_by_source.values() for r in records.values()
    ]
    return dict(build.count_surface_forms(skill_lists))


def _table_path(artifacts_dir: Path) -> Path:
    return paths.alias_dir(artifacts_dir) / TABLE_FILE


def _load_saved_table(artifacts_dir: Path) -> alias_table.AliasTable:
    path = _table_path(artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(f"alias table not found: {path} (run alias-table build first)")
    return alias_table.load_table(path)


def _run_build(args: argparse.Namespace) -> int:
    records = _read_structured(args.structured_dir)
    curated = build.load_curated(args.curated)
    forms = _surface_forms(records)
    table = build.build_table(forms, curated)
    report = build.build_report(forms, table, len(curated))
    meta = {
        "n_curated_pairs": len(curated),
        "n_records": {source: len(recs) for source, recs in records.items()},
    }
    alias_table.save_table(_table_path(args.artifacts_dir), table, meta)
    fileio.write_json_atomic(paths.alias_dir(args.artifacts_dir) / REPORT_FILE, report)
    print(json.dumps(report))
    return EXIT_OK


def _run_apply(args: argparse.Namespace) -> int:
    table = _load_saved_table(args.artifacts_dir)
    records = _read_structured(args.structured_dir)
    summaries = [
        apply.canonicalize_source(
            source, recs, table, paths.canonical_dir(args.artifacts_dir, source)
        )
        for source, recs in records.items()
    ]
    print(json.dumps(summaries))
    return EXIT_OK


def _propose_config(
    args: argparse.Namespace, bundle: factory.ClientBundle
) -> propose.ProposeConfig:
    return propose.ProposeConfig(
        model=bundle.model,
        backend=bundle.backend,
        batch_size=args.batch_size,
        reference_size=args.reference_size,
        max_skills=args.max_skills,
        max_tokens=args.max_tokens,
        extra_body=factory.parse_extra_body(args.extra_body),
    )


def _write_draft(
    out_dir: Path, result: propose.ProposeResult, cfg: propose.ProposeConfig
) -> dict:
    rows = [[row[column] for column in DRAFT_COLUMNS] for row in result.rows]
    fileio.write_csv_atomic(out_dir / DRAFT_FILE, DRAFT_COLUMNS, rows)
    summary = {
        "n_rows": len(result.rows),
        "failed_batches": result.failed_batches,
        "calls": result.stats.calls,
        "retries": result.stats.retries,
        "rate_limited": result.stats.rate_limited,
        "model": cfg.model,
        "backend": cfg.backend,
    }
    fileio.write_json_atomic(out_dir / DRAFT_SUMMARY_FILE, summary)
    return summary


def _run_propose(
    args: argparse.Namespace, client: LLMClient | None, environ: Mapping[str, str] | None
) -> int:
    bundle = factory.make_client(
        backend=args.backend,
        model=args.model,
        env_file=Path(args.env_file),
        environ=os.environ if environ is None else environ,
        client=client,
    )
    cfg = _propose_config(args, bundle)
    table = _load_saved_table(args.artifacts_dir)
    forms = _surface_forms(_read_structured(args.structured_dir))
    try:
        result = propose.propose_aliases(
            bundle.client,
            cfg,
            build.canonical_counts(forms, table),
            table,
            secrets=bundle.secrets,
        )
    except FatalLLMError as exc:
        print(f"error: {describe_exception(exc, bundle.secrets)}", file=sys.stderr)
        return EXIT_FATAL_LLM
    summary = _write_draft(paths.alias_dir(args.artifacts_dir), result, cfg)
    print(json.dumps(summary))
    return EXIT_FAILED_BATCHES if result.failed_batches else EXIT_OK


def main(
    argv: list[str] | None = None,
    *,
    client: LLMClient | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Exit codes: 0 ok, 1 propose finished with failed batches, 2 usage/config error,
    3 fatal LLM error. argparse usage errors raise SystemExit(2)."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build":
            return _run_build(args)
        if args.command == "apply":
            return _run_apply(args)
        return _run_propose(args, client, environ)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
