"""evaluate report: render report.md from whatever outputs exist so far."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from getwork import fileio
from getwork.evaluation import args as eval_args
from getwork.evaluation import judge_run, layout, report
from getwork.evaluation.args import EXIT_OK, Context

COMMANDS = ("report",)
OPTIONAL_INPUTS: dict[str, str] = {
    "metrics": layout.METRICS_FILE,
    "agreement": layout.AGREEMENT_FILE,
    "ari": layout.ARI_FILE,
    "frozen": layout.RUBRIC_FROZEN_FILE,
}


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("report", help="write report.md from the outputs so far")
    eval_args.add_common_args(parser)


def _read_optional(out_dir: Path, name: str) -> dict | None:
    path = out_dir / name
    return fileio.read_json(path) if path.exists() else None


def _count_truncated(path: Path, rubric_hash: str | None) -> int | str:
    if rubric_hash is None or not path.exists():
        return "n/a"
    rows = judge_run.read_judge_rows(path, rubric_hash).values()
    return sum(1 for row in rows if row["status"] == judge_run.OK_STATUS and row["truncated"])


def _with_truncated(manifest: Mapping, out_dir: Path, frozen: Mapping | None) -> dict:
    """The manifest carries no truncated counts, so they are read from the judge files."""
    rubric_hash = (frozen or {}).get("rubric_hash") or manifest.get("rubric_hash")
    judges = []
    for judge in manifest.get("judges", []):
        path = layout.judge_path(out_dir, judge["id"])
        judges.append({**judge, "truncated": _count_truncated(path, rubric_hash)})
    return {**manifest, "judges": judges}


def run(args: argparse.Namespace, ctx: Context) -> int:
    manifest = layout.read_manifest(args.out_dir)
    inputs = {key: _read_optional(args.out_dir, name) for key, name in OPTIONAL_INPUTS.items()}
    text = report.render_report(
        manifest=_with_truncated(manifest, args.out_dir, inputs["frozen"]), **inputs
    )
    target = args.out_dir / layout.REPORT_FILE
    fileio.atomic_write(target, lambda tmp: tmp.write_text(text, encoding="utf-8"))
    not_run = [key for key, value in inputs.items() if value is None]
    print(json.dumps({"report": str(target), "sections_not_run": not_run}))
    return EXIT_OK
