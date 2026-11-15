"""evaluate freeze-rubric and evaluate labels: pin the rubric, rebuild the label table."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from getwork import fileio
from getwork.evaluation import args as eval_args
from getwork.evaluation import defaults, judge_run, judges, labels, layout
from getwork.evaluation.args import EXIT_OK, EXIT_USAGE, Context
from getwork.evaluation.judges import Rubric

COMMANDS = ("freeze-rubric", "labels")
FROZEN_AT_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def add_parser(sub: argparse._SubParsersAction) -> None:
    freeze = sub.add_parser("freeze-rubric", help="record the rubric hash before the test stage")
    eval_args.add_common_args(freeze)
    freeze.add_argument("--rubric-file", type=Path, default=None)
    freeze.add_argument("--force", action="store_true", help="overwrite an earlier freeze")
    rebuild = sub.add_parser("labels", help="rebuild labels_judge.parquet from the judge files")
    eval_args.add_common_args(rebuild)
    rebuild.add_argument("--rubric-file", type=Path, default=None)


def rebuild_labels(out_dir: Path, rubric: Rubric) -> int:
    """Write labels_judge.parquet from both judge files; returns the number of labeled pairs."""
    rows = []
    for judge_id in defaults.DEFAULT_JUDGE_IDS:
        path = layout.judge_path(out_dir, judge_id)
        if not path.exists():
            raise FileNotFoundError(f"judge file not found: {path} (run evaluate judge)")
        rows.append(judge_run.read_judge_rows(path, rubric.hash))
    table = labels.build_labels_table(*rows)
    fileio.write_parquet_atomic(out_dir / layout.LABELS_JUDGE_FILE, table)
    return table.num_rows


def _run_freeze(args: argparse.Namespace) -> int:
    rubric = judges.load_rubric(args.rubric_file)
    target = args.out_dir / layout.RUBRIC_FROZEN_FILE
    if target.exists() and not args.force:
        print(f"error: {target} already exists; use --force to freeze again", file=sys.stderr)
        return EXIT_USAGE
    frozen_at = datetime.now(timezone.utc).strftime(FROZEN_AT_FORMAT)
    fileio.write_json_atomic(target, {
        "rubric_version": rubric.version, "rubric_hash": rubric.hash, "frozen_at": frozen_at,
    })
    layout.update_manifest(args.out_dir, {"rubric_hash": rubric.hash})
    print(rubric.hash)
    return EXIT_OK


def _run_labels(args: argparse.Namespace) -> int:
    rubric = judges.load_rubric(args.rubric_file)
    n_pairs = rebuild_labels(args.out_dir, rubric)
    print(json.dumps({"labeled_pairs": n_pairs, "rubric_hash": rubric.hash}, sort_keys=True))
    return EXIT_OK


def run(args: argparse.Namespace, ctx: Context) -> int:
    if args.command == "freeze-rubric":
        return _run_freeze(args)
    return _run_labels(args)
