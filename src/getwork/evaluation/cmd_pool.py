"""evaluate pool: rank the stage's queries in the capped pool and write the blind pair files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa

from getwork import fileio
from getwork.evaluation import args as eval_args
from getwork.evaluation import layout, loading, pairs, pooling
from getwork.evaluation.args import EXIT_OK, EXIT_USAGE, Context
from getwork.retrieval import b1plus

COMMANDS = ("pool",)
HUMAN_STAGE = "human"
DEFAULT_NOTE = "B1+ weights are the defaults (not tuned)"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("pool", help="rank the stage queries and write the blind pairs")
    eval_args.add_common_args(parser)
    parser.add_argument("--stage", choices=("dev", "test"), required=True)
    parser.add_argument("--k", type=_positive_int, default=pooling.K)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--weights", help="B1+ weights skills,title,level, e.g. 0.7,0.15,0.15")
    group.add_argument("--weights-file", type=Path, help="JSON file with skills, title, level")
    parser.add_argument("--force", action="store_true", help="overwrite earlier outputs")


def _resolve_weights(args: argparse.Namespace) -> tuple[b1plus.B1PlusWeights, str]:
    if args.weights is not None:
        return b1plus.parse_weights(args.weights), "cli"
    if args.weights_file is not None:
        return b1plus.load_weights(args.weights_file), f"file:{args.weights_file}"
    return b1plus.DEFAULT_WEIGHTS, "default"


def _output_paths(out_dir: Path, stage: str) -> list[Path]:
    return [
        layout.rankings_path(out_dir, stage), layout.pairs_path(out_dir, stage),
        layout.blind_path(out_dir, stage),
    ]


def _read_json_input(out_dir: Path, name: str) -> dict:
    path = out_dir / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found ({layout.RUN_HINT})")
    return fileio.read_json(path)


def _write_human_file(
    out_dir: Path, *, sets: dict, rankings: pa.Table, pair_table: pa.Table, seed: int
) -> int:
    human = sets["stages"][HUMAN_STAGE]
    ids = pairs.human_pair_ids(
        rankings, human_resumes=human["resumes"], human_postings=human["postings"], seed=seed
    )
    rows = pairs.blind_rows(pair_table, seed=seed, purpose=HUMAN_STAGE, only=ids)
    fileio.write_jsonl_atomic(layout.blind_path(out_dir, HUMAN_STAGE), rows)
    return len(rows)


def run(args: argparse.Namespace, ctx: Context) -> int:
    manifest = layout.read_manifest(args.out_dir)
    sets = _read_json_input(args.out_dir, layout.SETS_FILE)
    pool_ids = _read_json_input(args.out_dir, layout.POOL_POSTINGS_FILE)["ids"]
    weights, source = _resolve_weights(args)
    loaded = loading.load_eval_inputs(args.artifacts_dir, args.processed_dir)
    layout.check_stamps(manifest, layout.stamps_of(loaded.data))
    targets = _output_paths(args.out_dir, args.stage)
    if not args.force:
        existing = [path for path in targets if path.exists()]
        if existing:
            print(f"error: {existing[0]} already exists; use --force to overwrite",
                  file=sys.stderr)
            return EXIT_USAGE
    queries = sets["stages"][args.stage]
    rankings = pooling.stage_rankings(
        loaded.data, stage=args.stage, resume_queries=queries["resumes"],
        posting_queries=queries["postings"], pool_ids=pool_ids, k=args.k, weights=weights,
    )
    seed = manifest["seed"]
    pair_table = pairs.pairs_from_rankings(rankings, seed)
    fileio.write_parquet_atomic(targets[0], rankings)
    fileio.write_parquet_atomic(targets[1], pair_table)
    fileio.write_jsonl_atomic(
        targets[2], pairs.blind_rows(pair_table, seed=seed, purpose=args.stage)
    )
    summary = {"stage": args.stage, "rows": rankings.num_rows, "pairs": pair_table.num_rows}
    if args.stage == "test":
        summary["human_pairs"] = _write_human_file(
            args.out_dir, sets=sets, rankings=rankings, pair_table=pair_table, seed=seed
        )
        if source == "default":
            print(DEFAULT_NOTE, file=sys.stderr)
    stored = {**manifest.get("weights", {}), args.stage: {**weights.as_dict(), "source": source}}
    layout.update_manifest(args.out_dir, {"weights": stored})
    print(json.dumps(summary, sort_keys=True))
    return EXIT_OK
