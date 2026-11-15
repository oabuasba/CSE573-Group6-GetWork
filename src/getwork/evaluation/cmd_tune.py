"""evaluate tune: plan the pairs that tuning B1+ needs, then pick the best weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from getwork import fileio
from getwork.evaluation import args as eval_args
from getwork.evaluation import labels, layout, loading, pairs, pooling, scoring, tune
from getwork.evaluation.args import EXIT_OK, EXIT_USAGE, Context
from getwork.retrieval.b1plus import B1PlusWeights

COMMANDS = ("tune",)
DEFAULT_STEP = 0.1
TUNE_STAGE = "tune"
DEV_STAGE = "dev"
RESULTS_FILE = "tune_results.csv"
WEIGHTS_FILE = "b1plus_weights.json"
JUDGE_HINT = "run evaluate judge --stage tune (and --stage dev)"
POOL_HINT = "run evaluate pool --stage dev"


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("tune", help="plan the tuning pairs and pick the B1+ weights")
    actions = parser.add_subparsers(dest="action", required=True)
    for name, text in (("plan", "write the pairs the candidates add to dev"),
                       ("pick", "choose the best weights from the judged tune pairs")):
        action = actions.add_parser(name, help=text)
        eval_args.add_common_args(action)
        action.add_argument("--step", type=float, default=DEFAULT_STEP)
        action.add_argument("--k", type=_positive_int, default=pooling.K)


def _require(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found ({hint})")
    return path


def _candidate_tables(
    args: argparse.Namespace,
) -> tuple[list[B1PlusWeights], list[pa.Table], dict]:
    candidates = tune.candidate_weights(args.step)
    out_dir = args.out_dir
    manifest = layout.read_manifest(out_dir)
    sets = fileio.read_json(_require(out_dir / layout.SETS_FILE, layout.RUN_HINT))
    pool_ids = fileio.read_json(
        _require(out_dir / layout.POOL_POSTINGS_FILE, layout.RUN_HINT)
    )["ids"]
    _require(layout.pairs_path(out_dir, DEV_STAGE), POOL_HINT)
    loaded = loading.load_eval_inputs(args.artifacts_dir, args.processed_dir)
    layout.check_stamps(manifest, layout.stamps_of(loaded.data))
    dev = sets["stages"][DEV_STAGE]
    tables = tune.candidate_rankings(
        loaded.data, resume_queries=dev["resumes"], posting_queries=dev["postings"],
        pool_ids=pool_ids, weights=candidates, k=args.k,
    )
    return candidates, tables, {"manifest": manifest, "dev": dev}


def _plan(args: argparse.Namespace) -> int:
    candidates, tables, context = _candidate_tables(args)
    seed = context["manifest"]["seed"]
    dev_pairs = pq.read_table(layout.pairs_path(args.out_dir, DEV_STAGE))
    candidate_pairs = pairs.pairs_from_rankings(pa.concat_tables(tables), seed)
    known = dev_pairs.column("pair_id")
    new_pairs = candidate_pairs.filter(pc.invert(pc.is_in(candidate_pairs["pair_id"], known)))
    fileio.write_parquet_atomic(layout.pairs_path(args.out_dir, TUNE_STAGE), new_pairs)
    fileio.write_jsonl_atomic(
        layout.blind_path(args.out_dir, TUNE_STAGE),
        pairs.blind_rows(new_pairs, seed=seed, purpose=TUNE_STAGE),
    )
    summary = {
        "candidates": len(candidates), "new_pairs": new_pairs.num_rows,
        "calls_if_judged": 2 * new_pairs.num_rows,
    }
    print(json.dumps(summary, sort_keys=True))
    return EXIT_OK


def _pair_id_map(out_dir: Path) -> dict[tuple[str, str], str]:
    tune_path = _require(layout.pairs_path(out_dir, TUNE_STAGE), "run evaluate tune plan")
    both = pa.concat_tables(
        [pq.read_table(layout.pairs_path(out_dir, DEV_STAGE)), pq.read_table(tune_path)]
    )
    return pairs.pair_id_map(both)


def _pick(args: argparse.Namespace) -> int:
    out_dir = args.out_dir
    label_path = _require(out_dir / layout.LABELS_JUDGE_FILE, JUDGE_HINT)
    candidates, tables, context = _candidate_tables(args)
    dev = context["dev"]
    try:
        best, rows = tune.pick_weights(
            tables=tables, candidates=candidates, pair_ids=_pair_id_map(out_dir),
            labels=labels.read_labels(label_path),
            resume_queries=dev["resumes"], posting_queries=dev["postings"],
        )
    except scoring.MissingLabelsError as exc:
        print(f"error: {exc.count} top-10 pairs have no label; {JUDGE_HINT}", file=sys.stderr)
        return EXIT_USAGE
    ndcg10 = next(r["ndcg10"] for r in rows if r["skills"] == best.skills
                  and r["title"] == best.title and r["level"] == best.level)
    tune_dir = layout.tune_dir(out_dir)
    fileio.write_json_atomic(tune_dir / WEIGHTS_FILE, best.as_dict())
    fileio.write_csv_atomic(
        tune_dir / RESULTS_FILE, tune.RESULT_KEYS,
        [[r[key] for key in tune.RESULT_KEYS] for r in rows],
    )
    record = {"step": args.step, "weights": best.as_dict(), "ndcg10": ndcg10}
    layout.update_manifest(out_dir, {"tune": record})
    print(json.dumps(record, sort_keys=True))
    return EXIT_OK


def run(args: argparse.Namespace, ctx: Context) -> int:
    return _plan(args) if args.action == "plan" else _pick(args)
