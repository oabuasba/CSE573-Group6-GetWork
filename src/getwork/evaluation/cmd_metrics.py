"""evaluate metrics: score the three systems with one set of labels into metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from getwork import fileio
from getwork.evaluation import args as eval_args
from getwork.evaluation import human, labels, layout, pairs, scoring
from getwork.evaluation.args import EXIT_OK, EXIT_USAGE, Context

COMMANDS = ("metrics",)
JUDGE_LABELS = "judge"
QUERY_CHOICES = ("test", "human")
DEFAULT_RESAMPLES = 10000
HEADLINE_METRIC = "nDCG@10"
JUDGE_HINT = "run evaluate judge --stage test"
EMPTY_METRICS: dict = {"schema": 1, "groups": {}}


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("metrics", help="score B1+, B2 and Full with judge or author labels")
    eval_args.add_common_args(parser)
    parser.add_argument("--labels", required=True, help="judge or an author slug")
    parser.add_argument("--queries", required=True, choices=QUERY_CHOICES)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--n-boot", type=_positive_int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--n-perm", type=_positive_int, default=DEFAULT_RESAMPLES)


def _query_ids(out_dir: Path, queries: str) -> dict[str, list[str]]:
    stage = fileio.read_json(out_dir / layout.SETS_FILE)["stages"][queries]
    return {
        "resume_to_posting": sorted(stage["resumes"]),
        "posting_to_resume": sorted(stage["postings"]),
    }


def _human_pair_ids(out_dir: Path) -> set[str]:
    path = layout.blind_path(out_dir, human.HUMAN_STAGE)
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["pair_id"] for line in f if line.strip()}


def _load_labels(args: argparse.Namespace) -> dict[str, int]:
    if args.labels == JUDGE_LABELS:
        path = args.out_dir / layout.LABELS_JUDGE_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found ({JUDGE_HINT})")
        return labels.read_labels(path)
    if args.queries != "human":
        raise ValueError(f"--labels {args.labels} needs --queries human")
    author_labels = human.read_human_labels(args.out_dir, args.labels)
    missing = len(_human_pair_ids(args.out_dir) - set(author_labels))
    if missing and not args.allow_missing:
        raise ValueError(
            f"author {args.labels} has not labeled {missing} human pairs; finish labeling "
            f"or pass --allow-missing"
        )
    return author_labels


def _build_group(args: argparse.Namespace, label_map: dict[str, int]) -> dict:
    out_dir = args.out_dir
    seed = layout.read_manifest(out_dir)["seed"]
    rankings = pq.read_table(layout.rankings_path(out_dir, "test"))
    pair_ids = pairs.pair_id_map(pq.read_table(layout.pairs_path(out_dir, "test")))
    group: dict = {
        "labels": args.labels, "queries": args.queries, "n_boot": args.n_boot,
        "n_perm": args.n_perm, "seed": seed, "allow_missing": args.allow_missing,
        "n_missing": 0, "n_no_relevant": {}, "directions": {},
    }
    for direction, query_ids in _query_ids(out_dir, args.queries).items():
        try:
            scores = scoring.score_queries(
                rankings=rankings, direction=direction, query_ids=query_ids, pair_ids=pair_ids,
                labels=label_map, allow_missing=args.allow_missing,
            )
        except scoring.MissingLabelsError as exc:
            raise ValueError(
                f"{exc.count} top-10 pairs have no label in {direction}; {JUDGE_HINT}"
            ) from exc
        group["n_missing"] += scores.n_missing
        group["n_no_relevant"][direction] = scores.n_no_relevant
        group["directions"][direction] = scoring.aggregate_group(
            scores, n_boot=args.n_boot, n_perm=args.n_perm, seed=seed
        )
    return group


def _headline(group: dict) -> dict:
    return {
        direction: {s: v[HEADLINE_METRIC]["mean"] for s, v in block["systems"].items()}
        for direction, block in group["directions"].items()
    }


def _merge_group(out_dir: Path, key: str, group: dict) -> None:
    path = out_dir / layout.METRICS_FILE
    metrics = fileio.read_json(path) if path.exists() else dict(EMPTY_METRICS)
    metrics["groups"] = {**metrics["groups"], key: group}
    fileio.write_json_atomic(path, metrics)


def run(args: argparse.Namespace, ctx: Context) -> int:
    label_map = _load_labels(args)
    group = _build_group(args, label_map)
    key = f"{args.labels}@{args.queries}"
    _merge_group(args.out_dir, key, group)
    print(json.dumps({"group": key, HEADLINE_METRIC: _headline(group)}, sort_keys=True))
    return EXIT_OK
