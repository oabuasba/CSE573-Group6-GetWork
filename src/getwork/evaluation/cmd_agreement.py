"""evaluate agreement: judge-vs-judge and judge-vs-author agreement into agreement.json."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import pyarrow.parquet as pq

from getwork import fileio
from getwork.evaluation import agreement, defaults, human, judge_run, judges, labels, layout
from getwork.evaluation import args as eval_args
from getwork.evaluation import pairs
from getwork.evaluation.args import EXIT_OK, Context

COMMANDS = ("agreement",)
SCOPES = ("judges", "human")
HALVES = ("all", "first", "second")
JUDGE_STAGES = ("dev", "test")
FINAL = "final"
MAX_AUTHORS = 2


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("agreement", help="agreement between judges and between authors")
    eval_args.add_common_args(parser)
    parser.add_argument("--scope", required=True, choices=SCOPES)
    parser.add_argument("--stage", choices=JUDGE_STAGES, default=None)
    parser.add_argument("--half", choices=HALVES, default="all")
    parser.add_argument("--rubric-file", type=Path, default=None)


def _rubric_hash(args: argparse.Namespace) -> str:
    frozen = args.out_dir / layout.RUBRIC_FROZEN_FILE
    if args.rubric_file is None and frozen.exists():
        return fileio.read_json(frozen)["rubric_hash"]
    return judges.load_rubric(args.rubric_file).hash


def _judge_maps(out_dir: Path, rubric_hash: str) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for judge_id in defaults.DEFAULT_JUDGE_IDS:
        path = layout.judge_path(out_dir, judge_id)
        if not path.exists():
            raise FileNotFoundError(f"judge file not found: {path} (run evaluate judge)")
        rows = judge_run.read_judge_rows(path, rubric_hash)
        maps[judge_id] = {
            p: row["effective_label"]
            for p, row in rows.items()
            if row["status"] == labels.OK_STATUS
        }
    return maps


def _restrict(label_map: Mapping[str, int], pair_ids: set[str]) -> dict[str, int]:
    return {p: v for p, v in label_map.items() if p in pair_ids}


def _run_judges(args: argparse.Namespace) -> dict:
    if args.stage is None:
        raise ValueError("--scope judges needs --stage dev or test")
    stage_pairs = pq.read_table(layout.pairs_path(args.out_dir, args.stage))
    pair_ids = set(pairs.pair_id_map(stage_pairs).values())
    maps = _judge_maps(args.out_dir, _rubric_hash(args))
    first, second = (_restrict(maps[j], pair_ids) for j in defaults.DEFAULT_JUDGE_IDS)
    block = agreement.agreement_stats(first, second)
    block["threshold"] = agreement.THRESHOLD
    block["meets_threshold"] = block["kappa"] is not None and block["kappa"] >= agreement.THRESHOLD
    return {"judges": {args.stage: block}}


def _human_pair_ids(out_dir: Path) -> set[str]:
    path = layout.blind_path(out_dir, human.HUMAN_STAGE)
    if not path.exists():
        raise FileNotFoundError(f"human blind file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["pair_id"] for line in f if line.strip()}


def _selected_pairs(args: argparse.Namespace) -> set[str]:
    out_dir = args.out_dir
    if args.half == "all":
        return _human_pair_ids(out_dir)
    pair_ids = pairs.pair_id_map(pq.read_table(layout.pairs_path(out_dir, "test")))
    halves = agreement.split_halves(fileio.read_json(out_dir / layout.SETS_FILE))
    rankings = pq.read_table(layout.rankings_path(out_dir, "test"))
    return agreement.half_pair_ids(rankings, halves, pair_ids)[args.half]


def _author_labels(out_dir: Path, pair_ids: set[str]) -> dict[str, dict[str, int]]:
    authors = human.list_authors(out_dir)
    if not authors:
        raise ValueError(f"no author label files in {out_dir / human.HUMAN_DIR}")
    if len(authors) > MAX_AUTHORS:
        raise ValueError(f"at most {MAX_AUTHORS} authors are supported, found {authors}")
    result = {}
    for author in authors:
        author_map = human.read_human_labels(out_dir, author)
        missing = len(pair_ids - set(author_map))
        if missing:
            raise ValueError(f"author {author} has not labeled {missing} of {len(pair_ids)} pairs")
        result[author] = _restrict(author_map, pair_ids)
    return result


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _tau(args: argparse.Namespace, authors: list[str]) -> dict | None:
    path = args.out_dir / layout.METRICS_FILE
    if args.half != "all" or not path.exists():
        return None
    return agreement.tau_blocks(fileio.read_json(path), authors)


def _run_human(args: argparse.Namespace) -> dict:
    out_dir = args.out_dir
    pair_ids = _selected_pairs(args)
    by_author = _author_labels(out_dir, pair_ids)
    authors = list(by_author)
    sources = _judge_maps(out_dir, _rubric_hash(args))
    sources[FINAL] = labels.read_labels(out_dir / layout.LABELS_JUDGE_FILE)
    sources = {name: _restrict(m, pair_ids) for name, m in sources.items()}
    vs_author = {
        name: {a: agreement.agreement_stats(m, by_author[a]) for a in authors}
        for name, m in sources.items()
    }
    pair = (
        agreement.agreement_stats(*(by_author[a] for a in authors))
        if len(authors) == MAX_AUTHORS else None
    )
    author_kappa = pair["kappa"] if pair else None
    judge_kappas = [
        vs_author[j][a]["kappa"] for j in defaults.DEFAULT_JUDGE_IDS for a in authors
    ]
    human_judge_kappa = _mean(judge_kappas)
    return {"human": {
        "half": args.half, "n_pairs": len(pair_ids), "authors": authors,
        "author_vs_author": pair, "judge_vs_author": vs_author,
        "author_kappa": author_kappa, "human_judge_kappa": human_judge_kappa,
        "fallback": {
            "threshold": agreement.THRESHOLD,
            "decision": agreement.fallback_decision(author_kappa, human_judge_kappa),
        },
        "tau": _tau(args, authors),
    }}


def _merge(out_dir: Path, block: dict) -> dict:
    path = out_dir / layout.AGREEMENT_FILE
    current = fileio.read_json(path) if path.exists() else {"schema": 1}
    ((key, value),) = block.items()
    if key == "judges":
        value = {**current.get("judges", {}), **value}
    merged = {**current, key: value}
    fileio.write_json_atomic(path, merged)
    return merged


def run(args: argparse.Namespace, ctx: Context) -> int:
    block = _run_judges(args) if args.scope == "judges" else _run_human(args)
    merged = _merge(args.out_dir, block)
    key = args.scope
    print(json.dumps({key: merged[key]}, sort_keys=True))
    return EXIT_OK
