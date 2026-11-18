"""evaluate compare-rubrics: judge agreement per rubric on the pairs judged under all of them."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow.parquet as pq

from getwork.evaluation import agreement, defaults, judge_run, layout, stats
from getwork.evaluation.args import EXIT_OK, Context

COMMANDS = ("compare-rubrics",)
FORMATS = ("json", "table")
JUDGE_A, JUDGE_B = defaults.DEFAULT_JUDGE_IDS
BINARY_CATEGORIES = (0, 1)
DISAGREEMENT_KEYS = {(1, 2): "a1_b2", (2, 1): "a2_b1"}
UNKNOWN_DIRECTION = "unknown"


def add_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "compare-rubrics", help="compare judge agreement across rubrics on identical pairs"
    )
    parser.add_argument("--out-dir", type=Path, action="append", required=True, dest="out_dirs",
                        help="an eval folder; repeat to pool judge rows of one run split in parts")
    parser.add_argument("--rubric-hash", action="append", default=None,
                        help="hash to compare (repeatable); default: every hash found")
    parser.add_argument("--format", choices=FORMATS, default="json",
                        help="json (default) or one summary line per rubric")


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"judge file not found: {path} (run evaluate judge)")
    return judge_run.read_all_rows(path)


def _rows_by_judge(out_dirs: Sequence[Path]) -> dict[str, list[dict]]:
    return {
        judge_id: [row for d in out_dirs for row in _read_rows(layout.judge_path(d, judge_id))]
        for judge_id in defaults.DEFAULT_JUDGE_IDS
    }


def _hash_names(rows: Mapping[str, list[dict]]) -> dict[str, str]:
    """rubric_hash -> rubric_version, over every row of both judges."""
    return {r["rubric_hash"]: r.get("rubric_version", "?")
            for judge_rows in rows.values() for r in judge_rows if r.get("rubric_hash")}


def _ok_labels(rows: list[dict], rubric_hash: str) -> dict[str, int]:
    best = judge_run.best_rows(rows, rubric_hash)
    return {p: r["effective_label"] for p, r in best.items() if r["status"] == judge_run.OK_STATUS}


def _pair_directions(out_dirs: Sequence[Path]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for out_dir in out_dirs:
        for stage in layout.STAGES:
            path = layout.pairs_path(out_dir, stage)
            if path.exists():
                table = pq.read_table(path, columns=["pair_id", "directions"])
                found.update(zip(table.column("pair_id").to_pylist(),
                                 table.column("directions").to_pylist()))
    return found


def _binary_kappa(a: list[int], b: list[int], keep: str) -> float | None:
    test = (lambda x: x == 2) if keep == "two" else (lambda x: x >= 1)
    return stats.cohen_kappa([int(test(x)) for x in a], [int(test(x)) for x in b],
                             categories=BINARY_CATEGORIES)


def _disagreements(
    shared: Sequence[str], a: Mapping[str, int], b: Mapping[str, int],
    directions: Mapping[str, list[str]],
) -> dict:
    """Counts of judge-a=1/judge-b=2 (a1_b2) and the reverse, in total and per direction."""
    counts: dict = {"total": {"a1_b2": 0, "a2_b1": 0, "other": 0}, "by_direction": {}}
    for pair in shared:
        if a[pair] == b[pair]:
            continue
        key = DISAGREEMENT_KEYS.get((a[pair], b[pair]), "other")
        counts["total"][key] += 1
        for direction in directions.get(pair) or [UNKNOWN_DIRECTION]:
            bucket = counts["by_direction"].setdefault(
                direction, {"a1_b2": 0, "a2_b1": 0, "other": 0})
            bucket[key] += 1
    return counts


def _distribution(labels: Sequence[int]) -> dict[str, int]:
    return {str(c): sum(1 for x in labels if x == c) for c in agreement.CATEGORIES}


def _summarize(
    shared: Sequence[str], a: Mapping[str, int], b: Mapping[str, int],
    directions: Mapping[str, list[str]],
) -> dict:
    xs, ys = [a[p] for p in shared], [b[p] for p in shared]
    base = agreement.agreement_stats({p: a[p] for p in shared}, {p: b[p] for p in shared})
    return {
        "n": base["n"], "pct_agree": base["pct_agree"], "kappa": base["kappa"],
        "kappa_quadratic": base["kappa_quadratic"],
        "kappa_label2_vs_not": _binary_kappa(xs, ys, "two") if shared else None,
        "kappa_ge1_vs_0": _binary_kappa(xs, ys, "one") if shared else None,
        "disagreements": _disagreements(shared, a, b, directions),
        "distribution": {JUDGE_A: _distribution(xs), JUDGE_B: _distribution(ys)},
        "confusion_rows_a_cols_b": base["confusion"],
    }


def compare(out_dirs: Sequence[Path], rubric_hashes: Sequence[str] | None) -> dict:
    rows = _rows_by_judge(out_dirs)
    names = _hash_names(rows)
    chosen = list(rubric_hashes) if rubric_hashes else sorted(names)
    unknown = [h for h in chosen if h not in names]
    if unknown:
        raise ValueError(f"rubric hash {unknown[0]} has no judge rows; found {sorted(names)}")
    labels = {h: {j: _ok_labels(rows[j], h) for j in rows} for h in chosen}
    both = {h: set(labels[h][JUDGE_A]) & set(labels[h][JUDGE_B]) for h in chosen}
    overlap = sorted(set.intersection(*both.values())) if chosen else []
    directions = _pair_directions(out_dirs)
    return {
        "out_dirs": [str(d) for d in out_dirs],
        "overlap_pairs": len(overlap),
        "pairs_judged_by_both_judges": {h: len(both[h]) for h in chosen},
        "rubrics": {
            h: {"version": names[h],
                **_summarize(overlap, labels[h][JUDGE_A], labels[h][JUDGE_B], directions)}
            for h in chosen
        },
    }


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _distribution_text(dist: Mapping[str, int]) -> str:
    return "/".join(str(dist[str(c)]) for c in agreement.CATEGORIES)


def _directions_text(disagreements: Mapping) -> str:
    by_direction = disagreements["by_direction"]
    parts = [f"{d}={c['a1_b2']}" for d, c in sorted(by_direction.items())]
    return f"{disagreements['total']['a1_b2']} ({', '.join(parts) or 'none'})"


def format_table(result: Mapping) -> str:
    """One line per rubric: n, agreement, kappas, a1_b2 counts and the label distributions."""
    lines = [f"overlap pairs: {result['overlap_pairs']}"]
    for rubric_hash, block in result["rubrics"].items():
        pct = block["pct_agree"]
        lines.append(
            f"{block['version']} {rubric_hash}: n={block['n']} "
            f"agree={'n/a' if pct is None else f'{pct:.0%}'} kappa={_fmt(block['kappa'])} "
            f"wkappa={_fmt(block['kappa_quadratic'])} k2={_fmt(block['kappa_label2_vs_not'])} "
            f"a1_b2={_directions_text(block['disagreements'])} "
            f"a(0/1/2)={_distribution_text(block['distribution'][JUDGE_A])} "
            f"b(0/1/2)={_distribution_text(block['distribution'][JUDGE_B])}"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace, ctx: Context) -> int:
    result = compare(args.out_dirs, args.rubric_hash)
    print(format_table(result) if args.format == "table" else json.dumps(result, indent=2))
    return EXIT_OK
