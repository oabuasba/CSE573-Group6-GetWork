"""Read-only peek at dev-stage ranking metrics from the judge labels found so far.

Usage: uv run python scripts/python/other/peek_dev_metrics.py [--dir DIR ...] [--rubric-hash HASH]

Labels are the rows of one rubric hash (default: the packaged judge_v1) with status ok, taken
from every --dir (default eval/preliminary_results and eval/other/v7_heldout) and combined as the
harness builds labels_judge.parquet. Prints markdown tables only; writes nothing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from getwork import fileio
from getwork.evaluation import defaults, judge_run, judges, labels, layout, metrics, pairs, scoring
from getwork.retrieval.inputs import DIRECTIONS, SYSTEM_IDS

DEFAULT_DIRS = (Path("eval/preliminary_results"), Path("eval/other/v7_heldout"))
DEV_STAGE = "dev"
QUERY_SIDE = {"resume_to_posting": "resumes", "posting_to_resume": "postings"}
SHOWN = ("P@10", "nDCG@10", "MRR")
GAP_NAMES = ("B2-B1+", "Full-B2", "Full-B1+")
CAVEAT = (
    "Caveats: dev stage only (20 queries); labels are incomplete (see coverage); "
    "rubric judge_v1; B1+ untuned (default weights). Numbers only, no significance claim."
)


class InputMismatchError(ValueError):
    """The folders do not describe the same dev pairs."""


def _dev_pairs(out_dir: Path) -> list[str]:
    table = pq.read_table(layout.pairs_path(out_dir, DEV_STAGE), columns=["pair_id"])
    return sorted(table.column("pair_id").to_pylist())


def check_same_dev_pairs(out_dirs: Sequence[Path]) -> None:
    first = _dev_pairs(out_dirs[0])
    for other in out_dirs[1:]:
        if _dev_pairs(other) != first:
            raise InputMismatchError(
                f"the dev pair ids of {out_dirs[0]} and {other} differ; the folders were not "
                f"made from the same seed and artifacts, so their labels cannot be pooled"
            )


def collect_labels(out_dirs: Sequence[Path], rubric_hash: str) -> dict[str, int]:
    """Combined label per pair judged ok by both judges under rubric_hash, over all folders."""
    rows: dict[str, dict[str, dict]] = {judge_id: {} for judge_id in defaults.DEFAULT_JUDGE_IDS}
    for out_dir in out_dirs:
        for judge_id in rows:
            path = layout.judge_path(out_dir, judge_id)
            found = judge_run.read_judge_rows(path, rubric_hash)
            rows[judge_id].update({p: r for p, r in found.items() if r["status"] == "ok"})
    table = labels.build_labels_table(*rows.values())
    return dict(zip(table.column("pair_id").to_pylist(), table.column("label").to_pylist()))


def _load_dev(out_dir: Path) -> tuple:
    """(rankings, pair id map, dev query ids per direction) of the dev stage."""
    rankings = pq.read_table(layout.rankings_path(out_dir, DEV_STAGE))
    pair_ids = pairs.pair_id_map(pq.read_table(layout.pairs_path(out_dir, DEV_STAGE)))
    sets = fileio.read_json(out_dir / layout.SETS_FILE)["stages"][DEV_STAGE]
    return rankings, pair_ids, {d: sorted(sets[QUERY_SIDE[d]]) for d in DIRECTIONS}


def _top(rankings: pa.Table, pair_ids: Mapping, queries: Mapping) -> dict:
    return {
        d: scoring.top_pair_ids(rankings, direction=d, query_ids=queries[d], pair_ids=pair_ids)
        for d in DIRECTIONS
    }


def coverage(top: Mapping, queries: Mapping, label_map: Mapping[str, int]) -> dict:
    result: dict = {}
    for direction in DIRECTIONS:
        per_system = {}
        for system in SYSTEM_IDS:
            per_query = [
                sum(p in label_map for p in top[direction][system][q]) / metrics.TOP_K
                for q in queries[direction]
            ]
            per_system[system] = {
                "slots": sum(per_query) / len(per_query), "min": min(per_query),
                "median": statistics.median(per_query),
            }
        pooled = {p for system in SYSTEM_IDS for ps in top[direction][system].values()
                  for p in ps}
        result[direction] = {
            "systems": per_system, "pooled": len(pooled),
            "labeled": sum(p in label_map for p in pooled),
        }
    return result


def lower_bound(
    rankings: pa.Table, pair_ids: Mapping, queries: Mapping, label_map: Mapping[str, int],
    **kw: int,
) -> dict:
    """Unlabeled slots count as gain 0: the harness scorer with allow_missing."""
    group = {}
    for direction in DIRECTIONS:
        scores = scoring.score_queries(
            rankings=rankings, direction=direction, query_ids=queries[direction],
            pair_ids=pair_ids, labels=label_map, allow_missing=True,
        )
        group[direction] = scoring.aggregate_group(scores, **kw)
    return group


def _labeled_query(
    slots: Sequence[int | None], ideal: Sequence[int]
) -> dict[str, float]:
    """Metrics over the labeled slots only, in rank order; P@k divides by the slots counted."""
    kept = [x for x in slots if x is not None]
    scored = metrics.query_metrics(kept, ideal)
    for k in (5, 10):
        head = kept[:k]
        scored[f"P@{k}"] = sum(x == metrics.RELEVANT for x in head) / len(head)
    return scored


def labeled_only(top: Mapping, queries: Mapping, label_map: Mapping[str, int], **kw: int) -> dict:
    """Per direction: aggregate over queries with a labeled slot in every system, and the skips."""
    result = {}
    for direction in DIRECTIONS:
        usable = [
            q for q in queries[direction]
            if all(any(p in label_map for p in top[direction][s][q]) for s in SYSTEM_IDS)
        ]
        if not usable:
            result[direction] = {"n_used": 0, "n_skipped": len(queries[direction]), "group": None}
            continue
        scores = {s: {} for s in SYSTEM_IDS}
        for q in usable:
            pooled = {p for s in SYSTEM_IDS for p in top[direction][s][q] if p in label_map}
            ideal = metrics.ideal_labels(label_map[p] for p in pooled)
            for s in SYSTEM_IDS:
                slots = [label_map.get(p) for p in top[direction][s][q]]
                scores[s][q] = _labeled_query(slots, ideal)
        group = scoring.aggregate_group(
            scoring.QueryScores(scores=scores, n_missing=0, n_no_relevant=0), **kw
        )
        result[direction] = {
            "n_used": len(usable), "n_skipped": len(queries[direction]) - len(usable),
            "group": group,
        }
    return result


def _ci(entry: Mapping) -> str:
    low, high = entry["ci"]
    return f"{entry['mean']:.3f} [{low:.3f}, {high:.3f}]"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return [*lines, *("| " + " | ".join(row) + " |" for row in rows), ""]


def render_coverage(cov: Mapping, n_dev_pairs: int, n_labeled_total: int) -> list[str]:
    rows = []
    for direction in DIRECTIONS:
        for system in SYSTEM_IDS:
            c = cov[direction]["systems"][system]
            rows.append([direction, system, f"{c['slots']:.1%}", f"{c['min']:.0%}",
                         f"{c['median']:.0%}"])
    lines = _table(["Direction", "System", "Top-10 slots labeled", "Query min", "Query median"],
                   rows)
    for direction in DIRECTIONS:
        c = cov[direction]
        lines.append(f"- {direction}: {c['labeled']} of {c['pooled']} distinct pooled pairs "
                     f"labeled, {c['pooled'] - c['labeled']} unlabeled")
    lines.append(f"- all dev pairs: {n_labeled_total} labeled of {n_dev_pairs} "
                 f"({n_labeled_total / n_dev_pairs:.1%})")
    return [*lines, ""]


def render_group(groups: Mapping, *, skipped: Mapping | None = None) -> list[str]:
    lines: list[str] = []
    for direction in DIRECTIONS:
        group = groups[direction]
        if group is None:
            lines += [f"### {direction}: no usable queries", ""]
            continue
        note = f", {skipped[direction]} queries skipped" if skipped else ""
        lines += [f"### {direction} ({group['n_queries']} queries{note})", ""]
        rows = [[s, *(_ci(group["systems"][s][m]) for m in SHOWN)] for s in SYSTEM_IDS]
        lines += _table(["System", *SHOWN], rows)
    return lines


def render_gaps(groups: Mapping) -> list[str]:
    rows = []
    for direction in DIRECTIONS:
        if groups[direction] is None:
            continue
        for name in GAP_NAMES:
            gap = groups[direction]["gaps"][name]["nDCG@10"]
            low, high = gap["ci"]
            rows.append([direction, name, f"{gap['diff']:+.3f}", f"[{low:.3f}, {high:.3f}]",
                         f"{gap['p']:.3f}"])
    return _table(["Direction", "Gap", "nDCG@10 diff", "95% CI (bootstrap)",
                   "randomization p"], rows)


def build_report(
    out_dirs: Sequence[Path], rubric_hash: str, *, n_boot: int, n_perm: int
) -> str:
    check_same_dev_pairs(out_dirs)
    label_map = collect_labels(out_dirs, rubric_hash)
    rankings, pair_ids, queries = _load_dev(out_dirs[0])
    top = _top(rankings, pair_ids, queries)
    seed = layout.read_manifest(out_dirs[0])["seed"]
    kw = {"n_boot": n_boot, "n_perm": n_perm, "seed": seed}
    cov = coverage(top, queries, label_map)
    lower = lower_bound(rankings, pair_ids, queries, label_map, **kw)
    only = labeled_only(top, queries, label_map, **kw)
    lines = [
        "# Dev metrics peek", "", CAVEAT, f"Rubric hash {rubric_hash}; folders "
        f"{', '.join(str(d) for d in out_dirs)}; bootstrap {n_boot}, seed {seed}.", "",
        "Queries: " + ", ".join(f"{d} {len(queries[d])}" for d in DIRECTIONS), "",
        "## Coverage", "",
        *render_coverage(cov, len(_dev_pairs(out_dirs[0])), len(label_map)),
        "## Lower bound (unlabeled = not relevant)", "", *render_group(lower),
        "## Labeled slots only", "",
        *render_group({d: only[d]["group"] for d in DIRECTIONS},
                      skipped={d: only[d]["n_skipped"] for d in DIRECTIONS}),
        "## nDCG@10 gaps, lower bound", "", *render_gaps(lower),
        "## nDCG@10 gaps, labeled slots only", "",
        *render_gaps({d: only[d]["group"] for d in DIRECTIONS}),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, action="append", dest="dirs", default=None)
    parser.add_argument("--rubric-hash", default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=10000)
    args = parser.parse_args(argv)
    rubric_hash = args.rubric_hash or judges.load_rubric().hash
    try:
        text = build_report(args.dirs or list(DEFAULT_DIRS), rubric_hash,
                            n_boot=args.n_boot, n_perm=args.n_perm)
    except (InputMismatchError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
