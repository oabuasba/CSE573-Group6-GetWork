"""Agreement between label sources, the fallback rule, human-set halves and tau blocks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pyarrow as pa

from getwork.evaluation import pairs, stats
from getwork.retrieval.inputs import DIRECTIONS, SYSTEM_IDS

THRESHOLD: float = 0.6
CATEGORIES: tuple[int, ...] = (0, 1, 2)
GAP_KEYS: tuple[str, ...] = ("B2-B1+", "Full-B2", "Full-B1+")
SIDES: tuple[str, ...] = ("resumes", "postings")


def agreement_stats(a: Mapping[str, int], b: Mapping[str, int]) -> dict:
    shared = sorted(set(a) & set(b))
    xs, ys = [a[i] for i in shared], [b[i] for i in shared]
    confusion = [[0] * len(CATEGORIES) for _ in CATEGORIES]
    for x, y in zip(xs, ys):
        confusion[CATEGORIES.index(x)][CATEGORIES.index(y)] += 1
    if not shared:
        return {"n": 0, "pct_agree": None, "kappa": None, "kappa_quadratic": None,
                "confusion": confusion}  # fmt: skip
    return {
        "n": len(shared),
        "pct_agree": sum(x == y for x, y in zip(xs, ys)) / len(shared),
        "kappa": stats.cohen_kappa(xs, ys),
        "kappa_quadratic": stats.cohen_kappa(xs, ys, weights="quadratic"),
        "confusion": confusion,
    }


def fallback_decision(
    author_kappa: float | None, human_judge_kappa: float | None, threshold: float = THRESHOLD
) -> str:
    if author_kappa is None or human_judge_kappa is None:
        return "undetermined"
    if human_judge_kappa >= threshold:
        return "judges_reliable"
    return "revise_once" if author_kappa >= threshold else "subjective"


def split_halves(sets: Mapping[str, object]) -> dict[str, dict[str, list[str]]]:
    human = sets["stages"]["human"]  # type: ignore[index]
    ids = {side: sorted(human[side]) for side in SIDES}
    cut = {side: math.ceil(len(ids[side]) / 2) for side in SIDES}
    return {
        "first": {side: ids[side][: cut[side]] for side in SIDES},
        "second": {side: ids[side][cut[side] :] for side in SIDES},
    }


def half_pair_ids(
    rankings_test: pa.Table, halves: Mapping[str, Mapping[str, list[str]]],
    pair_ids: Mapping[tuple[str, str], str],
) -> dict[str, set[str]]:
    side_of = {"resume_to_posting": "resumes", "posting_to_resume": "postings"}
    member = {
        half: {side: set(halves[half][side]) for side in SIDES} for half in ("first", "second")
    }
    found: dict[str, set[str]] = {"first": set(), "second": set()}
    columns = ("direction", "query_id", "doc_id")
    for direction, query, doc in zip(*(rankings_test.column(c).to_pylist() for c in columns)):
        side = side_of[direction]
        pair_id = pair_ids[pairs.pair_key(direction, query, doc)]
        for half in found:
            if query in member[half][side]:
                found[half].add(pair_id)
    return {"first": found["first"], "second": found["second"] - found["first"]}


def _system_means(group: Mapping, direction: str, metric: str) -> list[float]:
    systems = group["directions"][direction]["systems"]
    return [systems[s][metric]["mean"] for s in SYSTEM_IDS]


def _direction_taus(
    groups: Mapping, authors: Sequence[str], direction: str, metric: str
) -> dict[str, float | None]:
    means = {"judge": _system_means(groups["judge@human"], direction, metric)}
    for author in authors:
        means[author] = _system_means(groups[f"{author}@human"], direction, metric)
    taus = {f"judge-vs-{a}": stats.kendall_tau(means["judge"], means[a]) for a in authors}
    if len(authors) == 2:
        first, second = authors
        taus[f"{first}-vs-{second}"] = stats.kendall_tau(means[first], means[second])
    return taus


def _gap_diffs(group: Mapping, direction: str, metric: str) -> dict[str, float]:
    gaps = group["directions"][direction]["gaps"]
    return {name: gaps[name][metric]["diff"] for name in GAP_KEYS}


def tau_blocks(metrics: Mapping, authors: Sequence[str], metric: str = "nDCG@10") -> dict | None:
    groups = metrics.get("groups", {})
    keys = ["judge@human", *(f"{a}@human" for a in authors)]
    if any(key not in groups for key in keys):
        return None
    return {
        metric: {d: _direction_taus(groups, authors, d, metric) for d in DIRECTIONS},
        "gaps": {k: {d: _gap_diffs(groups[k], d, metric) for d in DIRECTIONS} for k in keys},
    }
