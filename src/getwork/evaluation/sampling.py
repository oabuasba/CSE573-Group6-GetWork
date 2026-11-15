"""Seeded, order-independent draws of the pool and the dev, test and human query sets."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SampleParams:
    seed: int = 0
    pool_cap: int = 5000
    dev_resumes: int = 10
    dev_postings: int = 10
    test_resumes: int = 100
    test_postings: int = 100
    human_resumes: int = 10
    human_postings: int = 10


def rng_for(seed: int, purpose: str) -> random.Random:
    return random.Random(f"{seed}/{purpose}")


def sample_pool(posting_ids: Sequence[str], cap: int, seed: int) -> list[str]:
    ordered = sorted(posting_ids)
    if len(ordered) <= cap:
        return ordered
    return sorted(rng_for(seed, "pool").sample(ordered, cap))


def sample_ids(ids: Sequence[str], n: int, seed: int, purpose: str) -> list[str]:
    if n > len(ids):
        raise ValueError(f"cannot draw {n} ids for {purpose} from only {len(ids)}")
    return sorted(rng_for(seed, purpose).sample(sorted(ids), n))


def _quotas(sizes: Mapping[str, int], n: int) -> dict[str, int]:
    """Equal allocation over sorted labels, capped by group size, shortfall round-robin."""
    names = sorted(sizes)
    base, extra = divmod(n, len(names))
    quotas = {
        name: min(base + (1 if i < extra else 0), sizes[name]) for i, name in enumerate(names)
    }
    remaining = n - sum(quotas.values())
    while remaining > 0:
        for name in names:
            if remaining > 0 and quotas[name] < sizes[name]:
                quotas[name] += 1
                remaining -= 1
    return quotas


def stratified_sample(
    labels: Mapping[str, str | None], n: int, seed: int, purpose: str
) -> list[str]:
    groups: dict[str, list[str]] = {}
    for id_ in sorted(labels):
        if labels[id_]:
            groups.setdefault(labels[id_], []).append(id_)
    n_labeled = sum(len(ids) for ids in groups.values())
    if n > n_labeled:
        raise ValueError(f"cannot draw {n} labeled resumes for {purpose}: only {n_labeled}")
    if n == 0:
        return []
    quotas = _quotas({name: len(ids) for name, ids in groups.items()}, n)
    rng = rng_for(seed, purpose)
    chosen = [id_ for name in sorted(groups) for id_ in rng.sample(groups[name], quotas[name])]
    return sorted(chosen)


def _check_enough(*, what: str, available: int, dev: int, test: int) -> None:
    if available < dev + test:
        raise ValueError(
            f"not enough {what}: {available} available, dev {dev} + test {test} = {dev + test}"
        )


def build_sets(
    *, resume_labels: Mapping[str, str | None], posting_ids: Sequence[str], params: SampleParams
) -> dict:
    p = params
    labeled = {id_: label for id_, label in resume_labels.items() if label}
    pool = sample_pool(posting_ids, p.pool_cap, p.seed)
    _check_enough(what="labeled resumes", available=len(labeled), dev=p.dev_resumes,
                  test=p.test_resumes)
    _check_enough(what="pool postings", available=len(pool), dev=p.dev_postings,
                  test=p.test_postings)
    dev_resumes = stratified_sample(labeled, p.dev_resumes, p.seed, "dev-resumes")
    dev_postings = sample_ids(pool, p.dev_postings, p.seed, "dev-postings")
    rest_resumes = {i: lab for i, lab in labeled.items() if i not in set(dev_resumes)}
    test_resumes = stratified_sample(rest_resumes, p.test_resumes, p.seed, "test-resumes")
    test_postings = sample_ids(
        [i for i in pool if i not in set(dev_postings)], p.test_postings, p.seed, "test-postings"
    )
    human_resumes = stratified_sample(
        {i: labeled[i] for i in test_resumes}, p.human_resumes, p.seed, "human-resumes"
    )
    human_postings = sample_ids(test_postings, p.human_postings, p.seed, "human-postings")
    return {
        "schema": SCHEMA_VERSION,
        "seed": p.seed,
        "pool_cap": p.pool_cap,
        "n_resumes_total": len(resume_labels),
        "n_resumes_labeled": len(labeled),
        "n_resumes_unlabeled": len(resume_labels) - len(labeled),
        "n_postings_total": len(posting_ids),
        "n_pool": len(pool),
        "pool": pool,
        "stages": {
            "dev": {"resumes": dev_resumes, "postings": dev_postings},
            "test": {"resumes": test_resumes, "postings": test_postings},
            "human": {"resumes": human_resumes, "postings": human_postings},
        },
    }
