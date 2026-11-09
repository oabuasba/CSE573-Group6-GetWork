"""Read stored rankings and compute the ranking of a single query."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from getwork.retrieval import run, topk
from getwork.retrieval import inputs as retrieval_inputs
from getwork.retrieval.b1plus import DEFAULT_WEIGHTS, B1PlusWeights

RUN_HINT = "run retrieve run"


def read_topk(root: Path, system: str, direction: str) -> dict[str, list[tuple[str, float]]]:
    """query_id -> [(doc_id, score)] in rank order, queries in file order."""
    path = run.topk_path(root=root, system=system, direction=direction)
    if not path.exists():
        raise FileNotFoundError(f"ranking file not found: {path} ({RUN_HINT})")
    table = pq.read_table(path).to_pydict()
    matches: dict[str, list[tuple[str, float]]] = {}
    for query_id, doc_id, score in zip(
        table["query_id"], table["doc_id"], table["score"], strict=True
    ):
        matches.setdefault(query_id, []).append((doc_id, score))
    return matches


def query_matches(
    inputs: retrieval_inputs.RetrievalInputs,
    system: str,
    direction: str,
    query_id: str,
    k: int = run.DEFAULT_K,
    weights: B1PlusWeights = DEFAULT_WEIGHTS,
) -> list[tuple[str, float]]:
    """The run's ranking for one query, computed from one block of one row."""
    plan = run.make_plan(inputs=inputs, system=system, direction=direction, weights=weights)
    if query_id not in plan.query_ids:
        raise KeyError(f"{query_id!r} is not a query of {direction}")
    position = plan.query_ids.index(query_id)
    indices, scores = topk.topk_from_blocks(
        block=lambda start, stop: plan.block(position + start, position + stop),
        n_queries=1, n_docs=len(plan.doc_ids), k=k,
    )
    return [(plan.doc_ids[i], float(s)) for i, s in zip(indices[0], scores[0], strict=True)]
