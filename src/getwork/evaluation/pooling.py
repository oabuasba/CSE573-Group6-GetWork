"""Rank the sampled queries inside the capped posting pool with the retrieval code (ADR-37)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pyarrow as pa

from getwork.embed.store import VectorSet
from getwork.retrieval import inputs as retrieval_inputs
from getwork.retrieval import run
from getwork.retrieval.b1plus import DEFAULT_WEIGHTS, B1PlusWeights
from getwork.retrieval.inputs import SYSTEM_IDS, RetrievalInputs

K: int = 10
RANKINGS_SCHEMA: pa.Schema = pa.schema([
    ("stage", pa.string()),
    *run.TOPK_PARQUET_SCHEMA,
])


def _check_known(ids: Sequence[str], table: dict[str, dict], what: str) -> None:
    for id_ in ids:
        if id_ not in table:
            raise ValueError(f"unknown {what} id {id_!r}")


def _restrict_vectors(vset: VectorSet, keep: set[str]) -> VectorSet:
    rows = [i for i, id_ in enumerate(vset.ids) if id_ in keep]
    return VectorSet(
        ids=[vset.ids[i] for i in rows], vectors=vset.vectors[np.array(rows, dtype=int)],
        meta=vset.meta,
    )


def restricted_inputs(
    data: RetrievalInputs, *, resume_ids: Sequence[str] | None, posting_ids: Sequence[str] | None
) -> RetrievalInputs:
    keep = {"resumes": resume_ids, "postings": posting_ids}
    tables = {"resumes": data.resumes, "postings": data.postings}
    for source, ids in keep.items():
        if ids is not None:
            _check_known(ids, tables[source], source[:-1])
    kept = {source: None if ids is None else set(ids) for source, ids in keep.items()}
    new_tables = {
        source: table if kept[source] is None
        else {id_: doc for id_, doc in table.items() if id_ in kept[source]}
        for source, table in tables.items()
    }
    new_vectors = {
        (source, system): vset if kept[source] is None else _restrict_vectors(vset, kept[source])
        for (source, system), vset in data.vectors.items()
    }
    return replace(
        data, resumes=new_tables["resumes"], postings=new_tables["postings"],
        vectors=new_vectors,
    )


def _direction_inputs(
    data: RetrievalInputs, *, direction: str, resume_queries: Sequence[str],
    posting_queries: Sequence[str], pool_ids: Sequence[str],
) -> RetrievalInputs:
    if direction == "resume_to_posting":
        return restricted_inputs(data, resume_ids=resume_queries, posting_ids=pool_ids)
    return restricted_inputs(data, resume_ids=None, posting_ids=posting_queries)


def stage_rankings(
    data: RetrievalInputs, *, stage: str, resume_queries: Sequence[str],
    posting_queries: Sequence[str], pool_ids: Sequence[str],
    systems: Sequence[str] = SYSTEM_IDS, k: int = K, weights: B1PlusWeights = DEFAULT_WEIGHTS,
) -> pa.Table:
    _check_known(resume_queries, data.resumes, "resume query")
    _check_known(posting_queries, data.postings, "posting query")
    _check_known(pool_ids, data.postings, "pool posting")
    parts = []
    for system in systems:
        for direction in retrieval_inputs.DIRECTIONS:
            restricted = _direction_inputs(
                data, direction=direction, resume_queries=resume_queries,
                posting_queries=posting_queries, pool_ids=pool_ids,
            )
            plan = run.make_plan(
                inputs=restricted, system=system, direction=direction, weights=weights
            )
            parts.append(run.ranking_table(run.rank(plan=plan, k=k)))
    table = pa.concat_tables(parts)
    stage_column = pa.array([stage] * table.num_rows, type=pa.string())
    return table.add_column(0, "stage", stage_column).cast(RANKINGS_SCHEMA)
