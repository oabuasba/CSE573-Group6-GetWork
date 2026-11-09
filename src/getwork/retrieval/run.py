"""Scoring plans, exact top-k ranking and the retrieval run (all systems, both directions)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from getwork import fileio, paths
from getwork.docembed.documents import COMPARE_KEYS
from getwork.retrieval import b1plus, topk
from getwork.retrieval import inputs as retrieval_inputs
from getwork.retrieval.b1plus import DEFAULT_WEIGHTS, B1PlusWeights

SYSTEM_SLUGS: dict[str, str] = {"B1+": "b1plus", "B2": "b2", "Full": "full"}
DEFAULT_K: int = 10
SYSTEM_CHOICES = list(retrieval_inputs.SYSTEM_IDS)
DIRECTION_CHOICES = list(retrieval_inputs.DIRECTIONS)
SUMMARY_FILE = "retrieval_summary.json"
TOPK_PARQUET_SCHEMA: pa.Schema = pa.schema([
    ("system", pa.string()),
    ("direction", pa.string()),
    ("query_id", pa.string()),
    ("rank", pa.int32()),
    ("doc_id", pa.string()),
    ("score", pa.float32()),
])
_QUERY_SOURCE = {"resume_to_posting": ("resumes", "postings"),
                 "posting_to_resume": ("postings", "resumes")}  # fmt: skip


@dataclass(frozen=True)
class ScoringPlan:
    system: str
    direction: str
    query_ids: list[str]
    doc_ids: list[str]
    block: Callable[[int, int], np.ndarray]


@dataclass(frozen=True)
class Ranking:
    plan: ScoringPlan
    indices: np.ndarray
    scores: np.ndarray


def _embedding_plan(
    *, data: retrieval_inputs.RetrievalInputs, system: str, direction: str
) -> ScoringPlan:
    query_source, doc_source = _QUERY_SOURCE[direction]
    if (query_source, system) not in data.vectors or (doc_source, system) not in data.vectors:
        raise ValueError(f"vectors for system {system} were not loaded")
    queries = data.vectors[(query_source, system)]
    docs = data.vectors[(doc_source, system)]
    return ScoringPlan(
        system=system, direction=direction, query_ids=list(queries.ids),
        doc_ids=list(docs.ids),
        block=topk.matrix_block(queries=queries.vectors, docs=docs.vectors),
    )


def _b1plus_plan(
    *, data: retrieval_inputs.RetrievalInputs, direction: str, weights: B1PlusWeights
) -> ScoringPlan:
    resume_features, posting_features = b1plus.build_features(
        resume_docs=data.resumes, posting_docs=data.postings
    )
    if direction == "resume_to_posting":
        query_features, doc_features = resume_features, posting_features
    else:
        query_features, doc_features = posting_features, resume_features
    return ScoringPlan(
        system="B1+", direction=direction, query_ids=list(query_features.ids),
        doc_ids=list(doc_features.ids),
        block=lambda start, stop: b1plus.b1plus_block(
            q=query_features, d=doc_features, start=start, stop=stop, w=weights
        ),
    )


def make_plan(
    inputs: retrieval_inputs.RetrievalInputs,
    system: str,
    direction: str,
    weights: B1PlusWeights = DEFAULT_WEIGHTS,
) -> ScoringPlan:
    """ScoringPlan for one system and direction; ValueError if either name is unknown."""
    if system not in retrieval_inputs.SYSTEM_IDS:
        raise ValueError(f"unknown system {system!r}; choose from {SYSTEM_CHOICES}")
    if direction not in retrieval_inputs.DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; choose from {DIRECTION_CHOICES}")
    if system == "B1+":
        return _b1plus_plan(data=inputs, direction=direction, weights=weights)
    return _embedding_plan(data=inputs, system=system, direction=direction)


def rank(
    plan: ScoringPlan, k: int = DEFAULT_K, block_size: int = topk.DEFAULT_BLOCK_SIZE
) -> Ranking:
    indices, scores = topk.topk_from_blocks(
        block=plan.block, n_queries=len(plan.query_ids), n_docs=len(plan.doc_ids),
        k=k, block_size=block_size,
    )
    return Ranking(plan=plan, indices=indices, scores=scores)


def topk_path(root: Path, system: str, direction: str) -> Path:
    return paths.retrieval_dir(root) / f"topk_{SYSTEM_SLUGS[system]}_{direction}.parquet"


def ranking_table(ranking: Ranking) -> pa.Table:
    """Long-format rows ordered by query position then rank (1-based)."""
    plan = ranking.plan
    n_queries, k_eff = ranking.indices.shape
    doc_ids = np.array(plan.doc_ids, dtype=object)
    columns = {
        "system": [plan.system] * (n_queries * k_eff),
        "direction": [plan.direction] * (n_queries * k_eff),
        "query_id": np.repeat(np.array(plan.query_ids, dtype=object), k_eff).tolist(),
        "rank": np.tile(np.arange(1, k_eff + 1, dtype=np.int32), n_queries),
        "doc_id": doc_ids[ranking.indices.ravel()].tolist(),
        "score": ranking.scores.ravel(),
    }
    return pa.table(columns, schema=TOPK_PARQUET_SCHEMA)


def _summary(
    *,
    data: retrieval_inputs.RetrievalInputs,
    systems: Sequence[str],
    k: int,
    weights: B1PlusWeights,
    rankings: Sequence[Ranking],
    rows: dict[str, int],
) -> dict:
    embedding_systems = [s for s in systems if s != "B1+"]
    embedding = None
    if embedding_systems:
        embedding = data.vectors[("resumes", embedding_systems[0])].meta["embedding"]
    return {
        "k": k,
        "k_effective": min(r.indices.shape[1] for r in rankings),
        "systems": list(systems),
        "n_resumes": len(data.resumes),
        "n_postings": len(data.postings),
        "weights": weights.as_dict() if "B1+" in systems else None,
        "embedding": embedding,
        "provenance": {key: data.summaries["resumes"].get(key) for key in COMPARE_KEYS},
        "files": rows,
    }


def run_retrieval(
    inputs: retrieval_inputs.RetrievalInputs,
    root: Path,
    *,
    systems: Sequence[str] = retrieval_inputs.SYSTEM_IDS,
    k: int = DEFAULT_K,
    weights: B1PlusWeights = DEFAULT_WEIGHTS,
    block_size: int = topk.DEFAULT_BLOCK_SIZE,
) -> dict:
    """Rank every requested system and direction, then write the Parquet files and summary.

    All rankings are computed before the first file is written, so a failure leaves no files.
    """
    systems = list(dict.fromkeys(systems))
    rankings = [
        rank(
            plan=make_plan(inputs=inputs, system=system, direction=direction, weights=weights),
            k=k, block_size=block_size,
        )
        for system in systems
        for direction in retrieval_inputs.DIRECTIONS
    ]
    rows: dict[str, int] = {}
    for ranking in rankings:
        path = topk_path(
            root=root, system=ranking.plan.system, direction=ranking.plan.direction
        )
        table = ranking_table(ranking)
        fileio.write_parquet_atomic(path=path, table=table)
        rows[path.name] = table.num_rows
    summary = _summary(
        data=inputs, systems=systems, k=k, weights=weights, rankings=rankings, rows=rows
    )
    fileio.write_json_atomic(
        path=paths.retrieval_dir(root) / SUMMARY_FILE, obj=summary
    )
    return summary
