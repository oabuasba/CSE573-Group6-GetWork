"""Pair ids, pair tables and the blind rows that judges and annotators see (CC-8)."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Collection

import pyarrow as pa

from getwork.retrieval.inputs import DIRECTIONS

PAIRS_SCHEMA: pa.Schema = pa.schema([
    ("pair_id", pa.string()),
    ("resume_id", pa.string()),
    ("posting_id", pa.string()),
    ("directions", pa.list_(pa.string())),
])
ID_HEX_LENGTH = 12
RESUME_TO_POSTING = "resume_to_posting"


def make_pair_id(seed: int, resume_id: str, posting_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{resume_id}:{posting_id}".encode()).hexdigest()
    return "p-" + digest[:ID_HEX_LENGTH]


def pair_key(direction: str, query_id: str, doc_id: str) -> tuple[str, str]:
    """(resume_id, posting_id) of a ranking row."""
    if direction not in DIRECTIONS:
        raise ValueError(f"unknown direction {direction!r}; choose from {list(DIRECTIONS)}")
    return (query_id, doc_id) if direction == RESUME_TO_POSTING else (doc_id, query_id)


def _row_keys(rankings: pa.Table) -> list[tuple[tuple[str, str], str]]:
    columns = (rankings.column(name).to_pylist() for name in ("direction", "query_id", "doc_id"))
    return [(pair_key(d, q, doc), d) for d, q, doc in zip(*columns)]


def pairs_from_rankings(rankings: pa.Table, seed: int) -> pa.Table:
    directions: dict[tuple[str, str], set[str]] = {}
    for key, direction in _row_keys(rankings):
        directions.setdefault(key, set()).add(direction)
    rows: list[dict] = []
    owner: dict[str, tuple[str, str]] = {}
    for key in sorted(directions):
        pair_id = make_pair_id(seed, *key)
        if owner.setdefault(pair_id, key) != key:
            raise ValueError(f"pair id collision: {pair_id} for {owner[pair_id]} and {key}")
        rows.append({
            "pair_id": pair_id, "resume_id": key[0], "posting_id": key[1],
            "directions": sorted(directions[key]),
        })
    return pa.Table.from_pylist(rows, schema=PAIRS_SCHEMA)


def pair_id_map(pairs: pa.Table) -> dict[tuple[str, str], str]:
    columns = (pairs.column(name).to_pylist() for name in ("resume_id", "posting_id", "pair_id"))
    return {(resume, posting): pair_id for resume, posting, pair_id in zip(*columns)}


def blind_rows(
    pairs: pa.Table, *, seed: int, purpose: str, only: Collection[str] | None = None
) -> list[dict]:
    rows = [
        {"pair_id": r["pair_id"], "resume_id": r["resume_id"], "posting_id": r["posting_id"]}
        for r in pairs.to_pylist()
        if only is None or r["pair_id"] in only
    ]
    rows.sort(key=lambda row: row["pair_id"])
    random.Random(f"{seed}/blind/{purpose}").shuffle(rows)
    return rows


def human_pair_ids(
    rankings_test: pa.Table, *, human_resumes: Collection[str],
    human_postings: Collection[str], seed: int,
) -> set[str]:
    queries = {RESUME_TO_POSTING: set(human_resumes)}
    queries.update({d: set(human_postings) for d in DIRECTIONS if d != RESUME_TO_POSTING})
    query_ids = rankings_test.column("query_id").to_pylist()
    return {
        make_pair_id(seed, *key)
        for (key, direction), query in zip(_row_keys(rankings_test), query_ids)
        if query in queries[direction]
    }
