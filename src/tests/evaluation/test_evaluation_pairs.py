from __future__ import annotations

import pyarrow as pa
import pytest

from getwork.evaluation import pairs
from getwork.evaluation.pooling import RANKINGS_SCHEMA

pytestmark = pytest.mark.usefixtures("offline")

R2P, P2R = "resume_to_posting", "posting_to_resume"


def _rankings(rows: list[tuple[str, str, str, str]]) -> pa.Table:
    """Rows are (system, direction, query_id, doc_id)."""
    return pa.Table.from_pylist(
        [
            {"stage": "test", "system": s, "direction": d, "query_id": q, "rank": i + 1,
             "doc_id": doc, "score": 1.0}
            for i, (s, d, q, doc) in enumerate(rows)
        ],
        schema=RANKINGS_SCHEMA,
    )


def test_pair_id_stable() -> None:
    first = pairs.make_pair_id(0, "r-1", "j-2")
    assert first == pairs.make_pair_id(0, "r-1", "j-2")
    assert first.startswith("p-") and len(first) == 14
    assert int(first[2:], 16) >= 0
    assert first != pairs.make_pair_id(1, "r-1", "j-2")
    assert first != pairs.make_pair_id(0, "j-2", "r-1")


def test_pair_key_orientation() -> None:
    assert pairs.pair_key(R2P, "r-1", "j-2") == ("r-1", "j-2")
    assert pairs.pair_key(P2R, "j-2", "r-1") == ("r-1", "j-2")
    with pytest.raises(ValueError, match="sideways"):
        pairs.pair_key("sideways", "a", "b")


def test_pairs_from_rankings_dedup_and_directions() -> None:
    table = pairs.pairs_from_rankings(
        _rankings([
            ("B1+", R2P, "r-2", "j-1"), ("B2", R2P, "r-2", "j-1"), ("Full", R2P, "r-1", "j-3"),
            ("Full", P2R, "j-1", "r-2"), ("B2", P2R, "j-1", "r-1"),
        ]),
        seed=0,
    )
    assert table.schema == pairs.PAIRS_SCHEMA
    assert not {"system", "rank", "score"} & set(table.column_names)
    rows = table.to_pylist()
    assert [(r["resume_id"], r["posting_id"]) for r in rows] == [
        ("r-1", "j-1"), ("r-1", "j-3"), ("r-2", "j-1")
    ]
    assert [r["directions"] for r in rows] == [[P2R], [R2P], [P2R, R2P]]
    assert pairs.pair_id_map(table) == {
        (r["resume_id"], r["posting_id"]): r["pair_id"] for r in rows
    }


def test_pair_id_collision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pairs, "make_pair_id", lambda seed, resume_id, posting_id: "p-same")
    with pytest.raises(ValueError, match="collision"):
        pairs.pairs_from_rankings(
            _rankings([("B2", R2P, "r-1", "j-1"), ("B2", R2P, "r-2", "j-1")]), seed=0
        )


def _twenty_pairs() -> pa.Table:
    return pairs.pairs_from_rankings(
        _rankings([("B2", R2P, f"r-{i:02d}", "j-1") for i in range(20)]), seed=3
    )


def test_blind_rows_are_blind_and_shuffled() -> None:
    table = _twenty_pairs()
    rows = pairs.blind_rows(table, seed=3, purpose="dev")
    assert len(rows) == 20
    assert all(set(row) == {"pair_id", "resume_id", "posting_id"} for row in rows)
    forbidden = {"B1+", "B2", "Full", R2P, P2R}
    assert not forbidden & {x for row in rows for x in (*row, *row.values())}
    assert rows == pairs.blind_rows(table, seed=3, purpose="dev")
    ids = [row["pair_id"] for row in rows]
    assert ids != sorted(ids)
    assert ids != [row["pair_id"] for row in pairs.blind_rows(table, seed=3, purpose="test")]
    only = set(ids[:5])
    assert {r["pair_id"] for r in pairs.blind_rows(table, seed=3, purpose="dev", only=only)} == only


def test_human_pair_ids() -> None:
    table = _rankings([
        ("B2", R2P, "r-1", "j-1"), ("B2", R2P, "r-2", "j-1"),
        ("B2", P2R, "j-1", "r-3"), ("B2", P2R, "j-2", "r-1"),
    ])
    found = pairs.human_pair_ids(
        table, human_resumes={"r-1"}, human_postings={"j-1"}, seed=0
    )
    assert found == {pairs.make_pair_id(0, "r-1", "j-1"), pairs.make_pair_id(0, "r-3", "j-1")}
    all_ids = set(pairs.pairs_from_rankings(table, seed=0).column("pair_id").to_pylist())
    assert found < all_ids
