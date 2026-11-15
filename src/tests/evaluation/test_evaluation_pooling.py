from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from evaluation_helpers import make_eval_tree

from getwork.evaluation import loading, pooling
from getwork.retrieval import b1plus, query, run
from getwork.retrieval.b1plus import B1PlusWeights
from getwork.retrieval.inputs import DIRECTIONS, SYSTEM_IDS

pytestmark = pytest.mark.usefixtures("offline")

SCORE_TOLERANCE = 1e-6


@pytest.fixture
def data(tmp_path: Path):
    root, processed = make_eval_tree(tmp_path, n_resumes=14, n_postings=12)
    return loading.load_eval_inputs(root, processed).data


def _rows(table: pa.Table, **match: str) -> dict[str, list[tuple[str, float]]]:
    columns = table.to_pydict()
    out: dict[str, list[tuple[str, float]]] = {}
    for i in range(table.num_rows):
        if all(columns[key][i] == value for key, value in match.items()):
            out.setdefault(columns["query_id"][i], []).append(
                (columns["doc_id"][i], columns["score"][i])
            )
    return out


def test_restricted_inputs_keeps_order_and_vectors(data) -> None:
    snapshot = (list(data.resumes), list(data.postings), copy.deepcopy(data.vectors))
    resumes, postings = ["r-9", "r-2", "r-5"], ["p-7", "p-1"]
    sub = pooling.restricted_inputs(data, resume_ids=resumes, posting_ids=postings)
    assert list(sub.resumes) == ["r-2", "r-5", "r-9"] and list(sub.postings) == ["p-1", "p-7"]
    for (source, system), vset in sub.vectors.items():
        table = sub.resumes if source == "resumes" else sub.postings
        original = data.vectors[(source, system)]
        assert vset.ids == list(table) and vset.meta == original.meta
        for row, id_ in zip(vset.vectors, vset.ids):
            assert np.array_equal(row, original.vectors[original.ids.index(id_)])
    assert sub.summaries == data.summaries
    assert (list(data.resumes), list(data.postings)) == snapshot[:2]
    for key, vset in data.vectors.items():
        assert vset.ids == snapshot[2][key].ids
        assert np.array_equal(vset.vectors, snapshot[2][key].vectors)
    everything = pooling.restricted_inputs(data, resume_ids=None, posting_ids=None)
    assert list(everything.resumes) == list(data.resumes)
    assert everything.vectors[("resumes", "Full")].ids == data.vectors[("resumes", "Full")].ids


def test_restricted_inputs_unknown_id(data) -> None:
    with pytest.raises(ValueError, match="r-999"):
        pooling.restricted_inputs(data, resume_ids=["r-1", "r-999"], posting_ids=None)
    with pytest.raises(ValueError, match="p-999"):
        pooling.restricted_inputs(data, resume_ids=None, posting_ids=["p-999"])


def test_matches_retrieve_when_pool_is_everything(data, tmp_path: Path) -> None:
    run.run_retrieval(inputs=data, root=tmp_path / "retr")
    table = pooling.stage_rankings(
        data, stage="dev", resume_queries=["r-3", "r-1"], posting_queries=["p-2", "p-9"],
        pool_ids=list(data.postings),
    )
    assert table.schema == pooling.RANKINGS_SCHEMA
    assert set(table.column("stage").to_pylist()) == {"dev"}
    queries = {"resume_to_posting": {"r-1", "r-3"}, "posting_to_resume": {"p-2", "p-9"}}
    for system in SYSTEM_IDS:
        for direction in DIRECTIONS:
            stored = query.read_topk(tmp_path / "retr", system, direction)
            got = _rows(table, system=system, direction=direction)
            assert set(got) == queries[direction]
            for query_id, rows in got.items():
                assert [d for d, _ in rows] == [d for d, _ in stored[query_id]]
                for (_, a), (_, b) in zip(rows, stored[query_id]):
                    assert abs(a - b) < SCORE_TOLERANCE


def _expected_order(scores: np.ndarray, ids: list[str], k: int) -> list[str]:
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))
    return [ids[i] for i in order[:k]]


def test_pool_restricts_candidates_and_matches_brute_force(data) -> None:
    pool = ["p-11", "p-3", "p-5", "p-8", "p-2"]
    queries = ["r-4", "r-1", "r-7"]
    k = 3
    table = pooling.stage_rankings(
        data, stage="dev", resume_queries=queries, posting_queries=["p-1"], pool_ids=pool, k=k
    )
    pool_in_order = [p for p in data.postings if p in pool]
    for system in SYSTEM_IDS:
        got = _rows(table, system=system, direction="resume_to_posting")
        assert set(got) == set(queries)
        for query_id, rows in got.items():
            assert len(rows) == min(k, len(pool)) and {d for d, _ in rows} <= set(pool)
            assert [d for d, _ in rows] == _brute_force(data, system, query_id, pool_in_order, k)


def _brute_force(data, system: str, query_id: str, pool: list[str], k: int) -> list[str]:
    if system == "B1+":
        scores = np.array([
            b1plus.b1plus_score(data.resumes[query_id], data.postings[p], b1plus.DEFAULT_WEIGHTS)
            for p in pool
        ])
        return _expected_order(scores, pool, k)
    resumes, postings = data.vectors[("resumes", system)], data.vectors[("postings", system)]
    rows = [postings.ids.index(p) for p in pool]
    scores = postings.vectors[rows] @ resumes.vectors[resumes.ids.index(query_id)]
    return _expected_order(scores, pool, k)


def test_posting_to_resume_uses_all_resumes(data) -> None:
    table = pooling.stage_rankings(
        data, stage="test", resume_queries=["r-1"], posting_queries=["p-4", "p-6"],
        pool_ids=["p-1", "p-2"], k=len(data.resumes),
    )
    for system in SYSTEM_IDS:
        got = _rows(table, system=system, direction="posting_to_resume")
        assert set(got) == {"p-4", "p-6"}
        assert all(sorted(d for d, _ in rows) == sorted(data.resumes) for rows in got.values())


def test_weights_only_move_b1plus(data) -> None:
    common = dict(
        stage="dev", resume_queries=["r-1", "r-2", "r-3"], posting_queries=["p-1", "p-2"],
        pool_ids=list(data.postings),
    )
    default = pooling.stage_rankings(data, **common)
    shifted = pooling.stage_rankings(
        data, **common, weights=B1PlusWeights(skills=0.0, title=0.0, level=1.0)
    )
    for system in ("B2", "Full"):
        assert default.filter(pc.equal(default["system"], system)).equals(
            shifted.filter(pc.equal(shifted["system"], system))
        )
    mask = pc.equal(default["system"], "B1+")
    assert not default.filter(mask).equals(shifted.filter(mask))


def test_unknown_query_or_pool_id(data) -> None:
    common = dict(stage="dev", resume_queries=["r-1"], posting_queries=["p-1"],
                  pool_ids=["p-1"])
    for field, bad in (("resume_queries", "r-x"), ("posting_queries", "p-x"),
                       ("pool_ids", "p-y")):
        with pytest.raises(ValueError, match=bad):
            pooling.stage_rankings(data, **{**common, field: [bad]})
