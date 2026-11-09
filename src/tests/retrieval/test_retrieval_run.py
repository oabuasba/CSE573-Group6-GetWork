import dataclasses
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from retrieval_helpers import make_chain

from getwork import paths
from getwork.docembed.documents import COMPARE_KEYS
from getwork.retrieval import b1plus, inputs, run

pytestmark = pytest.mark.usefixtures("offline")

PAIRS = list(itertools.product(inputs.SYSTEM_IDS, inputs.DIRECTIONS))


@pytest.fixture
def chain(tmp_path: Path) -> tuple[Path, inputs.RetrievalInputs]:
    root = tmp_path / "art"
    make_chain(root=root, n_resumes=6, n_postings=5, dim=16)
    return root, inputs.load_inputs(root)


def _docs(data: inputs.RetrievalInputs, direction: str) -> tuple[dict, dict]:
    if direction == "resume_to_posting":
        return data.resumes, data.postings
    return data.postings, data.resumes


def test_make_plan_all_systems_and_directions(chain) -> None:
    _, data = chain
    for system, direction in PAIRS:
        plan = run.make_plan(inputs=data, system=system, direction=direction)
        query_docs, doc_docs = _docs(data, direction)
        assert (plan.system, plan.direction) == (system, direction)
        assert plan.query_ids == list(query_docs) and plan.doc_ids == list(doc_docs)
        assert plan.query_ids[0] == ("r-1" if direction == "resume_to_posting" else "p-1")
        block = plan.block(1, 4)
        assert block.dtype == np.float32
        assert block.shape == (3, len(plan.doc_ids))
        if system == "B1+":
            for row, q_id in enumerate(plan.query_ids[1:4]):
                for col, d_id in enumerate(plan.doc_ids):
                    pair = (q_id, d_id) if direction == "resume_to_posting" else (d_id, q_id)
                    want = b1plus.b1plus_score(
                        data.resumes[pair[0]], data.postings[pair[1]], b1plus.DEFAULT_WEIGHTS
                    )
                    assert block[row, col] == pytest.approx(want, abs=1e-6)
        else:
            q_source, d_source = (
                ("resumes", "postings") if direction == "resume_to_posting"
                else ("postings", "resumes")
            )
            q_vec = data.vectors[(q_source, system)].vectors
            d_vec = data.vectors[(d_source, system)].vectors
            np.testing.assert_allclose(block, q_vec[1:4] @ d_vec.T, atol=1e-6)
    weights = b1plus.B1PlusWeights(0.0, 1.0, 0.0)
    plan = run.make_plan(inputs=data, system="B1+", direction="resume_to_posting", weights=weights)
    default = run.make_plan(inputs=data, system="B1+", direction="resume_to_posting")
    assert not np.allclose(plan.block(0, 6), default.block(0, 6))


def test_make_plan_invalid(chain) -> None:
    _, data = chain
    with pytest.raises(ValueError, match="system"):
        run.make_plan(inputs=data, system="B3", direction="resume_to_posting")
    with pytest.raises(ValueError, match="direction"):
        run.make_plan(inputs=data, system="Full", direction="sideways")
    only_b1 = dataclasses.replace(data, vectors={})
    for system in ("B2", "Full"):
        with pytest.raises(ValueError, match=system):
            run.make_plan(inputs=only_b1, system=system, direction="resume_to_posting")
    run.make_plan(inputs=only_b1, system="B1+", direction="posting_to_resume")


def test_run_writes_six_files_with_schema_and_order(chain) -> None:
    root, data = chain
    run.run_retrieval(inputs=data, root=root, k=3)
    files = sorted(p.name for p in paths.retrieval_dir(root).iterdir())
    assert files == sorted(
        [f"topk_{slug}_{d}.parquet" for slug in ("b1plus", "b2", "full") for d in inputs.DIRECTIONS]
        + ["retrieval_summary.json"]
    )
    for system, direction in PAIRS:
        path = run.topk_path(root=root, system=system, direction=direction)
        assert path.name == f"topk_{run.SYSTEM_SLUGS[system]}_{direction}.parquet"
        table = pq.read_table(path)
        assert table.schema.equals(run.TOPK_PARQUET_SCHEMA)
        rows = table.to_pylist()
        query_docs, doc_docs = _docs(data, direction)
        assert len(rows) == len(query_docs) * 3
        assert [r["query_id"] for r in rows][::3] == list(query_docs)
        for query_index, q_id in enumerate(query_docs):
            mine = rows[query_index * 3 : query_index * 3 + 3]
            assert [r["rank"] for r in mine] == [1, 2, 3]
            assert [r["query_id"] for r in mine] == [q_id] * 3
            assert len({r["doc_id"] for r in mine}) == 3
            scores = [r["score"] for r in mine]
            assert scores == sorted(scores, reverse=True)
            assert {r["system"] for r in mine} == {system}
            assert {r["direction"] for r in mine} == {direction}
            assert all(r["doc_id"] in doc_docs for r in mine)


def test_full_top1_matches_matrix_argmax(chain) -> None:
    root, data = chain
    run.run_retrieval(inputs=data, root=root, systems=["Full"], k=2)
    resumes = data.vectors[("resumes", "Full")].vectors
    postings = data.vectors[("postings", "Full")].vectors
    matrix = resumes @ postings.T
    path = run.topk_path(root=root, system="Full", direction="resume_to_posting")
    rows = pq.read_table(path).to_pylist()
    top1 = [r for r in rows if r["rank"] == 1]
    assert [r["doc_id"] for r in top1] == [
        list(data.postings)[i] for i in matrix.argmax(axis=1)
    ]
    np.testing.assert_allclose([r["score"] for r in top1], matrix.max(axis=1), atol=1e-6)
    path = run.topk_path(root=root, system="Full", direction="posting_to_resume")
    cols = pq.read_table(path).to_pylist()
    best = [r for r in cols if r["rank"] == 1]
    assert [r["doc_id"] for r in best] == [list(data.resumes)[i] for i in matrix.argmax(axis=0)]
    assert not run.topk_path(root=root, system="B2", direction="resume_to_posting").exists()


def test_summary_and_determinism(chain) -> None:
    root, data = chain
    weights = b1plus.B1PlusWeights(0.5, 0.25, 0.25)
    summary = run.run_retrieval(inputs=data, root=root, k=50, weights=weights, block_size=2)
    summary_path = paths.retrieval_dir(root) / "retrieval_summary.json"
    assert json.loads(summary_path.read_text()) == summary
    assert set(summary) == {
        "k", "k_effective", "systems", "n_resumes", "n_postings", "weights", "embedding",
        "provenance", "files",
    }
    assert (summary["k"], summary["k_effective"]) == (50, 5)
    assert summary["systems"] == ["B1+", "B2", "Full"]
    assert (summary["n_resumes"], summary["n_postings"]) == (6, 5)
    assert summary["weights"] == weights.as_dict()
    assert summary["embedding"] == data.vectors[("resumes", "Full")].meta["embedding"]
    assert summary["provenance"] == {k: data.summaries["resumes"].get(k) for k in COMPARE_KEYS}
    assert summary["provenance"]["taxonomy_version"] is not None
    assert len(summary["files"]) == 6
    assert summary["files"]["topk_full_resume_to_posting.parquet"] == 6 * 5
    assert summary["files"]["topk_b2_posting_to_resume.parquet"] == 5 * 6
    first = {p.name: pq.read_table(p) for p in paths.retrieval_dir(root).glob("*.parquet")}
    again = run.run_retrieval(inputs=data, root=root, k=50, weights=weights, block_size=2)
    assert again == summary
    for name, table in first.items():
        assert pq.read_table(paths.retrieval_dir(root) / name).equals(table)
    different_blocks = run.run_retrieval(
        inputs=data, root=root, k=50, weights=weights, block_size=512
    )
    assert different_blocks == summary
    for name, table in first.items():
        other = pq.read_table(paths.retrieval_dir(root) / name)
        assert other.drop_columns(["score"]).equals(table.drop_columns(["score"]))
        np.testing.assert_allclose(other["score"].to_numpy(), table["score"].to_numpy(), atol=1e-6)
    emb_only = run.run_retrieval(inputs=data, root=root, systems=["Full"], k=3)
    assert emb_only["weights"] is None and emb_only["systems"] == ["Full"]
    assert emb_only["k_effective"] == 3
    b1_only = run.run_retrieval(inputs=data, root=root, systems=["B1+"], k=3)
    assert b1_only["embedding"] is None
    assert b1_only["weights"] == b1plus.DEFAULT_WEIGHTS.as_dict()
    assert sorted(b1_only["files"]) == [
        "topk_b1plus_posting_to_resume.parquet", "topk_b1plus_resume_to_posting.parquet",
    ]


def test_ranking_failure_writes_nothing(chain) -> None:
    root, data = chain
    bad = dataclasses.replace(
        data,
        vectors={
            **data.vectors,
            ("postings", "B2"): dataclasses.replace(
                data.vectors[("postings", "B2")],
                vectors=np.zeros((5, 8), dtype=np.float32),
            ),
        },
    )
    with pytest.raises(ValueError):
        run.run_retrieval(inputs=bad, root=root)
    assert not paths.retrieval_dir(root).exists()


def test_duplicate_systems_rank_once(chain) -> None:
    root, data = chain
    summary = run.run_retrieval(inputs=data, root=root, systems=["B1+", "B1+"], k=3)
    assert summary["systems"] == ["B1+"]
    assert sorted(summary["files"]) == [
        "topk_b1plus_posting_to_resume.parquet", "topk_b1plus_resume_to_posting.parquet",
    ]
    summary = run.run_retrieval(inputs=data, root=root, systems=["Full", "B1+", "Full"], k=3)
    assert summary["systems"] == ["Full", "B1+"]
    assert len(summary["files"]) == 4


@pytest.fixture
def wide_chain(tmp_path: Path) -> tuple[Path, inputs.RetrievalInputs]:
    root = tmp_path / "wide"
    make_chain(root=root, n_resumes=11, n_postings=12, dim=16)
    return root, inputs.load_inputs(root=root)


def test_numeric_id_order_and_block_size_handling(
    wide_chain, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data = wide_chain
    assert list(data.postings)[:3] == ["p-1", "p-2", "p-3"] and "p-10" in data.postings
    plan = run.make_plan(inputs=data, system="Full", direction="resume_to_posting")
    assert plan.query_ids == [f"r-{i}" for i in range(1, 12)]
    assert plan.doc_ids == [f"p-{i}" for i in range(1, 13)]

    calls: list[tuple[int, int]] = []

    def spy(start: int, stop: int) -> np.ndarray:
        calls.append((start, stop))
        return plan.block(start, stop)

    ranking = run.rank(plan=dataclasses.replace(plan, block=spy), k=3, block_size=4)
    assert calls == [(0, 4), (4, 8), (8, 11)]
    assert ranking.indices.shape == (11, 3)

    seen: dict[str, list[tuple[int, int]]] = {}
    real_make_plan = run.make_plan

    def spying_make_plan(**kwargs):
        made = real_make_plan(**kwargs)
        log = seen.setdefault(made.direction, [])
        return dataclasses.replace(
            made, block=lambda a, b: (log.append((a, b)), made.block(a, b))[1]
        )

    monkeypatch.setattr(run, "make_plan", spying_make_plan)
    run.run_retrieval(inputs=data, root=root, systems=["Full"], k=3, block_size=4)
    assert seen["resume_to_posting"] == [(0, 4), (4, 8), (8, 11)]
    assert seen["posting_to_resume"] == [(0, 4), (4, 8), (8, 12)]
    rows = pq.read_table(run.topk_path(root=root, system="Full", direction="resume_to_posting"))
    assert rows["query_id"].to_pylist()[::3] == [f"r-{i}" for i in range(1, 12)]
    assert set(rows["doc_id"].to_pylist()) <= set(data.postings)
