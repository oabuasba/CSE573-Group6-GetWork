import itertools
from pathlib import Path

import pyarrow as pa
import pytest
from retrieval_helpers import make_chain

from getwork import fileio
from getwork.retrieval import b1plus, inputs, query, run

pytestmark = pytest.mark.usefixtures("offline")

PAIRS = list(itertools.product(inputs.SYSTEM_IDS, inputs.DIRECTIONS))


@pytest.fixture
def chain(tmp_path: Path) -> tuple[Path, inputs.RetrievalInputs]:
    root = tmp_path / "art"
    make_chain(root=root, n_resumes=6, n_postings=5, dim=16)
    return root, inputs.load_inputs(root)


def test_read_topk_order_and_missing(tmp_path: Path) -> None:
    root = tmp_path / "art"
    with pytest.raises(FileNotFoundError, match="retrieve run") as info:
        query.read_topk(root=root, system="Full", direction="resume_to_posting")
    assert "topk_full_resume_to_posting.parquet" in str(info.value)
    rows = [
        ("q-2", 1, "d-9", 0.9), ("q-2", 2, "d-1", 0.5),
        ("q-10", 1, "d-3", 0.7), ("q-10", 2, "d-4", 0.7),
        ("q-1", 1, "d-2", 0.1),
    ]
    table = pa.table(
        {
            "system": ["Full"] * 5, "direction": ["resume_to_posting"] * 5,
            "query_id": [r[0] for r in rows],
            "rank": pa.array([r[1] for r in rows], pa.int32()),
            "doc_id": [r[2] for r in rows],
            "score": pa.array([r[3] for r in rows], pa.float32()),
        },
        schema=run.TOPK_PARQUET_SCHEMA,
    )
    path = run.topk_path(root=root, system="Full", direction="resume_to_posting")
    fileio.write_parquet_atomic(path, table)
    got = query.read_topk(root=root, system="Full", direction="resume_to_posting")
    assert list(got) == ["q-2", "q-10", "q-1"]
    assert got["q-2"] == [("d-9", pytest.approx(0.9)), ("d-1", pytest.approx(0.5))]
    assert [d for d, _ in got["q-10"]] == ["d-3", "d-4"]
    assert got["q-1"] == [("d-2", pytest.approx(0.1))]
    with pytest.raises(FileNotFoundError):
        query.read_topk(root=root, system="B2", direction="resume_to_posting")


def test_query_matches_equals_run_output(chain) -> None:
    root, data = chain
    run.run_retrieval(inputs=data, root=root, k=4)
    for system, direction in PAIRS:
        stored = query.read_topk(root, system, direction)
        plan = run.make_plan(inputs=data, system=system, direction=direction)
        assert list(stored) == plan.query_ids
        for query_id, expected in stored.items():
            got = query.query_matches(
                inputs=data, system=system, direction=direction, query_id=query_id, k=4
            )
            assert [d for d, _ in got] == [d for d, _ in expected], (system, direction, query_id)
            for (_, a), (_, b) in zip(got, expected, strict=True):
                assert a == pytest.approx(b, abs=1e-6)
    five = query.query_matches(
        inputs=data, system="Full", direction="resume_to_posting", query_id="r-3"
    )
    two = query.query_matches(
        inputs=data, system="B1+", direction="posting_to_resume", query_id="p-2", k=2
    )
    assert (len(five), len(two)) == (5, 2)


def test_query_matches_unknown_id(chain) -> None:
    _, data = chain
    with pytest.raises(KeyError, match="p-99"):
        query.query_matches(
            inputs=data, system="Full", direction="posting_to_resume", query_id="p-99"
        )
    with pytest.raises(KeyError, match="r-1"):
        query.query_matches(
            inputs=data, system="B1+", direction="posting_to_resume", query_id="r-1"
        )
    with pytest.raises(ValueError):
        query.query_matches(
            inputs=data, system="Nope", direction="posting_to_resume", query_id="p-1"
        )


def test_query_matches_uses_given_weights(chain) -> None:
    root, data = chain
    weights = b1plus.B1PlusWeights(0.0, 1.0, 0.0)
    run.run_retrieval(inputs=data, root=root, systems=["B1+"], k=4, weights=weights)
    differs = False
    for direction in inputs.DIRECTIONS:
        stored = query.read_topk(root=root, system="B1+", direction=direction)
        for query_id, expected in stored.items():
            got = query.query_matches(
                inputs=data, system="B1+", direction=direction, query_id=query_id, k=4,
                weights=weights,
            )
            assert [d for d, _ in got] == [d for d, _ in expected]
            for (_, a), (_, b) in zip(got, expected, strict=True):
                assert a == pytest.approx(b, abs=1e-6)
            default = query.query_matches(
                inputs=data, system="B1+", direction=direction, query_id=query_id, k=4
            )
            differs = differs or [s for _, s in default] != [s for _, s in got]
    assert differs
