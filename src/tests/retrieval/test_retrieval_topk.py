import numpy as np
import pytest

from getwork.retrieval import topk

pytestmark = pytest.mark.usefixtures("offline")


def _unit_vectors(rows: int, dim: int, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).normal(size=(rows, dim))
    return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)


def _oracle(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(scores, order, axis=1)


def test_matches_oracle_for_block_sizes() -> None:
    scores = np.random.default_rng(1).normal(size=(7, 12)).astype(np.float32)
    want_idx, want_scores = _oracle(scores, 5)
    calls: list[tuple[int, int]] = []

    def block(start: int, stop: int) -> np.ndarray:
        calls.append((start, stop))
        return scores[start:stop]

    for block_size in (1, 3, 7, 8, 100):
        calls.clear()
        idx, got = topk.topk_from_blocks(
            block=block, n_queries=7, n_docs=12, k=5, block_size=block_size
        )
        np.testing.assert_array_equal(idx, want_idx)
        np.testing.assert_array_equal(got, want_scores)
        assert (idx.dtype, got.dtype) == (np.int32, np.float32)
        assert len(calls) == -(-7 // block_size)
        assert calls[-1][1] == 7 and max(b - a for a, b in calls) <= block_size


def test_ties_break_by_position() -> None:
    base = np.array([[0.2, 0.9, 0.5], [0.9, 0.1, 0.5]], dtype=np.float32)
    scores = base[:, [0, 1, 1, 2, 1, 0]]  # duplicated columns
    for block_size in (1, 2, 5):
        idx, got = topk.topk_from_blocks(
            block=lambda a, b: scores[a:b], n_queries=2, n_docs=6, k=6, block_size=block_size
        )
        np.testing.assert_array_equal(idx[0], [1, 2, 4, 3, 0, 5])
        np.testing.assert_array_equal(idx[1], [0, 5, 3, 1, 2, 4])
        np.testing.assert_array_equal(got, np.take_along_axis(scores, idx, axis=1))
    idx, _ = topk.topk_from_blocks(block=lambda a, b: scores[a:b], n_queries=2, n_docs=6, k=2)
    np.testing.assert_array_equal(idx, [[1, 2], [0, 5]])


def test_ties_break_by_position_on_large_inputs() -> None:
    levels = np.random.default_rng(5).integers(0, 4, size=(3, 300)).astype(np.float32)
    want_idx, _ = _oracle(levels, 50)
    for block_size in (1, 2, 512):
        idx, _ = topk.topk_from_blocks(
            block=lambda a, b: levels[a:b], n_queries=3, n_docs=300, k=50, block_size=block_size
        )
        np.testing.assert_array_equal(idx, want_idx)
    ranked = levels[0][idx[0]]
    for score in np.unique(ranked):
        positions = idx[0][ranked == score]
        assert positions.tolist() == sorted(positions.tolist())


def test_k_clipping_and_empty() -> None:
    scores = np.random.default_rng(2).normal(size=(4, 3)).astype(np.float32)
    idx, got = topk.topk_from_blocks(block=lambda a, b: scores[a:b], n_queries=4, n_docs=3, k=10)
    assert idx.shape == got.shape == (4, 3)
    np.testing.assert_array_equal(idx, _oracle(scores, 3)[0])
    idx, got = topk.topk_from_blocks(block=lambda a, b: scores[a:b], n_queries=4, n_docs=3, k=3)
    assert idx.shape == (4, 3)
    idx, got = topk.topk_from_blocks(block=lambda a, b: scores[a:b], n_queries=4, n_docs=3, k=1)
    assert idx.shape == (4, 1)

    def never(a: int, b: int) -> np.ndarray:
        raise AssertionError("block must not be called")

    idx, got = topk.topk_from_blocks(block=never, n_queries=0, n_docs=3, k=10)
    assert idx.shape == got.shape == (0, 3)
    assert (idx.dtype, got.dtype) == (np.int32, np.float32)


def test_single_document() -> None:
    scores = np.array([[0.5], [0.1], [0.9]], dtype=np.float32)
    idx, got = topk.topk_from_blocks(block=lambda a, b: scores[a:b], n_queries=3, n_docs=1, k=5)
    assert idx.shape == got.shape == (3, 1)
    np.testing.assert_array_equal(idx, np.zeros((3, 1)))
    np.testing.assert_array_equal(got, scores)


def test_invalid_arguments() -> None:
    ok = np.zeros((2, 3), dtype=np.float32)
    kw = {"block": lambda a, b: ok[a:b], "n_queries": 2, "n_docs": 3, "k": 2}
    topk.topk_from_blocks(**kw)
    for bad, name in (
        ({"k": 0}, "k"), ({"k": -1}, "k"), ({"block_size": 0}, "block_size"),
        ({"n_docs": 0}, "n_docs"),
    ):
        with pytest.raises(ValueError, match=name):
            topk.topk_from_blocks(**{**kw, **bad})
    with pytest.raises(ValueError, match="n_docs"):
        topk.topk_from_blocks(**{**kw, "n_queries": 0, "n_docs": 0})
    with pytest.raises(ValueError, match="shape"):
        topk.topk_from_blocks(**{**kw, "block": lambda a, b: ok[a:b, :2]})
    with pytest.raises(ValueError, match="shape"):
        topk.topk_from_blocks(**{**kw, "block": lambda a, b: ok[:1]})


def test_similarity_matrix_and_validation() -> None:
    r, j = _unit_vectors(5, 8, seed=3), _unit_vectors(4, 8, seed=4)
    s = topk.similarity_matrix(resume_vectors=r, posting_vectors=j)
    assert s.dtype == np.float32 and s.shape == (5, 4)
    np.testing.assert_allclose(s, r @ j.T, atol=1e-6)
    with pytest.raises(ValueError):
        topk.similarity_matrix(resume_vectors=r[0], posting_vectors=j)
    with pytest.raises(ValueError):
        topk.similarity_matrix(resume_vectors=r, posting_vectors=j[0])
    with pytest.raises(ValueError):
        topk.similarity_matrix(resume_vectors=r.astype(np.float64), posting_vectors=j)
    with pytest.raises(ValueError):
        topk.similarity_matrix(resume_vectors=r, posting_vectors=j.astype(np.float64))
    with pytest.raises(ValueError, match="width"):
        topk.similarity_matrix(resume_vectors=r, posting_vectors=j[:, :7].copy())
    block = topk.matrix_block(queries=r, docs=j)
    wide = topk.matrix_block(queries=r.astype(np.float64), docs=j.astype(np.float64))(1, 4)
    assert wide.dtype == np.float32
    got = block(1, 4)
    assert got.dtype == np.float32 and got.shape == (3, 4)
    np.testing.assert_allclose(got, s[1:4], atol=1e-6)


def test_row_and_column_topk_match_matrix() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(size=(9, 16))
    j = rng.normal(size=(6, 16))
    r = (r / np.linalg.norm(r, axis=1, keepdims=True)).astype(np.float32)
    j = (j / np.linalg.norm(j, axis=1, keepdims=True)).astype(np.float32)
    s = topk.similarity_matrix(resume_vectors=r, posting_vectors=j)
    rows, _ = topk.topk_from_blocks(
        block=topk.matrix_block(queries=r, docs=j), n_queries=9, n_docs=6, k=4, block_size=2
    )
    np.testing.assert_array_equal(rows, _oracle(s, 4)[0])
    cols, col_scores = topk.topk_from_blocks(
        block=topk.matrix_block(queries=j, docs=r), n_queries=6, n_docs=9, k=4, block_size=4
    )
    np.testing.assert_array_equal(cols, _oracle(s.T, 4)[0])
    np.testing.assert_allclose(col_scores, _oracle(s.T, 4)[1], atol=1e-6)
