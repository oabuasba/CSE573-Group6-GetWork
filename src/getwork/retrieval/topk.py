"""Exact blockwise top-k: ties break by document position (stable sort of negated scores)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DEFAULT_BLOCK_SIZE = 512


def _check_float32_matrix(*, name: str, array: np.ndarray) -> None:
    if array.ndim != 2 or array.dtype != np.float32:
        raise ValueError(
            f"{name} must be a 2-D float32 array, got ndim={array.ndim} dtype={array.dtype}"
        )


def similarity_matrix(resume_vectors: np.ndarray, posting_vectors: np.ndarray) -> np.ndarray:
    """S = R @ J.T as float32, shape (m, n). Small inputs and tests only; the run never calls it.

    Raises ValueError unless both inputs are 2-D float32 with equal width.
    """
    _check_float32_matrix(name="resume_vectors", array=resume_vectors)
    _check_float32_matrix(name="posting_vectors", array=posting_vectors)
    if resume_vectors.shape[1] != posting_vectors.shape[1]:
        raise ValueError(
            f"vector widths differ: {resume_vectors.shape[1]} vs {posting_vectors.shape[1]}"
        )
    return (resume_vectors @ posting_vectors.T).astype(np.float32, copy=False)


def matrix_block(queries: np.ndarray, docs: np.ndarray) -> Callable[[int, int], np.ndarray]:
    """block(start, stop) = queries[start:stop] @ docs.T, cast to float32 whatever the inputs."""

    def block(start: int, stop: int) -> np.ndarray:
        return (queries[start:stop] @ docs.T).astype(np.float32, copy=False)

    return block


def topk_from_blocks(
    block: Callable[[int, int], np.ndarray],
    n_queries: int,
    n_docs: int,
    k: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Per query, the k_eff = min(k, n_docs) best documents: score descending, position ascending.

    Returns (indices int32, scores float32), both shape (n_queries, k_eff); n_queries == 0
    gives empty (0, k_eff) arrays. Raises ValueError for k < 1, block_size < 1, n_docs < 1,
    or a block whose shape is not (rows, n_docs).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if n_docs < 1:
        raise ValueError(f"n_docs must be >= 1, got {n_docs}")
    k_eff = min(k, n_docs)
    indices = np.empty((n_queries, k_eff), dtype=np.int32)
    scores_out = np.empty((n_queries, k_eff), dtype=np.float32)
    for start in range(0, n_queries, block_size):
        stop = min(start + block_size, n_queries)
        scores = np.asarray(block(start, stop))
        if scores.shape != (stop - start, n_docs):
            raise ValueError(
                f"block({start}, {stop}) returned shape {scores.shape}, "
                f"expected {(stop - start, n_docs)}"
            )
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k_eff]
        indices[start:stop] = order
        scores_out[start:stop] = np.take_along_axis(scores, order, axis=1)
    return indices, scores_out
