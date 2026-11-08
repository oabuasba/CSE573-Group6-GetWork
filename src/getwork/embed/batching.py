"""Longest-first batching that halves the batch size on out-of-memory errors."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from getwork.embed.client import EmbeddingResourceError


def encode_in_batches(
    encode: Callable[[Sequence[str]], np.ndarray],
    texts: Sequence[str],
    batch_size: int,
    *,
    is_oom: Callable[[BaseException], bool],
    on_oom: Callable[[int], None] | None = None,
) -> np.ndarray:
    """Encode texts longest-first in batches, halving the batch size on OOM.

    The reduced batch size stays for later slices. Rows come back in input order.
    """
    if not texts:
        raise ValueError("texts must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    order = sorted(range(len(texts)), key=lambda i: (-len(texts[i]), i))
    parts: list[np.ndarray] = []
    start = 0
    while start < len(order):
        indices = order[start : start + batch_size]
        try:
            rows = encode([texts[i] for i in indices])
        except Exception as exc:
            if not is_oom(exc):
                raise
            if batch_size == 1:
                raise EmbeddingResourceError("out of memory at batch size 1") from exc
            batch_size = max(1, batch_size // 2)
            if on_oom is not None:
                on_oom(batch_size)
            continue
        parts.append(np.asarray(rows, dtype=np.float32))
        start += len(indices)
    stacked = np.concatenate(parts, axis=0)
    result = np.empty_like(stacked)
    result[order] = stacked
    return result
