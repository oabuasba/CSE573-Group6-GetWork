"""Embedding client protocol, provenance record and errors (numpy only)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

EMBEDDING_BACKENDS: tuple[str, ...] = ("local", "voyager")
MIN_NORM = 1e-12


@dataclass(frozen=True)
class EmbeddingInfo:
    model: str
    backend: str  # "local" | "voyager" | "fake"
    dim: int
    max_seq_length: int | None = None
    dtype: str | None = None  # "float16", "float32", ... or None

    def as_dict(self) -> dict[str, object]:
        """Fields as a dict, keys in declared order: model, backend, dim, max_seq_length, dtype."""
        return asdict(self)


class EmbeddingError(Exception):
    pass


class EmbeddingResourceError(EmbeddingError):
    """Out of memory even at batch size 1."""


class EmbeddingMismatchError(EmbeddingError):
    """Vector sets from different model, backend or dimension."""


class EmbeddingClient(Protocol):
    @property
    def info(self) -> EmbeddingInfo: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """float32 array (len(texts), info.dim); row i embeds texts[i]; unit L2 rows.

        Empty input returns shape (0, info.dim). Raises EmbeddingError (HTTP backends may
        also raise the getwork.llm.client.LLMError subclasses).
        """
        ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """float32 copy with each row divided by its L2 norm; EmbeddingError on bad input."""
    if vectors.ndim != 2:
        raise EmbeddingError(f"expected a 2-D array, got {vectors.ndim}-D")
    if not np.all(np.isfinite(vectors)):
        raise EmbeddingError("vectors contain NaN or infinite values")
    data = vectors.astype(np.float32)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    if np.any(norms < MIN_NORM):
        raise EmbeddingError("a vector has (near) zero norm")
    return (data / norms).astype(np.float32)
