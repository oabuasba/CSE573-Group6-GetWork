"""Row-aligned vector files and the embedding identity check."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from getwork import fileio
from getwork.embed.client import EmbeddingMismatchError

NORM_TOLERANCE = 1e-3
IDENTITY_KEYS = ("model", "backend", "dim")


@dataclass(frozen=True)
class VectorSet:
    ids: list[str]
    vectors: np.ndarray  # float32 (n, dim), unit rows
    meta: dict


def _paths(directory: Path, stem: str) -> tuple[Path, Path, Path]:
    return (
        directory / f"{stem}.npy",
        directory / f"{stem}_ids.json",
        directory / f"{stem}_meta.json",
    )


def _validate(ids: Sequence[str], vectors: np.ndarray, meta: Mapping[str, object]) -> None:
    if vectors.ndim != 2 or vectors.dtype != np.float32:
        raise ValueError("vectors must be a 2-D float32 array")
    if vectors.shape[0] != len(ids):
        raise ValueError(f"{len(ids)} ids but {vectors.shape[0]} vector rows")
    if len(set(ids)) != len(ids):
        raise ValueError("ids are not unique")
    if not np.all(np.isfinite(vectors)):
        raise ValueError("vectors contain NaN or infinite values")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(norms - 1.0) > NORM_TOLERANCE):
        raise ValueError("every vector row must have L2 norm 1")
    embedding = meta.get("embedding")
    if not isinstance(embedding, dict) or embedding.get("dim") != vectors.shape[1]:
        raise ValueError("meta['embedding'] must be a dict whose 'dim' equals the vector width")


def write_vectors(
    directory: Path,
    stem: str,
    ids: Sequence[str],
    vectors: np.ndarray,
    meta: Mapping[str, object],
) -> None:
    _validate(ids, vectors, meta)
    npy_path, ids_path, meta_path = _paths(directory, stem)

    def write_npy(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.save(f, vectors)

    fileio.atomic_write(npy_path, write_npy)
    fileio.write_json_atomic(ids_path, list(ids))
    fileio.write_json_atomic(meta_path, {**meta, "n": len(ids), "dim": int(vectors.shape[1])})


def read_vectors(directory: Path, stem: str) -> VectorSet:
    npy_path, ids_path, meta_path = _paths(directory, stem)
    if not npy_path.is_file():
        raise FileNotFoundError(f"file not found: {npy_path}")
    ids = fileio.read_json(ids_path)
    meta = fileio.read_json(meta_path)
    vectors = np.load(npy_path, allow_pickle=False)
    if not isinstance(ids, list) or not isinstance(meta, dict):
        raise ValueError(f"malformed ids or meta file for {stem}")
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ValueError(f"{stem}: {len(ids)} ids but vectors have shape {vectors.shape}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{stem}: ids are not unique")
    if meta.get("n") != len(ids) or meta.get("dim") != vectors.shape[1]:
        raise ValueError(f"{stem}: meta n/dim disagree with the vectors")
    return VectorSet(ids=ids, vectors=vectors, meta=meta)


def check_same_embedding(a: Mapping[str, object], b: Mapping[str, object]) -> None:
    """Raise EmbeddingMismatchError naming each of model, backend, dim that differs."""
    emb_a = a.get("embedding") or {}
    emb_b = b.get("embedding") or {}
    diffs = [
        f"{key}: {emb_a.get(key)!r} != {emb_b.get(key)!r}"
        for key in IDENTITY_KEYS
        if emb_a.get(key) != emb_b.get(key)
    ]
    if diffs:
        raise EmbeddingMismatchError("embedding mismatch: " + "; ".join(diffs))
