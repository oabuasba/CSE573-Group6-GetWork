"""Resumable chunk cache for long embedding runs."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path

import numpy as np

from getwork import fileio
from getwork.embed.client import EmbeddingClient

KEY_LENGTH = 16


def _chunk_path(cache_dir: Path, index: int, info_dict: dict, chunk: Sequence[str]) -> Path:
    payload = json.dumps([info_dict, list(chunk)], ensure_ascii=False, sort_keys=True)
    key = sha256(payload.encode("utf-8")).hexdigest()[:KEY_LENGTH]
    return cache_dir / f"chunk_{index:06d}_{key}.npy"


def _load_chunk(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        rows = np.load(path, allow_pickle=False)
    except (OSError, ValueError):
        return None
    return rows if rows.shape == shape else None


def _save_chunk(path: Path, rows: np.ndarray) -> None:
    def write(tmp: Path) -> None:
        with open(tmp, "wb") as f:
            np.save(f, rows)

    fileio.atomic_write(path, write)


def embed_cached(
    client: EmbeddingClient,
    texts: Sequence[str],
    *,
    cache_dir: Path | None,
    chunk_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Embed texts in chunks, reusing chunk files keyed by chunk texts and client info."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    info = client.info
    chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]
    parts: list[np.ndarray] = []
    for index, chunk in enumerate(chunks):
        path = None if cache_dir is None else _chunk_path(cache_dir, index, info.as_dict(), chunk)
        rows = None if path is None else _load_chunk(path, (len(chunk), info.dim))
        if rows is None:
            rows = client.embed(chunk)
            if path is not None:
                _save_chunk(path, rows)
        parts.append(rows)
        if progress is not None:
            progress(index + 1, len(chunks))
    if not parts:
        return np.zeros((0, info.dim), np.float32)
    return np.concatenate(parts, axis=0)
