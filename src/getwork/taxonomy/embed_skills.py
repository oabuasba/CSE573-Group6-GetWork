"""Embed the selected skills, each alone and unprefixed (TX-2), into row-aligned vector files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from getwork.embed import cache, store
from getwork.embed.client import EmbeddingClient

SKILL_STEM = "skill_vectors"
VECTORS_KIND = "skills"


def embed_skills(
    client: EmbeddingClient,
    skills: Sequence[str],
    directory: Path,
    *,
    meta_extra: Mapping[str, object],
    cache_dir: Path | None,
    chunk_size: int = 256,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    vectors = cache.embed_cached(
        client, skills, cache_dir=cache_dir, chunk_size=chunk_size, progress=progress
    )
    meta = {"embedding": client.info.as_dict(), "kind": VECTORS_KIND, **meta_extra}
    store.write_vectors(directory, SKILL_STEM, skills, vectors, meta)
