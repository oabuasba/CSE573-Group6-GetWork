"""Document vectors: the Full and B2 texts of one source embedded into unit-norm vector sets."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path

from getwork import paths
from getwork.docembed import documents
from getwork.docembed.documents import COMPARE_KEYS, DocEmbedMismatchError
from getwork.embed import cache, store
from getwork.embed.client import EmbeddingClient

SYSTEMS: tuple[str, ...] = ("Full", "B2")
TEXT_FIELDS = {"Full": "text_full", "B2": "text_b2"}
HASH_HEX_LENGTH = 12
EMBED_HINT = "run doc-embed embed"


def text_field(system: str) -> str:
    if system not in TEXT_FIELDS:
        raise ValueError(f"unknown system {system!r}; expected one of {SYSTEMS}")
    return TEXT_FIELDS[system]


def vector_stem(source: str, system: str) -> str:
    text_field(system)
    return f"{paths.check_source(source)}_{system.lower()}"


def texts_hash(texts: Sequence[str]) -> str:
    payload = json.dumps(list(texts), ensure_ascii=False).encode("utf-8")
    return "sha256:" + sha256(payload).hexdigest()[:HASH_HEX_LENGTH]


def _texts_of(*, docs: Mapping[str, dict], system: str) -> list[str]:
    field = text_field(system)
    return [doc[field] for doc in docs.values()]


def embed_documents(
    client: EmbeddingClient,
    docs: Mapping[str, dict],
    summary: Mapping[str, object],
    source: str,
    out_dir: Path,
    *,
    cache_dir: Path | None,
    chunk_size: int = 256,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict:
    """Embed both systems, then write both vector sets; nothing is written if any pass fails."""
    if not docs:
        raise ValueError(f"no documents to embed for {source}")
    ids = list(docs)
    embedded = {}
    for system in SYSTEMS:
        texts = _texts_of(docs=docs, system=system)
        report = None if progress is None else _system_progress(progress, system)
        vectors = cache.embed_cached(
            client=client, texts=texts, cache_dir=cache_dir, chunk_size=chunk_size,
            progress=report,
        )
        embedded[system] = (texts, vectors)
    result = {}
    for system, (texts, vectors) in embedded.items():
        meta = {
            "kind": "document_vectors",
            "system": system,
            "source": source,
            "embedding": client.info.as_dict(),
            **{key: summary[key] for key in COMPARE_KEYS},
            "texts_hash": texts_hash(texts),
        }
        store.write_vectors(
            directory=out_dir, stem=vector_stem(source=source, system=system), ids=ids,
            vectors=vectors, meta=meta,
        )
        result[system] = {
            "n": len(ids), "dim": int(vectors.shape[1]), "texts_hash": meta["texts_hash"],
        }
    return result


def _system_progress(
    progress: Callable[[str, int, int], None], system: str
) -> Callable[[int, int], None]:
    return lambda done, total: progress(system, done, total)


def read_system_vectors(root: Path, source: str, system: str) -> store.VectorSet:
    try:
        return store.read_vectors(
            directory=paths.documents_dir(root, source),
            stem=vector_stem(source=source, system=system),
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{exc} ({EMBED_HINT})") from None


def check_vectors_current(
    meta: Mapping[str, object], docs: Mapping[str, dict], system: str
) -> None:
    if meta.get("system") != system:
        raise DocEmbedMismatchError(
            f"vectors are for system {meta.get('system')!r}, expected {system!r}"
        )
    if meta.get("texts_hash") != texts_hash(_texts_of(docs=docs, system=system)):
        raise DocEmbedMismatchError(
            f"{system} vectors do not match the current documents table (texts_hash differs); "
            f"run doc-embed build and doc-embed embed again"
        )


def check_vector_sets_comparable(
    a_meta: Mapping[str, object], b_meta: Mapping[str, object]
) -> None:
    store.check_same_embedding(a=a_meta, b=b_meta)
    documents.check_comparable(a=a_meta, b=b_meta)
    if a_meta.get("system") != b_meta.get("system"):
        raise DocEmbedMismatchError(
            f"vector sets are for different systems: {a_meta.get('system')!r} vs "
            f"{b_meta.get('system')!r}"
        )
