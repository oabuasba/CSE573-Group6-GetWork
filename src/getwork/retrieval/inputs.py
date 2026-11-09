"""Load documents tables and vector sets for retrieval and prove they belong together (RT-6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from getwork.docembed import documents, vectors
from getwork.embed.store import VectorSet

SYSTEM_IDS: tuple[str, ...] = ("B1+", "B2", "Full")
DIRECTIONS: tuple[str, ...] = ("resume_to_posting", "posting_to_resume")
SOURCES: tuple[str, ...] = ("resumes", "postings")


@dataclass(frozen=True)
class RetrievalInputs:
    resumes: dict[str, dict]
    postings: dict[str, dict]
    summaries: dict[str, dict]
    vectors: dict[tuple[str, str], VectorSet]


def _check_systems(systems: Sequence[str]) -> None:
    if not systems:
        raise ValueError("systems must not be empty")
    unknown = [s for s in systems if s not in SYSTEM_IDS]
    if unknown:
        raise ValueError(f"unknown system {unknown}; choose from {list(SYSTEM_IDS)}")


def _load_system_vectors(
    *, root: Path, system: str, docs_by_source: dict[str, dict[str, dict]]
) -> dict[tuple[str, str], VectorSet]:
    loaded = {}
    for source in SOURCES:
        docs = docs_by_source[source]
        vset = vectors.read_system_vectors(root=root, source=source, system=system)
        vectors.check_vectors_current(meta=vset.meta, docs=docs, system=system)
        if vset.ids != list(docs):
            raise documents.DocEmbedMismatchError(
                f"vector ids differ from the {source} documents table for {system}; "
                f"run doc-embed embed"
            )
        loaded[(source, system)] = vset
    vectors.check_vector_sets_comparable(
        a_meta=loaded[("resumes", system)].meta, b_meta=loaded[("postings", system)].meta
    )
    return loaded


def load_inputs(root: Path, systems: Sequence[str] = SYSTEM_IDS) -> RetrievalInputs:
    """Read and cross-check everything the requested systems need; B1+ needs no vectors."""
    _check_systems(systems)
    docs_by_source = {
        source: documents.read_documents(root=root, source=source) for source in SOURCES
    }
    summaries = {
        source: documents.read_documents_summary(root=root, source=source) for source in SOURCES
    }
    for source, docs in docs_by_source.items():
        if not docs:
            raise ValueError(f"the {source} documents table is empty")
    documents.check_comparable(a=summaries["resumes"], b=summaries["postings"])
    loaded: dict[tuple[str, str], VectorSet] = {}
    for system in systems:
        if system in vectors.SYSTEMS:
            loaded.update(
                _load_system_vectors(root=root, system=system, docs_by_source=docs_by_source)
            )
    return RetrievalInputs(
        resumes=docs_by_source["resumes"],
        postings=docs_by_source["postings"],
        summaries=summaries,
        vectors=loaded,
    )
