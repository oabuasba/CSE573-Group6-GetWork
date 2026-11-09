"""Synthetic chains (taxonomy, documents tables, fake vectors) for the retrieval tests."""

from __future__ import annotations

from pathlib import Path

from getwork import paths
from getwork.docembed import documents, vectors
from getwork.embed.fake import FakeEmbeddingClient
from getwork.relabel.records import RELABEL_KEYS
from getwork.taxonomy.model import Cluster, Taxonomy, make_taxonomy

SKILLS: tuple[str, ...] = (
    "Python", "SQL", "AWS", "Docker", "Vue", "CSS",
    "Excel", "Go", "Spark", "Java", "C++", "Kubernetes",
)
CLUSTER_OF: dict[str, str] = {
    "Python": "Data", "SQL": "Data", "Spark": "Data", "Excel": "Data",
    "AWS": "Cloud", "Docker": "Cloud", "Kubernetes": "Cloud",
    "Vue": "Front-end", "CSS": "Front-end",
    "Go": "Backend", "Java": "Backend", "C++": "Backend",
}
TITLES: tuple[str, ...] = (
    "Data Engineer", "Senior Data Analyst", "Cloud Platform Engineer",
    "Front-end Developer", "Backend Developer", "Data Scientist",
)
LISTED_LEVELS: tuple[str | None, ...] = (
    "Entry level", None, "Director", "Mid-Senior level", None, "Associate",
)
ALIAS_VERSION = "sha256:alias"


def make_chain_taxonomy() -> Taxonomy:
    """Four-cluster taxonomy over the 12 fixed skills."""
    clusters = [
        Cluster(i, name, tuple(s for s, c in CLUSTER_OF.items() if c == name))
        for i, name in enumerate(("Data", "Cloud", "Front-end", "Backend"))
    ]
    return make_taxonomy(clusters, noise=(), alias_table_version=ALIAS_VERSION, meta={})


def _record(*, source: str, index: int, tax: Taxonomy, title_suffix: str = "") -> dict:
    picked = [SKILLS[(index * 3 + step) % len(SKILLS)] for step in range(2 + index % 3)]
    record = dict.fromkeys(RELABEL_KEYS)
    record.update(
        id=f"{source[0]}-{index + 1}", source=source,
        title=TITLES[index % len(TITLES)] + title_suffix,
        years_experience=float(1 + 3 * index),
        skills=picked,
        skill_clusters=[{"skill": s, "cluster": CLUSTER_OF[s], "via": "lookup"} for s in picked],
        alias_table_version=ALIAS_VERSION, taxonomy_version=tax.version,
        relabel_mode="llm", relabel_model="m1", relabel_backend="local",
    )
    return record


def _build_source(
    *, root: Path, source: str, count: int, tax: Taxonomy, title_suffix: str = ""
) -> dict:
    records = {}
    for index in range(count):
        record = _record(source=source, index=index, tax=tax, title_suffix=title_suffix)
        records[record["id"]] = record
    listed = {
        id_: LISTED_LEVELS[i % len(LISTED_LEVELS)] for i, id_ in enumerate(records)
    }
    return documents.build_documents(
        source=source, relabeled=records, listed_levels=listed, tax=tax,
        out_dir=paths.documents_dir(root, source),
    )


def embed_source(*, root: Path, source: str, dim: int = 16, model: str | None = None) -> None:
    """Embed one source's current documents table with FakeEmbeddingClient into its directory."""
    client = (
        FakeEmbeddingClient(dim=dim) if model is None else FakeEmbeddingClient(dim=dim, model=model)
    )
    vectors.embed_documents(
        client=client,
        docs=documents.read_documents(root=root, source=source),
        summary=documents.read_documents_summary(root=root, source=source),
        source=source,
        out_dir=paths.documents_dir(root, source),
        cache_dir=None,
    )


def make_chain(
    root: Path, n_resumes: int = 6, n_postings: int = 5, dim: int = 16, embed: bool = True
) -> Taxonomy:
    """Build both documents tables under root and, if embed, fake vectors for both sources."""
    tax = make_chain_taxonomy()
    _build_source(root=root, source="resumes", count=n_resumes, tax=tax)
    _build_source(root=root, source="postings", count=n_postings, tax=tax)
    if embed:
        for source in ("resumes", "postings"):
            embed_source(root=root, source=source, dim=dim)
    return tax


def rebuild_posting_documents(root: Path, taxonomy: Taxonomy, changed_title: str) -> None:
    """Rebuild the postings table with every title suffixed by changed_title; vectors untouched."""
    count = len(documents.read_documents(root=root, source="postings"))
    _build_source(
        root=root, source="postings", count=count, tax=taxonomy,
        title_suffix=f" {changed_title}",
    )


def embed_other_model(root: Path, source: str) -> None:
    """Re-embed one source with FakeEmbeddingClient(dim=16, model="other")."""
    embed_source(root=root, source=source, dim=16, model="other")
