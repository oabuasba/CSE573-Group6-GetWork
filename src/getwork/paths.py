"""Artifact tree layout and trial defaults shared by every downstream stage (no third-party imports)."""

from __future__ import annotations

from pathlib import Path

DEFAULT_STRUCTURED_DIR: Path = Path("data/structured_trial")
DEFAULT_PROCESSED_DIR: Path = Path("data/processed")
DEFAULT_ARTIFACTS_DIR: Path = Path("artifacts_trial")
SOURCES: tuple[str, ...] = ("resumes", "postings")


def check_source(source: str) -> str:
    """Return source if it is in SOURCES, else ValueError naming it."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return source


def structured_jsonl(structured_dir: Path, source: str) -> Path:
    check_source(source)
    return structured_dir / source / f"{source}_structured.jsonl"


def processed_parquet(processed_dir: Path, source: str) -> Path:
    check_source(source)
    return processed_dir / source / f"{source}_clean.parquet"


def alias_dir(root: Path) -> Path:
    return root / "alias"


def canonical_dir(root: Path, source: str) -> Path:
    return root / "canonical" / check_source(source)


def skills_dir(root: Path) -> Path:
    return root / "skills"


def taxonomy_dir(root: Path) -> Path:
    return root / "taxonomy"


def relabeled_dir(root: Path, source: str) -> Path:
    return root / "relabeled" / check_source(source)


def relabel_cache_dir(root: Path) -> Path:
    return root / "relabeled" / "_cache"


def documents_dir(root: Path, source: str) -> Path:
    return root / "documents" / check_source(source)


def retrieval_dir(root: Path) -> Path:
    return root / "retrieval"
