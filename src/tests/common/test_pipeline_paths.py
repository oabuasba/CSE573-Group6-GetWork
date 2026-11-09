from pathlib import Path

import pytest

from getwork import paths
from getwork.structure import schema

pytestmark = pytest.mark.usefixtures("offline")

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_default_dirs_are_trial_dirs() -> None:
    assert paths.DEFAULT_STRUCTURED_DIR == Path("data/structured_trial")
    assert paths.DEFAULT_PROCESSED_DIR == Path("data/processed")
    assert paths.DEFAULT_ARTIFACTS_DIR == Path("artifacts_trial")


def test_stage_dir_layout() -> None:
    a = Path("a")
    assert paths.structured_jsonl(a, "resumes") == Path("a/resumes/resumes_structured.jsonl")
    assert paths.processed_parquet(a, "postings") == Path("a/postings/postings_clean.parquet")
    assert paths.alias_dir(a) == Path("a/alias")
    assert paths.canonical_dir(a, "resumes") == Path("a/canonical/resumes")
    assert paths.skills_dir(a) == Path("a/skills")
    assert paths.taxonomy_dir(a) == Path("a/taxonomy")
    assert paths.relabeled_dir(a, "postings") == Path("a/relabeled/postings")
    assert paths.relabel_cache_dir(a) == Path("a/relabeled/_cache")
    assert paths.documents_dir(a, "resumes") == Path("a/documents/resumes")
    assert paths.retrieval_dir(a) == Path("a/retrieval")


def test_unknown_source_rejected() -> None:
    assert paths.SOURCES == schema.SOURCES
    with pytest.raises(ValueError, match="jobs"):
        paths.check_source("jobs")
    for func in (
        paths.structured_jsonl,
        paths.processed_parquet,
        paths.canonical_dir,
        paths.relabeled_dir,
        paths.documents_dir,
    ):
        with pytest.raises(ValueError, match="jobs"):
            func(Path("a"), "jobs")


def test_gitignore_ignores_trial_artifacts_only() -> None:
    lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "artifacts_trial/" in lines
    assert ".env" in lines
    assert "data/structured_trial/" in lines
    assert "artifacts/" in lines  # full-data output tree, ignored per run_pipeline.sh
    assert "data/structured" not in lines
    assert not any(line.startswith("data/structured/") for line in lines)
    assert not any(line in (".env*", "*.env*", ".env.*") for line in lines)
