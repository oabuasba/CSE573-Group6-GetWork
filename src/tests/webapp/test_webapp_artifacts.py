from pathlib import Path

import pytest
from artifact_helpers import ALIASES, build_artifacts
from retrieval_helpers import rebuild_posting_documents

from getwork import paths
from getwork.alias.cli import TABLE_FILE
from getwork.alias.table import make_table, save_table
from getwork.docembed import documents
from getwork.embed.fake import FakeEmbeddingClient
from getwork.retrieval import inputs as retrieval_inputs
from getwork.retrieval import query
from getwork.structure.extract import ExtractConfig
from getwork.taxonomy.model import Cluster, make_taxonomy, save_taxonomy
from getwork.webapp import models
from getwork.webapp.artifacts import (
    ArtifactEngine,
    EmbedResources,
    StructureResources,
    resolve_artifacts_root,
)
from getwork.webapp.engine import EngineError, EngineLoadError

pytestmark = pytest.mark.usefixtures("offline")

PAIRS = [
    (system, direction)
    for system in retrieval_inputs.SYSTEM_IDS
    for direction in retrieval_inputs.DIRECTIONS
]
QUERY_SOURCE = {"resume_to_posting": "resumes", "posting_to_resume": "postings"}


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "art"
    build_artifacts(path)
    return path


def corpus(direction: str, doc_id: str, system: str = "Full", k: int = 10) -> models.MatchRequest:
    return models.MatchRequest.model_validate(
        {"direction": direction, "query": {"kind": "corpus", "id": doc_id}, "system": system, "k": k}
    )


def test_load_and_health(root: Path) -> None:
    embed = EmbedResources(factory=lambda: FakeEmbeddingClient(dim=16), describe="fake, 16")
    structure = StructureResources(client=object(), cfg=ExtractConfig(model="fake-model"))  # type: ignore[arg-type]
    engine = ArtifactEngine.load(root, structure=structure, embed=embed)
    health = engine.health()
    assert health.mode == "artifacts"
    assert health.artifacts_dir == str(root)
    assert health.counts == {"resumes": 6, "postings": 5}
    assert health.systems == ["B1+", "B2", "Full"]
    assert health.levels[-1] == "Not specified"
    versions = health.versions
    assert versions.alias_table_version.startswith("sha256:")
    assert versions.taxonomy_version.startswith("sha256:")
    assert (versions.serialization_version, versions.relabel_mode) == ("ser_v1", "llm")
    assert versions.embedding == models.EmbeddingIdentity(model="fake-embedding", backend="fake", dim=16)
    assert health.capabilities["structure"] == models.Capability(
        available=True, detail="voyager, fake-model"
    )
    assert health.capabilities["embed"] == models.Capability(available=True, detail="fake, 16")
    bare = ArtifactEngine.load(
        root, structure_reason="no key", embed_reason="Disabled with --no-embed."
    ).health()
    assert bare.capabilities["structure"] == models.Capability(available=False, reason="no key")
    assert bare.capabilities["embed"].reason == "Disabled with --no-embed."


def test_load_b1plus_only(tmp_path: Path) -> None:
    root = tmp_path / "art"
    build_artifacts(root, embed=False)
    engine = ArtifactEngine.load(root, systems=["B1+"])
    health = engine.health()
    assert health.systems == ["B1+"]
    assert health.versions.embedding is None
    result = engine.match(corpus("resume_to_posting", "r-1", system="B1+", k=3))
    assert len(result.matches) == 3
    with pytest.raises(EngineLoadError, match="doc-embed embed"):
        ArtifactEngine.load(root)


def test_load_errors_have_hints(root: Path, tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    build_artifacts(missing)
    (paths.documents_dir(missing, "postings") / "postings_documents.jsonl").unlink()
    with pytest.raises(EngineLoadError, match="doc-embed build"):
        ArtifactEngine.load(missing)

    stale = tmp_path / "stale"
    tax = build_artifacts(stale)[1]
    rebuild_posting_documents(stale, tax, "changed")
    with pytest.raises(EngineLoadError, match="doc-embed embed"):
        ArtifactEngine.load(stale)

    alias_diff = tmp_path / "alias"
    old_table, _ = build_artifacts(alias_diff)
    new_table = make_table({**ALIASES, "ml": "Machine Learning"})
    save_table(paths.alias_dir(alias_diff) / TABLE_FILE, new_table, {})
    with pytest.raises(EngineLoadError) as info:
        ArtifactEngine.load(alias_diff)
    assert new_table.version in str(info.value) and old_table.version in str(info.value)

    tax_diff = tmp_path / "tax"
    table, _ = build_artifacts(tax_diff)
    changed = make_taxonomy([Cluster(0, "Only", ("Python",))], (), table.version, {})
    save_taxonomy(paths.taxonomy_dir(tax_diff) / "taxonomy.json", changed)
    with pytest.raises(EngineLoadError) as info:
        ArtifactEngine.load(tax_diff)
    assert changed.version in str(info.value) and "taxonomy_version" in str(info.value)

    for bad in (["Nope"], []):
        with pytest.raises(EngineLoadError):
            ArtifactEngine.load(root, systems=bad)


@pytest.mark.parametrize(("system", "direction"), PAIRS)
def test_corpus_match_equals_query_matches(
    root: Path, system: str, direction: str
) -> None:
    engine = ArtifactEngine.load(root)
    oracle_inputs = retrieval_inputs.load_inputs(root)
    docs = oracle_inputs.resumes if QUERY_SOURCE[direction] == "resumes" else oracle_inputs.postings
    for doc_id in docs:
        expected = query.query_matches(oracle_inputs, system, direction, doc_id, k=4)
        result = engine.match(corpus(direction, doc_id, system=system, k=4))
        assert [m.id for m, _ in result.matches] == [i for i, _ in expected]
        for (_, got), (_, want) in zip(result.matches, expected, strict=True):
            assert got == pytest.approx(want, abs=1e-6)
        assert result.query.id == doc_id and result.notes == []


def test_corpus_match_needs_no_embedder(root: Path) -> None:
    calls = []

    def factory() -> FakeEmbeddingClient:
        calls.append(1)
        return FakeEmbeddingClient(dim=16)

    engine = ArtifactEngine.load(root, embed=EmbedResources(factory=factory, describe="fake"))
    for system in ("B1+", "B2", "Full"):
        engine.match(corpus("posting_to_resume", "p-2", system=system))
    assert calls == []
    only = ArtifactEngine.load(root, systems=["B1+"])
    with pytest.raises(EngineError) as info:
        only.match(corpus("resume_to_posting", "r-1", system="Full"))
    assert info.value.code == "capability_unavailable"


def test_documents_views_and_k_clipping(root: Path) -> None:
    engine = ArtifactEngine.load(root)
    first = engine.examples("resume", 2)
    assert [v.id for v in first] == ["r-1", "r-2"] and all(v.side == "resume" for v in first)
    assert [v.id for v in engine.examples("posting", 50)] == [f"p-{i}" for i in range(1, 6)]
    view = engine.get_document("p-1")
    assert view.side == "posting" and view.incomplete_reason is None
    assert view.skills and all(s.cluster != "Other" and s.via == "taxonomy" for s in view.skills)
    result = engine.match(corpus("resume_to_posting", "r-1", k=50))
    assert result.k_effective == 5 and len(result.matches) == 5
    for code, request in (
        ("not_found", corpus("resume_to_posting", "r-99")),
        ("side_mismatch", corpus("resume_to_posting", "p-1")),
    ):
        with pytest.raises(EngineError) as info:
            engine.match(request)
        assert info.value.code == code and info.value.field == "query.id"
    with pytest.raises(EngineError) as info:
        engine.get_document("x-1")
    assert info.value.code == "not_found"


def test_stale_vectors_message_mentions_both_commands(tmp_path: Path) -> None:
    root = tmp_path / "art"
    tax = build_artifacts(root)[1]
    rebuild_posting_documents(root, tax, "changed")
    with pytest.raises(documents.DocEmbedMismatchError):
        retrieval_inputs.load_inputs(root)
    with pytest.raises(EngineLoadError, match="do not match the current documents table"):
        ArtifactEngine.load(root)


def make_run(path: Path) -> Path:
    for name in ("alias", "documents", "taxonomy", "relabeled", "retrieval"):
        (path / name).mkdir(parents=True)
    return path


def test_resolve_artifacts_root_uses_complete_dir_as_is(tmp_path: Path) -> None:
    run = make_run(tmp_path / "v5run")
    make_run(run / "v9")
    assert resolve_artifacts_root(run) == run


def test_resolve_artifacts_root_picks_highest_complete_version(tmp_path: Path) -> None:
    base = tmp_path / "artifacts_trial"
    for name in ("v1", "v2", "v10"):
        make_run(base / name)
    (base / "v11").mkdir()
    (base / "v11" / "alias").mkdir()
    (base / "vx").mkdir()
    assert resolve_artifacts_root(base) == base / "v10"


def test_resolve_artifacts_root_errors_name_dir_and_fix(tmp_path: Path) -> None:
    with pytest.raises(EngineLoadError, match="not found") as info:
        resolve_artifacts_root(tmp_path / "nope")
    assert str(tmp_path / "nope") in str(info.value)
    base = tmp_path / "trial"
    (base / "v3" / "alias").mkdir(parents=True)
    with pytest.raises(EngineLoadError) as info:
        resolve_artifacts_root(base)
    text = str(info.value)
    assert str(base) in text and "documents" in text and "taxonomy" in text
    assert "v3 lacks" in text and "artifacts_trial/v5" in text
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(EngineLoadError, match="no vN subfolders"):
        resolve_artifacts_root(empty)


def test_load_resolves_versioned_root_and_reports_it(tmp_path: Path) -> None:
    base = tmp_path / "artifacts_trial"
    build_artifacts(base / "v2")
    build_artifacts(base / "v1", n_resumes=3)
    health = ArtifactEngine.load(base).health()
    assert health.artifacts_dir == str(base / "v2")
    assert health.counts["resumes"] == 6


def test_corrupt_relabeled_summary_is_load_error(root: Path) -> None:
    summary = paths.relabeled_dir(root, "resumes") / "resumes_relabeled_summary.json"
    summary.write_text("{not json", encoding="utf-8")
    with pytest.raises(EngineLoadError):
        ArtifactEngine.load(root)


def test_resolver_logs_choice_and_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    base = tmp_path / "trial"
    make_run(base / "v1")
    (base / "v2").mkdir(parents=True)
    with caplog.at_level("INFO"):
        resolve_artifacts_root(base)
    assert "v1" in caplog.text and "v2" in caplog.text
