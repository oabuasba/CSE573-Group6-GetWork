import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from getwork.docembed import documents, vectors
from getwork.embed import store
from getwork.embed.client import EmbeddingError, EmbeddingMismatchError
from getwork.embed.fake import FakeEmbeddingClient
from getwork.relabel.records import RELABEL_KEYS
from getwork.taxonomy.model import Cluster, make_taxonomy

pytestmark = pytest.mark.usefixtures("offline")

TAX = make_taxonomy(
    [Cluster(0, "Front-end", ("Vue", "CSS")), Cluster(1, "Cloud", ("AWS",))],
    noise=("Excel",),
    alias_table_version="sha256:t",
    meta={},
)
SKILL_SETS = {
    "r-10": [("AWS", "Cloud"), ("Vue", "Front-end")],
    "r-2": [("Excel", "Other"), ("CSS", "Front-end")],
    "r-1": [("AWS", "Cloud"), ("Excel", "Other"), ("Vue", "Front-end")],
}


def _record(id_: str, pairs: list[tuple[str, str]], source: str = "resumes") -> dict:
    record = dict.fromkeys(RELABEL_KEYS)
    record.update(
        id=id_, source=source, title=f"Engineer {id_}", years_experience=3.0,
        skills=[s for s, _ in pairs],
        skill_clusters=[{"skill": s, "cluster": c, "via": "lookup"} for s, c in pairs],
        alias_table_version="sha256:t", taxonomy_version=TAX.version, relabel_mode="llm",
        relabel_model="m1", relabel_backend="local",
    )
    return record


def _build(tmp_path: Path, source: str = "resumes", skill_sets=SKILL_SETS) -> tuple[dict, dict]:
    recs = {i: _record(i, pairs, source) for i, pairs in skill_sets.items()}
    out_dir = tmp_path / "docs" / source
    summary = documents.build_documents(
        source=source, relabeled=recs, listed_levels={}, tax=TAX, out_dir=out_dir
    )
    docs = {}
    for line in (out_dir / f"{source}_documents.jsonl").read_text().splitlines():
        row = json.loads(line)
        docs[row["id"]] = row
    return docs, summary


class FailOnSecondCall:
    def __init__(self) -> None:
        self.inner = FakeEmbeddingClient(dim=16)
        self.count = 0

    @property
    def info(self):
        return self.inner.info

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.count += 1
        if self.count == 2:
            raise EmbeddingError("boom")
        return self.inner.embed(texts)


def test_helpers_names_and_hash() -> None:
    assert vectors.text_field("Full") == "text_full"
    assert vectors.text_field("B2") == "text_b2"
    with pytest.raises(ValueError):
        vectors.text_field("full")
    assert vectors.vector_stem(source="postings", system="B2") == "postings_b2"
    assert vectors.vector_stem(source="resumes", system="Full") == "resumes_full"
    with pytest.raises(ValueError):
        vectors.vector_stem(source="jobs", system="Full")
    one = vectors.texts_hash(["a", "b"])
    assert one.startswith("sha256:") and len(one) == len("sha256:") + 12
    assert one == vectors.texts_hash(("a", "b"))
    assert one != vectors.texts_hash(["b", "a"])
    assert vectors.texts_hash(["é"]) != vectors.texts_hash(["e"])
    assert vectors.texts_hash(["é", "日本"]) == "sha256:051367365f95"


def test_embed_documents_files_and_meta(tmp_path: Path) -> None:
    docs, summary = _build(tmp_path)
    assert list(docs) == ["r-1", "r-2", "r-10"]
    out_dir = tmp_path / "docs" / "resumes"
    client = FakeEmbeddingClient(dim=16)
    result = vectors.embed_documents(
        client, docs, summary, "resumes", out_dir, cache_dir=None
    )
    assert set(result) == {"Full", "B2"}
    direct = FakeEmbeddingClient(dim=16)
    sets = {}
    for system, field in (("Full", "text_full"), ("B2", "text_b2")):
        stem = f"resumes_{system.lower()}"
        for suffix in (".npy", "_ids.json", "_meta.json"):
            assert (out_dir / f"{stem}{suffix}").is_file()
        vset = store.read_vectors(out_dir, stem)
        sets[system] = vset
        assert vset.ids == ["r-1", "r-2", "r-10"]
        assert vset.vectors.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vset.vectors, axis=1), 1.0, atol=1e-5)
        for row, doc_id in enumerate(vset.ids):
            expected = direct.embed([docs[doc_id][field]])[0]
            np.testing.assert_allclose(vset.vectors[row], expected, atol=1e-6)
        meta = vset.meta
        assert (meta["kind"], meta["system"], meta["source"]) == (
            "document_vectors", system, "resumes",
        )
        assert meta["embedding"] == client.info.as_dict()
        for key in documents.COMPARE_KEYS:
            assert meta[key] == summary[key]
        assert meta["texts_hash"] == vectors.texts_hash([d[field] for d in docs.values()])
        assert result[system] == {"n": 3, "dim": 16, "texts_hash": meta["texts_hash"]}
    assert not np.allclose(sets["Full"].vectors, sets["B2"].vectors)


def test_second_run_uses_cache(tmp_path: Path) -> None:
    docs, summary = _build(tmp_path)
    out_dir = tmp_path / "docs" / "resumes"
    cache_dir = out_dir / "_embed_cache"
    client = FakeEmbeddingClient(dim=16)
    steps: list[tuple[str, int, int]] = []
    vectors.embed_documents(
        client, docs, summary, "resumes", out_dir, cache_dir=cache_dir, chunk_size=2,
        progress=lambda system, done, total: steps.append((system, done, total)),
    )
    assert steps == [("Full", 1, 2), ("Full", 2, 2), ("B2", 1, 2), ("B2", 2, 2)]
    first_calls = len(client.calls)
    assert first_calls == 4
    first = store.read_vectors(out_dir, "resumes_full").vectors.copy()
    vectors.embed_documents(
        client, docs, summary, "resumes", out_dir, cache_dir=cache_dir, chunk_size=2
    )
    assert len(client.calls) == first_calls
    np.testing.assert_array_equal(store.read_vectors(out_dir, "resumes_full").vectors, first)


def test_empty_docs_and_client_failure_write_nothing(tmp_path: Path) -> None:
    docs, summary = _build(tmp_path)
    out_dir = tmp_path / "docs" / "resumes"
    with pytest.raises(ValueError):
        vectors.embed_documents(
            FakeEmbeddingClient(dim=16), {}, summary, "resumes", out_dir, cache_dir=None
        )
    failing = FailOnSecondCall()
    with pytest.raises(EmbeddingError):
        vectors.embed_documents(failing, docs, summary, "resumes", out_dir, cache_dir=None)
    assert failing.count == 2
    names = [p.name for p in out_dir.iterdir()]
    assert not [n for n in names if "_full" in n or "_b2" in n]


def _meta_pair(tmp_path: Path) -> tuple[dict, dict, dict]:
    docs, summary = _build(tmp_path)
    out_dir = tmp_path / "docs" / "resumes"
    vectors.embed_documents(
        FakeEmbeddingClient(dim=16), docs, summary, "resumes", out_dir, cache_dir=None
    )
    return docs, store.read_vectors(out_dir, "resumes_full").meta, store.read_vectors(
        out_dir, "resumes_b2"
    ).meta


def test_check_vectors_current(tmp_path: Path) -> None:
    docs, full, b2 = _meta_pair(tmp_path)
    vectors.check_vectors_current(full, docs, "Full")
    vectors.check_vectors_current(b2, docs, "B2")
    with pytest.raises(documents.DocEmbedMismatchError):
        vectors.check_vectors_current(full, docs, "B2")
    with pytest.raises(documents.DocEmbedMismatchError, match="system"):
        vectors.check_vectors_current({**full, "system": "B2"}, docs, "Full")
    changed = {k: dict(v) for k, v in docs.items()}
    changed["r-2"]["text_full"] += " x"
    with pytest.raises(documents.DocEmbedMismatchError, match="texts_hash"):
        vectors.check_vectors_current(full, changed, "Full")
    vectors.check_vectors_current(b2, changed, "B2")
    reordered = {k: docs[k] for k in reversed(list(docs))}
    with pytest.raises(documents.DocEmbedMismatchError):
        vectors.check_vectors_current(full, reordered, "Full")


def test_check_vector_sets_comparable(tmp_path: Path) -> None:
    _, full, b2 = _meta_pair(tmp_path)
    postings_docs, postings_summary = _build(tmp_path, "postings")
    post_dir = tmp_path / "docs" / "postings"
    vectors.embed_documents(
        FakeEmbeddingClient(dim=16), postings_docs, postings_summary, "postings", post_dir,
        cache_dir=None,
    )
    post_full = store.read_vectors(post_dir, "postings_full").meta
    vectors.check_vector_sets_comparable(full, post_full)
    for key, value in (("model", "other"), ("backend", "voyager"), ("dim", 32)):
        other = {**post_full, "embedding": {**post_full["embedding"], key: value}}
        with pytest.raises(EmbeddingMismatchError, match=key):
            vectors.check_vector_sets_comparable(full, other)
    for key, value in (
        ("taxonomy_version", "sha256:zzz"), ("relabel_mode", "lookup_only"),
        ("b2_order", "record"),
    ):
        with pytest.raises(documents.DocEmbedMismatchError, match=key):
            vectors.check_vector_sets_comparable(full, {**post_full, key: value})
    both = {**post_full, "taxonomy_version": "sha256:zzz",
            "embedding": {**post_full["embedding"], "model": "other"}}
    with pytest.raises(EmbeddingMismatchError):
        vectors.check_vector_sets_comparable(full, both)
    with pytest.raises(documents.DocEmbedMismatchError, match="system"):
        vectors.check_vector_sets_comparable(full, {**b2})


def test_read_system_vectors_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="doc-embed embed") as info:
        vectors.read_system_vectors(tmp_path, "resumes", "Full")
    assert "resumes_full.npy" in str(info.value)
    docs, summary = _build(tmp_path)
    vectors.embed_documents(
        FakeEmbeddingClient(dim=16), docs, summary, "resumes",
        tmp_path / "documents" / "resumes", cache_dir=None,
    )
    vset = vectors.read_system_vectors(tmp_path, "resumes", "B2")
    assert vset.ids == ["r-1", "r-2", "r-10"] and vset.meta["system"] == "B2"
