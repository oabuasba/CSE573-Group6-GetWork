import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from retrieval_helpers import (
    ALIAS_VERSION,
    CLUSTER_OF,
    embed_other_model,
    embed_source,
    make_chain,
    make_chain_taxonomy,
    rebuild_posting_documents,
)

from getwork import paths
from getwork.docembed.documents import DocEmbedMismatchError
from getwork.embed.client import EmbeddingMismatchError
from getwork.retrieval import inputs
from getwork.taxonomy.model import Cluster, make_taxonomy

pytestmark = pytest.mark.usefixtures("offline")


def _other_taxonomy():
    names = sorted(set(CLUSTER_OF.values()))
    clusters = [
        Cluster(i, name, tuple(s for s, c in CLUSTER_OF.items() if c == name))
        for i, name in enumerate(names)
    ]
    return make_taxonomy(clusters, noise=("Zzz",), alias_table_version=ALIAS_VERSION, meta={})


def test_constants() -> None:
    assert inputs.SYSTEM_IDS == ("B1+", "B2", "Full")
    assert inputs.DIRECTIONS == ("resume_to_posting", "posting_to_resume")


def test_check_order_and_empty_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bogus"):
        inputs.load_inputs(root=tmp_path / "nothing", systems=["bogus"])
    both = tmp_path / "both"
    tax = make_chain(both)
    rebuild_posting_documents(root=both, taxonomy=_other_taxonomy(), changed_title="")
    (paths.documents_dir(both, "resumes") / "resumes_full.npy").unlink()
    with pytest.raises(DocEmbedMismatchError, match="taxonomy_version"):
        inputs.load_inputs(root=both, systems=["Full"])
    empty = tmp_path / "empty_table"
    make_chain(empty)
    docs_dir = paths.documents_dir(empty, "postings")
    (docs_dir / "postings_documents.jsonl").write_text("")
    with pytest.raises(ValueError, match="postings.*empty"):
        inputs.load_inputs(root=empty, systems=["B1+"])
    assert tax.version != _other_taxonomy().version


def test_other_taxonomy_has_other_version() -> None:
    assert _other_taxonomy().version != make_chain_taxonomy().version


def test_load_inputs_complete_and_b1plus_only(tmp_path: Path) -> None:
    root = tmp_path / "art"
    make_chain(root=root, n_resumes=6, n_postings=5, dim=16)
    loaded = inputs.load_inputs(root)
    assert list(loaded.resumes) == [f"r-{i}" for i in range(1, 7)]
    assert list(loaded.postings) == [f"p-{i}" for i in range(1, 6)]
    assert set(loaded.summaries) == {"resumes", "postings"}
    assert loaded.summaries["resumes"]["source"] == "resumes"
    assert set(loaded.vectors) == {
        (source, system) for source in ("resumes", "postings") for system in ("B2", "Full")
    }
    for (source, _), vset in loaded.vectors.items():
        docs = loaded.resumes if source == "resumes" else loaded.postings
        assert vset.ids == list(docs)
        assert vset.vectors.dtype == np.float32 and vset.vectors.shape == (len(docs), 16)
    only_full = inputs.load_inputs(root=root, systems=["Full"])
    assert set(only_full.vectors) == {("resumes", "Full"), ("postings", "Full")}
    only_b1 = inputs.load_inputs(root=root, systems=["B1+"])
    assert only_b1.vectors == {} and len(only_b1.resumes) == 6

    bare = tmp_path / "bare"
    make_chain(root=bare, embed=False)
    assert inputs.load_inputs(root=bare, systems=["B1+"]).vectors == {}
    with pytest.raises(FileNotFoundError, match="doc-embed embed"):
        inputs.load_inputs(root=bare, systems=["B2"])
    with pytest.raises(ValueError):
        inputs.load_inputs(root=root, systems=[])
    with pytest.raises(ValueError, match="Other"):
        inputs.load_inputs(root=root, systems=["B1+", "Other"])
    with pytest.raises(ValueError):
        inputs.load_inputs(root=root, systems=["full"])
    with pytest.raises(ValueError):
        inputs.load_inputs(root=bare, systems=["bogus"])


def test_load_inputs_missing_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    with pytest.raises(FileNotFoundError, match="doc-embed build"):
        inputs.load_inputs(empty)
    with pytest.raises(FileNotFoundError, match="doc-embed build"):
        inputs.load_inputs(root=empty, systems=["B1+"])

    root = tmp_path / "art"
    make_chain(root)
    shutil.rmtree(paths.documents_dir(root, "postings"))
    with pytest.raises(FileNotFoundError, match="postings_documents"):
        inputs.load_inputs(root=root, systems=["B1+"])

    root = tmp_path / "half"
    make_chain(root)
    (paths.documents_dir(root, "postings") / "postings_full.npy").unlink()
    with pytest.raises(FileNotFoundError, match="doc-embed embed"):
        inputs.load_inputs(root=root, systems=["Full"])
    assert len(inputs.load_inputs(root=root, systems=["B2"]).vectors) == 2
    inputs.load_inputs(root=root, systems=["B1+"])


def test_load_inputs_stale_and_mismatched(tmp_path: Path) -> None:
    stale = tmp_path / "stale"
    tax = make_chain(stale)
    rebuild_posting_documents(root=stale, taxonomy=tax, changed_title="Lead")
    for system in ("B2", "Full"):
        with pytest.raises(DocEmbedMismatchError, match="texts_hash"):
            inputs.load_inputs(root=stale, systems=[system])
    assert len(inputs.load_inputs(root=stale, systems=["B1+"]).postings) == 5

    versions = tmp_path / "versions"
    make_chain(versions)
    rebuild_posting_documents(root=versions, taxonomy=_other_taxonomy(), changed_title="")
    with pytest.raises(DocEmbedMismatchError, match="taxonomy_version"):
        inputs.load_inputs(root=versions, systems=["B1+"])

    moved = tmp_path / "moved"
    make_chain(moved)
    vec_dir = paths.documents_dir(moved, "postings")
    ids_file = vec_dir / "postings_full_ids.json"
    ids_file.write_text(ids_file.read_text().replace('"p-1"', '"p-9"'))
    with pytest.raises(DocEmbedMismatchError, match="postings.*Full.*doc-embed embed"):
        inputs.load_inputs(root=moved, systems=["Full"])
    inputs.load_inputs(root=moved, systems=["B2"])

    swapped = tmp_path / "swapped"
    make_chain(swapped)
    ids_file = paths.documents_dir(swapped, "resumes") / "resumes_b2_ids.json"
    ids = json.loads(ids_file.read_text())
    ids[0], ids[1] = ids[1], ids[0]
    ids_file.write_text(json.dumps(ids))
    with pytest.raises(DocEmbedMismatchError, match="resumes.*B2"):
        inputs.load_inputs(root=swapped, systems=["B2"])

    model = tmp_path / "model"
    make_chain(model)
    embed_other_model(root=model, source="postings")
    with pytest.raises(EmbeddingMismatchError, match="model"):
        inputs.load_inputs(root=model, systems=["Full"])

    dim = tmp_path / "dim"
    make_chain(dim)
    embed_source(root=dim, source="postings", dim=8)
    with pytest.raises(EmbeddingMismatchError, match="dim"):
        inputs.load_inputs(root=dim, systems=["B2"])
