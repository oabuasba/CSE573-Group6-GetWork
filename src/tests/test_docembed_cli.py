import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from getwork import fileio, paths
from getwork.docembed import cli, documents, vectors
from getwork.embed.client import EmbeddingError
from getwork.embed.fake import FakeEmbeddingClient
from getwork.llm.client import FatalLLMError
from getwork.relabel.records import RELABEL_KEYS
from getwork.taxonomy.model import Cluster, make_taxonomy, save_taxonomy

pytestmark = pytest.mark.usefixtures("offline")

TAX = make_taxonomy(
    [Cluster(0, "Front-end", ("Vue", "CSS")), Cluster(1, "Cloud", ("AWS",))],
    noise=("Excel",),
    alias_table_version="sha256:t",
    meta={},
)
PAIRS = [("AWS", "Cloud"), ("Excel", "Other"), ("Vue", "Front-end")]


def _record(id_: str, source: str, years: float | None = 3.0, **over: object) -> dict:
    record = dict.fromkeys(RELABEL_KEYS)
    record.update(
        id=id_, source=source, title=f"Engineer {id_}", years_experience=years,
        skills=[s for s, _ in PAIRS],
        skill_clusters=[{"skill": s, "cluster": c, "via": "lookup"} for s, c in PAIRS],
        alias_table_version="sha256:t", taxonomy_version=TAX.version, relabel_mode="llm",
        relabel_model="m1", relabel_backend="local",
    )
    record.update(over)
    return record


def _setup(tmp_path: Path, *, with_parquet: bool = True, **over: dict) -> tuple[Path, Path]:
    root, processed = tmp_path / "art", tmp_path / "proc"
    save_taxonomy(paths.taxonomy_dir(root) / "taxonomy.json", TAX)
    rows = {
        "resumes": [_record("r-10", "resumes", 1.0), _record("r-2", "resumes", 6.0),
                    _record("r-1", "resumes", None)],
        "postings": [_record("p-1", "postings", 4.0), _record("p-2", "postings", 4.0)],
    }
    for source, source_rows in rows.items():
        patched = [{**row, **over.get(source, {})} for row in source_rows]
        path = paths.relabeled_dir(root, source) / f"{source}_relabeled.jsonl"
        fileio.write_jsonl_atomic(path, patched)
    if with_parquet:
        parquet = paths.processed_parquet(processed, "postings")
        parquet.parent.mkdir(parents=True)
        frame = pd.DataFrame({"id": ["p-1", "p-2"], "experience_level": ["Director", None]})
        frame.to_parquet(parquet)
    return root, processed


def _build(root: Path, processed: Path, *extra: str) -> int:
    return cli.main(
        ["build", "--artifacts-dir", str(root), "--processed-dir", str(processed), *extra]
    )


def _embed(root: Path, *extra: str, client=None, environ=None) -> int:
    return cli.main(
        ["embed", "--artifacts-dir", str(root), *extra], client=client,
        environ={} if environ is None else environ,
    )


def _jsonl(root: Path, source: str) -> list[dict]:
    path = paths.documents_dir(root, source) / f"{source}_documents.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tree(root: Path) -> list[str]:
    return sorted(str(p) for p in root.rglob("*")) if root.exists() else []


def test_build_both_sources_with_listed_levels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, processed = _setup(tmp_path)
    assert _build(root, processed) == 0
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [s["source"] for s in printed] == ["resumes", "postings"]
    for source in paths.SOURCES:
        folder = paths.documents_dir(root, source)
        for name in ("documents.jsonl", "documents.parquet", "documents_summary.json"):
            assert (folder / f"{source}_{name}").is_file()
    resumes = {d["id"]: d for d in _jsonl(root, "resumes")}
    assert list(resumes) == ["r-1", "r-2", "r-10"]
    assert [resumes[i]["level"] for i in resumes] == ["Not specified", "Mid-Senior level",
                                                      "Entry level"]
    postings = {d["id"]: d for d in _jsonl(root, "postings")}
    assert (postings["p-1"]["level"], postings["p-1"]["level_source"]) == ("Director", "listed")
    assert (postings["p-2"]["level"], postings["p-2"]["level_source"]) == ("Associate", "years")
    assert printed[1]["level_source_counts"] == {"listed": 1, "years": 1, "missing": 0}
    assert printed[0]["b2_order"] == "grouped" and printed[0]["level_bands"] == [2.0, 5.0]


def test_build_options_and_single_source(tmp_path: Path) -> None:
    root, processed = _setup(tmp_path)
    assert _build(root, processed, "--source", "resumes", "--level-bands", "1.5,7",
                  "--b2-order", "record") == 0
    assert not paths.documents_dir(root, "postings").exists()
    docs = {d["id"]: d for d in _jsonl(root, "resumes")}
    assert docs["r-2"]["level"] == "Associate" and docs["r-10"]["level"] == "Entry level"
    assert docs["r-1"]["text_b2"].endswith("Skills: AWS, Excel, Vue")
    summary = documents.read_documents_summary(root, "resumes")
    assert summary["level_bands"] == [1.5, 7.0] and summary["b2_order"] == "record"


def test_build_without_processed_parquet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, processed = _setup(tmp_path, with_parquet=False)
    assert _build(root, processed) == 0
    captured = capsys.readouterr()
    assert "clean.parquet" in captured.err and "years" in captured.err
    assert captured.err.count("note:") == 1 and "resumes" not in captured.err
    postings = {d["id"]: d for d in _jsonl(root, "postings")}
    assert {d["level"] for d in postings.values()} == {"Associate"}
    assert {d["level_source"] for d in postings.values()} == {"years"}


@pytest.mark.parametrize(
    ("case", "expect"),
    [
        ("no_taxonomy", "taxonomy build"),
        ("no_relabeled", "relabel"),
        ("taxonomy_version", "taxonomy_version"),
        ("alias_version", "alias_table_version"),
        ("bands", "bands"),
    ],
)
def test_build_input_errors_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str, expect: str
) -> None:
    over = {}
    if case == "taxonomy_version":
        over = {"postings": {"taxonomy_version": "sha256:other"}}
    if case == "alias_version":
        over = {"resumes": {"alias_table_version": "sha256:other"}}
    root, processed = _setup(tmp_path, **over)
    extra: tuple[str, ...] = ()
    if case == "no_taxonomy":
        (paths.taxonomy_dir(root) / "taxonomy.json").unlink()
    if case == "no_relabeled":
        (paths.relabeled_dir(root, "postings") / "postings_relabeled.jsonl").unlink()
    if case == "bands":
        extra = ("--level-bands", "5,2")
    assert _build(root, processed, *extra) == 2
    assert expect in capsys.readouterr().err
    assert not (root / "documents").exists()


@pytest.mark.parametrize("damage", ["no_column", "corrupt"])
def test_bad_postings_parquet_exit_2_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], damage: str
) -> None:
    root, processed = _setup(tmp_path)
    parquet = paths.processed_parquet(processed, "postings")
    if damage == "no_column":
        pd.DataFrame({"id": ["p-1", "p-2"]}).to_parquet(parquet)
    else:
        parquet.write_bytes(b"not a parquet file")
    assert _build(root, processed) == 2
    assert "postings_clean.parquet" in capsys.readouterr().err
    assert not (root / "documents").exists()


def test_build_unserializable_record_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = [{"skill": "AWS", "cluster": "Mystery", "via": "lookup"}]
    root, processed = _setup(tmp_path, postings={"skill_clusters": bad})
    assert _build(root, processed) == 2
    assert "Mystery" in capsys.readouterr().err
    assert not (root / "documents").exists()


@pytest.mark.parametrize("key", ["taxonomy_version", "alias_table_version"])
def test_build_single_source_version_mismatch_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], key: str
) -> None:
    root, processed = _setup(tmp_path, resumes={key: "sha256:other"})
    assert _build(root, processed, "--source", "resumes") == 2
    err = capsys.readouterr().err
    assert key in err and "r-" in err and "sha256:other" in err
    assert not (root / "documents").exists()


def test_build_incomparable_sources_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for over, key in (
        ({"postings": {"relabel_mode": "lookup_only"}}, "relabel_mode"),
        ({"postings": {"relabel_model": "m2"}}, "relabel_model"),
        ({"resumes": {"relabel_backend": "voyager"}}, "relabel_backend"),
    ):
        root, processed = _setup(tmp_path / key, **over)
        assert _build(root, processed) == 2
        assert key in capsys.readouterr().err
        assert not (root / "documents").exists()


def test_default_directories_are_trial_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = cli.build_parser().parse_args(["build"])
    assert args.artifacts_dir == Path("artifacts_trial")
    assert args.processed_dir == Path("data/processed")
    assert args.source == "both" and args.level_bands == "2,5" and args.b2_order == "grouped"
    assert cli.build_parser().parse_args(["embed"]).artifacts_dir == Path("artifacts_trial")
    assert cli.main(["build"]) == 2
    assert "taxonomy build" in capsys.readouterr().err
    assert _tree(tmp_path) == []
    assert cli.main(["embed"], client=FakeEmbeddingClient(dim=16), environ={}) == 2
    assert "doc-embed build" in capsys.readouterr().err
    assert _tree(tmp_path) == []


def _built(tmp_path: Path) -> Path:
    root, processed = _setup(tmp_path)
    assert _build(root, processed) == 0
    return root


def _vector_files(root: Path) -> list[Path]:
    found = []
    for source in paths.SOURCES:
        folder = paths.documents_dir(root, source)
        found += [p for p in folder.iterdir() if "_full" in p.name or "_b2" in p.name]
    return sorted(found)


def test_embed_both_sources_and_rerun_uses_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _built(tmp_path)
    capsys.readouterr()
    client = FakeEmbeddingClient(dim=16)
    assert _embed(root, "--chunk-size", "2", client=client) == 0
    captured = capsys.readouterr()
    printed = [json.loads(line) for line in captured.out.splitlines()]
    assert [p["source"] for p in printed] == ["resumes", "postings"]
    assert printed[0]["systems"]["Full"]["n"] == 3 and printed[0]["embedding"]["dim"] == 16
    assert "resumes Full: 100%" in captured.err and "postings B2: 100%" in captured.err
    calls = len(client.calls)
    assert calls > 0
    sets = {}
    for source in paths.SOURCES:
        folder = paths.documents_dir(root, source)
        assert len(list(folder.glob(f"{source}_full*"))) == 3
        assert len(list(folder.glob(f"{source}_b2*"))) == 3
        for system in vectors.SYSTEMS:
            vset = vectors.read_system_vectors(root, source, system)
            docs = documents.read_documents(root, source)
            assert vset.ids == list(docs)
            vectors.check_vectors_current(vset.meta, docs, system)
            direct = FakeEmbeddingClient(dim=16).embed(
                [d[vectors.text_field(system)] for d in docs.values()])
            np.testing.assert_allclose(vset.vectors, direct, atol=1e-6)
            sets[source, system] = vset
        assert (folder / "_embed_cache").is_dir()
    for system in vectors.SYSTEMS:
        vectors.check_vector_sets_comparable(
            sets["resumes", system].meta, sets["postings", system].meta)
    assert _embed(root, "--chunk-size", "2", client=client) == 0
    assert len(client.calls) == calls
    only = FakeEmbeddingClient(dim=16)
    assert _embed(root, "--source", "postings", "--no-cache", client=only) == 0
    assert len(only.calls) == 2


def test_embed_no_cache_option(tmp_path: Path) -> None:
    root = _built(tmp_path)
    client = FakeEmbeddingClient(dim=16)
    assert _embed(root, "--no-cache", client=client) == 0
    for source in paths.SOURCES:
        assert not (paths.documents_dir(root, source) / "_embed_cache").exists()
    first = len(client.calls)
    assert _embed(root, "--no-cache", client=client) == 0
    assert len(client.calls) == 2 * first


class _Failing:
    info = FakeEmbeddingClient(dim=16).info

    def embed(self, texts):
        raise EmbeddingError("gpu exploded")


def test_embed_errors_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _built(tmp_path)
    capsys.readouterr()
    fake = FakeEmbeddingClient(dim=16)
    summary = paths.documents_dir(root, "postings") / "postings_documents_summary.json"
    saved = summary.read_text()
    summary.unlink()
    assert _embed(root, client=fake) == 2
    assert "doc-embed build" in capsys.readouterr().err
    table = paths.documents_dir(root, "postings") / "postings_documents.jsonl"
    table_text = table.read_text()
    table.unlink()
    summary.write_text(saved)
    assert _embed(root, client=fake) == 2
    assert "doc-embed build" in capsys.readouterr().err
    table.write_text(table_text)
    changed = json.loads(saved) | {"taxonomy_version": "sha256:other"}
    summary.write_text(json.dumps(changed))
    assert _embed(root, client=fake) == 2
    assert "taxonomy_version" in capsys.readouterr().err
    summary.write_text(json.dumps(json.loads(saved) | {"n_documents": 5}))
    assert _embed(root, client=fake) == 2
    summary.write_text(saved)
    assert fake.calls == []
    assert _embed(root, client=_Failing()) == 3
    assert "gpu exploded" in capsys.readouterr().err
    assert _vector_files(root) == []
    assert _embed(root, "--embed-backend", "voyager") == 2
    err = capsys.readouterr().err
    assert "VOYAGER_API_KEY" in err and "VOYAGER_BASE_URL" in err
    assert _vector_files(root) == []
    assert _embed(root, "--chunk-size", "0", client=fake) == 2
    assert fake.calls == []


def test_embed_error_message_is_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _built(tmp_path)
    capsys.readouterr()

    def broken_factory(**kwargs: object) -> None:
        raise EmbeddingError("bad key s3cret-key in request")

    monkeypatch.setattr(cli.factory, "make_embedding_client", broken_factory)
    assert _embed(root, environ={"VOYAGER_API_KEY": "s3cret-key"}) == 3
    err = capsys.readouterr().err
    assert "s3cret-key" not in err and "bad key" in err


def test_embed_checks_inputs_before_building_the_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    assert _embed(root, "--embed-backend", "voyager") == 2
    err = capsys.readouterr().err
    assert "doc-embed build" in err and "VOYAGER" not in err


def test_embed_time_error_is_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _built(tmp_path)
    capsys.readouterr()

    class Leaky(_Failing):
        def embed(self, texts):
            raise EmbeddingError("rejected key s3cret-key")

    bundle = cli.factory.EmbeddingBundle(
        client=Leaky(), info=Leaky.info, secrets=["s3cret-key"]
    )
    monkeypatch.setattr(cli.factory, "make_embedding_client", lambda **kwargs: bundle)
    assert _embed(root) == 3
    err = capsys.readouterr().err
    assert "s3cret-key" not in err and "rejected key" in err
    assert _vector_files(root) == []


def test_chunk_size_option_reaches_the_cache(tmp_path: Path) -> None:
    assert cli.build_parser().parse_args(["embed"]).chunk_size == 256
    root = _built(tmp_path)
    default = FakeEmbeddingClient(dim=16)
    assert _embed(root, "--no-cache", client=default) == 0
    small = FakeEmbeddingClient(dim=16)
    assert _embed(root, "--no-cache", "--chunk-size", "2", client=small) == 0
    assert len(default.calls) == 4
    assert len(small.calls) == 6
    assert max(len(call) for call in small.calls) == 2


class _LLMFailing(_Failing):
    def embed(self, texts):
        raise FatalLLMError("auth failed for s3cret-key")


def test_llm_error_at_embed_time_exits_3_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _built(tmp_path)
    capsys.readouterr()
    bundle = cli.factory.EmbeddingBundle(
        client=_LLMFailing(), info=_LLMFailing.info, secrets=["s3cret-key"]
    )
    monkeypatch.setattr(cli.factory, "make_embedding_client", lambda **kwargs: bundle)
    assert _embed(root) == 3
    err = capsys.readouterr().err
    assert "auth failed" in err and "s3cret-key" not in err
    assert _vector_files(root) == []


def test_llm_error_at_construction_exits_3_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _built(tmp_path)
    capsys.readouterr()

    def refuse(**kwargs: object) -> None:
        raise FatalLLMError("auth failed for s3cret-key")

    monkeypatch.setattr(cli.factory, "make_embedding_client", refuse)
    assert _embed(root, environ={"VOYAGER_API_KEY": "s3cret-key"}) == 3
    err = capsys.readouterr().err
    assert "auth failed" in err and "s3cret-key" not in err
    assert _vector_files(root) == []
