from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from getwork import fileio
from getwork.alias import apply
from getwork.embed import factory, store
from getwork.embed.client import EmbeddingError
from getwork.embed.fake import FakeEmbeddingClient
from getwork.llm.client import FatalLLMError
from getwork.taxonomy import cli, counts, embed_skills

pytestmark = pytest.mark.usefixtures("offline")

VERSION = "sha256:aaaaaaaaaaaa"
SKILLS_BY_DOC = {
    "resumes": [["Python", "SQL"], ["Python", "AWS"], ["Python", "SQL", "Go"]],
    "postings": [["Python", "SQL"], ["AWS", "Go"], ["Rust"]],
}
# df_total: Python 4, SQL 3, AWS 2, Go 2, Rust 1 (dropped by min_df 2)
EXPECTED_SELECTED = ["Python", "SQL", "AWS", "Go"]


def _canonical_record(source: str, index: int, skills: list[str], version: str) -> dict:
    row = dict.fromkeys(apply.CANONICAL_KEYS)
    row.update(id=f"{source}-{index}", source=source, skills=skills, skills_raw=skills,
               alias_table_version=version)
    return row


def _write_canonical(root: Path, version: str | dict[str, str] = VERSION) -> None:
    for source, docs in SKILLS_BY_DOC.items():
        ver = version if isinstance(version, str) else version[source]
        rows = [_canonical_record(source, i, skills, ver) for i, skills in enumerate(docs)]
        path = root / "canonical" / source / f"{source}_canonical.jsonl"
        fileio.write_jsonl_atomic(path, rows)


def _tree(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")) if root.exists() else []


def _run_counts(root: Path, *extra: str) -> int:
    return cli.main(["counts", "--artifacts-dir", str(root), *extra])


def test_counts_writes_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _write_canonical(root)
    assert _run_counts(root) == 0
    summary = json.loads(capsys.readouterr().out)
    skills = root / "skills"
    assert counts.read_selected(skills) == EXPECTED_SELECTED
    on_disk = json.loads((skills / "counts_summary.json").read_text())
    assert on_disk == summary
    assert summary["alias_table_version"] == VERSION
    assert summary["n_docs"] == {"resumes": 3, "postings": 3}
    assert (summary["top_n"], summary["min_df"], summary["n_selected"]) == (1000, 2, 4)
    table = pq.read_table(skills / "skill_counts.parquet").to_pylist()
    assert len(table) == 5 and {r["skill"] for r in table if not r["selected"]} == {"Rust"}


def test_counts_options_are_applied(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _write_canonical(root)
    assert _run_counts(root, "--top-n", "2", "--min-df", "1") == 0
    assert counts.read_selected(root / "skills") == ["Python", "SQL"]


def test_counts_mixed_alias_versions_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _write_canonical(root, {"resumes": "sha256:111111111111", "postings": "sha256:222222222222"})
    before = _tree(root)
    assert _run_counts(root) == 2
    err = capsys.readouterr().err
    assert "sha256:111111111111" in err and "sha256:222222222222" in err
    assert _tree(root) == before and not (root / "skills").exists()


def test_counts_missing_input_exit_2_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _write_canonical(root)
    missing = root / "canonical" / "postings" / "postings_canonical.jsonl"
    missing.unlink()
    before = _tree(root)
    assert _run_counts(root) == 2
    assert str(missing) in capsys.readouterr().err
    assert _tree(root) == before


@pytest.mark.parametrize("flag", ["--top-n", "--min-df"])
def test_counts_non_positive_options_exit_2(tmp_path: Path, flag: str) -> None:
    root = tmp_path / "art"
    _write_canonical(root)
    assert _run_counts(root, flag, "0") == 2
    assert not (root / "skills").exists()


def test_missing_subcommand_exit_2() -> None:
    assert cli.main([]) == 2


def test_default_artifacts_dir_is_trial() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(["counts"]).artifacts_dir == Path("artifacts_trial")
    assert parser.parse_args(["counts"]).top_n == 1000
    assert parser.parse_args(["counts"]).min_df == 2


def test_default_run_in_empty_cwd_exits_2_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _tree(tmp_path)
    assert cli.main(["counts"]) == 2
    assert "artifacts_trial" in capsys.readouterr().err
    assert _tree(tmp_path) == before
    assert not (tmp_path / "artifacts_trial").exists()


def test_embed_skills_writes_aligned_vectors_with_meta(tmp_path: Path) -> None:
    client = FakeEmbeddingClient(dim=16)
    skills = ["Python", "SQL", "AWS"]
    embed_skills.embed_skills(
        client, skills, tmp_path, meta_extra={"alias_table_version": "v"}, cache_dir=None,
        chunk_size=2,
    )
    got = store.read_vectors(tmp_path, embed_skills.SKILL_STEM)
    assert embed_skills.SKILL_STEM == "skill_vectors"
    assert got.ids == skills
    assert got.meta["kind"] == "skills" and got.meta["alias_table_version"] == "v"
    assert got.meta["embedding"] == client.info.as_dict()
    assert (got.vectors == FakeEmbeddingClient(dim=16).embed(skills)).all()
    assert [len(c) for c in client.calls] == [2, 1]


def _counted_root(tmp_path: Path) -> Path:
    root = tmp_path / "art"
    _write_canonical(root)
    assert _run_counts(root) == 0
    return root


def _run_embed(root: Path, client: object, *extra: str) -> int:
    return cli.main(["embed", "--artifacts-dir", str(root), *extra], embed_client=client)


def test_embed_writes_vectors_with_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _counted_root(tmp_path)
    summary = json.loads((root / "skills" / "counts_summary.json").read_text())
    capsys.readouterr()
    client = FakeEmbeddingClient(dim=16)
    assert _run_embed(root, client) == 0
    got = store.read_vectors(root / "skills", "skill_vectors")
    assert got.ids == EXPECTED_SELECTED
    assert got.meta["embedding"] == client.info.as_dict()
    assert got.meta["kind"] == "skills"
    assert got.meta["selected_skills_version"] == summary["selected_skills_version"]
    assert got.meta["alias_table_version"] == VERSION
    expected = FakeEmbeddingClient(dim=16).embed(EXPECTED_SELECTED)
    assert (got.vectors == expected).all()
    assert json.loads(capsys.readouterr().out)["n_skills"] == 4
    assert [c for call in client.calls for c in call] == EXPECTED_SELECTED
    assert any((root / "skills" / "_embed_cache").glob("chunk_*.npy"))


def test_embed_rerun_uses_cache(tmp_path: Path) -> None:
    root = _counted_root(tmp_path)
    first = FakeEmbeddingClient(dim=16)
    assert _run_embed(root, first, "--chunk-size", "3") == 0
    assert [len(c) for c in first.calls] == [3, 1]
    second = FakeEmbeddingClient(dim=16)
    assert _run_embed(root, second, "--chunk-size", "3") == 0
    assert second.calls == []
    assert store.read_vectors(root / "skills", "skill_vectors").ids == EXPECTED_SELECTED


def test_embed_never_mixes_embedding_identities(tmp_path: Path) -> None:
    root = _counted_root(tmp_path)
    assert _run_embed(root, FakeEmbeddingClient(dim=16)) == 0
    other = FakeEmbeddingClient(dim=8, model="other-model")
    assert _run_embed(root, other) == 0
    assert other.calls, "a different model/dim must not reuse cached chunks"
    got = store.read_vectors(root / "skills", "skill_vectors")
    assert got.vectors.shape == (4, 8)
    assert got.meta["embedding"]["model"] == "other-model" and got.meta["dim"] == 8


def test_embed_without_counts_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    client = FakeEmbeddingClient(dim=16)
    assert _run_embed(root, client) == 2
    assert "skill_counts.parquet" in capsys.readouterr().err
    assert client.calls == [] and not root.exists()


def test_embed_without_summary_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _counted_root(tmp_path)
    (root / "skills" / "counts_summary.json").unlink()
    assert _run_embed(root, FakeEmbeddingClient(dim=16)) == 2
    assert "counts_summary.json" in capsys.readouterr().err
    assert not (root / "skills" / "skill_vectors.npy").exists()


def test_embed_config_error_exits_2_before_reading_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    code = cli.main(
        ["embed", "--artifacts-dir", str(root), "--embed-backend", "voyager",
         "--env-file", str(tmp_path / "none.env")],
        environ={},
    )
    err = capsys.readouterr().err
    assert code == 2 and "skill_counts" not in err and "VOYAGER" in err


@pytest.mark.parametrize("error_type", [EmbeddingError, FatalLLMError])
def test_embed_failure_exit_3_redacts_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    root = _counted_root(tmp_path)

    class Failing(FakeEmbeddingClient):
        def embed(self, texts):  # type: ignore[no-untyped-def]
            raise error_type("denied for key sk-SECRET")

    failing = Failing(dim=16)
    bundle = factory.EmbeddingBundle(client=failing, info=failing.info, secrets=["sk-SECRET"])
    monkeypatch.setattr(factory, "make_embedding_client", lambda **kw: bundle)
    capsys.readouterr()
    assert cli.main(["embed", "--artifacts-dir", str(root)]) == 3
    err = capsys.readouterr().err
    assert "sk-SECRET" not in err and "denied" in err
    assert not (root / "skills" / "skill_vectors.npy").exists()

@pytest.mark.parametrize("error_type", [EmbeddingError, FatalLLMError])
def test_embed_client_construction_failure_exit_3_redacts_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error_type: type[Exception],
) -> None:
    root = _counted_root(tmp_path)

    def failing_factory(**kwargs: object) -> object:
        raise error_type("cannot start with key sk-SECRET: denied")

    monkeypatch.setattr(factory, "make_embedding_client", failing_factory)
    capsys.readouterr()
    environ = {"VOYAGER_API_KEY": "sk-SECRET"}
    code = cli.main(["embed", "--artifacts-dir", str(root)], environ=environ)
    captured = capsys.readouterr()
    assert code == 3
    assert "sk-SECRET" not in captured.err + captured.out and "denied" in captured.err
    assert not (root / "skills" / "skill_vectors.npy").exists()

def test_missing_summary_key_and_os_errors_exit_2_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _counted_root(tmp_path)
    summary_path = root / "skills" / "counts_summary.json"
    summary = json.loads(summary_path.read_text())
    del summary["selected_skills_version"]
    summary_path.write_text(json.dumps(summary))
    capsys.readouterr()
    assert _run_embed(root, FakeEmbeddingClient(dim=16)) == 2
    assert "selected_skills_version" in capsys.readouterr().err
    secret_root = tmp_path / "sk-SECRET"
    path = secret_root / "canonical" / "resumes" / "resumes_canonical.jsonl"
    path.mkdir(parents=True)
    code = cli.main(
        ["counts", "--artifacts-dir", str(secret_root)], environ={"VOYAGER_API_KEY": "sk-SECRET"}
    )
    err = capsys.readouterr().err
    assert code == 2 and err.startswith("error:") and "sk-SECRET" not in err
