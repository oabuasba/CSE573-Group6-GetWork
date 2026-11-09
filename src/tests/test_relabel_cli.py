import json
from pathlib import Path

import pandas as pd
import pytest

from getwork import fileio, paths
from getwork.alias import table as alias_table
from getwork.llm import factory
from getwork.llm.client import FatalLLMError, LLMRequest
from getwork.llm.fake import FakeLLMClient
from getwork.relabel import cli, reextract, unseen
from getwork.taxonomy import model as taxonomy_model
from getwork.taxonomy.model import Cluster

pytestmark = pytest.mark.usefixtures("offline")

TABLE = alias_table.make_table({"js": "JavaScript"})
RECORDS = {
    "resumes": [("r-1", []), ("r-2", ["React", "Zig"]), ("r-3", ["React", "AWS", "Excel"])],
    "postings": [("j-1", ["React", "Zig"]), ("j-2", ["React", "AWS", "Excel"])],
}
TEXTS = {"r-1": "I built CI pipelines", "r-2": "React and Zig", "r-3": "x"}
REEXTRACT_REPLY = json.dumps({"additional_skills": [
    {"skill": "js", "cluster": "Cloud"}, {"skill": "Docker", "cluster": "Cloud"}]})


def _tax(alias_version: str = TABLE.version) -> taxonomy_model.Taxonomy:
    return taxonomy_model.make_taxonomy(
        [Cluster(0, "Front-end", ("React", "JavaScript")), Cluster(1, "Cloud", ("AWS",))],
        noise=("Excel",), alias_table_version=alias_version, meta={},
    )


def _record(id_: str, skills: list[str], source: str, alias_version: str) -> dict:
    return {
        "id": id_, "source": source, "title": "Dev", "skills": skills, "experience": [],
        "education": [], "years_experience": None, "model": "m0", "prompt_version": "p0",
        "schema_version": "1", "backend": "voyager", "skills_raw": skills,
        "alias_table_version": alias_version,
    }


def _setup(tmp_path: Path, *, canonical_version: str = TABLE.version) -> Path:
    root = tmp_path / "art"
    alias_table.save_table(paths.alias_dir(root) / "alias_table.json", TABLE, {})
    taxonomy_model.save_taxonomy(paths.taxonomy_dir(root) / "taxonomy.json", _tax())
    for source, items in RECORDS.items():
        rows = [_record(i, s, source, canonical_version) for i, s in items]
        fileio.write_jsonl_atomic(
            paths.canonical_dir(root, source) / f"{source}_canonical.jsonl", rows
        )
    return root


def _write_clean(tmp_path: Path) -> Path:
    processed = tmp_path / "processed"
    (processed / "resumes").mkdir(parents=True)
    frame = pd.DataFrame({"id": list(TEXTS), "text_clean": list(TEXTS.values())})
    frame.to_parquet(processed / "resumes" / "resumes_clean.parquet", index=False)
    return processed


def _args(tmp_path: Path, root: Path, *extra: str) -> list[str]:
    return ["--artifacts-dir", str(root), "--processed-dir", str(tmp_path / "processed"),
            "--env-file", str(tmp_path / "no.env"), *extra]


def _summaries(out: str) -> list[dict]:
    return [json.loads(line) for line in out.splitlines()]


def _llm_handler(request: LLMRequest) -> str:
    if request.system == reextract.load_reextract_prompt():
        return REEXTRACT_REPLY
    block = request.user.split("\n\nSKILLS:\n")[1]
    skills = [line[2:] for line in block.splitlines()]
    return json.dumps({"assignments": [{"skill": s, "cluster": "Cloud"} for s in skills]})


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _jsonl(root: Path, source: str) -> list[dict]:
    path = paths.relabeled_dir(root, source) / f"{source}_relabeled.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_lookup_only_both_sources(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _setup(tmp_path)
    assert cli.main(_args(tmp_path, root)) == 0
    summaries = _summaries(capsys.readouterr().out)
    assert [s["source"] for s in summaries] == ["resumes", "postings"]
    assert all(s["relabel_mode"] == "lookup_only" for s in summaries)
    assert summaries[0]["n_records"] == 3 and summaries[1]["n_records"] == 2
    for source in paths.SOURCES:
        names = sorted(p.name for p in paths.relabeled_dir(root, source).iterdir())
        assert names == sorted(f"{source}_{n}" for n in (
            "relabeled.jsonl", "relabeled.parquet", "relabeled_summary.json",
            "relabel_failures.csv"))
    assert not paths.relabel_cache_dir(root).exists()
    row = _jsonl(root, "resumes")[1]
    assert [(c["cluster"], c["via"]) for c in row["skill_clusters"]] == [
        ("Front-end", "taxonomy"), ("Other", "default")]
    assert row["relabel_mode"] == "lookup_only" and row["relabel_model"] is None
    assert cli.main(_args(tmp_path, root, "--min-known", "1.0")) == 0
    assert _jsonl(root, "resumes")[2]["incomplete_reason"] is None  # noise counts as known


def test_single_source(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _setup(tmp_path)
    assert cli.main(_args(tmp_path, root, "--source", "resumes")) == 0
    assert [s["source"] for s in _summaries(capsys.readouterr().out)] == ["resumes"]
    assert paths.relabeled_dir(root, "resumes").exists()
    assert not paths.relabeled_dir(root, "postings").exists()
    assert cli.main(_args(tmp_path, root, "--source", "postings")) == 0
    assert paths.relabeled_dir(root, "postings").exists()


def test_use_llm_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _setup(tmp_path)
    processed = _write_clean(tmp_path)
    assert processed.exists()
    client = FakeLLMClient(handler=_llm_handler)
    argv = _args(tmp_path, root, "--use-llm", "--model", "mod", "--backend", "local",
                 "--unseen-batch-size", "1", "--extra-body", '{"k": 1}',
                 "--source", "resumes")
    assert cli.main(argv, client=client, environ={}) == 0
    summary = _summaries(capsys.readouterr().out)[0]
    assert summary["relabel_mode"] == "llm"
    assert (summary["relabel_model"], summary["relabel_backend"]) == ("mod", "local")
    assert summary["relabel_prompt_version"] == "unseen_v1+reextract_v1"
    assert summary["n_reextracted"] == 1 and summary["n_reextract_failed"] == 0
    unseen_calls = [c for c in client.calls if c.system == unseen.load_unseen_prompt()]
    assert len(unseen_calls) == 1 and len(client.calls) == 2  # Zig once, one re-extraction
    assert all(c.extra_body == {"k": 1} and c.model == "mod" for c in client.calls)
    rows = {r["id"]: r for r in _jsonl(root, "resumes")}
    assert rows["r-1"]["skills"] == ["JavaScript", "Docker"] and rows["r-1"]["reextracted"]
    assert rows["r-1"]["added_skills"] == ["JavaScript", "Docker"]
    assert rows["r-2"]["skill_clusters"][1] == {"skill": "Zig", "cluster": "Cloud", "via": "llm"}
    assert (rows["r-2"]["relabel_model"], rows["r-2"]["relabel_backend"]) == ("mod", "local")
    cache = paths.relabel_cache_dir(root)
    assert sorted(p.name for p in cache.iterdir()) == ["reextract.json", "unseen_assignments.json"]
    again = FakeLLMClient(script=[])
    assert cli.main(argv, client=again, environ={}) == 0 and again.calls == []


def test_failures_exit_1(tmp_path: Path) -> None:
    root = _setup(tmp_path)
    bad_unseen = FakeLLMClient(handler=lambda r: "garbage")
    argv = _args(tmp_path, root, "--use-llm", "--model", "m", "--source", "postings")
    assert cli.main(argv, client=bad_unseen, environ={}) == 1
    summary = json.loads(
        (paths.relabeled_dir(root, "postings") / "postings_relabeled_summary.json").read_text()
    )
    assert summary["n_unseen_unresolved"] == 1
    assert paths.relabeled_dir(root, "postings").exists()
    # resumes: r-1 has no text (no clean parquet), so its re-extraction fails
    client = FakeLLMClient(handler=_llm_handler)
    argv = _args(tmp_path, root, "--use-llm", "--model", "m", "--source", "resumes")
    assert cli.main(argv, client=client, environ={}) == 1
    failures = (paths.relabeled_dir(root, "resumes") / "resumes_relabel_failures.csv")
    assert failures.read_text().splitlines()[1].startswith("r-1,no_text,")
    _write_clean(tmp_path)
    bad_reextract = FakeLLMClient(
        handler=lambda r: "garbage" if r.system == reextract.load_reextract_prompt()
        else _llm_handler(r))
    assert cli.main(argv, client=bad_reextract, environ={}) == 1
    assert failures.read_text().splitlines()[1].startswith("r-1,reextract,")


def test_fatal_exit_3_and_rerun_resumes(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    root = _setup(tmp_path)
    _write_clean(tmp_path)

    def fatal_on_reextract(request: LLMRequest) -> str:
        if request.system == reextract.load_reextract_prompt():
            raise FatalLLMError("connection refused", status=None)
        return _llm_handler(request)

    argv = _args(tmp_path, root, "--use-llm", "--model", "m")
    assert cli.main(argv, client=FakeLLMClient(handler=fatal_on_reextract), environ={}) == 3
    assert "connection refused" in capsys.readouterr().err
    cache = json.loads((paths.relabel_cache_dir(root) / "unseen_assignments.json").read_text())
    assert list(cache["entries"]) == ["zig"]
    assert not paths.relabeled_dir(root, "resumes").exists()
    resumed = FakeLLMClient(handler=_llm_handler)
    assert cli.main(argv, client=resumed, environ={}) == 0
    unseen_calls = [c for c in resumed.calls if c.system == unseen.load_unseen_prompt()]
    # every unseen skill (Zig, in both sources) is answered from the cache of the first run
    assert unseen_calls == []
    assert len(resumed.calls) == 1
    assert paths.relabeled_dir(root, "postings").exists()


def test_use_llm_without_settings_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    root = tmp_path / "missing_art"
    assert cli.main(_args(tmp_path, root, "--use-llm"), environ={}) == 2
    assert "error:" in capsys.readouterr().err
    assert not root.exists()
    client = FakeLLMClient(script=[])
    assert cli.main(_args(tmp_path, root, "--use-llm"), client=client, environ={}) == 2
    assert "--model" in capsys.readouterr().err
    assert not root.exists() and client.calls == []
    full = _setup(tmp_path)
    before = _tree(full)
    assert cli.main(_args(tmp_path, full, "--use-llm"), environ={}) == 2
    assert _tree(full) == before


def test_provenance_mismatch_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _setup(tmp_path)
    _write_clean(tmp_path)
    client = FakeLLMClient(handler=_llm_handler)
    argv = _args(tmp_path, root, "--use-llm", "--source", "postings")
    assert cli.main([*argv, "--model", "one"], client=client, environ={}) == 0
    capsys.readouterr()
    calls = len(client.calls)
    assert cli.main([*argv, "--model", "two"], client=client, environ={}) == 2
    assert "model" in capsys.readouterr().err and len(client.calls) == calls
    assert cli.main([*argv, "--model", "two", "--allow-mixed"], client=client, environ={}) == 0


def _remove(path: Path) -> None:
    path.unlink()


def _break_canonical_version(root: Path) -> None:
    rows = _jsonl_canonical(root)
    rows[0]["alias_table_version"] = "sha256:stale"
    fileio.write_jsonl_atomic(_canonical(root), rows)


def _canonical(root: Path) -> Path:
    return paths.canonical_dir(root, "resumes") / "resumes_canonical.jsonl"


def _jsonl_canonical(root: Path) -> list[dict]:
    return [json.loads(x) for x in _canonical(root).read_text().splitlines()]


def _break_postings_version(root: Path) -> None:
    path = paths.canonical_dir(root, "postings") / "postings_canonical.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    rows[1]["alias_table_version"] = "sha256:stale"
    fileio.write_jsonl_atomic(path, rows)


def _break_taxonomy_version(root: Path) -> None:
    taxonomy_model.save_taxonomy(
        paths.taxonomy_dir(root) / "taxonomy.json", _tax("sha256:other")
    )


@pytest.mark.parametrize(
    ("break_input", "needles"),
    [
        (lambda r: _remove(paths.taxonomy_dir(r) / "taxonomy.json"),
         ["taxonomy build", "taxonomy.json"]),
        (lambda r: _remove(paths.alias_dir(r) / "alias_table.json"),
         ["alias-table build", "alias_table.json"]),
        (lambda r: _remove(_canonical(r)), ["resumes_canonical.jsonl"]),
        (_break_canonical_version, ["sha256:stale", TABLE.version, "r-1"]),
        (_break_taxonomy_version, ["sha256:other", TABLE.version]),
        (_break_postings_version, ["sha256:stale", "j-2"]),  # resumes is valid: nothing written
    ],
)
def test_missing_and_mismatched_inputs_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture, break_input, needles: list[str]
) -> None:
    root = _setup(tmp_path)
    break_input(root)
    before = _tree(tmp_path)
    assert cli.main(_args(tmp_path, root)) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: ") and all(n in err for n in needles)
    assert _tree(tmp_path) == before
    assert not (root / "relabeled").exists()


def test_default_directories_are_trial_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    args = cli.build_parser().parse_args([])
    assert args.artifacts_dir == Path("artifacts_trial")
    assert args.processed_dir == Path("data/processed")
    assert args.source == "both" and args.use_llm is False and args.min_known == 0.5
    assert args.max_input_chars == 48000 and args.unseen_batch_size == 50
    assert args.allow_mixed is False
    assert cli.main([]) == 2
    assert "artifacts_trial" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("bad", [["--min-known", "1.5"], ["--source", "x"],
                                 ["--unseen-batch-size", "0"], ["--max-input-chars", "-1"]])
def test_invalid_options_are_usage_errors(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(bad)
    assert exc.value.code == 2


def _unseen_skills(request: LLMRequest) -> list[str]:
    block = request.user.split("\n\nSKILLS:\n")[1]
    return [line[2:] for line in block.splitlines()]


def _write_canonical(root: Path, source: str, items: list[tuple[str, list[str]]]) -> None:
    rows = [_record(i, s, source, TABLE.version) for i, s in items]
    fileio.write_jsonl_atomic(
        paths.canonical_dir(root, source) / f"{source}_canonical.jsonl", rows
    )


def test_fatal_error_message_is_redacted(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "sk-very-secret-123"

    def boom(request: LLMRequest) -> str:
        raise FatalLLMError(f"unauthorized, key {secret}", status=401)

    bundle = factory.ClientBundle(
        client=FakeLLMClient(handler=boom), backend="voyager", model="m", secrets=[secret]
    )
    monkeypatch.setattr(cli.factory, "make_client", lambda **kw: bundle)
    root = _setup(tmp_path)
    assert cli.main(_args(tmp_path, root, "--use-llm")) == 3
    err = capsys.readouterr().err
    assert "unauthorized" in err and secret not in err


def test_min_known_option_is_wired(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    root = _setup(tmp_path)  # r-2 has 1 unseen skill of 2: known fraction 0.5
    base = _args(tmp_path, root, "--source", "resumes")
    assert cli.main(base) == 0
    assert _jsonl(root, "resumes")[1]["incomplete_reason"] is None
    assert cli.main([*base, "--min-known", "0.5"]) == 0
    assert _jsonl(root, "resumes")[1]["incomplete_reason"] is None
    assert cli.main([*base, "--min-known", "1.0"]) == 0
    assert _jsonl(root, "resumes")[1]["incomplete_reason"] == "low_coverage"


def _unseen_calls_for(tmp_path: Path, *extra: str) -> list[list[str]]:
    root = _setup(tmp_path)
    _write_canonical(root, "resumes", [("r-1", ["React", "Zig", "Nim"])])
    argv = _args(tmp_path, root, "--use-llm", "--model", "m", "--source", "resumes",
                 "--min-known", "0.0", *extra)
    client = FakeLLMClient(handler=_llm_handler)
    assert cli.main(argv, client=client, environ={}) == 0
    return [_unseen_skills(c) for c in client.calls]


def test_unseen_batch_size_option_is_wired(tmp_path: Path) -> None:
    (tmp_path / "one").mkdir()
    (tmp_path / "default").mkdir()
    assert _unseen_calls_for(tmp_path / "one", "--unseen-batch-size", "1") == [["Nim"], ["Zig"]]
    assert _unseen_calls_for(tmp_path / "default") == [["Nim", "Zig"]]


def test_max_input_chars_option_is_wired(tmp_path: Path) -> None:
    root = _setup(tmp_path)
    _write_clean(tmp_path)
    client = FakeLLMClient(handler=_llm_handler)
    argv = _args(tmp_path, root, "--use-llm", "--model", "m", "--source", "resumes",
                 "--max-input-chars", "6")
    assert cli.main(argv, client=client, environ={}) == 0
    request = next(c for c in client.calls if c.system == reextract.load_reextract_prompt())
    assert request.user.endswith("\n\nTEXT:\nI buil")


def test_fatal_error_in_second_source_keeps_first_and_rerun_resumes(tmp_path: Path) -> None:
    root = _setup(tmp_path)
    _write_clean(tmp_path)
    _write_canonical(root, "postings", [("j-1", ["React", "Nim"])])

    def fatal_on_nim(request: LLMRequest) -> str:
        if request.system == unseen.load_unseen_prompt() and "Nim" in _unseen_skills(request):
            raise FatalLLMError("connection refused", status=None)
        return _llm_handler(request)

    argv = _args(tmp_path, root, "--use-llm", "--model", "m")
    assert cli.main(argv, client=FakeLLMClient(handler=fatal_on_nim), environ={}) == 3
    assert paths.relabeled_dir(root, "resumes").exists()
    assert not paths.relabeled_dir(root, "postings").exists()
    first = _tree(paths.relabeled_dir(root, "resumes"))
    resumed = FakeLLMClient(handler=_llm_handler)
    assert cli.main(argv, client=resumed, environ={}) == 0
    assert [_unseen_skills(c) for c in resumed.calls] == [["Nim"]]  # all else from the cache
    assert _tree(paths.relabeled_dir(root, "resumes")) == first
    assert (paths.relabeled_dir(root, "postings") / "postings_relabeled.jsonl").exists()


def test_null_text_is_a_no_text_failure(tmp_path: Path) -> None:
    root = _setup(tmp_path)
    processed = tmp_path / "processed"
    (processed / "resumes").mkdir(parents=True)
    frame = pd.DataFrame({"id": ["r-1", "r-2", "r-3"], "text_clean": [None, "t", float("nan")]})
    frame.to_parquet(processed / "resumes" / "resumes_clean.parquet", index=False)
    client = FakeLLMClient(handler=_llm_handler)
    argv = _args(tmp_path, root, "--use-llm", "--model", "m", "--source", "resumes")
    assert cli.main(argv, client=client, environ={}) == 1
    failures = paths.relabeled_dir(root, "resumes") / "resumes_relabel_failures.csv"
    rows = failures.read_text().splitlines()[1:]
    assert rows == ["r-1,no_text,no cleaned text found for the document"]
    assert not [c for c in client.calls if c.system == reextract.load_reextract_prompt()]
