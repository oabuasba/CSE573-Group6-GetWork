import json
from pathlib import Path

import pytest

from getwork import fileio
from getwork.alias import build, cli
from getwork.llm.client import FatalLLMError, LLMRequest
from getwork.llm.fake import FakeLLMClient
from getwork.structure import schema

pytestmark = pytest.mark.usefixtures("offline")

SKILLS = {
    "resumes": [["JS", "Java", "java"], ["Java", "Rust"]],
    "postings": [["js", "JavaScript"], ["Rust", "Go"]],
}


def _record(id_: str, skills: list[str], source: str) -> dict:
    output = schema.ModelOutput(
        title="Dev", skills=skills, experience=[], education=[], years_experience=None
    )
    record = schema.to_record(output, record_id=id_, source=source, model="m", prompt_version="p")
    record["skills"] = list(skills)
    return record


def _write_structured(structured: Path) -> None:
    for source, lists in SKILLS.items():
        rows = [_record(f"{source[0]}-{i}", skills, source) for i, skills in enumerate(lists)]
        fileio.write_jsonl_atomic(structured / source / f"{source}_structured.jsonl", rows)


def _dirs(tmp_path: Path) -> list[str]:
    return ["--structured-dir", str(tmp_path / "s"), "--artifacts-dir", str(tmp_path / "a")]


def _built(tmp_path: Path) -> list[str]:
    _write_structured(tmp_path / "s")
    assert cli.main(["build", *_dirs(tmp_path)]) == 0
    return _dirs(tmp_path)


def _tree(root: Path) -> dict[str, bytes]:
    return {str(p): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_build_writes_table_and_report(tmp_path: Path) -> None:
    _write_structured(tmp_path / "s")
    assert cli.main(["build", *_dirs(tmp_path)]) == 0
    alias_dir = tmp_path / "a" / "alias"
    table = json.loads((alias_dir / "alias_table.json").read_text())
    assert table["entries"]["js"] == "JavaScript"
    assert table["entries"]["java"] == "Java"
    assert table["meta"]["n_records"] == {"resumes": 2, "postings": 2}
    assert str(tmp_path) not in json.dumps(table["meta"])
    report = json.loads((alias_dir / "alias_build_report.json").read_text())
    assert report["alias_table_version"] == table["version"]

    other = tmp_path / "other.csv"
    other.write_text("alias,canonical\nrs,Rust\n", encoding="utf-8")
    assert cli.main(["build", *_dirs(tmp_path), "--curated", str(other)]) == 0
    entries = json.loads((alias_dir / "alias_table.json").read_text())["entries"]
    assert entries["rs"] == "Rust" and "k8s" not in entries


def test_build_missing_input_exits_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_structured(tmp_path / "s")
    missing = tmp_path / "s" / "postings" / "postings_structured.jsonl"
    missing.unlink()
    assert cli.main(["build", *_dirs(tmp_path)]) == 2
    assert str(missing) in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_build_conflicting_curated_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_structured(tmp_path / "s")
    bad = tmp_path / "bad.csv"
    bad.write_text("alias,canonical\njs,JavaScript\njs,Java\n", encoding="utf-8")
    assert cli.main(["build", *_dirs(tmp_path), "--curated", str(bad)]) == 2
    assert "js" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_apply_writes_both_sources(tmp_path: Path) -> None:
    dirs = _built(tmp_path)
    assert cli.main(["apply", *dirs]) == 0
    version = json.loads((tmp_path / "a/alias/alias_table.json").read_text())["version"]
    for source in SKILLS:
        out = tmp_path / "a" / "canonical" / source
        assert (out / f"{source}_canonical.parquet").exists()
        assert (out / f"{source}_canonical_summary.json").exists()
        lines = (out / f"{source}_canonical.jsonl").read_text().splitlines()
        records = [json.loads(line) for line in lines]
        assert {r["alias_table_version"] for r in records} == {version}
    path = tmp_path / "a/canonical/resumes/resumes_canonical.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    assert first["skills"] == ["JavaScript", "Java"]
    assert first["skills_raw"] == ["JS", "Java", "java"]


def test_apply_is_idempotent(tmp_path: Path) -> None:
    dirs = _built(tmp_path)
    assert cli.main(["apply", *dirs]) == 0
    before = _tree(tmp_path / "a" / "canonical")
    assert cli.main(["apply", *dirs]) == 0
    assert _tree(tmp_path / "a" / "canonical") == before and len(before) == 6


def test_apply_without_table_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_structured(tmp_path / "s")
    assert cli.main(["apply", *_dirs(tmp_path)]) == 2
    assert "alias-table build" in capsys.readouterr().err
    assert not (tmp_path / "a" / "canonical").exists()


def test_apply_corrupt_input_exits_2_and_leaves_input(tmp_path: Path) -> None:
    dirs = _built(tmp_path)
    resumes = tmp_path / "s" / "resumes" / "resumes_structured.jsonl"
    postings = tmp_path / "s" / "postings" / "postings_structured.jsonl"

    resumes.write_bytes(resumes.read_bytes() + b"{not json}\n")
    before = resumes.read_bytes()
    assert cli.main(["apply", *dirs]) == 2
    assert resumes.read_bytes() == before

    _write_structured(tmp_path / "s")
    bad = {"id": "p-9", "source": "postings", "surprise": 1}
    postings.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    before = postings.read_bytes()
    assert cli.main(["apply", *dirs]) == 2
    assert postings.read_bytes() == before
    assert not (tmp_path / "a" / "canonical").exists()


def _pairing_handler(request: LLMRequest) -> str:
    block = request.user.split("BATCH:\n")[1].split("\n\nREFERENCE:")[0]
    batch = [line[2:] for line in block.splitlines()]
    return json.dumps({"groups": [{"canonical": batch[0], "aliases": [batch[1]]}]})


def _propose_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["propose", *_dirs(tmp_path), "--model", "m", *extra]


def test_propose_writes_draft_and_leaves_curated(tmp_path: Path) -> None:
    dirs = _built(tmp_path)
    curated_before = build.CURATED_PATH.read_bytes()
    client = FakeLLMClient(handler=_pairing_handler)
    assert cli.main(["propose", *dirs, "--model", "m"], client=client) == 0
    lines = (tmp_path / "a/alias/alias_draft.csv").read_text().splitlines()
    assert lines[0] == ",".join(cli.DRAFT_COLUMNS)
    assert lines[0].startswith("alias,canonical,alias_count,canonical_count,batch,status,")
    assert len(lines) == 2 and lines[1].endswith(",new,m,voyager,alias_v1")
    summary = json.loads((tmp_path / "a/alias/alias_draft_summary.json").read_text())
    assert summary["n_rows"] == 1 and summary["failed_batches"] == []
    assert set(summary) == {
        "n_rows", "failed_batches", "calls", "retries", "rate_limited", "model", "backend"
    }
    assert build.CURATED_PATH.read_bytes() == curated_before


def test_propose_exit_codes(tmp_path: Path) -> None:
    dirs = _built(tmp_path)
    bad = FakeLLMClient(handler=lambda request: "not json")
    assert cli.main(["propose", *dirs, "--model", "m"], client=bad) == 1
    summary = json.loads((tmp_path / "a/alias/alias_draft_summary.json").read_text())
    assert summary["failed_batches"] == [0]
    fatal = FakeLLMClient(script=[FatalLLMError("denied", status=401)])
    assert cli.main(["propose", *dirs, "--model", "m"], client=fatal) == 3


def test_propose_requires_model_and_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeLLMClient(handler=_pairing_handler)
    argv = ["propose", *_dirs(tmp_path)]
    assert cli.main(argv, client=client) == 2
    assert "--model" in capsys.readouterr().err

    env_file = str(tmp_path / "none.env")
    assert cli.main([*argv, "--env-file", env_file], environ={}) == 2
    err = capsys.readouterr().err
    assert "VOYAGER_API_KEY" in err
    assert not (tmp_path / "a").exists()


def test_propose_conflict_status_and_backend_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = _built(tmp_path)
    client = FakeLLMClient(handler=_pairing_handler)
    assert cli.main(["propose", *dirs, "--model", "m", "--backend", "local"], client=client) == 0
    rows = (tmp_path / "a/alias/alias_draft.csv").read_text().splitlines()[1:]
    assert rows and all(row.endswith(",new,m,local,alias_v1") for row in rows)

    # An alias can only repeat across batches if the skill list repeats it, so feed rows directly.
    def row(canonical: str, batch: int, status: str) -> dict:
        values = ["x", canonical, 1, 2, batch, status, "m", "local", "p"]
        return dict(zip(cli.DRAFT_COLUMNS, values))

    fake_rows = [row("A", 0, "new"), row("B", 1, "conflict")]
    result = cli.propose.ProposeResult(rows=fake_rows, failed_batches=[])
    monkeypatch.setattr(cli.propose, "propose_aliases", lambda *a, **k: result)
    argv = ["propose", *dirs, "--model", "m", "--backend", "local"]
    assert cli.main(argv, client=client) == 0
    lines = (tmp_path / "a/alias/alias_draft.csv").read_text().splitlines()
    assert [line.split(",")[5] for line in lines[1:]] == ["new", "conflict"]


def test_default_directories_are_trial_directories() -> None:
    args = cli.build_parser().parse_args(["build"])
    assert args.structured_dir == Path("data/structured_trial")
    assert args.artifacts_dir == Path("artifacts_trial")


def test_default_run_in_empty_cwd_exits_2_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["build"]) == 2
    assert "data/structured_trial" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
