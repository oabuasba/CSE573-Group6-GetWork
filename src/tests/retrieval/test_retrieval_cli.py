import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from retrieval_helpers import embed_other_model, make_chain, rebuild_posting_documents

from getwork import paths
from getwork.retrieval import b1plus, cli, inputs, query, run

pytestmark = pytest.mark.usefixtures("offline")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    path = tmp_path / "art"
    make_chain(root=path, n_resumes=6, n_postings=5, dim=16)
    return path


def _run_cli(root: Path, *extra: str) -> list[str]:
    return ["run", "--artifacts-dir", str(root), *extra]


def _files(root: Path) -> list[str]:
    return sorted(p.name for p in paths.retrieval_dir(root).glob("*.parquet"))


def test_run_all_and_b1plus_only(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(_run_cli(root)) == 0
    out = capsys.readouterr()
    summary = json.loads(out.out)
    assert out.err == ""
    assert len(_files(root)) == 6
    assert (paths.retrieval_dir(root) / "retrieval_summary.json").is_file()
    assert summary["systems"] == ["B1+", "B2", "Full"] and summary["k"] == 10
    assert summary["files"].keys() == set(_files(root))

    bare = root.parent / "bare"
    make_chain(root=bare, embed=False)
    assert cli.main(_run_cli(bare, "--systems", "B1+")) == 0
    assert _files(bare) == [
        "topk_b1plus_posting_to_resume.parquet", "topk_b1plus_resume_to_posting.parquet",
    ]
    assert json.loads(capsys.readouterr().out)["embedding"] is None


def test_run_options_take_effect(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(_run_cli(root, "--systems", "B1+,Full", "--k", "2", "--block-size", "1")) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["systems"] == ["B1+", "Full"] and summary["k_effective"] == 2
    assert len(_files(root)) == 4
    table = pq.read_table(run.topk_path(root=root, system="Full", direction="resume_to_posting"))
    assert table.num_rows == 6 * 2

    assert cli.main(_run_cli(root, "--systems", "B1+", "--weights", "0.5,0.25,0.25")) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["weights"] == {"skills": 0.5, "title": 0.25, "level": 0.25}
    path = run.topk_path(root=root, system="B1+", direction="resume_to_posting")
    by_weights = pq.read_table(path).to_pylist()

    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"skills": 0.0, "title": 1.0, "level": 0.0}))
    assert cli.main(_run_cli(root, "--systems", "B1+", "--weights-file", str(weights_file))) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["weights"] == {"skills": 0.0, "title": 1.0, "level": 0.0}
    by_file = pq.read_table(path).to_pylist()
    assert by_file != by_weights

    assert cli.main(_run_cli(root, "--systems", "B1+")) == 0
    assert json.loads(capsys.readouterr().out)["weights"] == b1plus.DEFAULT_WEIGHTS.as_dict()
    assert cli.main(_run_cli(root, "--systems", "Full", "--k", "99")) == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["k"], summary["k_effective"]) == (99, 5)


def test_run_input_errors_exit_2(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    bare = tmp_path / "bare"
    make_chain(root=bare, embed=False)
    stale = tmp_path / "stale"
    tax = make_chain(stale)
    rebuild_posting_documents(root=stale, taxonomy=tax, changed_title="Lead")
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{"skills": 1}')
    cases = [
        (_run_cli(empty), "doc-embed build"),
        (_run_cli(bare), "doc-embed embed"),
        (_run_cli(stale), "doc-embed build"),
        (_run_cli(root, "--k", "0"), "positive"),
        (_run_cli(root, "--k", "x"), "integer"),
        (_run_cli(root, "--block-size", "0"), "positive"),
        (_run_cli(root, "--weights", "0.5,0.5"), "three"),
        (_run_cli(root, "--weights", "0.9,0.9,0.9"), "sum"),
        (_run_cli(root, "--weights-file", str(bad_file)), "bad.json"),
        (_run_cli(root, "--weights-file", str(tmp_path / "none.json")), "none.json"),
        (_run_cli(root, "--weights", "1,0,0", "--weights-file", str(bad_file)), "not allowed"),
        (_run_cli(root, "--systems", "B7"), "B7"),
        (["bogus"], "invalid choice"),
        ([], "required"),
    ]
    for argv, text in cases:
        capsys.readouterr()
        assert cli.main(argv) == 2, argv
        out = capsys.readouterr()
        assert text in out.err, (argv, out.err)
        assert out.out == ""
        for directory in (root, empty, bare, stale):
            assert not paths.retrieval_dir(directory).exists(), argv


def test_default_artifacts_dir_is_trial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run"]) == 2
    assert "doc-embed build" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
    make_chain(tmp_path / "artifacts_trial")
    assert cli.main(["run", "--systems", "B1+"]) == 0
    assert (tmp_path / "artifacts_trial" / "retrieval" / "retrieval_summary.json").is_file()
    assert cli.build_parser().parse_args(["run"]).artifacts_dir == Path("artifacts_trial")


def _show_args(root: Path, *extra: str) -> list[str]:
    return ["show", "--artifacts-dir", str(root), *extra]


def test_show_prints_matches(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = inputs.load_inputs(root)
    argv = _show_args(root, "--direction", "resume_to_posting", "--id", "r-2")
    assert cli.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    resume = data.resumes["r-2"]
    assert lines[0] == f"query r-2  [{resume['level']}]  {resume['title']}"
    assert lines[1] == f"  skills: {', '.join(resume['skills'])}"
    expected = query.query_matches(
        inputs=data, system="Full", direction="resume_to_posting", query_id="r-2", k=5
    )
    assert len(lines) == 2 + 5
    for rank, (line, (doc_id, score)) in enumerate(zip(lines[2:], expected, strict=True), 1):
        doc = data.postings[doc_id]
        head = f"{rank:>3}  {score:.4f}  {doc_id}  [{doc['level']}]  {doc['title']}  | shared: "
        assert line.startswith(head)
        wanted = b1plus.skill_keys(resume["skills"])
        shared = [s for s in doc["skills"] if b1plus.skill_keys([s]) & wanted]
        assert line == head + ", ".join(shared)

    for system in ("B1+", "B2"):
        argv = _show_args(
            root, "--direction", "posting_to_resume", "--id", "p-1", "--system", system, "--k", "2"
        )
        assert cli.main(argv) == 0
        lines = capsys.readouterr().out.splitlines()
        want = query.query_matches(
            inputs=data, system=system, direction="posting_to_resume", query_id="p-1", k=2
        )
        assert len(lines) == 2 + 2 and lines[0].startswith("query p-1")
        assert [line.split()[2] for line in lines[2:]] == [d for d, _ in want]


def test_shared_skills_capped_and_use_match_spelling() -> None:
    assert cli.MAX_SHARED_SKILLS == 8
    resume = {"skills": [f"S{i}" for i in range(12)]}
    order = [11, 3, 5, "x", 0, 1, 2, 4, 6, 7, 8]
    match = {"skills": [f"s{i}" for i in order]}
    assert cli._shared_skills(query_doc=resume, match_doc=match) == [
        "s11", "s3", "s5", "s0", "s1", "s2", "s4", "s6",
    ]
    assert cli._shared_skills(query_doc={"skills": []}, match_doc=match) == []


def test_show_errors_exit_2(root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = ["--direction", "resume_to_posting"]
    cases = [
        (_show_args(root, *base, "--id", "r-99"), "r-99"),
        (_show_args(root, *base, "--id", "p-1"), "p-1"),
        (_show_args(root, "--direction", "sideways", "--id", "r-1"), "invalid choice"),
        (_show_args(root, *base, "--id", "r-1", "--system", "Nope"), "invalid choice"),
        (_show_args(root, *base), "--id"),
        (_show_args(root, *base, "--id", "r-1", "--k", "0"), "positive"),
        (_show_args(tmp_path / "nothing", *base, "--id", "r-1"), "doc-embed build"),
    ]
    bare = tmp_path / "bare"
    make_chain(root=bare, embed=False)
    cases.append((_show_args(bare, *base, "--id", "r-1"), "doc-embed embed"))
    for argv, text in cases:
        capsys.readouterr()
        assert cli.main(argv) == 2, argv
        out = capsys.readouterr()
        assert text in out.err and out.out == "", (argv, out.err)


def test_run_embedding_mismatch_exit_2(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    embed_other_model(root=root, source="postings")
    assert cli.main(_run_cli(root)) == 2
    out = capsys.readouterr()
    assert "model" in out.err and out.out == ""
    assert not paths.retrieval_dir(root).exists()
    assert cli.main(_run_cli(root, "--systems", "B1+")) == 0


def test_show_b1plus_without_vectors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bare = tmp_path / "bare"
    make_chain(root=bare, embed=False)
    argv = _show_args(bare, "--direction", "resume_to_posting", "--id", "r-1", "--system", "B1+")
    assert cli.main(argv) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2 + 5
    full = _show_args(bare, "--direction", "resume_to_posting", "--id", "r-1")
    assert cli.main(full) == 2
    assert "doc-embed embed" in capsys.readouterr().err


def test_show_weights_options(
    root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = inputs.load_inputs(root=root, systems=["B1+"])
    base = ["--direction", "resume_to_posting", "--id", "r-2", "--system", "B1+", "--k", "5"]
    scores = {}
    weights_file = tmp_path / "w.json"
    weights_file.write_text(json.dumps({"skills": 0.0, "title": 1.0, "level": 0.0}))
    for name, extra in (
        ("default", []),
        ("inline", ["--weights", "0,1,0"]),
        ("file", ["--weights-file", str(weights_file)]),
    ):
        assert cli.main(_show_args(root, *base, *extra)) == 0
        lines = capsys.readouterr().out.splitlines()[2:]
        scores[name] = [line.split()[1] for line in lines]
    chosen = b1plus.B1PlusWeights(0.0, 1.0, 0.0)
    expected = query.query_matches(
        inputs=data, system="B1+", direction="resume_to_posting", query_id="r-2", k=5,
        weights=chosen,
    )
    assert scores["inline"] == scores["file"] == [f"{s:.4f}" for _, s in expected]
    assert scores["default"] != scores["inline"]
    assert cli.main(_show_args(root, *base, "--weights", "0.5,0.5")) == 2
    assert "three" in capsys.readouterr().err


def test_show_caps_shared_skills_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bare = tmp_path / "bare"
    make_chain(root=bare, embed=False)
    skills = [f"Skill{i}" for i in range(12)]
    for source, doc_id in (("resumes", "r-1"), ("postings", "p-1")):
        path = paths.documents_dir(bare, source) / f"{source}_documents.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["id"] == doc_id:
                row["skills"] = skills if source == "resumes" else skills[::-1]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    argv = _show_args(
        bare, "--direction", "resume_to_posting", "--id", "r-1", "--system", "B1+", "--k", "1"
    )
    assert cli.main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == f"  skills: {', '.join(skills)}"
    assert "p-1" in lines[2]
    shared = lines[2].split("| shared: ")[1].split(", ")
    assert shared == skills[::-1][:8]
