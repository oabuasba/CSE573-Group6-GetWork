from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from getwork import fileio
from getwork.alias import apply
from getwork.embed import store
from getwork.embed.fake import FakeEmbeddingClient
from getwork.llm import factory as llm_factory
from getwork.llm.client import FatalLLMError, LLMRequest
from getwork.llm.fake import FakeLLMClient
from getwork.taxonomy import build, cli, cli_name, fit, model, naming, overlap, report

pytestmark = pytest.mark.usefixtures("offline")

VERSION = "sha256:bbbbbbbbbbbb"
THEMES = ("frontend", "database", "cloud")
SKILLS = [f"{theme} item{n:02d}" for theme in THEMES for n in range(20)]
N_DOCS = 40
SKILLS_PER_DOC = 6
# one kmeans entry; the other methods cannot produce an eligible configuration
ALLOW = "--allow-member-names"
GRID_FILE = {
    "dbscan_eps": [0.001], "dbscan_min_samples": [5],
    "hdbscan_min_cluster_size": [100], "kmeans_k": [3],
}


def _reducer(vectors: np.ndarray, config: object) -> np.ndarray:
    return vectors[:, :3]


def _write_canonical(root: Path, skill_list: list[str] = SKILLS) -> None:
    for source, offset in (("resumes", 0), ("postings", N_DOCS // 2)):
        rows = []
        for j in range(offset, offset + N_DOCS // 2):
            skills = [skill_list[(j * 3 + k) % len(skill_list)] for k in range(SKILLS_PER_DOC)]
            row = dict.fromkeys(apply.CANONICAL_KEYS)
            row.update(id=f"{source}-{j}", source=source, skills=skills, skills_raw=skills,
                       alias_table_version=VERSION)
            rows.append(row)
        path = root / "canonical" / source / f"{source}_canonical.jsonl"
        fileio.write_jsonl_atomic(path, rows)


def _run(root: Path, *args: str, **kwargs: object) -> int:
    kwargs.setdefault("reduce_fn", _reducer)
    return cli.main([args[0], "--artifacts-dir", str(root), *args[1:]], **kwargs)  # type: ignore


def _through_grid(root: Path) -> None:
    _write_canonical(root)
    assert _run(root, "counts") == 0
    assert _run(root, "embed", embed_client=FakeEmbeddingClient(dim=16)) == 0
    grid_file = root.parent / f"{root.name}-grid.json"
    grid_file.write_text(json.dumps(GRID_FILE))
    assert _run(root, "grid", "--grid-file", str(grid_file)) == 0


def _members(root: Path) -> dict[int, list[str]]:
    out = root / "taxonomy"
    selection = report.read_selection(out)
    results, matrix = report.read_grid(out)
    row = [r.config_id for r in results].index(selection["config_id"])
    skills = [s for s in json.loads((out.parent / "skills" / "skill_vectors_ids.json").read_text())]
    members: dict[int, list[str]] = {}
    for skill, label in zip(skills, matrix[row]):
        if label >= 0:
            members.setdefault(int(label), []).append(skill)
    return members


def _names_rows(root: Path) -> list[dict[str, str]]:
    with open(root / "taxonomy" / "cluster_names.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_name_heuristic_writes_csv(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    members = _members(root)
    assert sorted(members) == [0, 1, 2]
    capsys.readouterr()
    assert _run(root, "name") == 0
    summary = json.loads(capsys.readouterr().out)
    assert (summary["n_clusters"], summary["n_fallback"], summary["method"]) == (3, 0, "heuristic")
    assert summary["n_member_list"] == 3
    path = root / "taxonomy" / "cluster_names.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(build.NAMES_COLUMNS)
    rows = _names_rows(root)
    assert [int(r["cluster_id"]) for r in rows] == [0, 1, 2]
    for row in rows:
        skills = members[int(row["cluster_id"])]
        assert row["name"] == naming.heuristic_name(skills)
        assert (row["size"], row["method"], row["status"]) == (
            str(len(skills)), "heuristic", "member_list")
        assert row["top_members"] == "; ".join(skills[:5])
        assert (row["model"], row["backend"], row["prompt_version"]) == ("", "", "")


def test_name_refuses_overwrite_without_force(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    assert _run(root, "name") == 0
    path = root / "taxonomy" / "cluster_names.csv"
    path.write_text("cluster_id,name\n0,My edit\n", encoding="utf-8")
    before = path.read_bytes()
    assert _run(root, "name") == 2
    assert path.read_bytes() == before
    assert _run(root, "name", "--force") == 0
    assert path.read_bytes() != before and len(_names_rows(root)) == 3


def _llm_handler(names_by_first_member: dict[str, str]):  # type: ignore[no-untyped-def]
    def handler(request: LLMRequest) -> str:
        first = request.user.splitlines()[1].removeprefix("- ")
        return json.dumps({"name": names_by_first_member[first]})

    return handler


def test_name_llm_columns_fallback_and_duplicates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    members = _members(root)
    wanted = {members[0][0]: "Frontend Tooling", members[1][0]: "frontend tooling",
              members[2][0]: "Other"}
    client = FakeLLMClient(handler=_llm_handler(wanted))
    capsys.readouterr()
    code = _run(root, "name", "--method", "llm", "--model", "m-1", "--backend", "local",
                "--fit-rounds", "0", llm_client=client)
    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert (printed["n_fallback"], printed["n_duplicate"], printed["n_clusters"]) == (2, 0, 3)
    assert printed["n_member_list"] == 2
    rows = _names_rows(root)
    assert [(r["method"], r["status"]) for r in rows] == [
        ("llm", "ok"), ("heuristic_fallback", "member_list"), ("heuristic_fallback", "member_list"),
    ]
    assert rows[0]["name"] == "Frontend Tooling"
    for i in (1, 2):
        assert rows[i]["name"] == naming.heuristic_name(members[i])
    assert (rows[0]["model"], rows[0]["backend"], rows[0]["prompt_version"]) == (
        "m-1", "local", "name_v2")
    for row in rows[1:]:
        assert (row["model"], row["backend"], row["prompt_version"]) == ("", "", "")
    assert {c.model for c in client.calls} == {"m-1"}


def test_name_llm_member_list_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    members = _members(root)
    listed = " / ".join(members[0][:3])
    wanted = {members[0][0]: listed, members[1][0]: "Data Stores", members[2][0]: "Cloud Hosting"}
    client = FakeLLMClient(handler=_llm_handler(wanted))
    capsys.readouterr()
    assert _run(root, "name", "--method", "llm", "--model", "m", "--overlap-rounds", "0",
                "--fit-rounds", "0",
                llm_client=client) == 1
    printed = json.loads(capsys.readouterr().out)
    assert (printed["n_fallback"], printed["n_member_list"]) == (1, 1)
    rows = _names_rows(root)
    assert [(r["method"], r["status"]) for r in rows] == [
        ("heuristic_fallback", "member_list"), ("llm", "ok"), ("llm", "ok"),
    ]
    assert rows[0]["name"] == naming.heuristic_name(members[0])


def test_name_llm_all_succeed_exit_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    members = _members(root)
    wanted = {members[i][0]: f"Group {i}" for i in range(3)}
    capsys.readouterr()
    code = _run(root, "name", "--method", "llm", "--model", "m",
                "--overlap-rounds", "0", "--fit-rounds", "0", "--top-members", "2",
                llm_client=FakeLLMClient(handler=_llm_handler(wanted)))
    assert code == 0
    assert {r["status"] for r in _names_rows(root)} == {"ok"}
    assert json.loads(capsys.readouterr().out)["n_member_list"] == 0
    assert [(r["name"], r["method"]) for r in _names_rows(root)] == [
        (f"Group {i}", "llm") for i in range(3)
    ]


def test_name_llm_requires_model_before_reading_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    client = FakeLLMClient(handler=lambda r: '{"name": "x"}')
    assert _run(root, "name", "--method", "llm", llm_client=client) == 2
    err = capsys.readouterr().err
    assert "--model" in err and "selection.json" not in err and client.calls == []
    args = ["name", "--method", "llm", "--env-file", str(tmp_path / "none.env")]
    assert _run(root, *args, environ={}) == 2
    assert "selection.json" not in capsys.readouterr().err
    assert not root.exists()


def test_name_llm_fatal_exit_3_redacted_and_nothing_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    client = FakeLLMClient(script=[FatalLLMError("denied sk-SECRET")])
    bundle = llm_factory.ClientBundle(client=client, backend="voyager", model="m",
                                      secrets=["sk-SECRET"])
    monkeypatch.setattr(llm_factory, "make_client", lambda **kw: bundle)
    capsys.readouterr()
    assert _run(root, "name", "--method", "llm", "--model", "m") == 3
    err = capsys.readouterr().err
    assert "sk-SECRET" not in err and "denied" in err
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_name_without_selection_exit_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    assert _run(root, "name") == 2
    assert "taxonomy grid" in capsys.readouterr().err
    assert not root.exists()


def _name_all(root: Path) -> Path:
    _through_grid(root)
    assert _run(root, "name") == 0
    return root / "taxonomy" / "cluster_names.csv"


def test_build_writes_loadable_taxonomy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    names_path = _name_all(root)
    text = names_path.read_text(encoding="utf-8").replace(
        naming.heuristic_name(_members(root)[0]), "Edited Name", 1
    )
    names_path.write_text(text, encoding="utf-8")
    capsys.readouterr()
    assert _run(root, "build", ALLOW) == 0
    summary = json.loads(capsys.readouterr().out)
    target = root / "taxonomy" / "taxonomy.json"
    tax = model.load_taxonomy(target)
    assert summary["version"] == tax.version and summary["n_clusters"] == len(tax.clusters) == 3
    assert tax.alias_table_version == VERSION
    assert tax.cluster_names()[0] == "Edited Name"
    selection = report.read_selection(root / "taxonomy")
    vector_meta = json.loads((root / "skills" / "skill_vectors_meta.json").read_text())
    assert tax.meta["selection"] == selection
    assert tax.meta["embedding"] == vector_meta["embedding"]
    assert tax.meta["names_file_sha256"] == hashlib.sha256(names_path.read_bytes()).hexdigest()
    members = _members(root)
    for cluster in tax.clusters:
        assert list(cluster.skills) == members[cluster.id]
    ranked = json.loads((root / "skills" / "skill_vectors_ids.json").read_text())
    assert all(tax.lookup(skill) is not None for skill in ranked)


def test_build_accepts_excel_bom_and_custom_names_path(tmp_path: Path) -> None:
    root = tmp_path / "art"
    names_path = _name_all(root)
    custom = tmp_path / "edited.csv"
    custom.write_bytes(b"\xef\xbb\xbf" + names_path.read_bytes())
    names_path.unlink()
    assert _run(root, "build", ALLOW) == 2
    assert _run(root, "build", ALLOW, "--names", str(custom)) == 0
    assert model.load_taxonomy(root / "taxonomy" / "taxonomy.json").cluster_names()


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda rows: rows[1].update(name=rows[0]["name"].upper()), "duplicate"),
        (lambda rows: rows[0].update(name="A, B"), "cluster ids [0]"),
        (lambda rows: rows[2].update(name="Other"), "cluster ids [2]"),
        (lambda rows: rows.pop(), "no name for cluster ids [2]"),
    ],
)
def test_build_rejects_bad_names_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], edit: object, message: str
) -> None:
    root = tmp_path / "art"
    names_path = _name_all(root)
    rows = _names_rows(root)
    edit(rows)  # type: ignore[operator]
    with open(names_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build.NAMES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    capsys.readouterr()
    assert _run(root, "build", ALLOW) == 2
    assert message in capsys.readouterr().err
    assert not (root / "taxonomy" / "taxonomy.json").exists()


def test_build_failure_keeps_existing_taxonomy(tmp_path: Path) -> None:
    root = tmp_path / "art"
    names_path = _name_all(root)
    assert _run(root, "build", ALLOW) == 0
    target = root / "taxonomy" / "taxonomy.json"
    before = target.read_bytes()
    names_path.write_text("cluster_id,name\n0,Only: one\n", encoding="utf-8")
    assert _run(root, "build", ALLOW) == 2
    assert target.read_bytes() == before


def test_stale_selection_after_failed_grid_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _name_all(root)
    all_noise = tmp_path / "noise.json"
    all_noise.write_text(json.dumps({**GRID_FILE, "kmeans_k": [100]}))
    assert _run(root, "grid", "--grid-file", str(all_noise)) == 2
    assert (root / "taxonomy" / "selection.json").exists()
    capsys.readouterr()
    assert _run(root, "build", ALLOW) == 2
    assert "does not belong to the current grid" in capsys.readouterr().err
    assert not (root / "taxonomy" / "taxonomy.json").exists()
    assert _run(root, "name", "--force") == 2
    assert "does not belong to the current grid" in capsys.readouterr().err


def test_pipeline_end_to_end_is_deterministic(tmp_path: Path) -> None:
    outputs = []
    for label in ("one", "two"):
        root = tmp_path / label / "art"
        _through_grid(root)
        assert _run(root, "name") == 0
        assert _run(root, "build", ALLOW) == 0
        outputs.append((root / "taxonomy" / "taxonomy.json").read_bytes())
        tax = model.load_taxonomy(root / "taxonomy" / "taxonomy.json")
        assert all(tax.lookup(skill) is not None for skill in SKILLS)
        assert tax.alias_table_version == VERSION
    assert outputs[0] == outputs[1]


def test_name_llm_top_members_and_extra_body_reach_the_request(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    members = _members(root)
    wanted = {members[i][0]: f"Group {i}" for i in range(3)}
    client = FakeLLMClient(handler=_llm_handler(wanted))
    code = _run(root, "name", "--method", "llm", "--model", "m", "--top-members", "2",
                "--overlap-rounds", "0", "--fit-rounds", "0", "--extra-body", '{"thinking": false}',
                llm_client=client)
    assert code == 0 and len(client.calls) == 3
    for call in client.calls:
        assert call.user.split("\n\nUSED NAMES:")[0].count("\n- ") == 2
        assert call.extra_body == {"thinking": False}
        assert call.temperature == 0
    assert _run(root, "name", "--method", "llm", "--model", "m", "--force",
                "--extra-body", "[1]", llm_client=client) == 2
    assert len(client.calls) == 3


def test_build_uses_alias_version_from_counts_summary_not_selection(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _name_all(root)
    summary_path = root / "skills" / "counts_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["alias_table_version"] = "sha256:cccccccccccc"
    summary_path.write_text(json.dumps(summary))
    selection = report.read_selection(root / "taxonomy")
    assert selection["context"]["alias_table_version"] == VERSION
    assert _run(root, "build", ALLOW) == 0
    tax = model.load_taxonomy(root / "taxonomy" / "taxonomy.json")
    assert tax.alias_table_version == "sha256:cccccccccccc"


def test_build_refuses_vectors_from_another_selected_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _name_all(root)
    vectors = store.read_vectors(root / "skills", "skill_vectors")
    meta = {**vectors.meta, "selected_skills_version": "sha256:000000000000"}
    store.write_vectors(root / "skills", "skill_vectors", vectors.ids, vectors.vectors, meta)
    capsys.readouterr()
    assert _run(root, "build", ALLOW) == 2
    assert "stale" in capsys.readouterr().err
    assert not (root / "taxonomy" / "taxonomy.json").exists()


def test_name_and_build_refuse_selection_for_a_different_skill_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    names_path = _name_all(root)
    names_before = names_path.read_bytes()
    _write_canonical(root, [skill.replace("item", "thing") for skill in SKILLS])
    assert _run(root, "counts") == 0
    assert _run(root, "embed", embed_client=FakeEmbeddingClient(dim=16)) == 0
    capsys.readouterr()
    assert _run(root, "name", "--force") == 2
    assert "taxonomy grid again" in capsys.readouterr().err
    assert _run(root, "build", ALLOW) == 2
    assert "taxonomy grid again" in capsys.readouterr().err
    assert names_path.read_bytes() == names_before
    assert not (root / "taxonomy" / "taxonomy.json").exists()


def _write_names(root: Path, names: dict[int, str]) -> None:
    rows = _names_rows(root)
    for row in rows:
        row["name"] = names[int(row["cluster_id"])]
    with open(root / "taxonomy" / "cluster_names.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build.NAMES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_refuses_member_list_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _name_all(root)
    heuristic = [r["name"] for r in _names_rows(root)]
    capsys.readouterr()
    assert _run(root, "build") == 2
    err = capsys.readouterr().err
    assert "cluster ids [0, 1, 2]" in err and "member skills" in err
    assert not any(name in err for name in heuristic)
    assert not (root / "taxonomy" / "taxonomy.json").exists()


def test_build_allow_member_names_flag(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _name_all(root)
    assert _run(root, "build", ALLOW) == 0
    assert (root / "taxonomy" / "taxonomy.json").exists()


def test_build_accepts_concept_and_single_member_names(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _name_all(root)
    members = _members(root)
    _write_names(root, {0: "Frontend Work", 1: members[1][0], 2: "Cloud Hosting"})
    assert _run(root, "build") == 0
    tax = model.load_taxonomy(root / "taxonomy" / "taxonomy.json")
    assert set(tax.cluster_names()) == {"Frontend Work", members[1][0], "Cloud Hosting"}


def _llm_run(root: Path, *extra: str, client: FakeLLMClient) -> int:
    # later options win; review tests pass their own
    defaults = ["--overlap-rounds", "0", "--fit-rounds", "0"]
    return _run(root, "name", "--method", "llm", "--model", "m", *defaults, *extra,
                llm_client=client)


def _own_name_client(root: Path) -> FakeLLMClient:
    wanted = {skills[0]: f"Concept {cid}" for cid, skills in _members(root).items()}
    return FakeLLMClient(handler=_llm_handler(wanted))


def test_name_workers_same_file_and_progress_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files, outputs = [], []
    for workers in ("2", "3"):
        root = tmp_path / f"art{workers}"
        _through_grid(root)
        capsys.readouterr()
        assert _llm_run(root, "--workers", workers, client=_own_name_client(root)) == 0
        out = capsys.readouterr()
        files.append((root / "taxonomy" / "cluster_names.csv").read_bytes())
        outputs.append(out)
    assert files[0] == files[1]
    assert len(outputs[1].out.splitlines()) == 1
    summary = json.loads(outputs[1].out)
    assert set(summary) == {"names_file", "method", "n_clusters", "n_fallback", "n_duplicate",
                            "n_member_list", "n_renamed", "n_overlap",
                            "n_fit_renamed", "n_weak_fit"}
    lines = outputs[1].err.splitlines()
    assert [line.split(" -> ")[0] for line in lines] == [
        f"named {i + 1}/3: cluster {i}" for i in range(3)
    ]
    assert all(line.endswith(" (attempts 1)") and '"Concept ' in line for line in lines)


def test_name_without_workers_sends_used_names(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    client = _own_name_client(root)
    assert _llm_run(root, client=client) == 0
    assert "USED NAMES" not in client.calls[0].user
    assert all("USED NAMES" in call.user for call in client.calls[1:])


def test_name_workers_invalid_exit_2(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    for bad in ("0", "x"):
        client = _own_name_client(root)
        assert _llm_run(root, "--workers", bad, client=client) == 2
        assert client.calls == []
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_name_workers_ignored_for_heuristic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    capsys.readouterr()
    assert _run(root, "name", "--workers", "3") == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["method"] == "heuristic" and "named" not in out.err


def test_name_fatal_error_still_exit_3_with_workers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)

    def handler(request: LLMRequest) -> str:
        raise FatalLLMError("denied")

    capsys.readouterr()
    assert _llm_run(root, "--workers", "2", client=FakeLLMClient(handler=handler)) == 3
    assert "denied" in capsys.readouterr().err
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_stderr_log_redacts_secrets(capsys: pytest.CaptureFixture[str]) -> None:
    cli_name._stderr_log(["sk-test-secret"])("named 1/1: cluster 0 -> sk-test-secret")
    out = capsys.readouterr()
    assert out.out == "" and out.err == "named 1/1: cluster 0 -> ***\n"


def test_name_rejection_lines_are_redacted_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    reply = json.dumps({"name": "sk-SECRET one two three four five"})
    client = FakeLLMClient(handler=lambda r: reply)
    bundle = llm_factory.ClientBundle(client=client, backend="voyager", model="m",
                                      secrets=["sk-SECRET"])
    monkeypatch.setattr(llm_factory, "make_client", lambda **kw: bundle)
    capsys.readouterr()
    assert _run(root, "name", "--method", "llm", "--model", "m", "--workers", "2") == 1
    err = capsys.readouterr().err
    assert "sk-SECRET" not in err and "rejected cluster 0 attempt 1: '***" in err


def _request_fields(tmp_path: Path, *extra: str) -> tuple[int, int, object, int]:
    root = tmp_path / "art"
    _through_grid(root)
    client = _own_name_client(root)
    code = _llm_run(root, *extra, client=client)
    call = client.calls[0]
    return code, call.max_tokens, call.reasoning_effort, len(client.calls)


def test_name_request_defaults_and_overrides(tmp_path: Path) -> None:
    assert _request_fields(tmp_path / "a") == (0, 1024, "none", 3)
    assert _request_fields(tmp_path / "b", "--max-tokens", "77", "--reasoning-effort", "low") == (
        0, 77, "low", 3,
    )
    assert _request_fields(tmp_path / "c", "--reasoning-effort", "omit")[2] is None


def test_name_reasoning_effort_default_follows_backend(tmp_path: Path) -> None:
    local = _request_fields(tmp_path / "a", "--backend", "local")
    assert local[2] is None
    voyager = _request_fields(tmp_path / "b", "--backend", "voyager")
    assert voyager[2] == "none"


def test_name_max_tokens_and_reasoning_validated(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    for bad in (("--max-tokens", "0"), ("--max-tokens", "x"), ("--reasoning-effort", "max")):
        client = _own_name_client(root)
        assert _llm_run(root, *bad, client=client) == 2 and client.calls == []


NAMED = ["Frontend Tools", "Frontend Tools Suite", "Cloud Platforms"]
RENAMED = {0: "Interface Engineering", 1: "Data Storage"}


def _review_client(
    root: Path, *, renames: dict[int, str] | None = None, fatal: bool = False
) -> FakeLLMClient:
    firsts = {skills[0]: cid for cid, skills in _members(root).items()}
    renames = RENAMED if renames is None else renames

    def handler(request: LLMRequest) -> str:
        if request.user.startswith("CLUSTERS:"):
            return '{"groups": []}'
        if "OVERLAPS WITH:" in request.user:
            if fatal:
                raise FatalLLMError("denied")
            first = request.user.split("MEMBERS:\n- ")[1].split("\n")[0]
            return json.dumps({"name": renames[firsts[first]]})
        first = request.user.splitlines()[1].removeprefix("- ")
        return json.dumps({"name": NAMED[firsts[first]]})

    return FakeLLMClient(handler=handler)


def _review_run(root: Path, *extra: str, client: FakeLLMClient) -> int:
    defaults = ["--fit-rounds", "0"]  # fit-review tests pass their own
    return _run(root, "name", "--method", "llm", "--model", "m", *defaults, *extra,
                llm_client=client)


def _kinds(client: FakeLLMClient) -> dict[str, int]:
    judge = sum(c.user.startswith("CLUSTERS:") for c in client.calls)
    rename = sum("OVERLAPS WITH:" in c.user for c in client.calls)
    return {"judge": judge, "rename": rename, "name": len(client.calls) - judge - rename}


def test_name_review_renames_overlapping_clusters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    capsys.readouterr()
    assert _review_run(root, client=_review_client(root)) == 0
    out = capsys.readouterr()
    summary = json.loads(out.out)
    assert (summary["n_renamed"], summary["n_overlap"], summary["n_fallback"]) == (2, 0, 0)
    rows = _names_rows(root)
    assert [r["name"] for r in rows] == ["Interface Engineering", "Data Storage", "Cloud Platforms"]
    renamed_version = "name_v2+rename_v1"
    assert [r["prompt_version"] for r in rows] == [renamed_version, renamed_version, "name_v2"]
    assert {r["status"] for r in rows} == {"ok"} and {r["method"] for r in rows} == {"llm"}
    lines = out.err.splitlines()
    assert any(line.startswith("renamed 1/2: cluster 0") for line in lines)
    assert lines[-1] == "overlap review done: no overlapping names"


def test_name_review_unresolved_marks_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    stubborn = {0: "Frontend Tool Kit", 1: "Frontend Tools Core"}
    capsys.readouterr()
    assert _review_run(root, client=_review_client(root, renames=stubborn)) == 0
    out = capsys.readouterr()
    rows = _names_rows(root)
    assert [r["name"] for r in rows] == NAMED
    assert [r["status"] for r in rows] == ["overlap", "overlap", "ok"]
    assert {r["prompt_version"] for r in rows} == {"name_v2"}
    summary = json.loads(out.out)
    assert (summary["n_renamed"], summary["n_overlap"]) == (0, 2)
    assert "overlap review done: 2 cluster(s) still overlapping: 0, 1" in out.err


def test_name_review_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    off = _review_client(root)
    capsys.readouterr()
    assert _review_run(root, "--overlap-rounds", "0", client=off) == 0
    assert _kinds(off) == {"judge": 0, "rename": 0, "name": 3}
    assert "overlap" not in capsys.readouterr().err
    no_judge = _review_client(root)
    assert _review_run(root, "--no-overlap-judge", "--force", client=no_judge) == 0
    assert _kinds(no_judge) == {"judge": 0, "rename": 2, "name": 3}
    one_round = _review_client(root)
    assert _review_run(root, "--overlap-rounds", "1", "--force", client=one_round) == 0
    assert _kinds(one_round)["judge"] == 1


def test_name_review_same_file_for_workers(tmp_path: Path) -> None:
    files = []
    for workers in ("2", "3"):
        root = tmp_path / f"art{workers}"
        _through_grid(root)
        assert _review_run(root, "--workers", workers, client=_review_client(root)) == 0
        files.append((root / "taxonomy" / "cluster_names.csv").read_bytes())
    assert files[0] == files[1]


def test_name_review_fatal_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    capsys.readouterr()
    assert _review_run(root, client=_review_client(root, fatal=True)) == 3
    assert "denied" in capsys.readouterr().err
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_overlap_rounds_invalid_exit_2(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    for bad in ("-1", "x"):
        client = _review_client(root)
        assert _review_run(root, "--overlap-rounds", bad, client=client) == 2
        assert client.calls == []
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_overlap_flags_ignored_for_heuristic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain, flagged = tmp_path / "plain", tmp_path / "flagged"
    for root, extra in ((plain, ()), (flagged, ("--overlap-rounds", "3", "--no-overlap-judge"))):
        _through_grid(root)
        capsys.readouterr()
        assert _run(root, "name", *extra) == 0
    out = capsys.readouterr()
    summary = json.loads(out.out)
    assert (summary["method"], summary["n_renamed"], summary["n_overlap"]) == ("heuristic", 0, 0)
    assert "overlap" not in out.err
    names = "taxonomy/cluster_names.csv"
    assert (plain / names).read_bytes() == (flagged / names).read_bytes()


def test_name_rows_keep_duplicate_status_for_unresolved_ids() -> None:
    members = {0: ["a1", "a2"], 1: ["b1", "b2"], 2: ["c1", "c2"]}
    llm = naming.NameResult(names={0: "Same Name", 1: "same name", 2: "Other Area"}, failed=[])
    review = overlap.ReviewResult(
        names=dict(llm.names), renamed={}, unresolved=[0, 1, 2], rounds_run=1,
        judge_failures=0, stats=naming.CallStats(),
    )
    bundle = llm_factory.ClientBundle(client=FakeLLMClient(script=[]), backend="voyager",
                                      model="m", secrets=[])
    rows = cli_name._name_rows(members=members, llm=llm, bundle=bundle, review=review)
    assert [row[5] for row in rows] == ["duplicate", "duplicate", "overlap"]


def test_review_skips_fallback_clusters(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    firsts = {skills[0]: cid for cid, skills in _members(root).items()}

    def handler(request: LLMRequest) -> str:
        first = request.user.splitlines()[1].removeprefix("- ")
        return json.dumps({"name": "Frontend Tools" if firsts[first] == 0 else "Other"})

    client = FakeLLMClient(handler=handler)
    assert _review_run(root, client=client) == 1
    assert _kinds(client)["judge"] == 0 and _kinds(client)["rename"] == 0
    assert [r["method"] for r in _names_rows(root)] == ["llm", "heuristic_fallback",
                                                         "heuristic_fallback"]


def _call_kind(request: LLMRequest) -> str:
    if request.user.startswith("NAMES TO CHECK:"):
        return "fit"
    if request.user.startswith("CLUSTERS:"):
        return "overlap_judge"
    if "OVERLAPS WITH:" in request.user:
        return "overlap_rename"
    return "refit" if "WHY:" in request.user else "name"


def _fit_client(
    root: Path,
    *,
    weak: list[int],
    refits: dict[int, str],
    groups: str = '{"groups": []}',
    fatal: bool = False,
) -> FakeLLMClient:
    firsts = {skills[0]: cid for cid, skills in _members(root).items()}
    renames = {0: "Interface Engineering", 1: "Data Storage", 2: "Cloud Hosting"}

    def handler(request: LLMRequest) -> str:
        kind = _call_kind(request)
        if kind == "fit":
            return json.dumps({"weak": [{"cluster": i, "reason": "mismatch"} for i in weak]})
        if kind == "overlap_judge":
            return groups
        if kind == "name":
            return json.dumps({"name": NAMED[firsts[request.user.splitlines()[1][2:]]]})
        if fatal and kind == "refit":
            raise FatalLLMError("denied")
        first = request.user.split("MEMBERS:\n- ")[1].split("\n")[0]
        table = refits if kind == "refit" else renames
        return json.dumps({"name": table[firsts[first]]})

    return FakeLLMClient(handler=handler)


def _fit_run(root: Path, *extra: str, client: FakeLLMClient) -> int:
    return _run(root, "name", "--method", "llm", "--model", "m", *extra, llm_client=client)


def test_name_fit_review_renames_and_orders_before_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    only, both = tmp_path / "only", tmp_path / "both"
    for root in (only, both):
        _through_grid(root)
    capsys.readouterr()
    client = _fit_client(only, weak=[2], refits={2: "Hosting Services"})
    assert _fit_run(only, "--overlap-rounds", "0", client=client) == 0
    rows = _names_rows(only)
    assert [r["name"] for r in rows] == [*NAMED[:2], "Hosting Services"]
    assert [r["prompt_version"] for r in rows] == ["name_v2", "name_v2", "name_v2+refit_v1"]
    assert [_call_kind(c) for c in client.calls] == ["name"] * 3 + ["fit", "refit"]
    capsys.readouterr()

    groups = '{"groups": [{"clusters": [1, 2], "reason": "same area"}]}'
    client = _fit_client(both, weak=[2], refits={2: "Hosting Services"}, groups=groups)
    assert _fit_run(both, "--overlap-rounds", "1", client=client) == 0
    out = capsys.readouterr()
    kinds = [_call_kind(c) for c in client.calls]
    assert kinds[:5] == ["name"] * 3 + ["fit", "refit"] and kinds[5] == "overlap_judge"
    assert set(kinds[6:]) == {"overlap_rename"}
    judge_user = client.calls[5].user
    assert "Hosting Services" in judge_user and "Cloud Platforms" not in judge_user
    rows = _names_rows(both)
    assert [r["prompt_version"] for r in rows] == [
        "name_v2+rename_v1", "name_v2+rename_v1", "name_v2+refit_v1+rename_v1"]
    assert [r["status"] for r in rows] == ["ok"] * 3
    lines = out.err.splitlines()
    first_overlap = next(i for i, x in enumerate(lines) if x.startswith("overlap review round"))
    assert any(x.startswith("fit review round 1/1: 1 of 3") for x in lines[:first_overlap])
    assert 'weak fit: cluster 2 "Cloud Platforms"' in lines[:first_overlap]
    assert "fit review done: no weak names left" in lines[:first_overlap]


def test_name_fit_unresolved_marks_weak_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    capsys.readouterr()
    client = _fit_client(root, weak=[2], refits={2: NAMED[2]})
    assert _fit_run(root, client=client) == 0
    out = capsys.readouterr()
    rows = _names_rows(root)
    assert rows[2]["name"] == "Cloud Platforms" and rows[2]["status"] == "weak_fit"
    assert rows[2]["prompt_version"] == "name_v2"
    assert [r["name"] for r in rows[:2]] == ["Interface Engineering", "Data Storage"]
    assert "fit review done: 1 cluster(s) still weak: 2" in out.err
    assert json.loads(out.out)["n_weak_fit"] == 1


def test_name_status_precedence_with_weak_fit() -> None:
    members = {0: ["a1", "a2"], 1: ["b1", "b2"], 2: ["c1", "c2"], 3: ["Git", "Jira"],
               4: ["d1", "d2"]}
    names = {0: "Same Name", 1: "same name", 2: "Other Area", 3: "Git; Jira", 4: "Fourth Area"}
    llm = naming.NameResult(names=dict(names), failed=[])
    stats = naming.CallStats()
    fitted = fit.FitResult(names=dict(names), renamed={}, unresolved=[0, 1, 2, 3], rounds_run=1,
                           judge_failures=0, stats=stats)
    review = overlap.ReviewResult(names=dict(names), renamed={}, unresolved=[2, 4], rounds_run=1,
                                  judge_failures=0, stats=stats)
    bundle = llm_factory.ClientBundle(client=FakeLLMClient(script=[]), backend="voyager",
                                      model="m", secrets=[])
    rows = cli_name._name_rows(members=members, llm=llm, bundle=bundle, fit_result=fitted,
                               review=review)
    assert [row[5] for row in rows] == [
        "duplicate", "duplicate", "weak_fit", "member_list", "overlap"]


def test_name_fit_summary_keys(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    capsys.readouterr()
    client = _fit_client(root, weak=[2], refits={2: "Hosting Services"})
    assert _fit_run(root, "--overlap-rounds", "0", client=client) == 0
    summary = json.loads(capsys.readouterr().out)
    assert list(summary) == ["names_file", "method", "n_clusters", "n_fallback", "n_duplicate",
                             "n_member_list", "n_renamed", "n_overlap", "n_fit_renamed",
                             "n_weak_fit"]
    assert (summary["n_fit_renamed"], summary["n_weak_fit"], summary["n_renamed"]) == (1, 0, 0)


def test_fit_rounds_zero_disables(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    client = _fit_client(root, weak=[2], refits={2: "Hosting Services"})
    capsys.readouterr()
    assert _fit_run(root, "--fit-rounds", "0", client=client) == 0
    kinds = {_call_kind(c) for c in client.calls}
    assert "fit" not in kinds and "refit" not in kinds and "overlap_judge" in kinds
    assert "fit review" not in capsys.readouterr().err
    assert _names_rows(root)[2]["name"] == "Cloud Platforms"


def test_fit_rounds_invalid_exit_2(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    for bad in ("-1", "x"):
        client = _fit_client(root, weak=[], refits={})
        assert _fit_run(root, "--fit-rounds", bad, client=client) == 2
        assert client.calls == []
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_fit_flag_ignored_for_heuristic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plain, flagged = tmp_path / "plain", tmp_path / "flagged"
    for root, extra in ((plain, ()), (flagged, ("--fit-rounds", "1"))):
        _through_grid(root)
        capsys.readouterr()
        assert _run(root, "name", *extra) == 0
    assert "fit review" not in capsys.readouterr().err
    names = "taxonomy/cluster_names.csv"
    assert (plain / names).read_bytes() == (flagged / names).read_bytes()


def test_name_fit_same_file_for_workers(tmp_path: Path) -> None:
    files = []
    for workers in ("2", "3"):
        root = tmp_path / f"art{workers}"
        _through_grid(root)
        client = _fit_client(root, weak=[0, 2], refits={0: "Interface Design", 2: "Hosting"})
        assert _fit_run(root, "--workers", workers, client=client) == 0
        files.append((root / "taxonomy" / "cluster_names.csv").read_bytes())
    assert files[0] == files[1]


def test_name_fit_fatal_exit_3(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    capsys.readouterr()
    client = _fit_client(root, weak=[2], refits={}, fatal=True)
    assert _fit_run(root, client=client) == 3
    assert "denied" in capsys.readouterr().err
    assert not (root / "taxonomy" / "cluster_names.csv").exists()


def test_fit_makes_no_call_without_model_named_clusters(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    client = FakeLLMClient(handler=lambda request: json.dumps({"name": "Other"}))
    assert _fit_run(root, client=client) == 1
    assert {_call_kind(c) for c in client.calls} == {"name"}


def _set_statuses(root: Path, statuses: dict[int, str]) -> None:
    path = root / "taxonomy" / "cluster_names.csv"
    rows = _names_rows(root)
    for row in rows:
        row["status"] = statuses.get(int(row["cluster_id"]), row["status"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=build.NAMES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def test_build_warns_but_builds_on_weak_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_client(*args: object, **kwargs: object) -> None:
        raise AssertionError("build must not create a model client")

    monkeypatch.setattr(llm_factory, "make_client", no_client)
    root = tmp_path / "art"
    _name_all(root)
    _write_names(root, {0: "Frontend Work", 1: "Database Storage", 2: "Cloud Platforms"})
    capsys.readouterr()
    assert _run(root, "build") == 0
    clean = capsys.readouterr()
    assert json.loads(clean.out)["n_weak_fit"] == 0 and "warning" not in clean.err
    plain = model.load_taxonomy(root / "taxonomy" / "taxonomy.json").cluster_names()

    _set_statuses(root, {2: "weak_fit", 0: "weak_fit"})
    assert _run(root, "build") == 0
    out = capsys.readouterr()
    warnings = [line for line in out.err.splitlines() if line.startswith("warning: cluster ")]
    assert len(warnings) == 2
    assert "cluster 0 name 'Frontend Work'" in warnings[0]
    assert "cluster 2 name 'Cloud Platforms'" in warnings[1]
    assert "weak_fit" in warnings[0] and "set the status to ok" in warnings[0]
    assert json.loads(out.out)["n_weak_fit"] == 2
    assert model.load_taxonomy(root / "taxonomy" / "taxonomy.json").cluster_names() == plain


def test_build_without_status_column_has_no_weak_fit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _name_all(root)
    path = root / "taxonomy" / "cluster_names.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "name"])
        writer.writerows([[0, "Frontend Work"], [1, "Database Storage"], [2, "Cloud Platforms"]])
    capsys.readouterr()
    assert _run(root, "build") == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["n_weak_fit"] == 0 and "warning" not in out.err


def test_nonnegative_int() -> None:
    import argparse

    from getwork.taxonomy import cli_common

    assert (cli_common.nonnegative_int("0"), cli_common.nonnegative_int("7")) == (0, 7)
    for bad in ("-1", "x", "1.5", ""):
        with pytest.raises(argparse.ArgumentTypeError):
            cli_common.nonnegative_int(bad)


def test_build_warns_but_builds_on_overlap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _name_all(root)
    _write_names(root, {0: "Frontend Tools", 1: "Frontend Tools Suite", 2: "Cloud Platforms"})
    capsys.readouterr()
    assert _run(root, "build") == 0
    out = capsys.readouterr()
    warnings = [line for line in out.err.splitlines() if line.startswith("warning: clusters")]
    assert len(warnings) == 1 and "clusters 0, 1 have overlapping names" in warnings[0]
    assert json.loads(out.out)["n_overlap_groups"] == 1
    assert (root / "taxonomy" / "taxonomy.json").exists()
    _write_names(root, {0: "Frontend Work", 1: "Database Storage", 2: "Cloud Platforms"})
    capsys.readouterr()
    assert _run(root, "build") == 0
    clean = capsys.readouterr()
    assert json.loads(clean.out)["n_overlap_groups"] == 0 and "warning" not in clean.err


def test_build_overlap_count_ignores_member_list_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "art"
    _name_all(root)
    first, second = _members(root)[1][:2]
    member_list = f"{first} / {second}"
    concept = f"{first} {second} Suite".title()
    # the member list's words are a subset of the concept name's: a rules overlap if it counted
    assert overlap.rule_overlap(member_list, concept) is not None
    _write_names(root, {0: "Frontend Work", 1: member_list, 2: concept})
    capsys.readouterr()
    assert _run(root, "build", ALLOW) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["n_overlap_groups"] == 0
    assert "warning: clusters" not in out.err
    _write_names(root, {0: "Frontend Work", 1: concept, 2: concept + " Extra"})
    capsys.readouterr()
    assert _run(root, "build", ALLOW) == 0
    assert json.loads(capsys.readouterr().out)["n_overlap_groups"] == 1


def test_build_ignores_overlap_status_column(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _name_all(root)
    _write_names(root, {0: "Frontend Work", 1: "Database Storage", 2: "Cloud Platforms"})
    path = root / "taxonomy" / "cluster_names.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("member_list", "overlap"),
                    encoding="utf-8")
    assert "overlap" in path.read_text(encoding="utf-8")
    assert _run(root, "build") == 0
