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
from getwork.taxonomy import build, cli, model, naming, report

pytestmark = pytest.mark.usefixtures("offline")

VERSION = "sha256:bbbbbbbbbbbb"
THEMES = ("frontend", "database", "cloud")
SKILLS = [f"{theme} item{n:02d}" for theme in THEMES for n in range(20)]
N_DOCS = 40
SKILLS_PER_DOC = 6
# one kmeans entry; the other methods cannot produce an eligible configuration
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
    path = root / "taxonomy" / "cluster_names.csv"
    assert path.read_text(encoding="utf-8").splitlines()[0] == ",".join(build.NAMES_COLUMNS)
    rows = _names_rows(root)
    assert [int(r["cluster_id"]) for r in rows] == [0, 1, 2]
    for row in rows:
        skills = members[int(row["cluster_id"])]
        assert row["name"] == naming.heuristic_name(skills)
        assert (row["size"], row["method"], row["status"]) == (str(len(skills)), "heuristic", "ok")
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
    wanted = {members[0][0]: "Same", members[1][0]: "same", members[2][0]: "Other"}
    client = FakeLLMClient(handler=_llm_handler(wanted))
    capsys.readouterr()
    code = _run(root, "name", "--method", "llm", "--model", "m-1", "--backend", "local",
                llm_client=client)
    assert code == 1
    printed = json.loads(capsys.readouterr().out)
    assert (printed["n_fallback"], printed["n_duplicate"], printed["n_clusters"]) == (1, 2, 3)
    rows = _names_rows(root)
    assert [(r["method"], r["status"]) for r in rows] == [
        ("llm", "duplicate"), ("llm", "duplicate"), ("heuristic_fallback", "ok"),
    ]
    assert [r["name"] for r in rows][:2] == ["Same", "same"]
    assert rows[2]["name"] == naming.heuristic_name(members[2])
    for row in rows[:2]:
        assert (row["model"], row["backend"], row["prompt_version"]) == ("m-1", "local", "name_v1")
    assert (rows[2]["model"], rows[2]["backend"], rows[2]["prompt_version"]) == ("", "", "")
    assert {c.model for c in client.calls} == {"m-1"}


def test_name_llm_all_succeed_exit_0(tmp_path: Path) -> None:
    root = tmp_path / "art"
    _through_grid(root)
    members = _members(root)
    wanted = {members[i][0]: f"Group {i}" for i in range(3)}
    code = _run(root, "name", "--method", "llm", "--model", "m",
                "--top-members", "2", llm_client=FakeLLMClient(handler=_llm_handler(wanted)))
    assert code == 0
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
    assert _run(root, "build") == 0
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
    assert _run(root, "build") == 2
    assert _run(root, "build", "--names", str(custom)) == 0
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
    assert _run(root, "build") == 2
    assert message in capsys.readouterr().err
    assert not (root / "taxonomy" / "taxonomy.json").exists()


def test_build_failure_keeps_existing_taxonomy(tmp_path: Path) -> None:
    root = tmp_path / "art"
    names_path = _name_all(root)
    assert _run(root, "build") == 0
    target = root / "taxonomy" / "taxonomy.json"
    before = target.read_bytes()
    names_path.write_text("cluster_id,name\n0,Only: one\n", encoding="utf-8")
    assert _run(root, "build") == 2
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
    assert _run(root, "build") == 2
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
        assert _run(root, "build") == 0
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
                "--extra-body", '{"thinking": false}', llm_client=client)
    assert code == 0 and len(client.calls) == 3
    for call in client.calls:
        assert call.user.count("\n- ") == 2
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
    assert _run(root, "build") == 0
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
    assert _run(root, "build") == 2
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
    assert _run(root, "build") == 2
    assert "taxonomy grid again" in capsys.readouterr().err
    assert names_path.read_bytes() == names_before
    assert not (root / "taxonomy" / "taxonomy.json").exists()
