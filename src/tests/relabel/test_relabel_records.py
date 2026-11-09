import copy
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from getwork.alias.apply import CANONICAL_KEYS, CANONICAL_PARQUET_SCHEMA
from getwork.alias.table import AliasTable, make_table
from getwork.relabel import records
from getwork.relabel.lookup import SkillAssignment
from getwork.taxonomy.model import Cluster, Taxonomy, make_taxonomy

pytestmark = pytest.mark.usefixtures("offline")

ALIAS_VERSION = make_table({"js": "JavaScript"}).version
VIA_KEYS = ("taxonomy", "noise", "llm", "default")


def _tax(alias_version: str = ALIAS_VERSION) -> Taxonomy:
    return make_taxonomy(
        [Cluster(0, "Front-end", ("React", "JavaScript")), Cluster(1, "Cloud", ("AWS",))],
        noise=("Excel",),
        alias_table_version=alias_version,
        meta={},
    )


def _table() -> AliasTable:
    return make_table({"js": "JavaScript"})


def _record(id_: str, skills: Sequence[str], alias_version: str = ALIAS_VERSION) -> dict:
    return {
        "id": id_, "source": "resumes", "title": "Dev", "skills": list(skills),
        "experience": [{"role": "r", "company": "c", "duration_months": 12}],
        "education": [{"degree": "BS", "field": "CS", "institution": "U"}],
        "years_experience": 3.5, "model": "m0", "prompt_version": "p0",
        "schema_version": "1", "backend": "voyager",
        "skills_raw": list(skills), "alias_table_version": alias_version,
    }


class FakeRelabeler:
    mode = "llm"

    def __init__(
        self,
        unseen: Mapping[str, str] | None = None,
        added: Mapping[str, list[tuple[str, str]] | None] | None = None,
    ) -> None:
        self.model, self.backend, self.prompt_version = "mod", "bk", "unseen_v1+reextract_v1"
        self.unseen = dict(unseen or {})
        self.added = dict(added or {})
        self.unseen_calls: list[list[str]] = []
        self.reextract_calls: list[tuple[str, str, list[str]]] = []

    def assign_unseen(self, skills: Sequence[str]) -> Mapping[str, str]:
        self.unseen_calls.append(list(skills))
        return {s: self.unseen[s] for s in skills if s in self.unseen}

    def reextract(self, doc_id: str, text: str, skills: Sequence[str]):
        self.reextract_calls.append((doc_id, text, list(skills)))
        return self.added.get(doc_id, [])


def _a(skill: str, cluster: str, via: str) -> SkillAssignment:
    return SkillAssignment(skill=skill, cluster=cluster, via=via)


def test_build_relabel_record_keys_and_values() -> None:
    record = _record("r-1", ["js", "Excel"])
    record["skills_raw"] = ["JS", "Excel", "x"]
    before = copy.deepcopy(record)
    assignments = [_a("JavaScript", "Front-end", "taxonomy"), _a("Excel", "Other", "noise")]
    out = records.build_relabel_record(
        record, assignments=assignments, added_skills=["Vue"], incomplete="few_skills",
        reextracted=True, tax=_tax(), relabeler=None,
    )
    assert record == before
    assert tuple(out) == records.RELABEL_KEYS
    assert records.RELABEL_KEYS[: len(CANONICAL_KEYS)] == CANONICAL_KEYS
    assert records.RELABEL_KEYS[len(CANONICAL_KEYS):] == (
        "added_skills", "skill_clusters", "incomplete_reason", "reextracted", "relabel_mode",
        "taxonomy_version", "relabel_model", "relabel_backend", "relabel_prompt_version",
    )
    assert out["skills"] == ["JavaScript", "Excel"]
    assert out["skills_raw"] == ["JS", "Excel", "x"]
    assert out["alias_table_version"] == ALIAS_VERSION
    assert out["added_skills"] == ["Vue"]
    assert out["skill_clusters"] == [
        {"skill": "JavaScript", "cluster": "Front-end", "via": "taxonomy"},
        {"skill": "Excel", "cluster": "Other", "via": "noise"},
    ]
    assert out["incomplete_reason"] == "few_skills" and out["reextracted"] is True
    assert out["relabel_mode"] == "lookup_only" and out["taxonomy_version"] == _tax().version
    assert [out[k] for k in ("relabel_model", "relabel_backend", "relabel_prompt_version")] == [
        None, None, None]
    out["experience"][0]["role"] = "changed"
    assert record["experience"][0]["role"] == "r"
    fake = FakeRelabeler()
    llm = records.build_relabel_record(
        record, assignments=assignments, added_skills=[], incomplete=None, reextracted=False,
        tax=_tax(), relabeler=fake,
    )
    assert llm["relabel_mode"] == "llm" and llm["incomplete_reason"] is None
    assert llm["reextracted"] is False and llm["added_skills"] == []
    assert (llm["relabel_model"], llm["relabel_backend"], llm["relabel_prompt_version"]) == (
        "mod", "bk", "unseen_v1+reextract_v1")


def test_parquet_schema_with_empty_lists() -> None:
    schema = records.RELABEL_PARQUET_SCHEMA
    assert schema.names == list(records.RELABEL_KEYS)
    assert schema.names[: len(CANONICAL_PARQUET_SCHEMA)] == CANONICAL_PARQUET_SCHEMA.names
    assert schema.field("added_skills").type == pa.list_(pa.string())
    assert schema.field("skill_clusters").type == pa.list_(pa.struct([
        ("skill", pa.string()), ("cluster", pa.string()), ("via", pa.string())]))
    assert schema.field("reextracted").type == pa.bool_()
    empty = _record("r-1", [])
    empty["experience"], empty["education"] = [], []
    rows = [records.build_relabel_record(
        empty, assignments=[], added_skills=[], incomplete="no_skills", reextracted=False,
        tax=_tax(), relabeler=None,
    )]
    assert pa.Table.from_pylist(rows, schema=schema).schema == schema


def _run(recs, tmp_path, relabeler=None, get_text=None, **kw):
    def no_text(_id: str) -> str | None:
        raise AssertionError("get_text must not be called")

    return records.relabel_source(
        "resumes", recs, _tax(), _table(), get_text or no_text, relabeler, tmp_path / "out", **kw
    )


def _read_rows(tmp_path: Path) -> list[dict]:
    text = (tmp_path / "out" / "resumes_relabeled.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def _recs(*recs: dict) -> dict[str, dict]:
    return {r["id"]: r for r in recs}


FILLER_IDS = [f"r-{n}" for n in range(10, 26)]


def _corpus() -> dict[str, dict]:
    """21 records: p10 of the counts is 2; r-1 empty, r-2 below p10, r-3 on the boundary."""
    fillers = [_record(i, ["React", "AWS", "JavaScript"]) for i in reversed(FILLER_IDS)]
    return _recs(
        *fillers,
        _record("r-5", ["Excel", "Zig"]),
        _record("r-4", ["Zig", "Nim", "Rust"]),
        _record("r-3", ["React", "Zig"]),
        _record("r-2", ["zig"]),
        _record("r-1", []),
    )


def test_lookup_only_assignments_and_flags(tmp_path: Path) -> None:
    summary = _run(_corpus(), tmp_path)
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    assert [(c["skill"], c["cluster"], c["via"]) for c in rows["r-5"]["skill_clusters"]] == [
        ("Excel", "Other", "noise"), ("Zig", "Other", "default")]
    assert [(c["skill"], c["cluster"], c["via"]) for c in rows["r-3"]["skill_clusters"]] == [
        ("React", "Front-end", "taxonomy"), ("Zig", "Other", "default")]
    assert rows["r-4"]["skills"] == ["Zig", "Nim", "Rust"]
    assert summary["p10_threshold"] == 2
    reasons = {i: r["incomplete_reason"] for i, r in rows.items()}
    assert reasons["r-1"] == "no_skills"
    assert reasons["r-2"] == "few_skills"  # also zero known: few_skills has priority
    assert reasons["r-3"] is None  # count == p10 and known == 0.5 are not flagged
    assert reasons["r-4"] == "low_coverage"
    assert reasons["r-5"] is None  # noise counts as known
    assert all(reasons[i] is None for i in FILLER_IDS)
    assert all(r["reextracted"] is False and r["added_skills"] == [] for r in rows.values())
    assert all(r["relabel_mode"] == "lookup_only" for r in rows.values())
    assert rows["r-1"]["skills"] == [] and rows["r-1"]["skill_clusters"] == []


def test_min_known_threshold(tmp_path: Path) -> None:
    _run(_corpus(), tmp_path, min_known=0.6)
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    assert rows["r-3"]["incomplete_reason"] == "low_coverage"
    assert rows["r-5"]["incomplete_reason"] == "low_coverage"


OUTPUT_NAMES = (
    "resumes_relabeled.jsonl", "resumes_relabeled.parquet",
    "resumes_relabeled_summary.json", "resumes_relabel_failures.csv",
)


def test_outputs_sorted_and_deterministic(tmp_path: Path) -> None:
    summary = _run(_corpus(), tmp_path)
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted(OUTPUT_NAMES)
    first = {name: (out / name).read_bytes() for name in OUTPUT_NAMES}
    rows = _read_rows(tmp_path)
    assert [r["id"] for r in rows] == ["r-1", "r-2", "r-3", "r-4", "r-5", *FILLER_IDS]
    assert all(tuple(r) == records.RELABEL_KEYS for r in rows)
    table = pq.read_table(out / "resumes_relabeled.parquet")
    assert table.schema == records.RELABEL_PARQUET_SCHEMA
    assert table.column("id").to_pylist() == [r["id"] for r in rows]
    assert pq.ParquetFile(out / "resumes_relabeled.parquet").metadata.row_group(0).column(
        0).compression == "ZSTD"
    assert table.to_pylist() == rows
    assert json.loads(first["resumes_relabeled_summary.json"]) == summary
    assert list(csv.reader((out / OUTPUT_NAMES[3]).open(encoding="utf-8"))) == [
        ["id", "stage", "message"]]
    reversed_input = dict(reversed(list(_corpus().items())))
    again = _run(reversed_input, tmp_path)
    assert again == summary
    assert {name: (out / name).read_bytes() for name in OUTPUT_NAMES} == first


def test_summary_keys_and_counts(tmp_path: Path) -> None:
    summary = _run(_corpus(), tmp_path)
    assert list(summary) == [
        "source", "relabel_mode", "taxonomy_version", "alias_table_version", "n_records",
        "p10_threshold", "n_incomplete", "incomplete_by_reason", "n_reextracted",
        "n_reextract_failed", "n_skills_final", "via_counts", "n_unseen_distinct",
        "n_unseen_unresolved", "mean_known_fraction", "relabel_model", "relabel_backend",
        "relabel_prompt_version",
    ]
    assert summary == {
        "source": "resumes", "relabel_mode": "lookup_only", "taxonomy_version": _tax().version,
        "alias_table_version": ALIAS_VERSION, "n_records": 21, "p10_threshold": 2,
        "n_incomplete": 3,
        "incomplete_by_reason": {"no_skills": 1, "few_skills": 1, "low_coverage": 1},
        "n_reextracted": 0, "n_reextract_failed": 0, "n_skills_final": 56,
        "via_counts": {"taxonomy": 49, "noise": 1, "llm": 0, "default": 6},
        "n_unseen_distinct": 3, "n_unseen_unresolved": 0,
        "mean_known_fraction": pytest.approx(0.85),
        "relabel_model": None, "relabel_backend": None, "relabel_prompt_version": None,
    }
    assert list(summary["incomplete_by_reason"]) == ["no_skills", "few_skills", "low_coverage"]
    assert list(summary["via_counts"]) == list(VIA_KEYS)


def test_empty_source_and_all_empty_skills(tmp_path: Path) -> None:
    empty = _run({}, tmp_path)
    assert empty["n_records"] == 0 and empty["n_incomplete"] == 0
    assert empty["mean_known_fraction"] == 0.0 and empty["p10_threshold"] == 0
    assert _read_rows(tmp_path) == []
    only = _run(_recs(_record("r-1", []), _record("r-2", [])), tmp_path / "x")
    assert only["mean_known_fraction"] == 0.0 and only["n_skills_final"] == 0
    assert only["incomplete_by_reason"] == {"no_skills": 2, "few_skills": 0, "low_coverage": 0}


def _texts(texts: Mapping[str, str | None]):
    calls: list[str] = []

    def get_text(id_: str) -> str | None:
        calls.append(id_)
        return texts.get(id_)

    get_text.calls = calls  # type: ignore[attr-defined]
    return get_text


def test_llm_mode_unseen_resolution(tmp_path: Path) -> None:
    fake = FakeRelabeler(unseen={"zig": "Cloud", "Nim": "Front-end", "Rust": "Other"})
    summary = _run(_corpus(), tmp_path, fake, _texts({}))
    assert fake.unseen_calls == [["Nim", "Rust", "zig"]]  # by key, first spelling
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    assert [(c["skill"], c["cluster"], c["via"]) for c in rows["r-4"]["skill_clusters"]] == [
        ("Zig", "Cloud", "llm"), ("Nim", "Front-end", "llm"), ("Rust", "Other", "llm")]
    assert rows["r-3"]["skill_clusters"][1]["via"] == "llm"  # Zig resolved through key of zig
    assert rows["r-5"]["skill_clusters"][0]["via"] == "noise"
    assert rows["r-4"]["relabel_mode"] == "llm" and rows["r-4"]["relabel_model"] == "mod"
    assert summary["relabel_mode"] == "llm" and summary["n_unseen_unresolved"] == 0
    assert summary["via_counts"]["llm"] == 6 and summary["via_counts"]["default"] == 0
    assert (summary["relabel_model"], summary["relabel_backend"]) == ("mod", "bk")
    assert summary["relabel_prompt_version"] == "unseen_v1+reextract_v1"


def test_llm_mode_unresolved_unseen_default_and_counted(tmp_path: Path) -> None:
    fake = FakeRelabeler(unseen={"Nim": "Cloud"})
    summary = _run(_corpus(), tmp_path, fake, _texts({}))
    assert summary["n_unseen_unresolved"] == 2 and summary["n_unseen_distinct"] == 3
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    assert [c["via"] for c in rows["r-4"]["skill_clusters"]] == ["default", "llm", "default"]
    assert rows["r-4"]["skill_clusters"][0]["cluster"] == "Other"
    assert summary["via_counts"]["llm"] == 1 and summary["via_counts"]["default"] == 5


def test_llm_mode_no_unseen_skills_makes_no_call(tmp_path: Path) -> None:
    fake = FakeRelabeler()
    _run(_recs(_record("r-1", ["React"])), tmp_path, fake, _texts({"r-1": "t"}))
    assert fake.unseen_calls == []


def test_reextraction_merges_canonicalizes_and_lookup_wins(tmp_path: Path) -> None:
    fake = FakeRelabeler(added={
        "r-1": [("js", "Cloud"), ("AWS", "Front-end"), ("Docker", "Cloud"), ("docker", "Other"),
                ("Excel", "Cloud"), ("", "Cloud")],
        "r-2": [("ZIG", "Cloud"), ("Terraform", "Cloud")],
        "r-4": [],
    })
    texts = _texts({"r-1": "text one", "r-2": "text two", "r-4": "text four"})
    summary = _run(_corpus(), tmp_path, fake, texts)
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    r1 = rows["r-1"]
    assert r1["skills"] == ["JavaScript", "AWS", "Docker", "Excel"]
    assert r1["added_skills"] == ["JavaScript", "AWS", "Docker", "Excel"]
    assert [(c["cluster"], c["via"]) for c in r1["skill_clusters"]] == [
        ("Front-end", "taxonomy"), ("Cloud", "taxonomy"), ("Cloud", "llm"), ("Other", "noise")]
    assert r1["reextracted"] is True and r1["incomplete_reason"] == "no_skills"
    r2 = rows["r-2"]
    assert r2["skills"] == ["zig", "Terraform"] and r2["added_skills"] == ["Terraform"]
    assert [(c["cluster"], c["via"]) for c in r2["skill_clusters"]] == [
        ("Other", "default"), ("Cloud", "llm")]
    assert r2["skills_raw"] == ["zig"]
    assert rows["r-4"]["reextracted"] is True and rows["r-4"]["added_skills"] == []
    assert fake.reextract_calls == [
        ("r-1", "text one", []), ("r-2", "text two", ["zig"]),
        ("r-4", "text four", ["Zig", "Nim", "Rust"]),
    ]
    assert summary["n_reextracted"] == 3 and summary["n_reextract_failed"] == 0
    assert summary["n_skills_final"] == 56 + 4 + 1
    assert summary["via_counts"] == {"taxonomy": 51, "noise": 2, "llm": 2, "default": 6}


def test_complete_documents_not_reextracted(tmp_path: Path) -> None:
    fake = FakeRelabeler(added={i: [("Extra", "Cloud")] for i in ("r-3", "r-5", *FILLER_IDS)})
    texts = _texts({"r-1": "a", "r-2": "b", "r-4": "c"})
    _run(_corpus(), tmp_path, fake, texts)
    assert texts.calls == ["r-1", "r-2", "r-4"]
    assert [c[0] for c in fake.reextract_calls] == ["r-1", "r-2", "r-4"]
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    assert all(rows[i]["reextracted"] is False for i in ("r-3", "r-5", *FILLER_IDS))


def test_reextract_failures_listed(tmp_path: Path) -> None:
    fake = FakeRelabeler(added={"r-2": None, "r-4": [("Docker", "Cloud")]})
    texts = _texts({"r-2": "text two", "r-4": "text four"})  # r-1 has no text
    summary = _run(_corpus(), tmp_path, fake, texts)
    out = tmp_path / "out"
    failures = list(csv.reader((out / "resumes_relabel_failures.csv").open(encoding="utf-8")))
    assert failures[0] == ["id", "stage", "message"]
    assert [row[:2] for row in failures[1:]] == [["r-1", "no_text"], ["r-2", "reextract"]]
    assert all(row[2] for row in failures[1:])
    rows = {r["id"]: r for r in _read_rows(tmp_path)}
    for id_, skills in (("r-1", []), ("r-2", ["zig"])):
        assert rows[id_]["skills"] == skills and rows[id_]["added_skills"] == []
        assert rows[id_]["reextracted"] is False
    assert rows["r-4"]["added_skills"] == ["Docker"] and rows["r-4"]["reextracted"] is True
    assert summary["n_reextract_failed"] == 2 and summary["n_reextracted"] == 1
    assert [c[0] for c in fake.reextract_calls] == ["r-2", "r-4"]


def test_alias_version_mismatch_raises_and_writes_nothing(tmp_path: Path) -> None:
    bad = _corpus()
    bad["r-3"] = _record("r-3", ["React"], alias_version="sha256:stale")
    with pytest.raises(ValueError) as exc:
        _run(bad, tmp_path)
    assert "sha256:stale" in str(exc.value) and ALIAS_VERSION in str(exc.value)
    assert "r-3" in str(exc.value)
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError) as exc2:
        records.relabel_source(
            "resumes", _corpus(), _tax("sha256:other"), _table(), lambda _i: None, None,
            tmp_path / "out2",
        )
    assert "sha256:other" in str(exc2.value) and ALIAS_VERSION in str(exc2.value)
    assert not (tmp_path / "out2").exists()


def _write_relabeled(root: Path, rows: list[dict]) -> Path:
    path = records.relabeled_path(root, "resumes")
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_read_relabeled_errors_and_order(tmp_path: Path) -> None:
    root = tmp_path / "art"
    assert records.relabeled_path(root, "resumes") == (
        root / "relabeled" / "resumes" / "resumes_relabeled.jsonl")
    with pytest.raises(FileNotFoundError) as missing:
        records.read_relabeled(root, "resumes")
    assert str(records.relabeled_path(root, "resumes")) in str(missing.value)
    assert "relabel" in str(missing.value)
    records.relabel_source(
        "resumes", _corpus(), _tax(), _table(), lambda _i: None, None,
        root / "relabeled" / "resumes",
    )
    loaded = records.read_relabeled(root, "resumes")
    assert list(loaded) == ["r-1", "r-2", "r-3", "r-4", "r-5", *FILLER_IDS]
    assert tuple(loaded["r-2"]) == records.RELABEL_KEYS
    assert loaded["r-4"]["skill_clusters"][0] == {"skill": "Zig", "cluster": "Other",
                                                   "via": "default"}
    full = loaded["r-2"]
    short = {k: v for k, v in full.items() if k != "reextracted"}
    other = tmp_path / "art2"
    path = _write_relabeled(other, [{**full, "id": "r-10"}, short, {**full, "id": "r-3"}])
    with pytest.raises(ValueError) as bad:
        records.read_relabeled(other, "resumes")
    assert "r-2" in str(bad.value) and "reextracted" in str(bad.value)
    assert str(path) in str(bad.value)
    path.write_text("".join(json.dumps(r) + "\n" for r in [
        {**full, "id": "r-10"}, {**full, "id": "r-3"}, full]), encoding="utf-8")
    assert list(records.read_relabeled(other, "resumes")) == ["r-2", "r-3", "r-10"]


def test_read_relabeled_truncated_line_is_ignored_without_repair(tmp_path: Path) -> None:
    root = tmp_path / "art"
    records.relabel_source(
        "resumes", _corpus(), _tax(), _table(), lambda _i: None, None,
        root / "relabeled" / "resumes",
    )
    path = records.relabeled_path(root, "resumes")
    expected = records.read_relabeled(root, "resumes")
    damaged = path.read_bytes() + b'{"id": "r-99", "skil'
    path.write_bytes(damaged)
    assert records.read_relabeled(root, "resumes") == expected
    assert path.read_bytes() == damaged  # repair=False: the file is never rewritten
