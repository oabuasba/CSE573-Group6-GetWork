import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from getwork.docembed import documents, levels, serialize
from getwork.relabel.records import RELABEL_KEYS
from getwork.taxonomy.model import Cluster, make_taxonomy

pytestmark = pytest.mark.usefixtures("offline")

TAX = make_taxonomy(
    [Cluster(id=0, name="Front-end", skills=("Vue", "CSS")),
     Cluster(id=1, name="Cloud", skills=("AWS",))],
    noise=("Excel",),
    alias_table_version="sha256:t",
    meta={},
)


def _record(
    id_: str,
    source: str = "resumes",
    skills: list[tuple[str, str]] | None = None,
    years: float | None = 3.0,
) -> dict:
    default = [("AWS", "Cloud"), ("Excel", "Other"), ("Vue", "Front-end")]
    pairs = default if skills is None else skills
    record = {key: None for key in RELABEL_KEYS}
    record.update(
        id=id_, source=source, title="Backend  Engineer", years_experience=years,
        skills=[s for s, _ in pairs], skill_clusters=[
            {"skill": s, "cluster": c, "via": "lookup"} for s, c in pairs
        ],
        alias_table_version="sha256:t", taxonomy_version=TAX.version, relabel_mode="llm",
        relabel_model="m1", relabel_backend="local",
    )
    return record


def test_build_document_fields_and_texts() -> None:
    record = _record("r-1")
    doc = documents.build_document(record=record, listed_level="Director", tax=TAX)
    assert tuple(doc) == documents.DOCUMENT_KEYS
    assert (doc["level"], doc["level_source"]) == ("Associate", "years")
    assert doc["text_full"] == (
        "Title: Backend Engineer\nLevel: Associate\n"
        "Skills: Front-end: Vue; Cloud: AWS; Other: Excel"
    )
    assert doc["text_b2"] == "Title: Backend Engineer\nLevel: Associate\nSkills: Vue, AWS, Excel"
    assert doc["title"] == "Backend  Engineer"
    assert doc["years_experience"] == 3.0
    assert doc["skills"] == ["AWS", "Excel", "Vue"]
    assert doc["skill_clusters"] == record["skill_clusters"]
    assert doc["serialization_version"] == serialize.SERIALIZATION_VERSION
    for key in documents.PROVENANCE_KEYS:
        assert doc[key] == record[key]
    assert (doc["relabel_model"], doc["relabel_backend"]) == ("m1", "local")
    posting = documents.build_document(
        record=_record("p-1", source="postings"),
        listed_level="Director",
        tax=TAX,
    )
    assert (posting["level"], posting["level_source"]) == ("Director", "listed")
    assert "Level: Director\n" in posting["text_full"]
    none_years = documents.build_document(
        record=_record("r-2", years=None),
        listed_level=None,
        tax=TAX,
    )
    assert none_years["years_experience"] is None
    assert (none_years["level"], none_years["level_source"]) == ("Not specified", "missing")
    bands = documents.build_document(
        record=_record("r-3"),
        listed_level=None,
        tax=TAX,
        bands=(4.0,
        6.0),
    )
    assert bands["level"] == "Entry level"
    rec = documents.build_document(
        record=_record("r-4"),
        listed_level=None,
        tax=TAX,
        b2_order="record",
    )
    assert rec["text_b2"].endswith("Skills: AWS, Excel, Vue")


def test_build_document_does_not_mutate() -> None:
    record = _record("r-1")
    before = copy.deepcopy(record)
    doc = documents.build_document(record=record, listed_level=None, tax=TAX)
    assert record == before
    doc["skills"].append("x")
    doc["skill_clusters"][0]["skill"] = "changed"
    assert record == before


def test_provenance_of() -> None:
    recs = {"r-1": _record("r-1"), "r-2": _record("r-2")}
    assert documents.provenance_of(recs) == {
        "alias_table_version": "sha256:t", "taxonomy_version": TAX.version,
        "relabel_mode": "llm", "relabel_model": "m1", "relabel_backend": "local",
    }
    with pytest.raises(ValueError, match="no records"):
        documents.provenance_of({})
    other = {**recs, "r-3": {**_record("r-3"), "taxonomy_version": "sha256:other"}}
    with pytest.raises(ValueError, match="taxonomy_version.*sha256:other"):
        documents.provenance_of(other)


def test_check_comparable_names_differing_keys() -> None:
    a = {key: f"a-{key}" for key in documents.COMPARE_KEYS}
    documents.check_comparable(a=a, b=dict(a))
    b = {key: f"b-{key}" for key in documents.COMPARE_KEYS}
    with pytest.raises(documents.DocEmbedMismatchError) as err:
        documents.check_comparable(a=a, b=b)
    for key in documents.COMPARE_KEYS:
        assert f"{key}: 'a-{key}' vs 'b-{key}'" in str(err.value)
    c = {**a, "b2_order": "record", "taxonomy_version": "x"}
    with pytest.raises(documents.DocEmbedMismatchError) as err:
        documents.check_comparable(a=a, b=c)
    assert "b2_order" in str(err.value) and "taxonomy_version" in str(err.value)
    assert "relabel_mode" not in str(err.value)
    missing = {k: v for k, v in a.items() if k != "b2_order"}
    with pytest.raises(documents.DocEmbedMismatchError, match="b2_order.*None"):
        documents.check_comparable(a=a, b=missing)
    documents.check_comparable(a=a, b=missing, keys=["relabel_mode"])


def _records() -> dict[str, dict]:
    return {
        "r-10": _record("r-10", years=6.0),
        "r-2": _record("r-2", skills=[], years=None),
        "r-1": _record("r-1", skills=[("Excel", "Other"), ("Vue", "Front-end")], years=1.0),
        "r-3": _record("r-3", years=3.0),
    }


def test_build_documents_outputs_sorted_and_deterministic(tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = documents.build_documents(
        source="resumes", relabeled=_records(), listed_levels={}, tax=TAX, out_dir=out
    )
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "resumes_documents.jsonl", "resumes_documents.parquet", "resumes_documents_summary.json",
    ]
    rows = [json.loads(x) for x in (out / "resumes_documents.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["r-1", "r-2", "r-3", "r-10"]
    assert all(tuple(r) == documents.DOCUMENT_KEYS for r in rows)
    assert rows[0]["text_b2"].endswith("Skills: Vue, Excel")
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    again = documents.build_documents(
        source="resumes", relabeled=_records(), listed_levels={}, tax=TAX, out_dir=out
    )
    assert again == summary
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first
    table = pq.read_table(out / "resumes_documents.parquet")
    assert table.column("id").to_pylist() == ["r-1", "r-2", "r-3", "r-10"]


def test_parquet_schema_with_empty_lists(tmp_path: Path) -> None:
    recs = {"r-1": _record("r-1", skills=[], years=None), "r-2": _record("r-2", skills=[])}
    documents.build_documents(
        source="resumes", relabeled=recs, listed_levels={}, tax=TAX, out_dir=tmp_path
    )
    table = pq.read_table(tmp_path / "resumes_documents.parquet")
    assert table.schema.equals(documents.DOCUMENT_PARQUET_SCHEMA)
    assert table.schema.field("years_experience").type == pa.float64()
    assert table.schema.field("skills").type == pa.list_(pa.string())
    assert table.schema.field("skill_clusters").type == pa.list_(
        pa.struct([("skill", pa.string()), ("cluster", pa.string()), ("via", pa.string())])
    )
    assert table.column("years_experience").to_pylist() == [None, 3.0]
    assert pq.ParquetFile(tmp_path / "resumes_documents.parquet").metadata.row_group(0).column(
        0
    ).compression == "ZSTD"


def test_summary_keys_and_counts(tmp_path: Path) -> None:
    posts = {
        "p-1": _record("p-1", source="postings", years=None),
        "p-2": _record("p-2", source="postings", years=1.0),
        "p-3": _record("p-3", source="postings", skills=[], years=None),
        "p-4": _record("p-4", source="postings", years=9.0),
    }
    listed = {"p-1": "Director", "p-4": "garbage", "p-9": "Executive"}
    summary = documents.build_documents(
        source="postings", relabeled=posts, listed_levels=listed, tax=TAX, out_dir=tmp_path,
        bands=(1.0, 8.0), b2_order="record",
    )
    assert list(summary) == [
        "source", "n_documents", *documents.PROVENANCE_KEYS, "serialization_version",
        "level_bands", "b2_order", "level_counts", "level_source_counts", "n_empty_skills",
        "mean_skills", "other_share", "examples",
    ]
    assert summary["source"] == "postings" and summary["n_documents"] == 4
    assert summary["level_bands"] == [1.0, 8.0] and summary["b2_order"] == "record"
    assert summary["serialization_version"] == "ser_v1"
    assert summary["relabel_model"] == "m1" and summary["taxonomy_version"] == TAX.version
    assert summary["level_counts"] == {
        "Internship": 0, "Entry level": 0, "Associate": 1, "Mid-Senior level": 1,
        "Director": 1, "Executive": 0, "Not specified": 1,
    }
    assert list(summary["level_counts"]) == [*levels.LEVELS, "Not specified"]
    assert summary["level_source_counts"] == {"listed": 1, "years": 2, "missing": 1}
    assert summary["n_empty_skills"] == 1
    assert summary["mean_skills"] == 2.25
    assert summary["other_share"] == 3 / 9
    assert [e["id"] for e in summary["examples"]] == ["p-1", "p-2", "p-3"]
    assert set(summary["examples"][0]) == {"id", "text_full", "text_b2"}
    assert summary["examples"][2]["text_full"].endswith("Skills: none")
    rows = [json.loads(x) for x in (tmp_path / "postings_documents.jsonl").read_text().splitlines()]
    assert rows[0]["text_b2"].endswith("Skills: AWS, Excel, Vue")
    assert rows[0]["text_full"].endswith("Skills: Front-end: Vue; Cloud: AWS; Other: Excel")
    stored = json.loads((tmp_path / "postings_documents_summary.json").read_text())
    assert stored == summary


def test_summary_empty_and_zero_skills(tmp_path: Path) -> None:
    only = {"r-1": _record("r-1", skills=[])}
    summary = documents.build_documents(
        source="resumes", relabeled=only, listed_levels={}, tax=TAX, out_dir=tmp_path
    )
    assert (summary["mean_skills"], summary["other_share"]) == (0.0, 0.0)
    assert summary["examples"][0]["id"] == "r-1" and len(summary["examples"]) == 1


def test_nonuniform_provenance_writes_nothing(tmp_path: Path) -> None:
    recs = _records()
    recs["r-3"] = {**recs["r-3"], "relabel_mode": "lookup_only"}
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="relabel_mode"):
        documents.build_documents(
            source="resumes", relabeled=recs, listed_levels={}, tax=TAX, out_dir=out
        )
    assert not out.exists()


def test_read_documents_order_and_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="doc-embed build") as err:
        documents.read_documents(root=tmp_path, source="resumes")
    assert str(documents.documents_path(root=tmp_path, source="resumes")) in str(err.value)
    out = tmp_path / "documents" / "resumes"
    summary = documents.build_documents(
        source="resumes", relabeled=_records(), listed_levels={}, tax=TAX, out_dir=out
    )
    path = documents.documents_path(root=tmp_path, source="resumes")
    assert path == out / "resumes_documents.jsonl"
    docs = documents.read_documents(root=tmp_path, source="resumes")
    assert list(docs) == ["r-1", "r-2", "r-3", "r-10"]
    assert tuple(docs["r-10"]) == documents.DOCUMENT_KEYS
    assert docs["r-10"]["level"] == "Mid-Senior level"
    assert documents.read_documents_summary(root=tmp_path, source="resumes") == summary
    path = documents.documents_path(root=tmp_path, source="resumes")
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    del rows[1]["text_b2"]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match=r"'r-2'.*text_b2"):
        documents.read_documents(root=tmp_path, source="resumes")


def test_read_summary_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="doc-embed build") as err:
        documents.read_documents_summary(root=tmp_path, source="postings")
    assert "postings_documents_summary.json" in str(err.value)


def test_read_documents_unsorted_file_and_bad_json(tmp_path: Path) -> None:
    out = tmp_path / "documents" / "resumes"
    documents.build_documents(
        source="resumes", relabeled=_records(), listed_levels={}, tax=TAX, out_dir=out
    )
    path = documents.documents_path(root=tmp_path, source="resumes")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(reversed(lines)) + "\n")
    assert list(documents.read_documents(root=tmp_path, source="resumes")) == [
        "r-1", "r-2", "r-3", "r-10",
    ]
    path.write_text(lines[0] + "\n{not json\n")
    with pytest.raises(ValueError, match=r"resumes_documents\.jsonl: line 2 is not valid JSON"):
        documents.read_documents(root=tmp_path, source="resumes")
