import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from getwork.taxonomy import counts

pytestmark = pytest.mark.usefixtures("offline")


def _doc(*skills: str) -> dict:
    return {"skills": list(skills)}


def _sc(skill: str, total: int, resumes: int = 0, postings: int = 0) -> counts.SkillCount:
    return counts.SkillCount(
        skill=skill, df_total=total, df_resumes=resumes, df_postings=postings
    )


def test_count_skills_document_frequency() -> None:
    records = {
        "resumes": {"r1": _doc("Java", "java", "Python"), "r2": _doc("Python")},
        "postings": {"p1": _doc("Python")},
    }
    result = {c.skill: c for c in counts.count_skills(records)}
    assert result["Java"] == _sc("Java", 1, resumes=1)
    assert result["Python"] == _sc("Python", 3, resumes=2, postings=1)
    assert counts.count_skills({"resumes": {}}) == []


def test_count_merges_spellings_and_sorts() -> None:
    records = {
        "resumes": {"r1": _doc("JAVA"), "r2": _doc("Java"), "r3": _doc("Java"), "r4": _doc("b")},
        "postings": {"p1": _doc("java"), "p2": _doc("B"), "p3": _doc("a")},
    }
    result = counts.count_skills(records)
    assert [c.skill for c in result] == ["Java", "B", "a"]
    assert result[0] == _sc("Java", 4, resumes=3, postings=1)
    tie = counts.count_skills({"resumes": {"r1": _doc("Zed", "alpha", "Beta")}})
    assert [c.skill for c in tie] == ["alpha", "Beta", "Zed"]
    spelling_tie = counts.count_skills({"resumes": {"r1": _doc("JAVA"), "r2": _doc("Java")}})
    assert spelling_tie[0].skill == "JAVA"


def test_select_top_filters_and_caps() -> None:
    items = [_sc("a", 5), _sc("b", 3), _sc("c", 2), _sc("d", 1)]
    assert [c.skill for c in counts.select_top(items, top_n=10, min_df=2)] == ["a", "b", "c"]
    assert [c.skill for c in counts.select_top(items, top_n=2, min_df=1)] == ["a", "b"]
    with pytest.raises(ValueError):
        counts.select_top(items, top_n=0, min_df=1)
    with pytest.raises(ValueError):
        counts.select_top(items, top_n=1, min_df=0)


def test_skills_version() -> None:
    version = counts.skills_version(["a", "b"])
    assert version == counts.skills_version(["a", "b"])
    assert version != counts.skills_version(["b", "a"])
    assert version.startswith("sha256:") and len(version) == len("sha256:") + 12


def test_counts_summary_keys_and_values() -> None:
    all_counts = [_sc("a", 4, resumes=2, postings=2), _sc("b", 2, resumes=2), _sc("c", 2, 0, 2)]
    selected = all_counts[:2]
    summary = counts.counts_summary(
        all_counts, selected, {"resumes": 5, "postings": 6}, "sha256:abc", 10, 2
    )
    assert set(summary) == {
        "alias_table_version",
        "top_n",
        "min_df",
        "n_docs",
        "n_distinct_skills",
        "n_selected",
        "mention_coverage",
        "one_corpus_only_fraction",
        "selected_skills_version",
    }
    assert summary["mention_coverage"] == pytest.approx(6 / 8)
    assert summary["one_corpus_only_fraction"] == pytest.approx(0.5)
    assert summary["n_docs"] == {"resumes": 5, "postings": 6}
    assert summary["selected_skills_version"] == counts.skills_version(["a", "b"])
    empty = counts.counts_summary([], [], {}, "v", 1, 1)
    assert empty["mention_coverage"] == 0.0 and empty["one_corpus_only_fraction"] == 0.0


def test_write_counts_and_read_selected(tmp_path: Path) -> None:
    all_counts = [_sc("a", 4, 2, 2), _sc("b", 3, 3, 0), _sc("c", 1, 0, 1)]
    selected = all_counts[:2]
    summary = counts.counts_summary(all_counts, selected, {"resumes": 1}, "v", 2, 2)
    directory = tmp_path / "skills"
    counts.write_counts(directory, all_counts, selected, summary)

    table = pq.read_table(directory / "skill_counts.parquet")
    assert table.column_names == [
        "skill", "df_total", "df_resumes", "df_postings", "selected", "rank"
    ]
    for name in ("df_total", "df_resumes", "df_postings"):
        assert table.schema.field(name).type == pa.int64()
    assert table.schema.field("skill").type == pa.string()
    assert table.schema.field("selected").type == pa.bool_()
    assert table.schema.field("rank").type == pa.int32()
    assert table.column("skill").to_pylist() == ["a", "b", "c"]
    assert table.column("rank").to_pylist() == [1, 2, None]
    assert table.column("selected").to_pylist() == [True, True, False]
    assert json.loads((directory / "counts_summary.json").read_text()) == summary
    assert counts.read_selected(directory) == ["a", "b"]


def test_read_selected_missing_names_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="skill_counts.parquet"):
        counts.read_selected(tmp_path)
