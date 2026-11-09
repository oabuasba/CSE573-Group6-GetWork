from pathlib import Path

import pytest

from getwork.alias import build, normalize
from getwork.alias.build import CuratedPair

pytestmark = pytest.mark.usefixtures("offline")


def test_count_surface_forms_is_document_frequency() -> None:
    counts = build.count_surface_forms([["Java", "Java", " Java "], ["Java", "java"], [" ", ""]])
    assert counts == {"Java": 2, "java": 1}


def test_pick_display_rules() -> None:
    assert build.pick_display({"JAVASCRIPT": 5, "JavaScript": 3}) == "JavaScript"
    assert build.pick_display({"JAVA": 5, "Java": 3}) == "JAVA"
    assert build.pick_display({"HTML": 58, "Html": 5}) == "HTML"
    assert build.pick_display({"Java": 2, "java": 2}) == "Java"
    assert build.pick_display({"AWS": 4, "Aws": 4}) == "AWS"
    assert build.pick_display({"JAVASCRIPT": 3}) == "JAVASCRIPT"
    assert build.pick_display({"Java": 1, "java": 4}) == "java"
    with pytest.raises(ValueError):
        build.pick_display({})


def test_canonical_counts_sorted() -> None:
    table = build.build_table({}, [CuratedPair("js", "JavaScript")])
    forms = {"js": 2, "JavaScript": 1, "Zed": 3, "apple": 3, "Apple": 0, " ": 9}
    assert build.canonical_counts(forms, table) == [
        ("apple", 3), ("JavaScript", 3), ("Zed", 3), ("Apple", 0),
    ]


def test_build_table_merges_case_variants() -> None:
    table = build.build_table({"JAVASCRIPT": 5, "JavaScript": 3, "javascript": 1, "Go": 2}, [])
    assert table.canonical("JAVASCRIPT") == "JavaScript"
    assert table.canonical("javascript") == "JavaScript"
    assert table.canonical("Go") == "Go"


def test_curated_rows_override_and_self_map() -> None:
    table = build.build_table(
        {"javascript": 9, "js": 1}, [CuratedPair("js", "JavaScript")]
    )
    assert table.canonical("js") == "JavaScript"
    assert table.canonical("javascript") == "JavaScript"
    assert table.canonical("JavaScript") == "JavaScript"


def test_curated_conflicts_and_chains_rejected() -> None:
    with pytest.raises(ValueError, match="js"):
        build.build_table({}, [CuratedPair("js", "JavaScript"), CuratedPair("js", "Script")])
    with pytest.raises(ValueError, match="B"):
        build.build_table({}, [CuratedPair("A", "B"), CuratedPair("B", "C")])


def test_build_report_keys() -> None:
    forms = {f"Skill{i}": 1 for i in range(30)}
    forms.update({f"skill{i}": 1 for i in range(30)})
    forms["js"] = 1
    forms["JavaScript"] = 1
    table = build.build_table(forms, [CuratedPair("js", "JavaScript")])
    report = build.build_report(forms, table, n_curated=1)
    assert set(report) == {
        "alias_table_version", "n_keys", "n_raw_forms", "n_curated_pairs",
        "n_forms_merged", "top_merges",
    }
    assert report["n_raw_forms"] == 62
    assert report["n_curated_pairs"] == 1
    assert report["n_keys"] == len(table.entries)
    assert report["n_forms_merged"] == 62 - 31
    assert len(report["top_merges"]) == 20
    assert set(report["top_merges"][0]) == {"canonical", "forms"}


def test_load_curated_parsing_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "c.csv"
    path.write_text(
        'alias,canonical\n# comment\n\nJS , JavaScript\n"a,b",C\n', encoding="utf-8"
    )
    assert build.load_curated(path) == [
        CuratedPair("JS", "JavaScript"), CuratedPair("a,b", "C"),
    ]
    path.write_text("a,b\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        build.load_curated(path)
    path.write_text("alias,canonical\nx,y\nonly\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        build.load_curated(path)
    path.write_text("alias,canonical\nx,y\n . ,z\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        build.load_curated(path)


def test_shipped_curated_file_is_consistent() -> None:
    assert build.CURATED_PATH.is_file()
    pairs = build.load_curated()
    table = build.build_table({}, pairs)
    assert table.canonical("js") == "JavaScript"
    assert table.canonical("k8s") == "Kubernetes"
    for pair in pairs:
        key = normalize.normalize_key(pair.canonical)
        assert table.entries[key] == normalize.clean_surface(pair.canonical)
