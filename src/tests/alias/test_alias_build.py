from pathlib import Path

import pytest

from getwork.alias import build, normalize
from getwork.alias.build import CuratedPair
from getwork.alias.versions import VersionFamily

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


def _families(*rows: tuple[str, str, bool]) -> list[VersionFamily]:
    return [VersionFamily(base, canonical, map_bare) for base, canonical, map_bare in rows]


WINDOWS_FAMILIES = _families(("windows", "Windows", True), ("windows server", "Windows Server", True))


def test_build_table_folds_versions() -> None:
    forms = {"Windows 7": 3, "windows 10": 2, "Windows Server 2019": 1, "Cisco 3750": 1}
    table = build.build_table(forms, [], WINDOWS_FAMILIES)
    assert table.canonical("Windows 7") == "Windows"
    assert table.canonical("windows 10") == "Windows"
    assert table.canonical("Windows Server 2019") == "Windows Server"
    assert table.canonical("Cisco 3750") == "Cisco 3750"


def test_build_table_map_bare_and_self_map() -> None:
    families = _families(("sql", "SQL Server", False), ("windows", "Windows", True))
    table = build.build_table({"SQL": 5, "SQL 2008": 1, "Windows": 1, "win": 1}, [], families)
    assert table.canonical("SQL") == "SQL"
    assert table.canonical("SQL 2008") == "SQL Server"
    assert table.canonical("SQL Server") == "SQL Server"
    assert table.entries["sql server"] == "SQL Server"
    assert table.canonical("windows") == "Windows"


def test_build_table_family_conflict_rejected() -> None:
    curated = [CuratedPair("windows", "Windows OS")]
    with pytest.raises(ValueError, match="'windows'"):
        build.build_table({"Windows 7": 1}, curated, _families(("windows", "Windows", True)))


def test_build_table_without_families_is_unchanged() -> None:
    forms = {"Windows 7": 2, "JAVA": 1, "js": 1}
    curated = [CuratedPair("js", "JavaScript")]
    plain = build.build_table(forms, curated)
    assert build.build_table(forms, curated, ()).entries == plain.entries
    assert build.build_table(forms, curated, ()).version == plain.version
    with_families = build.build_table(forms, curated, WINDOWS_FAMILIES)
    assert with_families.version != plain.version


def test_curated_row_beats_family_fold() -> None:
    forms = {"Win 7": 1, "Windows 7": 1}
    curated = [CuratedPair("win 7", "Windows Seven")]
    table = build.build_table(forms, curated, _families(("windows", "Windows", True)))
    assert table.canonical("Win 7") == "Windows Seven"
    assert table.canonical("Windows 7") == "Windows"
