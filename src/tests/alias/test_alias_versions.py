from pathlib import Path

import pytest

from getwork.alias import build, versions
from getwork.alias.versions import VersionFamily, fold_version, load_families

pytestmark = pytest.mark.usefixtures("offline")

FAMILIES = [
    VersionFamily("windows", "Windows"),
    VersionFamily("windows server", "Windows Server"),
    VersionFamily("microsoft windows server", "Windows Server"),
    VersionFamily("win", "Windows", map_bare=False),
    VersionFamily("server", "Windows Server", map_bare=False),
    VersionFamily("sql server", "SQL Server"),
    VersionFamily("sql", "SQL Server", map_bare=False),
    VersionFamily("oracle", "Oracle"),
    VersionFamily("angular", "Angular"),
    VersionFamily("python", "Python"),
    VersionFamily("java", "Java"),
    VersionFamily("html", "HTML"),
    VersionFamily("itil", "ITIL"),
]

FOLDS = {
    "Windows": ["Windows 7", "Windows 10", "Windows 11", "windows xp", "Windows NT 4.0",
                "Windows XP SP3", "Windows 7/8/10", "Windows 7, 8 and 10", "Win 7", "Win7",
                "Win2K"],
    "Windows Server": ["Windows Server 2019", "Windows Server 2008 R2", "Windows Server 2008R2",
                       "Microsoft Windows Server 2012 R2", "Server 2012"],
    "SQL Server": ["SQL Server 2019", "SQL 2008 R2"],
    "Oracle": ["Oracle 11gR2", "Oracle12c"],
    "Angular": ["Angular2", "Angular 1.x"],
    "Python": ["Python 3.9", "Python3"],
    "Java": ["Java 1.8"],
    "HTML": ["HTML5", "HTML 4.01"],
    "ITIL": ["ITIL v3", "ITILv3"],
}

NOT_FOLDED = [
    "Windows", "SQL", "Server", "Win", "Windows Phone", "Windows Azure", "Windows 7 Professional",
    "Java ME", "JavaScript", "Python Django", "SQL Server Express", "Spring Boot", "Cisco 3750",
    "S3", "ISO 27001", "", " . ", "Java 8 / Python 3",
]


@pytest.mark.parametrize(
    ("skill", "canonical"), [(s, c) for c, skills in FOLDS.items() for s in skills]
)
def test_fold_version_folds_version_forms(skill: str, canonical: str) -> None:
    assert fold_version(skill, FAMILIES) == canonical


@pytest.mark.parametrize("skill", NOT_FOLDED)
def test_fold_version_leaves_other_skills_alone(skill: str) -> None:
    assert fold_version(skill, FAMILIES) is None


def test_fold_version_normalization_and_longest_base() -> None:
    assert fold_version("  WINDOWS   Server  2012. ", FAMILIES) == "Windows Server"
    short, long = VersionFamily("x", "X"), VersionFamily("x 2", "X Two")
    assert fold_version("x 2-3", [short, long]) == "X Two"
    assert fold_version("x 2-3", [long, short]) == "X Two"
    first, second = VersionFamily("x", "First"), VersionFamily(" X ", "Second")
    assert fold_version("x 1", [first, second]) == "First"
    assert fold_version("x 1", [VersionFamily("x", "  Spaced   Name ")]) == "Spaced Name"


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "families.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_families_parsing_and_errors(tmp_path: Path) -> None:
    ok = _write(
        tmp_path,
        "base,canonical,map_bare\n# comment\n\nwindows,Windows,yes\nsql,SQL Server,NO\n"
        "java,Java,\nphp,PHP\n",
    )
    assert load_families(ok) == [
        VersionFamily("windows", "Windows", True),
        VersionFamily("sql", "SQL Server", False),
        VersionFamily("java", "Java", True),
        VersionFamily("php", "PHP", True),
    ]
    head = "base,canonical,map_bare\n"
    bad = {
        "": "line 1",
        "base,alias\nx,X\n": "line 1",
        head + "# c\n,X,yes\n": "line 3",
        head + "x,,yes\n": "line 2",
        head + "x, . ,yes\n": "line 2",
        head + "ok,OK,yes\nx,X,maybe\n": "line 3",
        head + "x,X,yes\n\n X ,Y,no\n": "line 4",
    }
    for text, line in bad.items():
        with pytest.raises(ValueError, match=line):
            load_families(_write(tmp_path, text))
    with pytest.raises(FileNotFoundError):
        load_families(tmp_path / "missing.csv")


def test_constants() -> None:
    assert versions.FAMILIES_HEADER == ["base", "canonical", "map_bare"]


def _report(forms: dict[str, int], curated: list[build.CuratedPair] | None = None) -> dict:
    table = build.build_table(forms, curated or [], FAMILIES)
    return versions.version_report(forms, FAMILIES, table)


def test_version_report_keys_and_folds() -> None:
    forms = {
        "Windows": 5, "Windows 7": 3, "Windows 10": 2, "Windows Server 2019": 9,
        "Python 3.9": 1, "Java": 1,
    }
    report = _report(forms)
    assert list(report) == [
        "alias_table_version", "n_families", "n_version_forms_folded",
        "top_version_folds", "version_candidates",
    ]
    assert report["n_families"] == len(FAMILIES)
    assert report["n_version_forms_folded"] == 4
    assert report["top_version_folds"] == [
        {"canonical": "Windows Server", "forms": ["Windows Server 2019"]},
        {"canonical": "Windows", "forms": ["Windows 10", "Windows 7"]},
        {"canonical": "Python", "forms": ["Python 3.9"]},
    ]
    assert report == _report(forms)
    many = {f"P{i:02d}x 1": 1 for i in range(30)}
    many_families = [VersionFamily(f"p{i:02d}x", f"Canon {i:02d}") for i in range(30)]
    table = build.build_table(many, [], many_families)
    capped = versions.version_report(many, many_families, table)
    assert len(capped["top_version_folds"]) == versions.TOP_VERSION_FOLDS


def test_version_report_candidates() -> None:
    forms = {
        "Cisco": 4, "Cisco 3750": 3, "Cisco 2960": 3, "Windows 7": 9, "Foo": 1, "Foo 2": 8,
        "Alone 5": 7, "Cisco 1900": 5, "Bare": 1, "Java": 1, "Java 8": 2, "S3": 1,
    }
    curated = [build.CuratedPair("Foo 2", "Foo")]
    report = _report(forms, curated)
    assert report["version_candidates"] == [
        {"skill": "Cisco 1900", "base": "Cisco", "count": 5},
        {"skill": "Cisco 2960", "base": "Cisco", "count": 3},
        {"skill": "Cisco 3750", "base": "Cisco", "count": 3},
    ]
    many = {"Base": 1} | {f"Base {i}": 1 for i in range(60)}
    capped = versions.version_report(many, [], build.build_table(many, []))
    assert len(capped["version_candidates"]) == versions.VERSION_CANDIDATE_LIMIT
