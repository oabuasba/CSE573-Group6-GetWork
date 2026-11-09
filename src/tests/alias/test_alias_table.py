from pathlib import Path

import pytest

from getwork.alias.normalize import clean_surface, normalize_key
from getwork.alias.table import AliasTable, load_table, make_table, save_table, table_version

pytestmark = pytest.mark.usefixtures("offline")


def test_normalize_key_cases() -> None:
    assert normalize_key("  Java ") == normalize_key("JAVA") == normalize_key("java.") == "java"
    assert normalize_key("C++") == "c++"
    assert normalize_key("C#") == "c#"
    assert normalize_key(".NET") == ".net"
    assert normalize_key("Node.js") == "node.js"
    assert normalize_key("Node.js.") == "node.js"
    assert normalize_key("ＪＡＶＡ") == "java"
    assert normalize_key("") == ""
    assert normalize_key(" . ") == ""


def test_clean_surface_keeps_case() -> None:
    assert clean_surface("  Node \t JS\n") == "Node JS"
    assert clean_surface("Ｊava.") == "Java."


def test_canonical_and_unknown_passthrough() -> None:
    table = make_table({"js": "JavaScript"})
    assert table.canonical("JS") == "JavaScript"
    assert table.canonical(" . ") is None
    assert table.canonical("  Rust  Lang ") == "Rust Lang"


def test_canonicalize_list_dedupes_by_key() -> None:
    table = make_table({"js": "JavaScript", "javascript": "JavaScript"})
    result = table.canonicalize_list(["Python", "", "JS", "python", "javascript", "Go"])
    assert result == ["Python", "JavaScript", "Go"]


def test_table_version_changes_with_entries() -> None:
    a = table_version({"a": "A"})
    assert a == table_version({"a": "A"})
    assert a != table_version({"a": "B"})
    assert a != table_version({"a": "A", "b": "B"})
    prefix, _, digest = a.partition(":")
    assert prefix == "sha256" and len(digest) == 12
    int(digest, 16)


def test_save_load_roundtrip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "alias" / "alias_table.json"
    table = make_table({"js": "JavaScript"})
    save_table(path, table, {"n_curated_pairs": 1})
    loaded = load_table(path)
    assert loaded == AliasTable({"js": "JavaScript"}, table.version)

    path.write_text(path.read_text().replace("JavaScript", "Javascript"), encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_table(path)

    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="nope.json"):
        load_table(missing)
