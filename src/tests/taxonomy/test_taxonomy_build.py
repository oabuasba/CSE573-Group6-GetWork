from pathlib import Path

import numpy as np
import pytest

from getwork.taxonomy import build

pytestmark = pytest.mark.usefixtures("offline")


def _write(path: Path, rows: list[str], header: str = "cluster_id,name,size") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_read_names_validation(tmp_path: Path) -> None:
    good = _write(tmp_path / "ok.csv", ["0, Front-end ,3", "10,Cloud,2"])
    assert build.read_names(good) == {0: "Front-end", 10: "Cloud"}
    with pytest.raises(FileNotFoundError, match="nope.csv.*taxonomy name"):
        build.read_names(tmp_path / "nope.csv")

    def fails(rows: list[str], match: str, secret: str = "") -> None:
        with pytest.raises(ValueError, match=match) as info:
            build.read_names(_write(tmp_path / "bad.csv", rows))
        if secret:
            assert secret not in str(info.value)

    fails(["0,Fine,1", "3,A: B,1", "7,Other,1"], r"cluster ids \[3, 7\]", secret="A: B")
    fails(["0,Fine,1", "4,,1"], r"cluster ids \[4\]")
    fails(["1,A,1", "1,B,1"], r"cluster ids \[1\]")
    fails(["x,A,1"], "not an integer")
    fails(["2,Same,1", "5,SAME,1", "6,Other name,1"], r"cluster ids \[2, 5\]", secret="Same")
    with pytest.raises(ValueError, match="cluster_id and name"):
        build.read_names(_write(tmp_path / "h.csv", ["a,b"], header="id,label"))


def test_name_status() -> None:
    names = {0: "Cloud", 1: "cloud", 2: "Data", 3: "CLOUD"}
    assert build.name_status(names) == {0: "duplicate", 1: "duplicate", 2: "ok", 3: "duplicate"}
    assert build.name_status({}) == {}
    assert build.NAMES_COLUMNS == (
        "cluster_id", "name", "size", "top_members", "method", "status", "model", "backend",
        "prompt_version",
    )


def test_build_taxonomy_from_labels() -> None:
    skills = ["Python", "Java", "AWS", "Excel", "Go", "Azure"]
    labels = np.array([0, 0, 1, -1, 0, 1])
    names = {1: "Cloud", 0: "Languages"}
    tax = build.build_taxonomy_from_labels(
        skills, labels, names, alias_table_version="sha256:abc", meta={"k": 1}
    )
    assert [(c.id, c.name, c.skills) for c in tax.clusters] == [
        (0, "Languages", ("Python", "Java", "Go")),
        (1, "Cloud", ("AWS", "Azure")),
    ]
    assert tax.noise == ("Excel",)
    assert tax.alias_table_version == "sha256:abc" and tax.meta == {"k": 1}
    assert tax.lookup("excel") == "Other" and tax.lookup("go") == "Languages"
    no_noise = build.build_taxonomy_from_labels(
        ["a", "b"], np.array([0, 1]), {0: "A", 1: "B"}, alias_table_version="v", meta={}
    )
    assert no_noise.noise == ()
    for bad_names in ({0: "A"}, {0: "A", 1: "B", 2: "C"}):
        with pytest.raises(ValueError, match="cluster ids"):
            build.build_taxonomy_from_labels(
                ["a", "b"], np.array([0, 1]), bad_names, alias_table_version="v", meta={}
            )
    with pytest.raises(ValueError, match=r"\[2\]"):
        build.build_taxonomy_from_labels(
            ["a", "b"], np.array([0, 1]), {0: "A", 1: "B", 2: "C"}, alias_table_version="v", meta={}
        )


def test_read_names_rejects_non_integer_and_empty_ids_with_line_number(tmp_path: Path) -> None:
    for bad_id, line in (("1.5", 3), ("", 2)):
        rows = ["0,Fine,1", f"{bad_id},Other name,1"] if line == 3 else [f"{bad_id},A,1"]
        with pytest.raises(ValueError, match=rf"line {line}.*not an integer"):
            build.read_names(_write(tmp_path / "ids.csv", rows))


def test_read_names_accepts_excel_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_bytes(b"\xef\xbb\xbfcluster_id,name\n0,Cloud\n")
    assert build.read_names(path) == {0: "Cloud"}


def test_build_taxonomy_from_labels_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        build.build_taxonomy_from_labels(
            ["a", "b", "c"], np.array([0, 1]), {0: "A", 1: "B"}, alias_table_version="v", meta={}
        )
