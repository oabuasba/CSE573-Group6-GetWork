import json
from pathlib import Path

import pytest

from getwork.taxonomy import model
from getwork.taxonomy.model import Cluster

pytestmark = pytest.mark.usefixtures("offline")


def _tax(meta: dict | None = None, name: str = "Web") -> model.Taxonomy:
    clusters = [
        Cluster(id=2, name="Cloud", skills=("AWS", "Azure")),
        Cluster(id=1, name=name, skills=("Vue", "React", "CSS")),
    ]
    return model.make_taxonomy(clusters, ["Excel"], "sha256:abc", meta or {})


def test_check_name_rules() -> None:
    assert model.check_name("  Web  ") == "Web"
    assert model.check_name("x" * 40) == "x" * 40
    for bad in ["", "  ", "x" * 41, "a:b", "a;b", "a,b", "a\nb", "a\rb", "other", "OTHER"]:
        with pytest.raises(ValueError) as e:
            model.check_name(bad)
        assert len(str(e.value)) <= 120


def test_member_list_reason() -> None:
    members = ["junit", "Selenium", "TDD", "JMS", "RabbitMQ", "Java", "Spring", "Windows",
               "Active Directory", "Windows 7", "AWS", "Python"]
    for listed in ["JUnit / Selenium / TDD", "JUnit Selenium", "JMS & RabbitMQ", "Java and Spring",
                   "Windows / Active Directory / Windows 7", "JUNIT  /  selenium"]:
        reason = model.member_list_reason(listed, members)
        assert reason is not None and "member skills" in reason
    for fine in ["Unit Testing", "Java Frameworks", "Cloud & AWS", "CI/CD Pipelines", "Front-end",
                 "Python"]:
        assert model.member_list_reason(fine, members) is None
    assert model.member_list_reason("JUnit / Selenium / TDD", []) is None


def test_make_taxonomy_validation() -> None:
    def make(clusters: list[Cluster], noise: list[str] | None = None) -> model.Taxonomy:
        return model.make_taxonomy(clusters, noise or [], "v", {})

    with pytest.raises(ValueError):
        make([])
    with pytest.raises(ValueError):
        make([Cluster(1, "Web", ("a",)), Cluster(2, "web", ("b",))])
    with pytest.raises(ValueError):
        make([Cluster(1, "Web", ("a",)), Cluster(1, "Cloud", ("b",))])
    with pytest.raises(ValueError):
        make([Cluster(1, "Web", ("a",)), Cluster(2, "Cloud", ("b",))], ["A"])
    with pytest.raises(ValueError, match="(?i)java"):
        make([Cluster(1, "Web", ("Java", "JAVA"))])
    with pytest.raises(ValueError):
        make([Cluster(1, "other", ("a",))])


def test_lookup_semantics() -> None:
    tax = _tax()
    assert tax.lookup("vue") == "Web"
    assert tax.lookup(" AWS. ") == "Cloud"
    assert tax.lookup("excel") == "Other"
    assert tax.lookup("cobol") is None
    assert tax.is_known("css") and not tax.is_known("cobol")
    assert tax.cluster_names() == ["Web", "Cloud"]
    assert tax.examples(2) == {"Web": ["Vue", "React"], "Cloud": ["AWS", "Azure"]}
    assert tax.examples()["Web"] == ["Vue", "React", "CSS"]


def test_version_changes_with_names_and_membership() -> None:
    base = _tax()
    assert base.version == _tax().version
    assert base.version.startswith("sha256:") and len(base.version) == len("sha256:") + 12
    assert _tax(meta={"seed": 1}).version == base.version
    assert _tax(name="Frontend").version != base.version
    moved = model.make_taxonomy(
        [Cluster(1, "Web", ("Vue", "React")), Cluster(2, "Cloud", ("AWS", "Azure", "CSS"))],
        ["Excel"],
        "sha256:abc",
        {},
    )
    assert moved.version != base.version


def test_save_load_roundtrip_and_tamper(tmp_path: Path) -> None:
    tax = _tax(meta={"method": "kmeans", "k": 2})
    path = tmp_path / "tax.json"
    model.save_taxonomy(path, tax)
    raw = json.loads(path.read_text())
    assert (raw["n_skills"], raw["n_clusters"], raw["n_noise"]) == (6, 2, 1)
    loaded = model.load_taxonomy(path)
    assert loaded == tax and loaded.lookup("vue") == "Web" and loaded.meta == tax.meta
    raw["clusters"][0]["name"] = "Renamed"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="version"):
        model.load_taxonomy(path)
    with pytest.raises(FileNotFoundError, match="missing.json"):
        model.load_taxonomy(tmp_path / "missing.json")


def test_load_rejects_malformed_types(tmp_path: Path) -> None:
    path = tmp_path / "tax.json"
    model.save_taxonomy(path, _tax())
    raw = json.loads(path.read_text())
    bad_name = {**raw, "clusters": [{**raw["clusters"][0], "name": 5}]}
    bad_noise = {**raw, "noise": "Excel"}
    for content in (bad_name, bad_noise):
        path.write_text(json.dumps(content))
        with pytest.raises(ValueError, match="tax.json"):
            model.load_taxonomy(path)
