import pandas as pd

from getwork.preprocess import domain


def test_title_families():
    f = domain.domain_families
    assert f(title="React Developer", industries=["Construction"]) == ["software"]
    assert f(title="Salesforce Developer", industries=[]) == ["software"]
    assert f(title="DevOps / Systems Engineer", industries=[]) == ["software", "sysadmin"]
    assert f(title="Postgres Database Administrator", industries=[]) == ["database"]
    assert f(title="Network Engineer", industries=[]) == ["network"]
    assert f(title="Cyber Security Analyst", industries=[]) == ["security"]


def test_exclusions_and_non_matches():
    for title in ["Real Estate Developer", "Business Development Manager",
                  "Sales Engineer", "HVAC Programmer", "Registered Nurse"]:
        assert domain.domain_families(title=title, industries=[]) == []


def test_project_manager_needs_it_signal():
    f = domain.domain_families
    assert f(title="Project Manager", industries=["Construction"]) == []
    assert f(title="Project Manager", industries=["IT Services and IT Consulting"]) == ["project_manager"]
    assert f(title="IT Project Manager", industries=["Construction"]) == ["project_manager"]
    assert f(title="Software Engineer", industries=["Construction"]) == ["software"]


def test_apply_domain_filter_partition():
    df = pd.DataFrame({
        "id": ["j-1", "j-2", "j-3", "j-4"],
        "title": ["Registered Nurse", "React Developer", "Project Manager", "IT Project Manager"],
        "industries": [[], ["Construction"], ["Construction"], []],
    })
    kept, dropped = domain.apply_domain_filter(df)
    assert list(kept["id"]) == ["j-2", "j-4"]
    assert list(kept["families"]) == [["software"], ["project_manager"]]
    assert list(kept.columns) == ["id", "title", "industries", "families"]
    assert list(dropped.columns) == ["id", "rule", "detail"]
    assert set(dropped["rule"]) == {"not_in_domain"}
    assert set(dropped["detail"]) == {""}
    assert set(kept["id"]).isdisjoint(dropped["id"])
    assert set(kept["id"]) | set(dropped["id"]) == set(df["id"])
