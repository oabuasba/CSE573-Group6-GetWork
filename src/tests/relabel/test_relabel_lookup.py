import pytest

from getwork.relabel import lookup
from getwork.relabel.lookup import SkillAssignment
from getwork.taxonomy.model import OTHER, Cluster, Taxonomy, make_taxonomy

pytestmark = pytest.mark.usefixtures("offline")


def _tax() -> Taxonomy:
    return make_taxonomy(
        [Cluster(0, "Front-end", ("React", "Vue.js")), Cluster(1, "Cloud", ("AWS",))],
        noise=("Excel",),
        alias_table_version="sha256:t",
        meta={},
    )


def test_split_known_via_and_order() -> None:
    skills = ["aws", "Rust", "REACT", "excel", "Go", "Vue.js"]
    assignments, unseen = lookup.split_known(skills=skills, tax=_tax())
    assert assignments == [
        SkillAssignment(skill="aws", cluster="Cloud", via="taxonomy"),
        SkillAssignment(skill="REACT", cluster="Front-end", via="taxonomy"),
        SkillAssignment(skill="excel", cluster=OTHER, via="noise"),
        SkillAssignment(skill="Vue.js", cluster="Front-end", via="taxonomy"),
    ]
    assert unseen == ["Rust", "Go"]
    assert lookup.split_known(skills=[], tax=_tax()) == ([], [])
    assert lookup.VIA_VALUES == ("taxonomy", "noise", "llm", "default")


def test_known_fraction() -> None:
    tax = _tax()
    assert lookup.known_fraction([], tax) is None
    assert lookup.known_fraction(["AWS", "Excel", "Rust", "Go"], tax) == 0.5
    assert lookup.known_fraction(["Excel"], tax) == 1.0
    assert lookup.known_fraction(["Rust"], tax) == 0.0


def test_percentile_threshold() -> None:
    assert lookup.percentile_threshold(list(range(1, 11))) == 1
    assert lookup.percentile_threshold([5] * 10) == 5
    assert lookup.percentile_threshold([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 50.0) == 6
    assert lookup.percentile_threshold([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) == 10
    assert lookup.percentile_threshold([0, 100]) == 0  # "lower", not "linear" (which gives 10)
    assert isinstance(lookup.percentile_threshold([3, 4]), int)
    with pytest.raises(ValueError):
        lookup.percentile_threshold([])


def test_incomplete_reason_rules_and_priority() -> None:
    reason = lookup.incomplete_reason
    assert reason(0, None, 3) == "no_skills"
    assert reason(0, None, 0) == "no_skills"
    assert reason(2, 0.1, 3) == "few_skills"
    assert reason(3, 1.0, 3) is None
    assert reason(3, 0.1, 3) == "low_coverage"
    assert reason(4, 0.49, 3) == "low_coverage"
    assert reason(4, 0.5, 3) is None
    assert reason(4, 0.6, 3, min_known=0.7) == "low_coverage"
    assert reason(4, None, 3) is None
