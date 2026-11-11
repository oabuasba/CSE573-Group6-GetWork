import pytest

from getwork.taxonomy import overlap
from getwork.taxonomy.overlap import OverlapGroup

pytestmark = pytest.mark.usefixtures("offline")

V4_NAMES = {
    0: "Front-end Development",
    1: "Desktop Applications",
    2: "Relational Databases",
    3: "Development Tools",
    4: "IT Service Management",
    5: "Software Development Tools",
    6: "Server-Side Programming",
    7: "Cloud Infrastructure",
    8: "Network Infrastructure",
    9: "Linux and UNIX Administration",
}


def test_name_words() -> None:
    assert overlap.name_words("Development Tools") == {"development", "tool"}
    assert overlap.name_words("Tool") == overlap.name_words("Tools") == {"tool"}
    assert overlap.name_words("Class") == {"class"}
    assert overlap.name_words("Administration of Databases") == overlap.name_words(
        "Database Administration"
    )
    assert overlap.name_words("Front-end Development") == {"front", "end", "development"}
    assert overlap.name_words("The and of") == frozenset() == overlap.name_words("!!! ---")


def test_rule_overlap_cases() -> None:
    reason = overlap.rule_overlap("Development Tools", "Software Development Tools")
    assert reason is not None and "contains the other" in reason
    assert "development, tool" in reason
    assert overlap.rule_overlap("Tool", "Tools") == "the names use the same words"
    assert overlap.rule_overlap("Administration of Databases", "Database Administration")
    jaccard = overlap.rule_overlap("Network Security Tools", "Network Security Monitoring")
    assert jaccard is not None and "share most of their words" in jaccard
    assert "network, security" in jaccard
    for first, second in [
        ("Cloud Infrastructure", "Network Infrastructure"),
        ("Front-end Development", "Server-Side Programming"),
        ("Python", "Java"),
        ("!!!", "!!!"),
        ("", "Tools"),
    ]:
        assert overlap.rule_overlap(first, second) is None
    for first, second in [("Security", "Network Security"), ("a b", "b a")]:
        found = overlap.rule_overlap(first, second)
        assert found is not None and "\n" not in found


def test_rule_groups_v4_names() -> None:
    groups = overlap.rule_groups(V4_NAMES)
    assert len(groups) == 1
    group = groups[0]
    assert (group.ids, group.source) == ((3, 5), "rules")
    assert "'Development Tools'" in group.reason and "'Software Development Tools'" in group.reason
    assert overlap.rule_groups(V4_NAMES) == groups


def test_rule_groups_chain_and_cap() -> None:
    chain = overlap.rule_groups({7: "Data Tools", 2: "Data", 5: "Data Tools Plus", 1: "Cloud"})
    assert [g.ids for g in chain] == [(2, 5, 7)]
    assert overlap.rule_groups({0: "Cloud", 1: "Linux"}) == []
    many = {i: "Data" if i == 0 else f"Data Topic{i}" for i in range(5)}
    group = overlap.rule_groups(many)[0]
    assert group.ids == (0, 1, 2, 3, 4) and group.reason.count(" and ") == 3
    assert group.reason.endswith("; ...") and len(group.reason) <= overlap.MAX_REASON_CHARS
    two = overlap.rule_groups({3: "Cloud Tools", 1: "Cloud Tool Kit", 9: "Linux Tool"})
    assert [g.ids for g in two] == [(1, 3)]
    ordered = overlap.rule_groups({5: "Data Tools", 6: "Data Tool", 1: "Web Apps", 2: "Web App"})
    assert [g.ids for g in ordered] == [(1, 2), (5, 6)]


def test_merge_groups() -> None:
    rules = [OverlapGroup((1, 2), "r12", "rules"), OverlapGroup((7, 8), "r78", "rules")]
    judge = [OverlapGroup((2, 3), "j23", "judge"), OverlapGroup((4, 5), "j45", "judge"),
             OverlapGroup((7, 8), "r78", "judge")]
    merged = overlap.merge_groups(rules, judge)
    assert merged == [
        OverlapGroup((1, 2, 3), "r12 | j23", "rules+judge"),
        OverlapGroup((4, 5), "j45", "judge"),
        OverlapGroup((7, 8), "r78", "rules+judge"),
    ]
    assert overlap.merge_groups(rules) == rules
    assert overlap.merge_groups() == []
    long = [OverlapGroup((0, 1), "x" * 300, "judge"), OverlapGroup((1, 2), "y" * 300, "judge")]
    assert len(overlap.merge_groups(long)[0].reason) == overlap.MAX_REASON_CHARS


def test_rule_groups_reason_cap_boundary() -> None:
    three = {0: "Data", 1: "Data X", 2: "Data Y", 3: "Data Z"}
    group = overlap.rule_groups(three)[0]
    assert group.ids == (0, 1, 2, 3) and group.reason.count(" and ") == 3
    assert not group.reason.endswith("...")
