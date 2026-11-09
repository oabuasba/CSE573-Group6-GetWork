import pytest

from getwork.webapp import explain, models

pytestmark = pytest.mark.usefixtures("offline")

Skills = list[tuple[str, str]]


def doc(
    side: models.Side, doc_id: str, title: str | None, level: str, skills: Skills
) -> models.DocumentView:
    return models.DocumentView(
        id=doc_id,
        side=side,
        title=title,
        level=level,
        level_source="listed",
        years_experience=None,
        skills=[
            models.DocSkill(skill=s, cluster=c, via="noise" if c == "Other" else "taxonomy")
            for s, c in skills
        ],
    )


def resume_ac4() -> models.DocumentView:
    skills = [
        ("Vue", "Front-end"),
        ("CSS", "Front-end"),
        ("JavaScript", "Front-end"),
        ("SQL", "Data"),
        ("Excel", "Other"),
    ]
    return doc("resume", "r-1", "Front-end developer", "Associate", skills)


def posting_ac4() -> models.DocumentView:
    skills = [
        ("React", "Front-end"),
        ("JavaScript", "Front-end"),
        ("CSS", "Front-end"),
        ("Kubernetes", "Cloud"),
        ("Excel", "Other"),
    ]
    return doc("posting", "j-1", "React developer", "Mid-Senior level", skills)


def test_evidence_shared_related_gaps() -> None:
    ev = explain.build_evidence("resume_to_posting", resume_ac4(), posting_ac4())
    assert [(s.skill, s.cluster) for s in ev.shared] == [
        ("JavaScript", "Front-end"),
        ("CSS", "Front-end"),
        ("Excel", "Other"),
    ]
    assert len(ev.related) == 1
    assert ev.related[0].cluster == "Front-end"
    assert ev.related[0].resume_skills == ["Vue"]
    assert ev.related[0].posting_skills == ["React"]
    assert [(g.skill, g.cluster) for g in ev.gaps] == [("Kubernetes", "Cloud")]
    assert ev.title_overlap == ["developer"]


def test_evidence_direction_independent() -> None:
    forward = explain.build_evidence("resume_to_posting", resume_ac4(), posting_ac4())
    backward = explain.build_evidence("posting_to_resume", posting_ac4(), resume_ac4())
    assert forward == backward


def test_evidence_other_never_related() -> None:
    resume = doc("resume", "r-1", "A", "Associate", [("Excel", "Other"), ("Word", "Other")])
    posting = doc("posting", "j-1", "B", "Associate", [("Tableau", "Other"), ("Excel", "Other")])
    ev = explain.build_evidence("resume_to_posting", resume, posting)
    assert [s.skill for s in ev.shared] == ["Excel"]
    assert ev.related == []
    assert [g.skill for g in ev.gaps] == ["Tableau"]


def test_evidence_edge_cases() -> None:
    resume = doc("resume", "r-1", "Data Scientist, the", "Associate", [("node.js ", "Back-end")])
    posting = doc("posting", "j-1", "Scientist of Data", "Associate", [("Node.js", "Back-end")])
    ev = explain.build_evidence("resume_to_posting", resume, posting)
    assert [s.skill for s in ev.shared] == ["Node.js"]
    assert ev.title_overlap == ["data", "scientist"]
    empty_resume = doc("resume", "r-2", None, "Associate", [])
    empty_posting = doc("posting", "j-2", "", "Associate", [])
    empty = explain.build_evidence("resume_to_posting", empty_resume, empty_posting)
    assert (empty.shared, empty.related, empty.gaps, empty.title_overlap) == ([], [], [], [])


@pytest.mark.parametrize(
    ("resume_level", "posting_level", "relation"),
    [
        ("Associate", "Associate", "same"),
        ("Associate", "Mid-Senior level", "adjacent"),
        ("Entry level", "Mid-Senior level", "different"),
        ("Not specified", "Associate", "unknown"),
        ("Associate", "Not specified", "unknown"),
    ],
)
def test_level_relations(resume_level: str, posting_level: str, relation: str) -> None:
    resume = doc("resume", "r-1", "x", resume_level, [])
    posting = doc("posting", "j-1", "x", posting_level, [])
    ev = explain.build_evidence("resume_to_posting", resume, posting)
    assert ev.level == models.LevelComparison(
        resume=resume_level, posting=posting_level, relation=relation
    )


def test_template_explanation_exact_text() -> None:
    result = explain.TemplateExplainer().explain(
        "resume_to_posting", resume_ac4(), posting_ac4()
    )
    assert result.summary == "3 shared skills, 1 related group, 1 gap."
    assert result.bullets == [
        "Shared skills: JavaScript, CSS, Excel.",
        "Related through Front-end: the resume lists Vue, the posting asks for React.",
        "The posting asks for skills the resume does not show: Kubernetes.",
        "Experience levels are adjacent (resume: Associate, posting: Mid-Senior level).",
        "Shared title words: developer.",
    ]
    assert result.explainer == models.ExplainerInfo(kind="template", model=None)
    assert explain.TemplateExplainer().info == result.explainer
    again = explain.TemplateExplainer().explain("resume_to_posting", resume_ac4(), posting_ac4())
    assert result == again


def test_template_overflow_and_fallback() -> None:
    names = [f"Skill{i}" for i in range(10)]
    both = [(n, "Other") for n in names]
    resume = doc("resume", "r-1", "Alpha", "Not specified", both)
    posting = doc("posting", "j-1", "Beta", "Not specified", both)
    result = explain.TemplateExplainer().explain("resume_to_posting", resume, posting)
    assert result.summary == "10 shared skills, 0 related groups, 0 gaps."
    assert result.bullets == [f"Shared skills: {', '.join(names[:8])} and 2 more."]
    lonely_resume = doc("resume", "r-1", "Alpha", "Not specified", [("A", "Other")])
    lonely_posting = doc("posting", "j-1", "Beta", "Not specified", [])
    fallback = explain.TemplateExplainer().explain(
        "resume_to_posting", lonely_resume, lonely_posting
    )
    assert fallback.summary == "0 shared skills, 0 related groups, 0 gaps."
    assert fallback.bullets == [
        "No shared or related skills were found; this match comes from overall document "
        "similarity."
    ]


def _names_in_text(result: models.Explanation) -> set[str]:
    names = {s.skill for s in result.evidence.shared} | {g.skill for g in result.evidence.gaps}
    for group in result.evidence.related:
        names |= set(group.resume_skills) | set(group.posting_skills)
    for bullet in result.bullets:
        if ": " in bullet and bullet.startswith(("Shared skills", "The posting asks")):
            names |= {n.strip() for n in bullet.split(": ", 1)[1].rstrip(".").split(", ")}
        if bullet.startswith("Related through"):
            lists = bullet.split("the resume lists ", 1)[1].rstrip(".")
            resume_part, posting_part = lists.split(", the posting asks for ")
            names |= set(resume_part.split(", ")) | set(posting_part.split(", "))
    return names


def test_explanation_names_only_known_skills() -> None:
    resume, posting = resume_ac4(), posting_ac4()
    known = {s.skill for s in resume.skills} | {s.skill for s in posting.skills}
    result = explain.TemplateExplainer().explain("resume_to_posting", resume, posting)
    found = _names_in_text(result)
    assert found
    assert found <= known
