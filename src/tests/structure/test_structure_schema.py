import copy
import json

import pytest

from getwork.structure import schema

pytestmark = pytest.mark.usefixtures("offline")

VALID = {
    "title": "Data Engineer",
    "skills": ["Python", "SQL"],
    "experience": [{"role": "Engineer", "company": "Acme", "duration_months": 12.0}],
    "education": [{"degree": "BS", "field": "CS", "institution": "ASU"}],
    "years_experience": 3,
}


def make(**overrides: object) -> dict:
    data = copy.deepcopy(VALID)
    data.update(overrides)
    return data


def make_entry(**overrides: object) -> dict:
    return make(experience=[{**VALID["experience"][0], **overrides}])


def parse(data: dict) -> schema.ModelOutput:
    return schema.parse_model_output(json.dumps(data))


def test_parse_accepts_valid_output() -> None:
    out = parse(VALID)
    assert out.title == "Data Engineer"
    assert out.experience[0].duration_months == 12
    assert isinstance(out.experience[0].duration_months, int)
    assert out.years_experience == 3.0
    assert isinstance(out.years_experience, float)


def _missing_key() -> dict:
    data = make()
    del data["skills"]
    return data


REJECTED = {
    "missing_key": _missing_key(),
    "extra_top_key": make(bonus=1),
    "extra_entry_key": make(experience=[{**VALID["experience"][0], "x": 1}]),
    "empty_title": make(title=""),
    "whitespace_title": make(title="   "),
    "role_company_blank": make(experience=[{"role": " ", "company": None, "duration_months": 1}]),
    "duration_negative": make_entry(duration_months=-1),
    "duration_721": make_entry(duration_months=721),
    "duration_fractional": make_entry(duration_months=12.5),
    "duration_string": make_entry(duration_months="12"),
    "duration_bool": make_entry(duration_months=True),
    "years_61": make(years_experience=61),
    "years_string": make(years_experience="5"),
    "years_bool": make(years_experience=True),
    "skill_too_long": make(skills=["x" * 101]),
    "too_many_skills": make(skills=[f"skill{i}" for i in range(201)]),
    "title_list": make(title=["a"]),
    "title_number": make(title=5),
}


@pytest.mark.parametrize("data", REJECTED.values(), ids=REJECTED.keys())
def test_parse_rejects_invalid_output(data: dict) -> None:
    with pytest.raises(schema.OutputValidationError):
        parse(data)


def test_skills_trimmed_deduped_in_order() -> None:
    out = parse(make(skills=["Python", " python", "", "Git"]))
    assert out.skills == ["Python", "Git"]


def test_blank_optional_fields_become_none() -> None:
    out = parse(
        make(
            experience=[{"role": "  ", "company": "Acme", "duration_months": None}],
            education=[{"degree": " ", "field": "", "institution": "  "}],
        )
    )
    assert out.experience[0].role is None
    assert out.experience[0].company == "Acme"
    edu = out.education[0]
    assert (edu.degree, edu.field, edu.institution) == (None, None, None)


def test_parse_strips_code_fence_and_think_block() -> None:
    body = json.dumps(VALID)
    assert parse_text(f"```json\n{body}\n```").title == "Data Engineer"
    assert parse_text(f"<think>step one\nstep two</think>\n{body}").title == "Data Engineer"


def parse_text(text: str) -> schema.ModelOutput:
    return schema.parse_model_output(text)


def test_parse_error_messages() -> None:
    with pytest.raises(schema.OutputValidationError, match=r"^empty reply"):
        parse_text("  \n")
    with pytest.raises(schema.OutputValidationError, match=r"^not valid JSON"):
        parse_text("{nope")
    with pytest.raises(schema.OutputValidationError, match=r"^expected a JSON object"):
        parse_text("[1, 2]")


def test_parse_error_message_omits_input_values() -> None:
    with pytest.raises(schema.OutputValidationError) as extra:
        parse(make(bonus="do-not-echo"))
    assert "do-not-echo" not in str(extra.value)
    with pytest.raises(schema.OutputValidationError) as title:
        parse(make(title=""))
    assert str(title.value).startswith("title: ")
