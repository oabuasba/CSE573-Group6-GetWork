import pytest

from getwork.llm.fake import make_output_json
from getwork.structure import schema

pytestmark = pytest.mark.usefixtures("offline")

LIMIT_KEYS = {
    "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "minItems", "maxItems",
}


def _objects(node: object) -> list[dict]:
    found: list[dict] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_objects(item))
    return found


def _all_keys(node: object, in_properties: bool = False) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if not in_properties:
                keys.add(key)
            keys |= _all_keys(value, in_properties=(key == "properties" and not in_properties))
    elif isinstance(node, list):
        for item in node:
            keys |= _all_keys(item)
    return keys


def test_response_schema_is_strict_shape() -> None:
    result = schema.response_schema()
    assert not _all_keys(result) & LIMIT_KEYS
    assert "title" in result["properties"]
    assert result["title"] == "ModelOutput"
    objects = _objects(result)
    assert len(objects) == 3
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_to_record_key_order_and_provenance() -> None:
    output = schema.parse_model_output(make_output_json(years_experience=3))
    record = schema.to_record(
        output, record_id="r-1", source="resumes", model="m", prompt_version="extract_v1"
    )
    assert list(record) == list(schema.RECORD_KEYS)
    assert record["id"] == "r-1"
    assert record["model"] == "m"
    assert record["prompt_version"] == "extract_v1"
    assert record["schema_version"] == schema.SCHEMA_VERSION
    assert record["skills"] == ["Python"]


def test_make_output_json_parses() -> None:
    assert schema.parse_model_output(make_output_json()).title == "Software Engineer"
    output = schema.parse_model_output(
        make_output_json(
            experience=[{"role": "Dev", "company": "Acme", "duration_months": 12}],
            education=[{"degree": "BS", "field": "CS", "institution": "ASU"}],
            years_experience=1.5,
        )
    )
    assert output.experience[0].company == "Acme"
    assert output.education[0].degree == "BS"
    assert output.years_experience == 1.5
