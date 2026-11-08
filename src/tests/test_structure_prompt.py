from pathlib import Path

import pytest

from getwork.structure import prompt
from getwork.structure.schema import ModelOutput

pytestmark = pytest.mark.usefixtures("offline")


def _template_block(text: str) -> str:
    lines = text.splitlines()
    start = lines.index("{")
    end = lines.index("}", start)
    return "\n".join(lines[start : end + 1])


def _property_names(schema: dict) -> set[str]:
    names = set(schema.get("properties", {}))
    for definition in schema.get("$defs", {}).values():
        names |= set(definition.get("properties", {}))
    return names


def test_prompt_template_names_every_schema_field() -> None:
    block = _template_block(prompt.load_system_prompt())
    names = _property_names(ModelOutput.model_json_schema())
    assert names
    for name in names:
        assert f'"{name}":' in block


def test_prompt_version_matches_file_stem() -> None:
    path = Path(prompt.__file__).parent / "prompts" / f"{prompt.PROMPT_VERSION}.md"
    assert path.is_file()
    assert prompt.load_system_prompt().strip()


def test_prompt_states_schema_limits() -> None:
    text = prompt.load_system_prompt()
    for limit in ("200", "100", "50", "20", "720", "60"):
        assert limit in text


def test_build_user_message_resume() -> None:
    assert (
        prompt.build_user_message(source="resumes", title="ignored", text="body")
        == "SOURCE: resume\n\nTEXT:\nbody"
    )


def test_build_user_message_posting() -> None:
    assert (
        prompt.build_user_message(source="postings", title="Data Engineer", text="body")
        == "SOURCE: job posting\nTITLE: Data Engineer\n\nTEXT:\nbody"
    )
    assert (
        prompt.build_user_message(source="postings", title=None, text="body")
        == "SOURCE: job posting\nTITLE: \n\nTEXT:\nbody"
    )
    with pytest.raises(ValueError):
        prompt.build_user_message(source="other", title=None, text="body")


def test_build_retry_message_truncates_error() -> None:
    result = prompt.build_retry_message(user="U", error="x" * 1000)
    assert result == (
        "U\n\nYour previous reply was rejected: "
        + "x" * 300
        + "\nReturn only the corrected JSON object."
    )


def test_unknown_prompt_version_raises() -> None:
    with pytest.raises(FileNotFoundError):
        prompt.load_system_prompt(version="no_such_version")
