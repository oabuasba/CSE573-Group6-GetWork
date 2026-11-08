"""Output contract for the structurer: models, validators and the reply parser."""

import json
import re
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from getwork.llm.client import MAX_ERROR_CHARS

SCHEMA_VERSION: str = "1"
SOURCES: tuple[str, ...] = ("resumes", "postings")
SOURCE_PREFIX: dict[str, str] = {"resumes": "r-", "postings": "j-"}

MAX_TITLE_CHARS = 200
MAX_NAME_CHARS = 200
MAX_SKILL_CHARS = 100
MAX_SKILLS = 200
MAX_EXPERIENCE = 50
MAX_EDUCATION = 20
MAX_DURATION_MONTHS = 720
MAX_YEARS_EXPERIENCE = 60
MAX_ERROR_ITEMS = 5

_THINK_BLOCK = re.compile(r"\A<think>.*?</think>", re.DOTALL)
_CODE_FENCE = re.compile(r"\A```[A-Za-z]*\s*\n?(.*?)\s*```\Z", re.DOTALL)


def _strip_str(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _blank_to_none(value: Any) -> Any:
    value = _strip_str(value)
    return None if value == "" else value


def _check_max_chars(value: str | None, limit: int) -> str | None:
    if value is not None and len(value) > limit:
        raise ValueError(f"must be at most {limit} characters")
    return value


def _reject_bool_and_str(value: Any) -> Any:
    if isinstance(value, (bool, str)):
        raise ValueError("must be a number or null")
    return value


class ExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str | None
    company: str | None
    duration_months: int | None

    @field_validator("role", "company", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("role", "company")
    @classmethod
    def _limit_name(cls, value: str | None) -> str | None:
        return _check_max_chars(value, MAX_NAME_CHARS)

    @field_validator("duration_months", mode="before")
    @classmethod
    def _normalize_duration(cls, value: Any) -> Any:
        value = _reject_bool_and_str(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    @field_validator("duration_months")
    @classmethod
    def _limit_duration(cls, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= MAX_DURATION_MONTHS:
            raise ValueError(f"must be between 0 and {MAX_DURATION_MONTHS}")
        return value

    @model_validator(mode="after")
    def _require_role_or_company(self) -> "ExperienceEntry":
        if self.role is None and self.company is None:
            raise ValueError("role and company cannot both be null")
        return self


class EducationEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    degree: str | None
    field: str | None
    institution: str | None

    @field_validator("degree", "field", "institution", mode="before")
    @classmethod
    def _normalize_name(cls, value: Any) -> Any:
        return _blank_to_none(value)

    @field_validator("degree", "field", "institution")
    @classmethod
    def _limit_name(cls, value: str | None) -> str | None:
        return _check_max_chars(value, MAX_NAME_CHARS)


class ModelOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    skills: list[str]
    experience: list[ExperienceEntry]
    education: list[EducationEntry]
    years_experience: float | None

    @field_validator("title", mode="before")
    @classmethod
    def _trim_title(cls, value: Any) -> Any:
        return _strip_str(value)

    @field_validator("title")
    @classmethod
    def _limit_title(cls, value: str) -> str:
        if not 1 <= len(value) <= MAX_TITLE_CHARS:
            raise ValueError(f"must be 1 to {MAX_TITLE_CHARS} characters")
        return value

    @field_validator("skills", mode="before")
    @classmethod
    def _clean_skills(cls, value: Any) -> Any:
        if not isinstance(value, list) or not all(isinstance(s, str) for s in value):
            return value
        seen: set[str] = set()
        cleaned: list[str] = []
        for skill in (s.strip() for s in value):
            if skill and skill.casefold() not in seen:
                seen.add(skill.casefold())
                cleaned.append(skill)
        return cleaned

    @field_validator("skills")
    @classmethod
    def _limit_skills(cls, value: list[str]) -> list[str]:
        if any(len(skill) > MAX_SKILL_CHARS for skill in value):
            raise ValueError(f"each skill must be at most {MAX_SKILL_CHARS} characters")
        if len(value) > MAX_SKILLS:
            raise ValueError(f"at most {MAX_SKILLS} skills")
        return value

    @field_validator("experience")
    @classmethod
    def _limit_experience(cls, value: list[ExperienceEntry]) -> list[ExperienceEntry]:
        if len(value) > MAX_EXPERIENCE:
            raise ValueError(f"at most {MAX_EXPERIENCE} entries")
        return value

    @field_validator("education")
    @classmethod
    def _limit_education(cls, value: list[EducationEntry]) -> list[EducationEntry]:
        if len(value) > MAX_EDUCATION:
            raise ValueError(f"at most {MAX_EDUCATION} entries")
        return value

    @field_validator("years_experience", mode="before")
    @classmethod
    def _reject_non_numbers(cls, value: Any) -> Any:
        return _reject_bool_and_str(value)

    @field_validator("years_experience")
    @classmethod
    def _limit_years(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= MAX_YEARS_EXPERIENCE:
            raise ValueError(f"must be between 0 and {MAX_YEARS_EXPERIENCE}")
        return value


class OutputValidationError(ValueError):
    """str(e) is a one-line, human-readable reason."""


def _strip_wrappers(text: str) -> str:
    text = _THINK_BLOCK.sub("", text.strip(), count=1).strip()
    fence = _CODE_FENCE.match(text)
    return fence.group(1).strip() if fence else text


def _describe_errors(exc: ValidationError) -> str:
    parts = [
        ".".join(str(p) for p in error["loc"]) + ": " + error["msg"]
        for error in exc.errors()[:MAX_ERROR_ITEMS]
    ]
    return "; ".join(parts)[:MAX_ERROR_CHARS]


def parse_model_output(text: str) -> ModelOutput:
    body = _strip_wrappers(text)
    if not body:
        raise OutputValidationError(
            "empty reply (max_tokens may be too small or reasoning used the budget)"
        )
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"not valid JSON: {exc.msg}") from None
    if not isinstance(data, dict):
        raise OutputValidationError("expected a JSON object")
    try:
        return ModelOutput.model_validate(data)
    except ValidationError as exc:
        raise OutputValidationError(_describe_errors(exc)) from None


_UNSUPPORTED_KEYWORDS = frozenset({
    "minLength", "maxLength", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "minItems", "maxItems",
})


def _drop_keywords(node: Any, *, in_properties: bool = False) -> Any:
    # A properties mapping holds property names, so its keys are never dropped.
    if isinstance(node, dict):
        return {
            key: _drop_keywords(value, in_properties=(key == "properties" and not in_properties))
            for key, value in node.items()
            if in_properties or key not in _UNSUPPORTED_KEYWORDS
        }
    if isinstance(node, list):
        return [_drop_keywords(item) for item in node]
    return node


def response_schema() -> dict:
    return _drop_keywords(ModelOutput.model_json_schema())


RECORD_KEYS: tuple[str, ...] = (
    "id", "source", "title", "skills", "experience", "education",
    "years_experience", "model", "prompt_version", "schema_version",
)

PARQUET_SCHEMA: pa.Schema = pa.schema([
    ("id", pa.string()), ("source", pa.string()), ("title", pa.string()),
    ("skills", pa.list_(pa.string())),
    ("experience", pa.list_(pa.struct([("role", pa.string()), ("company", pa.string()),
                                       ("duration_months", pa.int64())]))),
    ("education", pa.list_(pa.struct([("degree", pa.string()), ("field", pa.string()),
                                      ("institution", pa.string())]))),
    ("years_experience", pa.float64()),
    ("model", pa.string()), ("prompt_version", pa.string()), ("schema_version", pa.string()),
])


def to_record(
    output: ModelOutput, record_id: str, source: str, model: str, prompt_version: str
) -> dict:
    fields = output.model_dump()
    return {
        "id": record_id,
        "source": source,
        "title": fields["title"],
        "skills": fields["skills"],
        "experience": fields["experience"],
        "education": fields["education"],
        "years_experience": fields["years_experience"],
        "model": model,
        "prompt_version": prompt_version,
        "schema_version": SCHEMA_VERSION,
    }


FAILURE_STAGES: tuple[str, ...] = ("validation", "transient", "rate_limit", "permanent")
FAILURE_COLUMNS: tuple[str, ...] = ("id", "stage", "error_type", "message", "attempts")


@dataclass(frozen=True)
class Failure:
    id: str
    stage: str
    error_type: str
    message: str  # redacted, at most MAX_ERROR_CHARS characters
    attempts: int
