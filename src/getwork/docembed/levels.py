"""Seniority level resolution for resumes and postings."""

from __future__ import annotations

import math

from getwork import paths

LEVELS: tuple[str, ...] = (
    "Internship",
    "Entry level",
    "Associate",
    "Mid-Senior level",
    "Director",
    "Executive",
)
NOT_SPECIFIED: str = "Not specified"
LEVEL_SOURCES: tuple[str, ...] = ("listed", "years", "missing")
DEFAULT_BANDS: tuple[float, float] = (2.0, 5.0)

_LEVEL_BY_KEY: dict[str, str] = {
    **{level.casefold(): level for level in LEVELS},
    "mid-senior": "Mid-Senior level",
    "entry": "Entry level",
}


def level_rank(level: str) -> int | None:
    """Index of level in LEVELS, None for NOT_SPECIFIED or anything else."""
    return LEVELS.index(level) if level in LEVELS else None


def normalize_listed_level(raw: object) -> str | None:
    """Map a listed level (any case, padded, or short form) to a LEVELS entry, else None."""
    if not isinstance(raw, str):
        return None
    return _LEVEL_BY_KEY.get(raw.strip().casefold())


def level_from_years(
    years: float | None, bands: tuple[float, float] = DEFAULT_BANDS
) -> str | None:
    """None for None, NaN or negative; Entry, Associate or Mid-Senior by the bands."""
    if years is None or math.isnan(years) or years < 0:
        return None
    if years < bands[0]:
        return "Entry level"
    if years < bands[1]:
        return "Associate"
    return "Mid-Senior level"


def parse_bands(text: str) -> tuple[float, float]:
    """Parse "2,5" into (2.0, 5.0); ValueError unless 0 < first < second."""
    parts = text.split(",")
    try:
        values = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"bands must be two numbers like '2,5', got {text!r}") from e
    if len(values) != 2 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"bands must be two numbers like '2,5', got {text!r}")
    first, second = values
    if not 0 < first < second:
        raise ValueError(f"bands need 0 < first < second, got {text!r}")
    return first, second


def resolve_level(
    source: str,
    listed_raw: object,
    years: float | None,
    bands: tuple[float, float] = DEFAULT_BANDS,
) -> tuple[str, str]:
    """Return (level, level_source); resumes ignore listed_raw, postings prefer it."""
    paths.check_source(source)
    if source == "postings":
        listed = normalize_listed_level(listed_raw)
        if listed is not None:
            return listed, "listed"
    derived = level_from_years(years, bands)
    if derived is not None:
        return derived, "years"
    return NOT_SPECIFIED, "missing"
