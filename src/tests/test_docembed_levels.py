import math

import pytest

from getwork.docembed import levels

pytestmark = pytest.mark.usefixtures("offline")


def test_level_rank_and_normalize() -> None:
    assert [levels.level_rank(x) for x in levels.LEVELS] == [0, 1, 2, 3, 4, 5]
    assert levels.level_rank("Not specified") is None
    assert levels.level_rank("whatever") is None
    for raw, want in [
        ("Associate", "Associate"),
        ("ASSOCIATE", "Associate"),
        ("  director ", "Director"),
        ("mid-senior", "Mid-Senior level"),
        ("Entry", "Entry level"),
        ("Mid-Senior level", "Mid-Senior level"),
    ]:
        assert levels.normalize_listed_level(raw) == want
    for bad in [None, math.nan, "", "   ", "guru", 3]:
        assert levels.normalize_listed_level(bad) is None


def test_level_from_years_boundaries() -> None:
    f = levels.level_from_years
    assert f(0) == f(1.99) == "Entry level"
    assert f(2.0) == f(4.99) == "Associate"
    assert f(5.0) == f(30) == "Mid-Senior level"
    assert f(None) is None and f(math.nan) is None and f(-1) is None
    assert f(2.5, (3.0, 6.0)) == "Entry level"
    assert f(5.0, (3.0, 6.0)) == "Associate"
    assert f(6.0, (3.0, 6.0)) == "Mid-Senior level"


def test_parse_bands() -> None:
    assert levels.parse_bands("2,5") == (2.0, 5.0)
    assert levels.parse_bands(" 1.5 , 6 ") == (1.5, 6.0)
    for bad in ["5,2", "0,5", "2", "a,b", "1,2,3"]:
        with pytest.raises(ValueError):
            levels.parse_bands(bad)


def test_resolve_level_resumes() -> None:
    assert levels.resolve_level("resumes", "Director", 1.0) == ("Entry level", "years")
    assert levels.resolve_level("resumes", "Director", None) == ("Not specified", "missing")
    assert levels.resolve_level("resumes", None, math.nan) == ("Not specified", "missing")
    with pytest.raises(ValueError):
        levels.resolve_level("other", None, 1.0)


def test_resolve_level_postings_fallbacks() -> None:
    assert levels.resolve_level("postings", "Director", 1.0) == ("Director", "listed")
    assert levels.resolve_level("postings", math.nan, 3.0) == ("Associate", "years")
    assert levels.resolve_level("postings", "guru", 6.0) == ("Mid-Senior level", "years")
    assert levels.resolve_level("postings", "guru", None) == ("Not specified", "missing")
