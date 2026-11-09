import json
import math
from pathlib import Path

import pytest

from getwork.retrieval import b1plus

pytestmark = pytest.mark.usefixtures("offline")


def test_title_tokens() -> None:
    assert b1plus.title_tokens("Senior C++ Developer") == {"senior", "c++", "developer"}
    assert b1plus.title_tokens("C# and Node.js Engineer.") == {"c#", "node.js", "engineer"}
    assert b1plus.title_tokens("Head of the Data at Acme") == {"head", "data", "acme"}
    assert b1plus.title_tokens(None) == frozenset()
    assert b1plus.title_tokens("   ") == frozenset()


def test_jaccard_and_skill_keys() -> None:
    assert b1plus.jaccard({"a", "b"}, {"b", "c", "d"}) == 0.25
    assert b1plus.jaccard(set(), set()) == 0.0
    assert b1plus.skill_keys(["Java", "JAVA", " java. ", "", " . ", "C++"]) == {"java", "c++"}


def test_seniority_similarity() -> None:
    s = b1plus.seniority_similarity
    assert s("Associate", "Associate") == 1.0
    assert s("Associate", "Mid-Senior level") == pytest.approx(0.8)
    assert s("Internship", "Executive") == 0.0
    assert s("Not specified", "Director") == 0.5
    assert s("Director", "Not specified") == 0.5


def test_weights_validation_and_defaults() -> None:
    assert b1plus.DEFAULT_WEIGHTS == b1plus.B1PlusWeights(0.7, 0.15, 0.15)
    assert b1plus.DEFAULT_WEIGHTS.as_dict() == {"skills": 0.7, "title": 0.15, "level": 0.15}
    for bad in [(-0.1, 0.6, 0.5), (math.nan, 0.5, 0.5), (0.5, 0.5, 0.5), (0.2, 0.2, 0.2)]:
        with pytest.raises(ValueError):
            b1plus.B1PlusWeights(*bad)


def test_parse_and_load_weights(tmp_path: Path) -> None:
    assert b1plus.parse_weights("0.7,0.15,0.15") == b1plus.DEFAULT_WEIGHTS
    for bad in ["0.5,0.5", "0.4,0.3,0.2,0.1", "a,b,c", "0.5,0.5,0.5"]:
        with pytest.raises(ValueError):
            b1plus.parse_weights(bad)
    good = tmp_path / "w.json"
    good.write_text(json.dumps({"skills": 0.5, "title": 0.25, "level": 0.25}))
    assert b1plus.load_weights(good) == b1plus.B1PlusWeights(0.5, 0.25, 0.25)
    cases = {
        "extra": {"skills": 0.5, "title": 0.25, "level": 0.25, "x": 0},
        "missing": {"skills": 0.5, "title": 0.5},
        "invalid": {"skills": 0.9, "title": 0.9, "level": 0.9},
        "strings": {"skills": "a", "title": "b", "level": "c"},
    }
    for name, content in cases.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content))
        with pytest.raises(ValueError, match=name):
            b1plus.load_weights(path)
    broken = tmp_path / "broken.json"
    broken.write_text("{nope")
    with pytest.raises(ValueError, match="broken.json"):
        b1plus.load_weights(broken)
    with pytest.raises(FileNotFoundError, match="absent.json"):
        b1plus.load_weights(tmp_path / "absent.json")


def test_grid_search_weights() -> None:
    seen: list[b1plus.B1PlusWeights] = []

    def record(w: b1plus.B1PlusWeights) -> float:
        seen.append(w)
        return 0.0

    best, value = b1plus.grid_search_weights(record, step=0.1)
    assert len(seen) == 66 and len(set(seen)) == 66
    assert all(abs(sum(w.as_dict().values()) - 1) < 1e-6 for w in seen)
    assert (seen[0].skills, seen[0].title, seen[0].level) == (1.0, 0.0, 0.0)
    assert best == seen[0] and value == 0.0

    target = b1plus.B1PlusWeights(0.3, 0.2, 0.5)
    best, value = b1plus.grid_search_weights(
        lambda w: -abs(w.skills - 0.3) - abs(w.title - 0.2), step=0.1
    )
    assert best == target and value == pytest.approx(0.0)

    for bad in [0, 0.3, 0.7]:
        with pytest.raises(ValueError):
            b1plus.grid_search_weights(record, step=bad)
