import json
import math
from pathlib import Path

import numpy as np
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


def _doc(skills: list[str], title: str | None, level: str) -> dict:
    return {"skills": skills, "title": title, "level": level}


RESUMES = {
    "r-1": _doc(["Python", "SQL"], "Data Engineer", "Associate"),
    "r-2": _doc([], "Data Analyst", "Not specified"),
    "r-3": _doc(["python", "AWS", "Docker"], "", "Director"),
    "r-4": _doc(["C++", "Go"], "Senior C++ Developer", "Mid-Senior level"),
    "r-5": _doc(["SQL", "sql", " Excel "], None, "Internship"),
    "r-6": _doc([], None, "Not specified"),
    "r-10": _doc(["Go", "Docker", "Python"], "Platform Engineer", "Executive"),
    "r-11": _doc(["Excel"], "Data Engineer", "Entry level"),
}
POSTINGS = {
    "p-1": _doc(["SQL"], "Data Analyst", "Mid-Senior level"),
    "p-2": _doc(["PYTHON", "sql", "Spark"], "Senior Data Engineer", "Associate"),
    "p-3": _doc([], "", "Not specified"),
    "p-4": _doc(["Go", "Docker"], "Platform Engineer", "Director"),
    "p-5": _doc(["c++"], "C++ Developer", "Executive"),
    "p-6": _doc(["Excel", "aws"], None, "Internship"),
    "p-10": _doc([], "Data Engineer", "Not specified"),
}


def test_b1plus_score_hand_computed() -> None:
    resume = _doc(["Python", "SQL"], "Data Engineer", "Associate")
    posting = _doc(["sql"], "Data Analyst", "Mid-Senior level")
    expected = 0.7 * 0.5 + 0.15 * (1 / 3) + 0.15 * 0.8
    got = b1plus.b1plus_score(resume, posting, b1plus.DEFAULT_WEIGHTS)
    assert got == pytest.approx(expected, abs=1e-12)
    weights = b1plus.B1PlusWeights(0.2, 0.3, 0.5)
    assert b1plus.b1plus_score(resume, posting, weights) == pytest.approx(
        0.2 * 0.5 + 0.3 / 3 + 0.5 * 0.8
    )
    empty = _doc([], None, "Not specified")
    assert b1plus.b1plus_score(empty, empty, b1plus.DEFAULT_WEIGHTS) == pytest.approx(0.075)


def test_block_matches_reference_for_all_pairs() -> None:
    assert (len(RESUMES), len(POSTINGS)) == (8, 7)
    resume_features, posting_features = b1plus.build_features(RESUMES, POSTINGS)
    weights = b1plus.B1PlusWeights(0.5, 0.3, 0.2)
    for q, d, q_docs, d_docs in (
        (resume_features, posting_features, RESUMES, POSTINGS),
        (posting_features, resume_features, POSTINGS, RESUMES),
    ):
        for start, stop in ((0, len(q.ids)), (3, 7), (5, 6), (len(q.ids) - 1, len(q.ids))):
            got = b1plus.b1plus_block(q, d, start, stop, weights)
            assert got.dtype == np.float32 and got.shape == (stop - start, len(d.ids))
            for row, q_id in enumerate(q.ids[start:stop]):
                for col, d_id in enumerate(d.ids):
                    want = b1plus.b1plus_score(q_docs[q_id], d_docs[d_id], weights)
                    assert got[row, col] == pytest.approx(want, abs=1e-6), (q_id, d_id)
    single = {name: 1.0 for name in ("skills", "title", "level")}
    for name in single:
        w = b1plus.B1PlusWeights(**{k: float(k == name) for k in single})
        got = b1plus.b1plus_block(resume_features, posting_features, 0, 8, w)
        for row, q_id in enumerate(resume_features.ids):
            for col, d_id in enumerate(posting_features.ids):
                want = b1plus.b1plus_score(RESUMES[q_id], POSTINGS[d_id], w)
                assert got[row, col] == pytest.approx(want, abs=1e-6), (name, q_id, d_id)


def test_build_features_deterministic() -> None:
    a_res, a_post = b1plus.build_features(RESUMES, POSTINGS)
    b_res, b_post = b1plus.build_features(dict(RESUMES), dict(POSTINGS))
    assert a_res.ids == list(RESUMES) and a_post.ids == list(POSTINGS)
    for a, b in ((a_res, b_res), (a_post, b_post)):
        assert a.ids == b.ids
        assert (a.skills != b.skills).nnz == 0 and (a.titles != b.titles).nnz == 0
        np.testing.assert_array_equal(a.skill_counts, b.skill_counts)
        np.testing.assert_array_equal(a.title_counts, b.title_counts)
        np.testing.assert_array_equal(a.ranks, b.ranks)
    assert a_res.skills.shape[1] == a_post.skills.shape[1]
    assert a_res.titles.shape[1] == a_post.titles.shape[1]
    skill_vocab = {"aws", "c++", "docker", "excel", "go", "python", "spark", "sql"}
    assert a_res.skills.shape[1] == len(skill_vocab)
    assert a_res.skills.dtype == np.float32 and a_res.skill_counts.dtype == np.float32
    assert a_res.titles.dtype == np.float32 and a_res.ranks.dtype == np.float32
    # sorted vocabulary: r-1 has python and sql, columns 5 and 7
    assert a_res.skills[0].indices.tolist() == [5, 7]
    np.testing.assert_array_equal(a_res.skill_counts, [2, 0, 3, 2, 2, 0, 3, 1])
    np.testing.assert_array_equal(a_res.title_counts, [2, 2, 0, 3, 0, 0, 2, 2])
    np.testing.assert_array_equal(a_post.skill_counts, [1, 3, 0, 2, 1, 2, 0])
    assert a_res.skills.max() == 1.0
    np.testing.assert_array_equal(a_res.ranks, [2, np.nan, 4, 3, 0, np.nan, 5, 1])
    # swapping sides keeps one shared vocabulary
    s_post, s_res = b1plus.build_features(POSTINGS, RESUMES)
    assert (s_post.skills != a_post.skills).nnz == 0 and (s_res.titles != a_res.titles).nnz == 0
