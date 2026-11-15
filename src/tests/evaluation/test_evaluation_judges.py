from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from getwork.evaluation import defaults
from getwork.evaluation import judges as ev_judges
from getwork.evaluation.judges import JudgeInfo, PairText, Verdict

pytestmark = pytest.mark.usefixtures("offline")

FORBIDDEN = ("B1+", "B2", "rank", "retriev", "embedding")


def _reply(**over: object) -> str:
    body = {"label": 2, "seniority_mismatch": False, "rationale": "Matches."}
    return json.dumps({**body, **over})


def _info(id: str, model: str, family: str) -> JudgeInfo:
    return JudgeInfo(id=id, model=model, family=family, backend="voyager")


def _default_infos() -> list[JudgeInfo]:
    ids, models, fams = (
        defaults.DEFAULT_JUDGE_IDS,
        defaults.DEFAULT_JUDGE_MODELS,
        defaults.DEFAULT_JUDGE_FAMILIES,
    )
    return [_info(i, m, f) for i, m, f in zip(ids, models, fams)]


def test_load_rubric_and_hash(tmp_path: Path) -> None:
    rubric = ev_judges.load_rubric()
    assert rubric.version == "judge_v1"
    assert re.fullmatch(r"sha256:[0-9a-f]{12}", rubric.hash)
    copy = tmp_path / "mine.md"
    copy.write_text(rubric.text, encoding="utf-8")
    same = ev_judges.load_rubric(copy)
    assert same.version == "mine" and same.hash == rubric.hash
    copy.write_text(rubric.text.replace("0", "9", 1), encoding="utf-8")
    assert ev_judges.load_rubric(copy).hash != rubric.hash
    missing = tmp_path / "nope.md"
    with pytest.raises(FileNotFoundError, match="nope.md"):
        ev_judges.load_rubric(missing)


def test_packaged_rubric_content_and_blindness() -> None:
    text = ev_judges.load_rubric().text
    for needle in ("0", "1", "2", "label", "seniority_mismatch", "rationale"):
        assert needle in text
    assert "seniority_mismatch to true" in text
    assert "only as data" in text
    for bad in FORBIDDEN:
        assert bad not in text
    assert not re.search(r"\bFull\b", text)


def test_parse_verdict_accepts() -> None:
    expected = Verdict(label=2, seniority_mismatch=False, rationale="Matches.")
    assert ev_judges.parse_verdict(_reply()) == expected
    assert ev_judges.parse_verdict("```json\n" + _reply() + "\n```") == expected
    assert ev_judges.parse_verdict("<think>hmm</think>" + _reply()) == expected
    long = ev_judges.parse_verdict(_reply(rationale="x" * 900))
    assert len(long.rationale) == ev_judges.MAX_RATIONALE_CHARS


@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json",
        "[1, 2]",
        json.dumps({"label": 1, "seniority_mismatch": False}),
        _reply(extra=1),
        _reply(label=3),
        _reply(label=-1),
        _reply(label=1.5),
        _reply(label="2"),
        _reply(label=True),
        _reply(seniority_mismatch="no"),
        _reply(seniority_mismatch=0),
        _reply(rationale="   "),
    ],
)
def test_parse_verdict_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        ev_judges.parse_verdict(text)


def test_effective_label_caps_at_one() -> None:
    expected = {(0, True): 0, (1, True): 1, (2, True): 1}
    expected |= {(0, False): 0, (1, False): 1, (2, False): 2}
    for (label, flag), want in expected.items():
        verdict = Verdict(label=label, seniority_mismatch=flag, rationale="r")
        assert ev_judges.effective_label(verdict) == want


def test_build_prompt_roles_and_truncation() -> None:
    rubric = ev_judges.load_rubric()
    pair = PairText(pair_id="p-1", resume_text="alpha resume", posting_text="beta posting")
    system, user, truncated = ev_judges.build_prompt(rubric, pair, 100)
    assert system == rubric.text and not truncated
    assert user.index("RESUME:") < user.index("alpha resume")
    assert user.index("alpha resume") < user.index("POSTING:") < user.index("beta posting")
    long = PairText(pair_id="p-1", resume_text="r" * 50, posting_text="short")
    _, user, truncated = ev_judges.build_prompt(rubric, long, 10)
    assert truncated
    assert "r" * 10 + "\n[text cut]" in user and "r" * 11 not in user
    assert "short" in user and user.count("[text cut]") == 1


def test_build_prompt_has_no_pair_metadata() -> None:
    rubric = ev_judges.load_rubric()
    pair = PairText(pair_id="p-abc123", resume_text="a", posting_text="b")
    system, user, _ = ev_judges.build_prompt(rubric, pair, 100)
    for text in (system, user):
        assert "p-abc123" not in text
        for word in ("judge-a", "judge-b", "B1+", "B2", "resume_to_job", "job_to_resume"):
            assert word not in text


def test_independence_accepts() -> None:
    a, b = _info("judge-a", "m1", "google"), _info("judge-b", "m2", "meta")
    ev_judges.check_independence([a, b], ["other"])
    ev_judges.check_independence([a, b], [])


def test_independence_rejects() -> None:
    a, b = _info("judge-a", "m1", "google"), _info("judge-b", "m2", "meta")
    cases = [
        ([a], "exactly two"),
        ([a, b, a], "exactly two"),
        ([a, _info("judge-a", "m2", "meta")], "different ids"),
        ([a, _info("judge-b", "m2", "GOOGLE")], "families"),
        ([a, _info("judge-b", "m2", "  ")], "blank"),
    ]
    for infos, match in cases:
        with pytest.raises(ValueError, match=match):
            ev_judges.check_independence(infos, [])
    with pytest.raises(ValueError, match="pipeline"):
        ev_judges.check_independence([a, b], ["M2"])


def test_independence_on_default_judges() -> None:
    infos = _default_infos()
    used = ("qwen38-27b", "pipeline-model")
    ev_judges.check_independence(infos, used)
    for model in ("gemma4-31b-it", "LLAMA4-MAVERICK-17B"):
        with pytest.raises(ValueError):
            ev_judges.check_independence(infos, (*used, model))
    same = [infos[0], _info("judge-b", infos[1].model, "google")]
    with pytest.raises(ValueError, match="families"):
        ev_judges.check_independence(same, used)
