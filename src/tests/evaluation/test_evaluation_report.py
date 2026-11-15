from __future__ import annotations

import re

import pytest

from getwork.evaluation import report
from getwork.evaluation.metrics import METRICS
from getwork.retrieval.inputs import DIRECTIONS, SYSTEM_IDS

pytestmark = pytest.mark.usefixtures("offline")

GAPS = ("B2-B1+", "Full-B2", "Full-B1+")


def _direction(offset: float) -> dict:
    systems = {
        s: {m: {"mean": 0.1 * (i + 1) + offset, "ci": [0.05 + offset, 0.5]} for m in METRICS}
        for i, s in enumerate(SYSTEM_IDS)
    }
    gaps = {
        g: {m: {"diff": 0.123 + offset, "ci": [-0.01, 0.2], "p": 0.0004 if g == "Full-B2" else 0.5}
            for m in METRICS}
        for g in GAPS
    }
    return {"n_queries": 100, "systems": systems, "gaps": gaps, "order": {}}


def _group(offset: float = 0.0) -> dict:
    return {"directions": {d: _direction(offset) for d in DIRECTIONS}}


def _metrics() -> dict:
    return {"groups": {
        "judge@test": _group(), "judge@human": _group(0.01),
        "omar@human": _group(0.02), "sara@human": _group(0.03),
    }}


def _manifest(b2_order: str = "grouped") -> dict:
    return {
        "seed": 7,
        "params": {"pool_cap": 5000, "dev_resumes": 10},
        "stamps": {
            "documents": {"taxonomy_version": "tax-v3", "relabel_mode": "llm",
                          "b2_order": b2_order},
            "n_resumes": 1234, "n_postings": 4321,
        },
        "weights": {"test": {"skills": 0.5, "title": 0.3, "level": 0.2,
                             "source": "file:tune/b1plus_weights.json"}},
        "judges": [
            {"id": "judge-a", "model": "gemma4-31b-it", "family": "google", "truncated": 3},
            {"id": "judge-b", "model": "llama4-maverick-17b", "family": "meta", "truncated": 0},
        ],
    }


def _agreement() -> dict:
    tau = {"nDCG@10": {d: {"judge-vs-omar": 1.0, "judge-vs-sara": 0.333} for d in DIRECTIONS},
           "gaps": {}}
    return {
        "judges": {"dev": {"kappa": 0.71}, "test": {"kappa": 0.655}},
        "human": {
            "half": "all", "n_pairs": 40, "authors": ["omar", "sara"], "author_kappa": 0.64,
            "human_judge_kappa": 0.58,
            "fallback": {"threshold": 0.6, "decision": "revise_once"}, "tau": tau,
        },
    }


def _ari() -> dict:
    block = {"dims": 12, "ari_by_seed": [0.2], "ari_mean": 0.234, "ari_sd": 0.011}
    return {
        "k": 10, "seeds": [0, 1], "n_resumes": 90, "n_unlabeled_excluded": 2,
        "clusters": block, "raw_skills": {**block, "dims": 300, "ari_mean": 0.1},
        "taxonomy": {"config_id": "cfg-1", "n_clusters": 12, "silhouette": 0.31,
                     "noise_fraction": 0.05},
        "vocabulary_gap": {"n_retained": 100, "n_one_corpus": 40, "fraction_one_corpus": 0.4},
    }


def _render(**overrides: object) -> str:
    inputs = {"manifest": _manifest(), "metrics": _metrics(), "agreement": _agreement(),
              "ari": _ari(), "frozen": {"rubric_hash": "sha256:abc123"}, **overrides}
    return report.render_report(**inputs)  # type: ignore[arg-type]


def test_report_sections() -> None:
    text = _render()
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == [
        "## Run", "## Judges", "## Ranking quality (judge labels, test queries)",
        "## System gaps", "## Human check", "## Taxonomy validity (ARI)", "## Caveats",
    ]
    assert text.startswith("# Evaluation report")
    for needle in ("Seed: 7", "capped at 5000", "software and adjacent IT postings",
                   "tax-v3", "llm", "grouped", "file:tune/b1plus_weights.json", "1234", "4321",
                   "gemma4-31b-it", "meta", "sha256:abc123", "0.710", "0.655",
                   "0.100 [0.050, 0.500]", "resume_to_posting", "posting_to_resume",
                   "<0.001", "0.123", "0.640", "0.580", "revise_once", "omar@human",
                   "sara@human", "judge@human", "0.333", "0.234", "cfg-1", "0.400"):
        assert needle in text, needle
    assert "SWE postings" not in text
    for caveat in ("lower-label rule", "1/3", "no multiplicity correction", "LLM judges",
                   "small"):
        assert caveat in text
    assert "| judge-a | gemma4-31b-it | google | 3 |" in text


def test_report_handles_missing_inputs() -> None:
    for missing in ("metrics", "agreement", "ari", "frozen"):
        text = _render(**{missing: None})
        assert report.NOT_RUN == "_not run yet_"
        assert "_not run yet_" in text
    text = _render(metrics=None, agreement=None, ari=None, frozen=None)
    assert text.count("_not run yet_") >= 5
    assert "## Caveats" in text


def test_report_is_deterministic_and_b2_caveat() -> None:
    assert _render() == _render()
    assert "Full grouping order" in _render()
    assert "Full grouping order" not in _render(manifest=_manifest(b2_order="record"))
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", _render())
