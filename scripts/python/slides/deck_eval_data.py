"""Read evaluation results (metrics.json, agreement.json) into slide-ready strings.

Structure read (written by ``evaluate metrics`` / ``evaluate agreement``):
  metrics.json   groups["judge@test"]["directions"][direction]
                   ["systems"][system][metric] = {"mean", "ci": [lo, hi]},
                   ["gaps"]["B2-B1+" | "Full-B2" | "Full-B1+"][metric] = {"diff", "ci", "p"}
  agreement.json judges["test"] = {"n", "pct_agree", "kappa", "kappa_quadratic", ...}
Anything missing leaves the slide placeholder in place and adds a warning; nothing raises.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

JUDGE_GROUP = "judge@test"
JUDGE_STAGE = "test"
DEFAULT_DIRECTION = "resume_to_posting"
POSTING_DIRECTION = "posting_to_resume"
SYSTEMS = ("B1+", "B2", "Full")
RESULT_METRICS = ("nDCG@10", "P@10", "MRR")
GAP_METRIC = "nDCG@10"
GAP_KEYS = {"B2 - B1+": "B2-B1+", "Full - B2": "Full-B2", "Full - B1+": "Full-B1+"}
P_FLOOR = 0.001


@dataclass
class EvalValues:
    """Strings to drop into the evaluation slides; empty fields keep their placeholders."""

    direction: str = DEFAULT_DIRECTION
    results: dict[tuple[str, str], str] = field(default_factory=dict)
    gaps: dict[str, tuple[str, str]] = field(default_factory=dict)
    direction_gaps: dict[str, dict[str, tuple[str, str]]] = field(default_factory=dict)
    exact_agreement: str | None = None
    kappa: str | None = None
    kappa_weighted: str | None = None
    pairs_judged: str | None = None
    warnings: list[str] = field(default_factory=list)


def _read_json(path: Path, warnings: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        warnings.append(f"{path}: cannot read ({error}); keeping placeholders")
        return None


def _num(value: float) -> str:
    return f"{float(value):.3f}"


def _p_text(value: float) -> str:
    return f"p < {P_FLOOR}" if float(value) < P_FLOOR else f"p = {_num(value)}"


def _gap_strings(block: dict) -> dict[str, tuple[str, str]]:
    gaps = {}
    for label, key in GAP_KEYS.items():
        gap = block["gaps"][key][GAP_METRIC]
        gaps[label] = (f"{float(gap['diff']):+.3f}", _p_text(gap["p"]))
    return gaps


def _fill_direction_gaps(values: EvalValues, directions: dict) -> None:
    """nDCG@10 gaps for both directions; a direction that cannot be read is skipped."""
    for name in (DEFAULT_DIRECTION, POSTING_DIRECTION):
        try:
            values.direction_gaps[name] = _gap_strings(directions[name])
        except (KeyError, TypeError, ValueError):
            continue


def _fill_metrics(values: EvalValues, metrics: dict) -> None:
    try:
        block = metrics["groups"][JUDGE_GROUP]["directions"][values.direction]
        for system in SYSTEMS:
            for metric in RESULT_METRICS:
                entry = block["systems"][system][metric]
                low, high = entry["ci"]
                values.results[(system, metric)] = (
                    f"{_num(entry['mean'])} [{_num(low)}, {_num(high)}]")
        values.gaps = _gap_strings(block)
        _fill_direction_gaps(values, metrics["groups"][JUDGE_GROUP]["directions"])
    except (KeyError, TypeError, ValueError):
        values.warnings.append("metrics.json: results table incomplete; placeholders kept")


def _fill_agreement(values: EvalValues, agreement: dict) -> None:
    try:
        block = agreement["judges"][JUDGE_STAGE]
        values.pairs_judged = str(block["n"])
        values.exact_agreement = f"{float(block['pct_agree']) * 100:.0f}%"
        values.kappa = f"{float(block['kappa']):.2f}"
        values.kappa_weighted = f"{float(block['kappa_quadratic']):.2f}"
    except (KeyError, TypeError, ValueError):
        values.warnings.append("agreement.json: no usable judges.test block; placeholders kept")


def load_eval_values(eval_dir: Path | None, *, direction: str = DEFAULT_DIRECTION) -> EvalValues:
    """Collect values from ``eval_dir`` (read-only); no directory means all placeholders."""
    values = EvalValues(direction=direction)
    if eval_dir is None:
        return values
    metrics = _read_json(eval_dir / "metrics.json", values.warnings)
    if metrics is not None:
        _fill_metrics(values, metrics)
    agreement = _read_json(eval_dir / "agreement.json", values.warnings)
    if agreement is not None:
        _fill_agreement(values, agreement)
    return values
