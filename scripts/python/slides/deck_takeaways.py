"""Takeaway sentences for the results slide, derived only from the numbers it shows."""

from __future__ import annotations

from deck_eval_data import DEFAULT_DIRECTION, POSTING_DIRECTION, EvalValues

SIGNIFICANCE = 0.05
KAPPA_TARGET = 0.6
PLACEHOLDER_HEADLINE = "[Headline: one sentence finding goes here]"
PLACEHOLDER_TAKEAWAYS = ["[Finding 1]", "[Finding 2]", "[Caveat: ...]"]
GENERIC_HEADLINE = "Preliminary results from the judge-labeled test run"
CAVEAT = "Caveat: B1+ uses untuned default weights; table shows resume to posting"
PAIRS = (("B2 - B1+", "B2", "B1+"), ("Full - B2", "Full", "B2"))


def _is_significant(p_text: str) -> bool:
    if p_text.startswith("p <"):
        return True
    return float(p_text.split("=")[1]) < SIGNIFICANCE


def _verdict(values: EvalValues, *, label: str) -> str:
    """'better', 'worse' or 'unclear' for the nDCG@10 gap named ``label``."""
    diff, p_text = values.gaps[label]
    if not _is_significant(p_text):
        return "unclear"
    return "better" if float(diff) > 0 else "worse"


def _gap_sentence(values: EvalValues, *, label: str, system: str, other: str) -> str:
    diff, p_text = values.gaps[label]
    verdict = _verdict(values, label=label)
    if verdict == "unclear":
        return f"{system} vs. {other}: no clear gap ({diff}, {p_text})"
    verb = "beats" if verdict == "better" else "trails"
    return f"{system} {verb} {other} on nDCG@10 ({diff}, {p_text})"


def _has_all(values: EvalValues) -> bool:
    return bool(values.gaps) and values.kappa is not None and values.pairs_judged is not None


def _direction_sentence(values: EvalValues) -> str | None:
    """B2 over B1+ and Full vs. B2 in each direction, or None unless both are available."""
    forward = values.direction_gaps.get(DEFAULT_DIRECTION)
    reverse = values.direction_gaps.get(POSTING_DIRECTION)
    if not forward or not reverse:
        return None
    verdicts = {name: _verdict(EvalValues(gaps=gaps), label="Full - B2")
                for name, gaps in (("forward", forward), ("reverse", reverse))}
    kinds = set(verdicts.values())
    full = ("Full vs. B2 unclear in both" if kinds == {"unclear"}
            else "Full vs. B2 same in both" if len(kinds) == 1 else "Full vs. B2 differs")
    return (f"Posting to resume: B2 over B1+ {reverse['B2 - B1+'][0]} (vs. "
            f"{forward['B2 - B1+'][0]}); {full}")


def headline(values: EvalValues) -> str:
    if not _has_all(values):
        return PLACEHOLDER_HEADLINE
    if (_verdict(values, label="B2 - B1+") == "better"
            and _verdict(values, label="Full - B2") != "better"):
        return "Embeddings beat keyword matching; the taxonomy adds no clear gain yet"
    return GENERIC_HEADLINE


def takeaways(values: EvalValues) -> list[str]:
    if not _has_all(values):
        return PLACEHOLDER_TAKEAWAYS
    sentences = [_gap_sentence(values, label=label, system=system, other=other)
                 for label, system, other in PAIRS]
    sentences.append(CAVEAT)
    direction = _direction_sentence(values)
    if direction:
        sentences.append(direction)
    below = float(values.kappa) < KAPPA_TARGET
    sentences.append(f"Judge kappa {values.kappa} is {'below' if below else 'at or above'} "
                     f"the {KAPPA_TARGET} target")
    return sentences
