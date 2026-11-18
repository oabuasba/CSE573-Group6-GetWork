"""Takeaway sentences for the results slide, derived only from the numbers it shows."""

from __future__ import annotations

from deck_eval_data import EvalValues

SIGNIFICANCE = 0.05
KAPPA_TARGET = 0.6
PLACEHOLDER_HEADLINE = "[Headline: one sentence finding goes here]"
PLACEHOLDER_TAKEAWAYS = ["[Finding 1]", "[Finding 2]", "[Caveat: ...]"]
GENERIC_HEADLINE = "Preliminary results from the judge-labeled test run"
CAVEAT = "Caveat: B1+ used untuned default weights; results are resume-to-posting queries"
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
    phrase = {"better": f"{system} beats {other}", "worse": f"{system} trails {other}",
              "unclear": f"{system} vs. {other}: no clear gap"}[_verdict(values, label=label)]
    return f"{phrase} on nDCG@10 ({diff}, {p_text})"


def _has_all(values: EvalValues) -> bool:
    return bool(values.gaps) and values.kappa is not None and values.pairs_judged is not None


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
    below = float(values.kappa) < KAPPA_TARGET
    sentences.append(f"Judge kappa {values.kappa} is {'below' if below else 'at or above'} "
                     f"the {KAPPA_TARGET} target on {values.pairs_judged} pairs: preliminary")
    return sentences
