"""Slides 6-7: evaluation plan (real content) and preliminary results (filled from --eval-dir).

Bracketed text such as ``[N]`` is deliberate: it is easy to find and replace by hand, and it
is replaced automatically when ``--eval-dir`` supplies metrics.json / agreement.json values.
"""

from __future__ import annotations

import deck_eval_data
import deck_helpers as dh
import deck_slides
import deck_takeaways
import deck_tables
from deck_eval_data import EvalValues
from deck_helpers import GRAY, LIGHT, MAROON, Rect, TextLine
from deck_slides import CARD_GAP, CARD_PT, CARD_TITLE_PT, CONTENT_TOP, CONTENT_WIDTH, MARGIN_LEFT
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

SECTION = "Evaluation Plan"
NOTE_PT = 11
TABLE_PT = 11
ROW_HEIGHT = 330000
PLACEHOLDER_CELL = "0.000 [0.000, 0.000]"
_values = EvalValues()


def use_values(values: EvalValues) -> None:
    """Set the values the evaluation slides are filled from (placeholders when empty)."""
    global _values
    _values = values


def _bullets(items: list[tuple[str, str]], *, space: int = 4) -> list[TextLine]:
    return [TextLine(text, CARD_PT, lead=lead, bullet=True, space_before_pt=space)
            for lead, text in items]


def _card(slide: Slide, *, rect: Rect, title: str, body: list[TextLine]) -> None:
    lines = [TextLine(title, CARD_TITLE_PT, bold=True, color=MAROON), *body]
    dh.add_box(slide, rect=rect, fill=LIGHT, lines=lines, align=PP_ALIGN.LEFT,
               anchor=MSO_ANCHOR.TOP)


PRELIMINARY_ITEMS = [
    ("Systems: ", "B1+ (keyword, untuned default weights), B2 (embeddings), "
                  "Full (embeddings + taxonomy)"),
    ("Queries: ", "test stage, 30 resumes + 30 postings, both directions"),
    ("Pool: ", "top-10 per system, pooled and deduplicated, blind pairs; "
               "1,286 unique pairs judged"),
    ("Judges: ", "Gemma 4 31B (Google) and Llama 4 Maverick 17B (Meta), different families"),
    ("Labels: ", "0 not relevant, 1 partial, 2 relevant; rubric judge_v1 frozen before the "
                 "test run; lower of the two labels wins"),
    ("Metrics: ", "nDCG@10 (graded gain 0/1/2), P@10, MRR; bootstrap 95% CIs and a paired "
                  "randomization test for system gaps"),
]
UPCOMING_ITEMS = [
    ("Skills retained: ", "vary the top-N skills kept"),
    ("Clusters: ", "vary the number of clusters k"),
    ("Judge prompts: ", "compare rubric versions"),
    ("Judge models: ", "try other models"),
    ("Directions: ", "do system gaps differ for resume to posting vs. posting to "
                     "resume?"),
    ("Substitution test (not yet run): ", "React to Vue vs. React to Java rank drop"),
    ("Explanation faithfulness (not yet run): ", "judge rating 1-5, hallucination rate, "
                                                 "hand review"),
    ("Human evaluation: ", "the system is built, so the focus shifts here; both authors "
                           "label blind, judges checked against them"),
]
LEFT_WIDTH = 4900000
CARDS_BOTTOM = 4800000


def add_eval_design_slide(prs: PresentationType) -> Slide:
    """Slide 6: what the reported test run did, and what is planned next."""
    slide = deck_slides.start_slide(
        prs, section=SECTION, headline="Three systems, blind LLM judges, ranked-list metrics",
        keep_body=False)
    height = CARDS_BOTTOM - CONTENT_TOP
    _card(slide, rect=Rect(MARGIN_LEFT, CONTENT_TOP, LEFT_WIDTH, height),
          title="Preliminary evaluation plan", body=_bullets(PRELIMINARY_ITEMS))
    right = MARGIN_LEFT + LEFT_WIDTH + CARD_GAP
    width = MARGIN_LEFT + CONTENT_WIDTH - right
    _card(slide, rect=Rect(right, CONTENT_TOP, width, height),
          title="Upcoming evaluation plan", body=_bullets(UPCOMING_ITEMS))
    return slide


RESULT_COLUMNS = ("System", "nDCG@10", "P@10", "MRR")
RESULT_WIDTHS = (700000, 1600000, 1600000, 1600000)
GAP_LABELS = ("B2 - B1+", "Full - B2", "Full - B1+")
GAP_COLUMNS = ("Difference", "nDCG@10 gap", "p-value")
GAP_WIDTHS = (1500000, 2000000, 2000000)
RIGHT_LEFT = 6000000
NOTE_TOP, NOTE_HEIGHT = 4560000, 380000
AGREEMENT_HEIGHT, RIGHT_GAP, TAKEAWAY_PT = 1200000, 100000, 10
RESULTS_SECTION = "Preliminary Results"
NOT_RUN_NOTE = ("Note: the full evaluation plan has not been run yet. Parameter tuning "
                "(skills retained, number of clusters, judge prompts, judge models) is still in "
                "progress ahead of the final report. Not yet reported: substitution test, "
                "explanation faithfulness, human check.")


def _result_rows() -> list[list[str]]:
    return [[system, *(_values.results.get((system, metric), PLACEHOLDER_CELL)
                       for metric in deck_eval_data.RESULT_METRICS)]
            for system in deck_eval_data.SYSTEMS]


def _gap_rows() -> list[list[str]]:
    return [[label, *_values.gaps.get(label, ("+0.000", "p = 0.000"))] for label in GAP_LABELS]


def _table_label(slide: Slide, *, top: int, text: str) -> None:
    dh.add_label(slide, rect=Rect(MARGIN_LEFT, top, sum(RESULT_WIDTHS), 260000),
                 lines=[TextLine(text, 12, bold=True, color=MAROON)])


def add_eval_results_slide(prs: PresentationType) -> Slide:
    """Slide 7: result tables, judge agreement, takeaways and the not-yet-run note."""
    slide = deck_slides.start_slide(
        prs, section=RESULTS_SECTION, headline=deck_takeaways.headline(_values),
        keep_body=False)
    suffix = " (resume to posting queries)" if _values.results else ""
    _table_label(slide, top=CONTENT_TOP - 40000, text=f"Ranking metrics, mean [95% CI]{suffix}")
    deck_tables.add_table(slide, rect=Rect(MARGIN_LEFT, CONTENT_TOP + 240000, 0, 0),
                          header=RESULT_COLUMNS, rows=_result_rows(), col_widths=RESULT_WIDTHS,
                          size_pt=TABLE_PT, row_height=ROW_HEIGHT)
    _table_label(slide, top=2900000, text="Differences")
    deck_tables.add_table(slide, rect=Rect(MARGIN_LEFT, 3180000, 0, 0), header=GAP_COLUMNS,
                          rows=_gap_rows(), col_widths=GAP_WIDTHS, size_pt=TABLE_PT,
                          row_height=ROW_HEIGHT)
    _add_side_cards(slide)
    dh.add_label(slide, rect=Rect(MARGIN_LEFT, NOTE_TOP, CONTENT_WIDTH, NOTE_HEIGHT),
                 anchor=MSO_ANCHOR.TOP, lines=[TextLine(NOT_RUN_NOTE, NOTE_PT, color=dh.INK)])
    return slide


def _add_side_cards(slide: Slide) -> None:
    width = MARGIN_LEFT + CONTENT_WIDTH - RIGHT_LEFT
    exact = _values.exact_agreement or "[00%]"
    kappa = _values.kappa or "[0.00]"
    weighted = _values.kappa_weighted or "[0.00]"
    pairs = _values.pairs_judged or "[N]"
    _card(slide, rect=Rect(RIGHT_LEFT, CONTENT_TOP, width, AGREEMENT_HEIGHT),
          title="Judge agreement",
          body=_bullets([("", f"Exact agreement: {exact}"),
                         ("", f"Cohen's kappa: {kappa} (weighted {weighted})"),
                         ("", f"Pairs judged: {pairs}")], space=3))
    top = CONTENT_TOP + AGREEMENT_HEIGHT + RIGHT_GAP
    _card(slide, rect=Rect(RIGHT_LEFT, top, width, NOTE_TOP - 80000 - top), title="Takeaways",
          body=[TextLine(text, TAKEAWAY_PT, bullet=True, space_before_pt=2)
                for text in deck_takeaways.takeaways(_values)])
