"""Slide 10: project plan, proposal weeks marked done or upcoming, plus division of work."""

from __future__ import annotations

import deck_helpers as dh
import deck_slides
import deck_tables
from deck_helpers import MAROON, Rect, TextLine
from deck_slides import CARD_GAP, CARD_PT, CARD_TITLE_PT, CONTENT_TOP, CONTENT_WIDTH, MARGIN_LEFT
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

DONE, UPCOMING = "Done", "Upcoming"
PLAN_COLUMNS = ("Wk", "Task", "Status")
PLAN_ROWS = [
    ("1", "Clean data, filter postings, structuring script", DONE),
    ("2", "LLM structuring, alias table, dev/test sets", DONE),
    ("3", "Skill embeddings, clustering, taxonomy", DONE),
    ("4", "Re-labeling, B1+/B2/Full rankings, retrieval", DONE),
    ("5", "Web app, judge pipeline and rubric", DONE),
    ("6", "Match agent, preliminary judge-labeled test run", DONE),
    ("6-7", "Parameter tuning (skills kept, clusters, judges)", UPCOMING),
    ("6", "Human evaluation (both authors, blind)", UPCOMING),
    ("7", "Full evaluation run: metrics, CIs, ARI", UPCOMING),
    ("8", "Final report and demo", UPCOMING),
]
PLAN_WIDTHS = (520000, 3980000, 1000000)
PLAN_PT, PLAN_ROW_HEIGHT = 10, 290000
DIVISION_TITLE = "Division of work"
DIVISION_ITEMS = [
    "Both authors contribute to every stage, working jointly within each week",
    "Both label the human set (about 7 hours each) and review explanations",
    "Shared repository; weekly progress review to adjust the timeline",
]
REPO_LINK = "GitHub: <repo link>"
CARD_TOP, CARD_HEIGHT = CONTENT_TOP, 2650000
LINK_TOP = CARD_TOP + CARD_HEIGHT + 120000


def add_plan_slide(prs: PresentationType) -> Slide:
    """Slide 10: proposal timeline with status on the left, division of work on the right."""
    slide = deck_slides.start_slide(
        prs, section="Project Plan",
        headline="Build is done; tuning, human evaluation and the report remain",
        keep_body=False)
    deck_tables.add_table(slide, rect=Rect(MARGIN_LEFT, CONTENT_TOP, 0, 0), header=PLAN_COLUMNS,
                          rows=[list(row) for row in PLAN_ROWS], col_widths=PLAN_WIDTHS,
                          size_pt=PLAN_PT, row_height=PLAN_ROW_HEIGHT)
    left = MARGIN_LEFT + sum(PLAN_WIDTHS) + CARD_GAP
    width = MARGIN_LEFT + CONTENT_WIDTH - left
    body = [TextLine(DIVISION_TITLE, CARD_TITLE_PT, bold=True, color=MAROON)]
    body += [TextLine(text, CARD_PT, bullet=True, space_before_pt=4) for text in DIVISION_ITEMS]
    dh.add_box(slide, rect=Rect(left, CARD_TOP, width, CARD_HEIGHT), fill=dh.LIGHT, lines=body,
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)
    dh.add_label(slide, rect=Rect(left, LINK_TOP, width, 300000), anchor=MSO_ANCHOR.TOP,
                 lines=[TextLine(REPO_LINK, CARD_PT, bold=True, color=MAROON)])
    return slide
