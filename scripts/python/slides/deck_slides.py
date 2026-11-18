"""Shared slide layout helpers plus the problem and datasets slides of the GetWork deck.

Styled after the previous group deck: the template's "title + subtitle" layout with the section
name in the (gold-highlighted) title and the headline in the (black-highlighted) subtitle, light
rounded cards, maroon/gold/ink diagram boxes, and explicit RGB colors.
"""

from __future__ import annotations

import deck_helpers as dh
from deck_helpers import GOLD, GRAY, INK, LIGHT, MAROON, PAPER, Rect, TextLine
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

LAYOUT_SECTION_HEADLINE = "CUSTOM_22_1_1_1_1_1_1_3"
IDX_BODY = 1
IDX_SECOND_BODY = 2
IDX_HEADLINE = 3

MARGIN_LEFT = 338328
CONTENT_WIDTH = 8467344
CONTENT_TOP = 1097280
SECTION_PT = 30
HEADLINE_PT = 12
BODY_PT = 14
CARD_PT = 12
CARD_TITLE_PT = 15
CARD_GAP = 180000
LABEL_PT = 10
COLUMN_LEFTS = (338328, 3200400, 6062472)
COLUMN_WIDTH = 2743200


def start_slide(prs: PresentationType, *, section: str, headline: str,
                 keep_body: bool) -> Slide:
    slide = dh.new_slide(prs, layout_name=LAYOUT_SECTION_HEADLINE)
    dh.set_title(slide, section, size_pt=SECTION_PT)
    dh.set_text(slide, idx=IDX_HEADLINE, text=headline, size_pt=HEADLINE_PT)
    dh.remove_placeholder(slide, idx=IDX_SECOND_BODY)
    if not keep_body:
        dh.remove_placeholder(slide, idx=IDX_BODY)
    return slide


def column_rect(column: int, *, top: int, height: int) -> Rect:
    return Rect(left=COLUMN_LEFTS[column], top=top, width=COLUMN_WIDTH, height=height)


def add_problem_slide(prs: PresentationType) -> Slide:
    """Slide 2: bullets on the left; vocabulary-gap graphic and research question right."""
    slide = start_slide(prs, section="Problem Definition",
                         headline="Same skills, different vocabulary", keep_body=True)
    dh.set_bullets(slide, idx=IDX_BODY, size_pt=BODY_PT, items=[
        "Hiring is one matching problem from two sides: candidates sift postings, "
        "recruiters screen applicants",
        "Keyword screening gives no credit to related skills and can reject qualified people",
        "Embedding matching finds similarity, but one vector per document is hard to explain, "
        "and most systems search one direction only",
    ])
    dh.place(dh.get_placeholder(slide, idx=IDX_BODY),
             rect=Rect(MARGIN_LEFT, CONTENT_TOP, 4343400, 3200400))
    _add_vocabulary_gap(slide)
    _add_research_question(slide, top=2743200)
    return slide


def _add_vocabulary_gap(slide: Slide) -> None:
    left, box_width, gap, height = 4937760, 1783080, 301752, 914400
    job = Rect(left, CONTENT_TOP, box_width, height)
    resume = Rect(job.right + gap, CONTENT_TOP, box_width, height)
    for rect, label, text in ((job, "Job posting:", "Requires React"),
                              (resume, "Resume:", "Built SPAs in Vue")):
        dh.add_box(slide, rect=rect, fill=LIGHT, lines=[
            TextLine(label, 12, bold=True, color=MAROON, space_before_pt=0),
            TextLine(text, BODY_PT, bold=True, space_before_pt=4)])
    dh.add_label(slide, rect=Rect(job.right, CONTENT_TOP + 250000, gap, 365760),
                 align=PP_ALIGN.CENTER, lines=[TextLine("\u2194", 18, bold=True, color=MAROON)])
    dh.add_label(slide, rect=Rect(left, 2103120, 3867912, 502920), align=PP_ALIGN.LEFT,
                 anchor=MSO_ANCHOR.TOP, lines=[TextLine(
                     "Keyword overlap: none. Same cluster (Front-end): partial credit.",
                     12, bold=True)])


def _add_research_question(slide: Slide, *, top: int) -> None:
    dh.add_box(slide, rect=Rect(4937760, top, 3867912, 1554480), fill=MAROON,
               align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, lines=[
                   TextLine("Research question:", 12, bold=True, color=GOLD),
                   TextLine("Does a corpus-generated skill taxonomy improve two-way matching "
                            "over keyword and embedding baselines, while producing explanations "
                            "people can act on?", 13, color=PAPER, space_before_pt=4)])


def add_datasets_slide(prs: PresentationType) -> Slide:
    """Slide 5: stat tiles for the two corpora and the collection gap, plus notes."""
    slide = start_slide(prs, section="Data Set Description",
                         headline="Two public datasets, one per side of the match",
                         keep_body=True)
    tiles = [
        ("29,783", "IT resumes, 10 occupation labels. 28,842 after cleaning, "
                   "**10,567** LLM-structured so far"),
        ("~124K", "LinkedIn job postings, 2023-2024. 8,423 retained, **8,416** structured"),
        ("3+ yrs", "gap between resume and posting collection (pre-2021 vs. 2023-2024)"),
    ]
    for column, (number, caption) in enumerate(tiles):
        dh.add_box(slide, rect=column_rect(column, top=CONTENT_TOP, height=1300000),
                   fill=LIGHT, align=PP_ALIGN.LEFT, lines=[
                       TextLine(number, 30, bold=True, color=MAROON),
                       TextLine(caption, CARD_PT, space_before_pt=2)])
    dh.set_bullets(slide, idx=IDX_BODY, size_pt=BODY_PT, items=[
        "Resumes need no domain filter; occupation labels serve as a reference for our clusters",
        "Postings are filtered to software and adjacent IT roles by title and industry, with no "
        "LLM; duplicate, short, corrupted and non-English documents are dropped from both",
        "Structuring is time-limited: **10,567** resumes took about 15 hours of LLM time. We moved "
        "ahead on this subset and will structure the rest before the final report",
    ])
    dh.place(dh.get_placeholder(slide, idx=IDX_BODY),
             rect=Rect(MARGIN_LEFT, 2600000, CONTENT_WIDTH, 1950000))
    dh.add_label(slide, rect=Rect(MARGIN_LEFT, 4650000, CONTENT_WIDTH, 250000), lines=[
        TextLine("Sources: Jiechieu and Tsopze 2021 (resumes); arshkon 2024, Kaggle (postings)",
                 LABEL_PT, color=GRAY)])
    return slide
