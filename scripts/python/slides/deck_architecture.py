"""Slide 3: the three-band system diagram, styled and color-coded like the previous deck.

Fill = approach category (same mapping as the previous deck's pipeline slide):
gold = Classical, maroon = Modern, ink = Both.
"""

from __future__ import annotations

import deck_helpers as dh
import deck_slides
from deck_helpers import GOLD, GRAY, INK, LIGHT, MAROON, PAPER, Rect, TextLine
from deck_slides import CONTENT_WIDTH, MARGIN_LEFT
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

CLASSICAL, MODERN, BOTH = "Classical", "Modern", "Both"
CATEGORY_STYLE = {CLASSICAL: (GOLD, INK), MODERN: (MAROON, PAPER), BOTH: (INK, PAPER)}
LEGEND = [
    (CLASSICAL, "Classical (rules, statistics, non-neural ML)"),
    (MODERN, "Modern (LLMs, neural embeddings)"),
    (BOTH, "Both"),
]

BUILD_STEPS = [
    ("1  Filter", CLASSICAL, "regex clean, langdetect, dedup, IT filter"),
    ("2  Structure", BOTH, "qwen38-27b extracts fields; alias merge"),
    ("3  Select skills", BOTH, "top 1,000 by frequency; each embedded"),
    ("4  Cluster", BOTH, "UMAP + k-means (k=10); LLM names clusters"),
    ("5  Re-label", CLASSICAL, "taxonomy lookup; unseen skills to Other"),
    ("6  Embed docs", MODERN, "qwen3-embedding-8b, 4,096-d per document"),
]
QUERY_STEPS = [
    ("7  Two-way matching", BOTH, "B1+ Jaccard; B2, Full cosine top-k, both directions"),
    ("8  Explain", BOTH, "rule-based match cards; LangGraph chat agent, 4 tools"),
]
SERVING_STEPS = [
    ("React 18 + Vite UI", "find, results, chat pages"),
    ("FastAPI + uvicorn API", "find, explain, chat"),
]
CLUSTER_STEP = 3
STEP_WIDTH = 1250000
STEP_GAP = (CONTENT_WIDTH - len(BUILD_STEPS) * STEP_WIDTH) // (len(BUILD_STEPS) - 1)
STEP_INSET_X = 91440
BUILD_LABEL_TOP, BUILD_TOP, BUILD_HEIGHT = 1000000, 1300000, 1050000
CALLOUT_GAP, CALLOUT_HEIGHT = 100000, 480000
QUERY_LABEL_TOP, QUERY_TOP, QUERY_HEIGHT = 2960000, 3260000, 700000
FEED_BEND_Y = 2950000
QUERY_WIDTH = 2100000
QUERY_LEFTS = (4300000, 6700000)
SERVING_LABEL_TOP, SERVING_TOP, SERVING_HEIGHT = 3980000, 4270000, 700000
SERVING_LEFTS = QUERY_LEFTS
SERVING_WIDTH = QUERY_WIDTH
BRACKET_Y = 4115000
LEGEND_TEXT_LEFT, LEGEND_TEXT_WIDTH, LEGEND_ROW = 640000, 3500000, 230000
SWATCH_WIDTH, SWATCH_HEIGHT = 201168, 164592
VOYAGER_DISCLAIMER = ("Disclaimer: the models shown were chosen based on what was available "
                      "on ASU Voyager.")
DISCLAIMER_RECT = Rect(MARGIN_LEFT, 4500000, 3600000, 420000)
FOOTER_NOTE = "New documents pass through steps 2, 5 and 6 before matching."


def add_architecture_slide(prs: PresentationType) -> Slide:
    """Build band, query band, serving band, legend, and the arrows between them."""
    slide = deck_slides.start_slide(prs, section="System Architecture",
                                    headline="Build once, then match per request",
                                    keep_body=False)
    build = _add_build_band(slide)
    queries = _add_query_band(slide, feeder=build[-1])
    _add_serving_band(slide, queries=queries)
    _add_legend(slide)
    dh.add_label(slide, rect=DISCLAIMER_RECT, anchor=MSO_ANCHOR.TOP,
                 lines=[TextLine(VOYAGER_DISCLAIMER, 10, color=GRAY)])
    dh.add_speaker_notes(slide, FOOTER_NOTE)
    return slide


def _band_label(slide: Slide, *, top: int, text: str) -> None:
    dh.add_label(slide, rect=Rect(MARGIN_LEFT, top, 3600000, 274320),
                 lines=[TextLine(text, 12, bold=True, color=MAROON)])


def _step_box(slide: Slide, *, rect: Rect, step: tuple[str, str, str]) -> None:
    name, category, detail = step
    fill, ink = CATEGORY_STYLE[category]
    dh.add_box(slide, rect=rect, fill=fill, inset_x=STEP_INSET_X, lines=[
        TextLine(name, 13, bold=True, color=ink),
        TextLine(detail, 10, color=ink, space_before_pt=3)])


def _add_build_band(slide: Slide) -> list[Rect]:
    _band_label(slide, top=BUILD_LABEL_TOP, text="Build phase (runs once on the corpus)")
    rects = [Rect(MARGIN_LEFT + i * (STEP_WIDTH + STEP_GAP), BUILD_TOP, STEP_WIDTH,
                  BUILD_HEIGHT) for i in range(len(BUILD_STEPS))]
    for rect, step in zip(rects, BUILD_STEPS):
        _step_box(slide, rect=rect, step=step)
    middle = BUILD_TOP + BUILD_HEIGHT // 2
    for left, right in zip(rects, rects[1:]):
        dh.add_line(slide, start=(left.right, middle), end=(right.left, middle))
    _add_cluster_callout(slide, anchor=rects[CLUSTER_STEP])
    return rects


def _add_cluster_callout(slide: Slide, *, anchor: Rect) -> None:
    callout = Rect(anchor.center_x - 1450000, anchor.bottom + CALLOUT_GAP, 2900000,
                   CALLOUT_HEIGHT)
    dh.add_line(slide, start=(anchor.center_x, anchor.bottom),
                end=(anchor.center_x, callout.top), tail=False)
    dh.add_box(slide, rect=callout, fill=LIGHT, lines=[TextLine(
        "Compared: k-means, DBSCAN, HDBSCAN. Chosen: k-means, k=10.", 11, color=INK)])


def _add_query_band(slide: Slide, *, feeder: Rect) -> list[Rect]:
    _band_label(slide, top=QUERY_LABEL_TOP, text="Query phase (runs per request)")
    boxes = [Rect(left, QUERY_TOP, QUERY_WIDTH, QUERY_HEIGHT) for left in QUERY_LEFTS]
    for rect, step in zip(boxes, QUERY_STEPS):
        _step_box(slide, rect=rect, step=step)
    middle = QUERY_TOP + QUERY_HEIGHT // 2
    dh.add_line(slide, start=(boxes[0].right, middle), end=(boxes[1].left, middle))
    x_from, x_to = feeder.center_x, boxes[0].center_x
    dh.add_line(slide, start=(x_from, feeder.bottom), end=(x_from, FEED_BEND_Y), tail=False)
    dh.add_line(slide, start=(x_from, FEED_BEND_Y), end=(x_to, FEED_BEND_Y), tail=False)
    dh.add_line(slide, start=(x_to, FEED_BEND_Y), end=(x_to, QUERY_TOP))
    return boxes


def _add_serving_band(slide: Slide, *, queries: list[Rect]) -> None:
    """Neutral infrastructure boxes; a bracket of arrows shows they serve steps 7 and 8."""
    _band_label(slide, top=SERVING_LABEL_TOP, text="Serving layer (web app)")
    boxes = [Rect(left, SERVING_TOP, SERVING_WIDTH, SERVING_HEIGHT) for left in SERVING_LEFTS]
    for rect, (name, detail) in zip(boxes, SERVING_STEPS):
        dh.add_box(slide, rect=rect, fill=LIGHT, outline=MAROON, inset_x=STEP_INSET_X, lines=[
            TextLine(name, 13, bold=True, color=INK),
            TextLine(detail, 10, color=INK, space_before_pt=3)])
    middle = SERVING_TOP + SERVING_HEIGHT // 2
    dh.add_line(slide, start=(boxes[0].right, middle), end=(boxes[1].left, middle), head=True)
    api_x = boxes[1].center_x
    bottom = QUERY_TOP + QUERY_HEIGHT
    dh.add_line(slide, start=(api_x, SERVING_TOP), end=(api_x, bottom))
    dh.add_line(slide, start=(api_x, BRACKET_Y), end=(queries[0].center_x, BRACKET_Y),
                tail=False)
    dh.add_line(slide, start=(queries[0].center_x, BRACKET_Y), end=(queries[0].center_x, bottom))


def _add_legend(slide: Slide) -> None:
    for position, (category, text) in enumerate(LEGEND):
        top = QUERY_TOP + position * (LEGEND_ROW + 5000)
        fill, _ = CATEGORY_STYLE[category]
        dh.add_box(slide, rect=Rect(MARGIN_LEFT, top + 20000, SWATCH_WIDTH, SWATCH_HEIGHT),
                   fill=fill, lines=[], inset_x=0)
        dh.add_label(slide, rect=Rect(LEGEND_TEXT_LEFT, top, LEGEND_TEXT_WIDTH, LEGEND_ROW),
                     anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT,
                     lines=[TextLine(text, 11)])
