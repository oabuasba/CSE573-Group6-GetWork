"""Slide 4: how the pipeline works, as four step cards (structure, taxonomy, re-label, retrieval).

Every number comes from the code defaults and the v7 taxonomy artifacts: UMAP 5 dimensions and
15 neighbors, noise cap 30%, 29 grid configurations, k-means with 10 clusters chosen by review.
"""

from __future__ import annotations

import deck_helpers as dh
import deck_slides
from deck_helpers import LIGHT, MAROON, Rect, TextLine
from deck_slides import CARD_GAP, CONTENT_TOP, CONTENT_WIDTH, MARGIN_LEFT
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

TITLE_PT, BODY_PT = 13, 11
BOTTOM = 4950000
ROW_GAP = 80000
CARD_WIDTH = (CONTENT_WIDTH - CARD_GAP) // 2
TOP_HEIGHT = 1900000
EXAMPLE = '"Front-end Development: JavaScript, HTML, CSS"'

STEPS = [
    ("1  Structure and canonicalize", [
        "An LLM extracts title, skills, experience and education (reasoning off, temperature 0, "
        "schema-validated, up to 3 attempts)",
        "An alias table canonicalizes skills: JS becomes JavaScript, Windows 7/10/11 fold "
        "into one",
    ]),
    ("2  Build the skill taxonomy", [
        "Top 1,000 canonical skills by count across both corpora, each embedded alone",
        "UMAP (5 dimensions, 15 neighbors), then 29 configurations of k-means, DBSCAN, HDBSCAN",
        "Best mean silhouette (cosine, clustered skills only) under a 30% noise cap",
        "Review chose k-means, 10 clusters (HDBSCAN scored higher, left skills unclustered); "
        "an LLM names them",
    ]),
    ("3  Re-label and serialize", [
        'Known skills: cluster lookup; noise skills: "Other"',
        "LLM only for unseen skills and incomplete documents (few skills, or under half known)",
        f"Text: title, level, then skills by cluster, e.g. {EXAMPLE}; this gives related "
        "skills partial credit",
        "Level: listed for postings, from years of experience for resumes",
    ]),
    ("4  Retrieval", [
        "Document vectors are L2-normalized, so cosine is an inner product: S = R · J^T",
        "Top-k by row (jobs for a resume) or by column (candidates for a posting): "
        "one matrix, both directions",
        "B1+: skill Jaccard + title token overlap + seniority distance (default weights, "
        "untuned)",
        "B2: same text without cluster labels; Full: with them. B1+ vs. B2 isolates "
        "embeddings, B2 vs. Full the taxonomy",
    ]),
]


def _card(slide: Slide, *, rect: Rect, title: str, points: list[str]) -> None:
    body = [TextLine(title, TITLE_PT, bold=True, color=MAROON)]
    body += [TextLine(text, BODY_PT, bullet=True, space_before_pt=3) for text in points]
    dh.add_box(slide, rect=rect, fill=LIGHT, lines=body, align=PP_ALIGN.LEFT,
               anchor=MSO_ANCHOR.TOP)


def add_algorithms_slide(prs: PresentationType) -> Slide:
    """Slide 4: 2x2 cards, left to right and top to bottom in pipeline order."""
    slide = deck_slides.start_slide(prs, section="Algorithms",
                                    headline="From raw text to a ranked list in four steps",
                                    keep_body=False)
    bottom_top = CONTENT_TOP + TOP_HEIGHT + ROW_GAP
    rows = ((CONTENT_TOP, TOP_HEIGHT), (bottom_top, BOTTOM - bottom_top))
    for position, (title, points) in enumerate(STEPS):
        top, height = rows[position // 2]
        left = MARGIN_LEFT + (position % 2) * (CARD_WIDTH + CARD_GAP)
        _card(slide, rect=Rect(left, top, CARD_WIDTH, height), title=title, points=points)
    return slide
