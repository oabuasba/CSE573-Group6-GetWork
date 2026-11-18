"""Slides 8-9: UI and visualization slides with two screenshots each.

Uses the template's two-picture layout, which (unlike the three-picture one) carries no
decorative dashed card shapes. Screenshots come from the images folder; a missing file leaves
that picture placeholder empty (click its icon to insert by hand) and prints a warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import deck_helpers as dh
from deck_helpers import GOLD, INK, MAROON, PAPER, Rect, TextLine
from deck_slides import CARD_GAP, CONTENT_WIDTH, MARGIN_LEFT
from pptx.enum.text import MSO_ANCHOR
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

LAYOUT_TWO_PICTURES = "CUSTOM_22_1_1_1_1_1_1_1_1_1_1_1_1_1"
IDX_TITLE, IDX_BODY, IDX_HEADLINE = 0, 1, 5
PICTURE_IDX = (2, 3)
UNUSED_IDX = (4,)
TITLE_RECT = Rect(338325, 320050, 8467200, 440100)
HEADLINE_RECT = Rect(337900, 726175, 8467200, 165900)
SECTION_PT, HEADLINE_PT, CAPTION_PT, BODY_PT = 30, 12, 12, 12
CAPTION_TOP, CAPTION_HEIGHT = 1000000, 380000
SLOT_TOP, SLOT_HEIGHT = 1420000, 2900000
BULLETS_TOP, BULLETS_HEIGHT = 4400000, 660000
COLUMN_WIDTH = (CONTENT_WIDTH - CARD_GAP) // 2
TAXONOMY_FALLBACK = "cluster_map.png"
TAXONOMY_NOTE = ("Positions are a separate 2-D UMAP of the same skill vectors; distances are "
                 "only a rough guide to the 5-D clustering space.")
_images_dir = Path("docs/presentation/screenshots")


def use_images_dir(path: Path) -> None:
    """Folder the screenshots are read from."""
    global _images_dir
    _images_dir = path


def _find_image(name: str, *, fallback: str | None = None) -> Path | None:
    for candidate in (name, fallback):
        if candidate and (_images_dir / candidate).is_file():
            if candidate != name:
                print(f"warning: {name} missing, using {candidate}", file=sys.stderr)
            return _images_dir / candidate
    print(f"warning: no screenshot {name} in {_images_dir}; picture slot left empty",
          file=sys.stderr)
    return None


def _build_ui_slide(prs: PresentationType, *, section: str, headline: str,
                    shots: list[tuple[str, str | None, str]], bullets: list[str]) -> Slide:
    """``shots`` holds (file name, fallback file name, caption) for the left and right slot."""
    slide = dh.new_slide(prs, layout_name=LAYOUT_TWO_PICTURES)
    dh.style_heading(slide, idx=IDX_TITLE, text=section, size_pt=SECTION_PT, color=INK,
                     highlight=GOLD)
    dh.place(slide.shapes.title, rect=TITLE_RECT)
    dh.style_heading(slide, idx=IDX_HEADLINE, text=headline, size_pt=HEADLINE_PT, color=PAPER,
                     highlight=INK, anchor=MSO_ANCHOR.BOTTOM)
    dh.place(dh.get_placeholder(slide, idx=IDX_HEADLINE), rect=HEADLINE_RECT)
    for idx in UNUSED_IDX:
        dh.remove_placeholder(slide, idx=idx)
    for column, (idx, (name, fallback, caption)) in enumerate(zip(PICTURE_IDX, shots)):
        left = MARGIN_LEFT + column * (COLUMN_WIDTH + CARD_GAP)
        dh.add_label(slide, rect=Rect(left, CAPTION_TOP, COLUMN_WIDTH, CAPTION_HEIGHT),
                     anchor=MSO_ANCHOR.TOP,
                     lines=[TextLine(caption, CAPTION_PT, bold=True, color=MAROON)])
        slot = Rect(left, SLOT_TOP, COLUMN_WIDTH, SLOT_HEIGHT)
        image = _find_image(name, fallback=fallback)
        if image is None:
            dh.place(dh.get_placeholder(slide, idx=idx), rect=slot)
        else:
            dh.insert_fitted_picture(slide, idx=idx, image=image, slot=slot)
    dh.set_bullets(slide, idx=IDX_BODY, items=bullets, size_pt=BODY_PT, space_before_pt=3)
    dh.place(dh.get_placeholder(slide, idx=IDX_BODY),
             rect=Rect(MARGIN_LEFT, BULLETS_TOP, CONTENT_WIDTH, BULLETS_HEIGHT))
    return slide


def add_ui_find_slide(prs: PresentationType) -> Slide:
    """Slide 8: query form and results screenshots."""
    return _build_ui_slide(
        prs, section="UI Interfaces", headline="Find matches in both directions",
        shots=[("find_query.png", None,
                "Find jobs: paste a resume, fill in a form, or pick an example"),
               ("find_results.png", None,
                "Top-10 postings, each with shared, related and missing skills")],
        bullets=["Submit a resume to find jobs, or a posting to find candidates",
                 "Each result shows a similarity score and which skills match, relate or "
                 "are missing",
                 "\"Related\" skills earn partial credit through shared skill clusters"])


def add_ui_chat_slide(prs: PresentationType) -> Slide:
    """Slide 9: chat and taxonomy screenshots; notes carry the UMAP caveat."""
    slide = _build_ui_slide(
        prs, section="UI Interfaces and Data Visualization",
        headline="Chat explanations and the skill taxonomy",
        shots=[("chat.png", None,
                "Chat about a suggestion: ask why it matched or how to improve the resume"),
               ("taxonomy.png", TAXONOMY_FALLBACK,
                "Skill clusters: 1,000 skills in 10 clusters (k-means, 2-D UMAP view)")],
        bullets=["The assistant answers follow-up questions about the matched posting",
                 "It warns that it is an AI and can be wrong, and points users to the skills "
                 "on the card",
                 "Clusters group related skills, such as Front-end, Operating Systems and "
                 "Database"])
    dh.add_speaker_notes(slide, TAXONOMY_NOTE)
    return slide
