"""Build the GetWork course presentation from the ASU template with python-pptx.

Usage: uv run python scripts/python/slides/build_deck.py [--template PATH] [--out PATH] [--eval-dir DIR]

Each slide is one ``add_*_slide(prs)`` function listed in ``SLIDE_BUILDERS``; later slides
append their builder there. Text goes into the layout's real placeholders so fonts, colors
and positions come from the ASU template. The template itself is never modified.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import deck_algorithms
import deck_architecture
import deck_eval
import deck_eval_data
import deck_helpers as dh
import deck_plan
import deck_slides
import deck_ui
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = REPO_ROOT / "docs"
TEMPLATE_GLOB = "__2026 Official ASU presentation template*.pptx"
DEFAULT_OUT = DOCS_DIR / "presentation" / "getwork_group6.pptx"
DEFAULT_IMAGES = DOCS_DIR / "presentation" / "screenshots"

# Layout name of the template's "Opening title" slide (Google Slides export naming).
LAYOUT_OPENING_TITLE = "CUSTOM_23_1_1_1_1_1_1_2_1_2"

COVER_TITLE = "GetWork: Two-Way Resume and Job Matching with a Skill Taxonomy"
COVER_GROUP = "CSE 573 · Group 6 · Project 3"
COVER_MEMBERS = "Omar Abuasba and Prathyush Vasa"
COVER_TITLE_PT = 40
AI_DISCLOSURE = ("AI disclosure: Claude (generative AI, Anthropic) was used in producing the "
                 "code, documentation, presentation and other project materials.")
DISCLOSURE_RECT = dh.Rect(left=338325, top=3480000, width=4500000, height=540000)
DISCLOSURE_PT = 11

# Placeholder idx values on the opening-title layout.
IDX_TITLE = 0
IDX_PRESENTERS = 1
IDX_SUBTITLE = 2


def find_default_template(docs_dir: Path = DOCS_DIR) -> Path:
    """Return the ASU template found in ``docs_dir`` (error if absent or ambiguous)."""
    matches = sorted(docs_dir.glob(TEMPLATE_GLOB))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {TEMPLATE_GLOB!r} in {docs_dir}, found {len(matches)}"
        )
    return matches[0]


def add_cover_slide(prs: PresentationType) -> Slide:
    """Slide 1: title, course/group line, and presenters on the opening-title layout."""
    slide = dh.new_slide(prs, layout_name=LAYOUT_OPENING_TITLE)
    dh.set_title(slide, COVER_TITLE, size_pt=COVER_TITLE_PT)
    dh.set_text(slide, idx=IDX_SUBTITLE, text=COVER_GROUP)
    dh.set_text(slide, idx=IDX_PRESENTERS, text=COVER_MEMBERS)
    dh.add_label(slide, rect=DISCLOSURE_RECT, anchor=MSO_ANCHOR.TOP,
                 lines=[dh.TextLine(AI_DISCLOSURE, DISCLOSURE_PT, color=dh.INK)])
    return slide


SLIDE_BUILDERS: list[Callable[[PresentationType], Slide]] = [
    add_cover_slide,
    deck_slides.add_problem_slide,
    deck_architecture.add_architecture_slide,
    deck_algorithms.add_algorithms_slide,
    deck_slides.add_datasets_slide,
    deck_eval.add_eval_design_slide,
    deck_eval.add_eval_results_slide,
    deck_ui.add_ui_find_slide,
    deck_ui.add_ui_chat_slide,
    deck_plan.add_plan_slide,
]


def build_deck(*, template: Path, out: Path, eval_dir: Path | None = None,
               eval_direction: str = deck_eval_data.DEFAULT_DIRECTION,
               images_dir: Path = DEFAULT_IMAGES) -> Path:
    """Build the deck from ``template`` and save to ``out`` (never to the template).

    ``eval_dir`` optionally supplies metrics.json / agreement.json to fill the evaluation
    slides; problems are printed as warnings and the placeholders stay.
    """
    if out.resolve() == template.resolve():
        raise ValueError("refusing to overwrite the template; choose a different --out")
    values = deck_eval_data.load_eval_values(eval_dir, direction=eval_direction)
    for warning in values.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    deck_eval.use_values(values)
    deck_ui.use_images_dir(images_dir)
    prs = Presentation(str(template))
    dh.clear_slides(prs)
    for builder in SLIDE_BUILDERS:
        builder(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--template", type=Path, default=None, help="ASU .pptx template")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output .pptx")
    parser.add_argument("--eval-dir", type=Path, default=None,
                        help="evaluation dir with metrics.json and agreement.json (read-only)")
    parser.add_argument("--eval-direction", default=deck_eval_data.DEFAULT_DIRECTION,
                        choices=[deck_eval_data.DEFAULT_DIRECTION,
                                 deck_eval_data.POSTING_DIRECTION],
                        help="query direction shown in the slide 7 results table")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES,
                        help="folder with find_query/find_results/chat/taxonomy .png files")
    args = parser.parse_args(argv)
    template = args.template or find_default_template()
    saved = build_deck(template=template, out=args.out, eval_dir=args.eval_dir,
                       eval_direction=args.eval_direction, images_dir=args.images_dir)
    print(f"wrote {saved} ({len(SLIDE_BUILDERS)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
