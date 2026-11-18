"""Geometry checks for the built deck (no renderer needed).

Usage: uv run python scripts/python/slides/check_deck_layout.py [DECK.pptx]

Reports per slide: shape bounds, out-of-bounds shapes, overlapping shapes, and text boxes
whose estimated wrapped height exceeds the box (Arial, average glyph width heuristic).
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import deck_textfit
from pptx import Presentation
from pptx.oxml.ns import qn

SLIDE_WIDTH, SLIDE_HEIGHT = 9144000, 5143500
OVERLAP_TOLERANCE_EMU = 40000  # template's own title/subtitle boxes overlap by ~34000
BODY_PLACEHOLDER_INDENT_EMU = 457200


def is_line(shape) -> bool:
    return shape.shape_type is not None and "LINE" in str(shape.shape_type)


def out_of_bounds(shape) -> bool:
    return (shape.left < 0 or shape.top < 0 or shape.left + shape.width > SLIDE_WIDTH
            or shape.top + shape.height > SLIDE_HEIGHT)


def overlaps(first, second) -> bool:
    across_x = min(first.left + first.width, second.left + second.width) - max(first.left,
                                                                              second.left)
    across_y = min(first.top + first.height, second.top + second.height) - max(first.top,
                                                                              second.top)
    return across_x > OVERLAP_TOLERANCE_EMU and across_y > OVERLAP_TOLERANCE_EMU


def _insets(shape) -> tuple[int, int]:
    frame = shape.text_frame
    if shape.is_placeholder:
        return 0, 0
    return frame.margin_left + frame.margin_right, frame.margin_top + frame.margin_bottom


def _layout_line_spacing(shape) -> float:
    """Line-spacing factor from the layout placeholder's level-1 style (1.0 if unset)."""
    if not shape.is_placeholder:
        return 1.0
    for layout_shape in shape.part.slide_layout.placeholders:
        if layout_shape.placeholder_format.idx == shape.placeholder_format.idx:
            percent = layout_shape._element.find(
                ".//" + qn("a:lvl1pPr") + "/" + qn("a:lnSpc") + "/" + qn("a:spcPct"))
            return int(percent.get("val")) / 100000 if percent is not None else 1.0
    return 1.0


def estimated_text_height_emu(shape) -> float:
    """Wrapped text height in EMU (word wrap, bullets, paragraph spacing, insets)."""
    width_inset, height_inset = _insets(shape)
    is_body = shape.is_placeholder and shape.placeholder_format.idx == 1
    usable = shape.width - width_inset - (BODY_PLACEHOLDER_INDENT_EMU if is_body else 0)
    return deck_textfit.frame_height_emu(
        shape.text_frame, width_emu=usable, insets_y=height_inset,
        spacing=deck_textfit.LINE_HEIGHT_EM * _layout_line_spacing(shape))


def estimated_cell_height_emu(cell, *, width: int) -> float:
    """Wrapped height of one table cell's text, same estimator as for text boxes."""
    return deck_textfit.frame_height_emu(
        cell.text_frame, width_emu=width - cell.margin_left - cell.margin_right,
        insets_y=cell.margin_top + cell.margin_bottom, spacing=deck_textfit.LINE_HEIGHT_EM)


def table_problems(shape) -> list[str]:
    problems = []
    table = shape.table
    for row_number, row in enumerate(table.rows):
        for column_number, column in enumerate(table.columns):
            cell = table.cell(row_number, column_number)
            if estimated_cell_height_emu(cell, width=column.width) > row.height:
                problems.append(f"table cell overflow: {shape.name!r} r{row_number} "
                                f"c{column_number}: {cell.text_frame.text!r}")
    return problems


def check_slide(slide) -> list[str]:
    problems = []
    shapes = list(slide.shapes)
    for shape in shapes:
        if out_of_bounds(shape):
            problems.append(f"out of bounds: {shape.name}")
        if shape.has_table:
            problems += table_problems(shape)
            continue
        if shape.has_text_frame and shape.text_frame.text.strip() and not is_line(shape):
            needed, have = estimated_text_height_emu(shape), shape.height
            if needed > have:
                problems.append(f"text overflow: {shape.name!r} needs {needed / 914400:.2f}in, "
                                f"has {have / 914400:.2f}in: {shape.text_frame.text[:40]!r}")
    boxes = [s for s in shapes if not is_line(s)]
    for first, second in combinations(boxes, 2):
        if overlaps(first, second):
            problems.append(f"overlap: {first.name!r} / {second.name!r}")
    return problems


def describe(slide) -> list[str]:
    return [f"  {s.name[:22]:22} x={s.left:>8} y={s.top:>8} w={s.width:>8} h={s.height:>8}"
            for s in slide.shapes]


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path("docs/presentation/getwork_group6.pptx")
    prs = Presentation(str(path))
    failed = 0
    for number, slide in enumerate(prs.slides, start=1):
        print(f"slide {number} ({slide.slide_layout.name})")
        print("\n".join(describe(slide)))
        problems = check_slide(slide)
        failed += len(problems)
        print("\n".join(f"  PROBLEM {p}" for p in problems) or "  ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
