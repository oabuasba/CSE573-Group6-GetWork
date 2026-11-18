"""Native PowerPoint tables in the previous deck's palette (maroon header, light rows)."""

from __future__ import annotations

from collections.abc import Sequence

import deck_helpers as dh
from deck_helpers import INK, LIGHT, MAROON, PAPER, Rect
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.util import Emu, Pt

CELL_MARGIN_X_EMU = 64008
CELL_MARGIN_Y_EMU = 36576


def _style_cell(cell, *, text: str, size_pt: int, bold: bool, color: RGBColor,
                fill: RGBColor, align: PP_ALIGN) -> None:
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = cell.margin_right = Emu(CELL_MARGIN_X_EMU)
    cell.margin_top = cell.margin_bottom = Emu(CELL_MARGIN_Y_EMU)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame = cell.text_frame
    frame.text = text
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    run = paragraph.runs[0]
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.name = dh.FONT_NAME
    run.font.color.rgb = color


def add_table(slide: Slide, *, rect: Rect, header: Sequence[str], rows: Sequence[Sequence[str]],
              col_widths: Sequence[int], size_pt: int, row_height: int):
    """Header row in maroon/white, body rows alternating F4F1F2 and white, first column bold."""
    shape = slide.shapes.add_table(len(rows) + 1, len(header), Emu(rect.left), Emu(rect.top),
                                   Emu(sum(col_widths)), Emu(row_height * (len(rows) + 1)))
    table = shape.table
    table.horz_banding = False
    for column, width in enumerate(col_widths):
        table.columns[column].width = Emu(width)
    for row in table.rows:
        row.height = Emu(row_height)
    for column, text in enumerate(header):
        _style_cell(table.cell(0, column), text=text, size_pt=size_pt, bold=True, color=PAPER,
                    fill=MAROON, align=_align(column))
    for position, values in enumerate(rows, start=1):
        fill = LIGHT if position % 2 else PAPER
        for column, text in enumerate(values):
            _style_cell(table.cell(position, column), text=text, size_pt=size_pt,
                        bold=column == 0, color=INK, fill=fill, align=_align(column))
    return shape


def _align(column: int) -> PP_ALIGN:
    return PP_ALIGN.LEFT if column == 0 else PP_ALIGN.CENTER


def cell_fill_hex(cell) -> str | None:
    """RRGGBB of a cell's solid fill, or None."""
    fill = cell._tc.tcPr.find(qn("a:solidFill")) if cell._tc.tcPr is not None else None
    return None if fill is None else fill[0].get("val")
