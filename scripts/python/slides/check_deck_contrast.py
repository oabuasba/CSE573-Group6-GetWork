"""WCAG contrast check for every text run in the deck, resolving colors like PowerPoint does.

Usage: uv run python scripts/python/slides/check_deck_contrast.py [DECK.pptx]

Text color: the run's own fill, else the layout placeholder's default, else tx1. Background:
the run's highlight, else the shape's solid fill, else the slide background (bg1). Scheme colors
go through the master's color map (e.g. bg2 -> dk2) and then the theme, with lumMod/lumOff.
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

from pptx import Presentation
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn

MIN_CONTRAST = 4.5
DEFAULT_DECK = Path("docs/presentation/getwork_group6.pptx")
FALLBACK_TEXT, FALLBACK_BACKGROUND = "tx1", "bg1"


def theme_colors(prs) -> dict[str, str]:
    """Map theme slot names (dk1, lt1, accent1, ...) to RRGGBB for the master's theme."""
    theme = prs.slide_master.part.part_related_by(RT.THEME)
    from lxml import etree

    root = etree.fromstring(theme.blob)
    scheme = root.find(".//" + qn("a:clrScheme"))
    colors = {}
    for slot in scheme:
        child = slot[0]
        colors[etree.QName(slot).localname] = child.get("val") or child.get("lastClr")
    return colors


def color_map(prs) -> dict[str, str]:
    """The master's clrMap, e.g. bg2 -> dk2 (the template swaps background/text roles)."""
    return dict(prs.slide_master._element.find(qn("p:clrMap")).attrib)


def _apply_luminance(rgb: str, element) -> str:
    red, green, blue = (int(rgb[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hue, light, sat = colorsys.rgb_to_hls(red, green, blue)
    for modifier in element:
        value = int(modifier.get("val")) / 100000
        if modifier.tag == qn("a:lumMod"):
            light *= value
        elif modifier.tag == qn("a:lumOff"):
            light += value
    red, green, blue = colorsys.hls_to_rgb(hue, min(max(light, 0), 1), sat)
    return "".join(f"{round(c * 255):02X}" for c in (red, green, blue))


def resolve(fill_holder, *, theme: dict[str, str], cmap: dict[str, str]) -> str | None:
    """RRGGBB for the first color child of a solidFill/highlight element, or None."""
    if fill_holder is None or not len(fill_holder):
        return None
    color = fill_holder[0]
    if color.tag == qn("a:srgbClr"):
        return color.get("val").upper()
    name = color.get("val")
    slot = cmap.get(name, name)
    return _apply_luminance(theme[slot], color)


def luminance(rgb: str) -> float:
    channels = []
    for i in (0, 2, 4):
        value = int(rgb[i:i + 2], 16) / 255
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def contrast_ratio(first: str, second: str) -> float:
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _layout_default(shape, tag: str):
    """The layout placeholder's lvl1 default run property child (solidFill or highlight)."""
    if not shape.is_placeholder:
        return None
    for layout_shape in shape.part.slide_layout.placeholders:
        if layout_shape.placeholder_format.idx == shape.placeholder_format.idx:
            level = layout_shape._element.find(".//" + qn("a:lvl1pPr"))
            defaults = None if level is None else level.find(qn("a:defRPr"))
            return None if defaults is None else defaults.find(qn(tag))
    return None


def check_run(shape, run, *, theme, cmap) -> tuple[str, str, float]:
    props = run._r.find(qn("a:rPr"))
    own = None if props is None else props.find(qn("a:solidFill"))
    text = resolve(own if own is not None else _layout_default(shape, "a:solidFill"),
                   theme=theme, cmap=cmap) or resolve_slot(FALLBACK_TEXT, theme, cmap)
    shape_fill = shape._element.find(qn("p:spPr")).find(qn("a:solidFill"))
    mark = None if props is None else props.find(qn("a:highlight"))
    back = (resolve(mark if mark is not None else _layout_default(shape, "a:highlight"),
                    theme=theme, cmap=cmap)
            or resolve(shape_fill, theme=theme, cmap=cmap)
            or resolve_slot(FALLBACK_BACKGROUND, theme, cmap))
    return text, back, contrast_ratio(text, back)


def check_cell_run(cell, run, *, theme, cmap) -> tuple[str, str, float]:
    """Contrast of a table-cell run against the cell's own fill (explicit colors only)."""
    props = run._r.find(qn("a:rPr"))
    own = None if props is None else props.find(qn("a:solidFill"))
    text = resolve(own, theme=theme, cmap=cmap) or resolve_slot(FALLBACK_TEXT, theme, cmap)
    properties = cell._tc.tcPr
    fill = None if properties is None else properties.find(qn("a:solidFill"))
    back = resolve(fill, theme=theme, cmap=cmap) or resolve_slot(FALLBACK_BACKGROUND, theme, cmap)
    return text, back, contrast_ratio(text, back)


def resolve_slot(name: str, theme: dict[str, str], cmap: dict[str, str]) -> str:
    return theme[cmap.get(name, name)]


def _runs(slide):
    """Yield (shape, run, checker) for every text run, including table cells."""
    for shape in slide.shapes:
        if shape.has_table:
            for row in shape.table.rows:
                for cell in row.cells:
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            yield cell, run, check_cell_run
        elif shape.has_text_frame:
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    yield shape, run, check_run


def main(argv: list[str]) -> int:
    prs = Presentation(str(Path(argv[1]) if len(argv) > 1 else DEFAULT_DECK))
    theme, cmap = theme_colors(prs), color_map(prs)
    failures = 0
    for number, slide in enumerate(prs.slides, start=1):
        for owner, run, checker in _runs(slide):
            if not run.text.strip():
                continue
            text, back, ratio = checker(owner, run, theme=theme, cmap=cmap)
            status = "ok  " if ratio >= MIN_CONTRAST else "FAIL"
            failures += status == "FAIL"
            print(f"{status} s{number} {text} on {back} {ratio:5.1f} {run.text[:34]!r}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
