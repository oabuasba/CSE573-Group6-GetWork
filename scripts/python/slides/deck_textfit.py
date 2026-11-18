"""Word-wrap text-height estimator for the deck checks.

Measures with real Arial (or metric-compatible Liberation Sans / Arimo) when one is installed;
otherwise falls back to a conservative average glyph width. Wrapping is greedy by word, and
the height includes line spacing, paragraph space-before, bullet indent and frame insets.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from pptx.oxml.ns import qn

EMU_PER_POINT = 12700
LINE_HEIGHT_EM = 1.2
FALLBACK_EM = 0.55
FALLBACK_BOLD_EM = 0.60
DEFAULT_SIZE_PT = 14
MEASURE_PX = 200
FONT_CANDIDATES = {
    False: ("/mnt/c/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf"),
    True: ("/mnt/c/Windows/Fonts/arialbd.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
           "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf"),
}


@lru_cache(maxsize=None)
def _font(bold: bool):
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES[bold]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, MEASURE_PX)
    return None


def text_width_pt(text: str, *, size_pt: float, bold: bool) -> float:
    font = _font(bold)
    if font is None:
        return len(text) * size_pt * (FALLBACK_BOLD_EM if bold else FALLBACK_EM)
    return font.getlength(text) * size_pt / MEASURE_PX


Piece = tuple[str, float, bool]


def _words(paragraph) -> Iterator[list[Piece]]:
    """Words as lists of (text, size, bold) pieces; a word may span several runs."""
    word: list[Piece] = []
    for run in paragraph.runs:
        size = run.font.size.pt if run.font.size else DEFAULT_SIZE_PT
        for position, part in enumerate(run.text.split(" ")):
            if position:
                yield word
                word = []
            if part:
                word.append((part, size, bool(run.font.bold)))
    yield word


def wrapped_line_count(paragraph, *, width_emu: float) -> int:
    """Lines the paragraph needs in ``width_emu`` (greedy word wrap, bold runs measured bold)."""
    limit = width_emu / EMU_PER_POINT
    lines, used = 1, 0.0
    for word in _words(paragraph):
        if not word:
            continue
        word_width = sum(text_width_pt(t, size_pt=size, bold=bold) for t, size, bold in word)
        space = text_width_pt(" ", size_pt=word[0][1], bold=False)
        if used and used + space + word_width > limit:
            lines, used = lines + 1, word_width
        else:
            used += (space if used else 0) + word_width
    return lines


def _indent_emu(paragraph) -> int:
    properties = paragraph._p.find(qn("a:pPr"))
    return int(properties.get("marL", 0)) if properties is not None else 0


def paragraph_height_emu(paragraph, *, width_emu: float, spacing: float) -> float:
    """Height of one paragraph: wrapped lines at the largest run size plus space before."""
    if not paragraph.runs:
        return 0.0
    size = max((r.font.size.pt if r.font.size else DEFAULT_SIZE_PT) for r in paragraph.runs)
    factor = (LINE_HEIGHT_EM * paragraph.line_spacing
              if isinstance(paragraph.line_spacing, float) else spacing)
    lines = wrapped_line_count(paragraph, width_emu=width_emu - _indent_emu(paragraph))
    before = paragraph.space_before or 0
    return lines * size * factor * EMU_PER_POINT + before


def frame_height_emu(frame, *, width_emu: float, spacing: float, insets_y: float) -> float:
    total = insets_y
    for paragraph in frame.paragraphs:
        total += paragraph_height_emu(paragraph, width_emu=width_emu, spacing=spacing)
    return math.ceil(total)
