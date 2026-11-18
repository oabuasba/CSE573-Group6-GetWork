"""Reusable python-pptx helpers for the GetWork deck: placeholders, shapes, arrows, notes.

Colors are explicit RGB copied from the previous deck. Theme references are avoided because
the template's color map swaps background/text roles (bg2 -> dk2, tx2 -> lt2), so a theme
"light" color such as BACKGROUND_2 renders as dark gray.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide, SlideLayout
from pptx.util import Emu, Pt

GOLD = RGBColor(0xFF, 0xC6, 0x27)
MAROON = RGBColor(0x8C, 0x1D, 0x40)
INK = RGBColor(0x19, 0x19, 0x19)
GRAY = RGBColor(0x74, 0x74, 0x74)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF4, 0xF1, 0xF2)
FONT_NAME = "Arial"

INSET_X_EMU = 137150
INSET_Y_EMU = 54850
BULLET_INDENT_EMU = 164592
HEADING_LINE_SPACING = 0.9
OUTLINE_WIDTH_PT = 1
ARROW_WIDTH_PT = 2.25
CORNER_RADIUS = 0.08


@dataclass(frozen=True)
class Rect:
    """Position and size in EMU."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> int:
        return self.left + self.width // 2


@dataclass(frozen=True)
class TextLine:
    """One paragraph: optional bold lead-in, then text, in one size and color."""

    text: str
    size_pt: float
    bold: bool = False
    italic: bool = False
    color: RGBColor = INK
    lead: str = ""
    space_before_pt: float = 0
    bullet: bool = False


def clear_slides(prs: PresentationType) -> None:
    """Delete every slide and its relationship so no orphaned slide parts are saved."""
    slide_ids = prs.slides._sldIdLst
    for slide_id in list(slide_ids):
        prs.part.drop_rel(slide_id.rId)
        slide_ids.remove(slide_id)


def get_layout(prs: PresentationType, *, name: str) -> SlideLayout:
    """Return the slide layout called ``name``."""
    for layout in prs.slide_layouts:
        if layout.name == name:
            return layout
    raise KeyError(f"layout {name!r} not found in template")


def new_slide(prs: PresentationType, *, layout_name: str) -> Slide:
    """Add a slide from the named layout."""
    return prs.slides.add_slide(get_layout(prs, name=layout_name))


def get_placeholder(slide: Slide, *, idx: int):
    """Return the placeholder with ``idx`` on ``slide``."""
    for placeholder in slide.placeholders:
        if placeholder.placeholder_format.idx == idx:
            return placeholder
    raise KeyError(f"placeholder idx {idx} not on slide layout {slide.slide_layout.name!r}")


def place(shape, *, rect: Rect) -> None:
    """Move/resize a shape or placeholder to ``rect``."""
    shape.left, shape.top = Emu(rect.left), Emu(rect.top)
    shape.width, shape.height = Emu(rect.width), Emu(rect.height)


def set_title(slide: Slide, text: str, *, size_pt: int | None = None) -> None:
    """Fill the title placeholder; ``size_pt`` overrides the layout size when given."""
    frame = slide.shapes.title.text_frame
    frame.text = text
    if size_pt is not None:
        frame.paragraphs[0].runs[0].font.size = Pt(size_pt)


def style_heading(slide: Slide, *, idx: int, text: str, size_pt: int, color: RGBColor,
                  highlight: RGBColor, anchor: MSO_ANCHOR = MSO_ANCHOR.TOP) -> None:
    """Explicitly style a heading as in the old deck (bold, colored text on a highlight bar).

    Needed on layouts whose title/subtitle placeholders lack that styling.
    """
    frame = get_placeholder(slide, idx=idx).text_frame
    frame.text = text
    frame.vertical_anchor = anchor
    frame.paragraphs[0].line_spacing = HEADING_LINE_SPACING
    run = frame.paragraphs[0].runs[0]
    run.font.size = Pt(size_pt)
    run.font.bold = True
    run.font.color.rgb = color
    solid = run._r.get_or_add_rPr().find(qn("a:solidFill"))
    mark = solid.makeelement(qn("a:highlight"), {})
    mark.append(mark.makeelement(qn("a:srgbClr"), {"val": str(highlight)}))
    solid.addnext(mark)


def fit_rect(slot: Rect, *, aspect: float) -> Rect:
    """Largest rect of width/height ``aspect`` that fits ``slot``, centered in it."""
    width = min(slot.width, int(slot.height * aspect))
    height = int(width / aspect)
    return Rect(slot.left + (slot.width - width) // 2, slot.top + (slot.height - height) // 2,
                width, height)


def insert_fitted_picture(slide: Slide, *, idx: int, image: Path, slot: Rect) -> Rect:
    """Fill picture placeholder ``idx`` with ``image``, uncropped, centered in ``slot``."""
    with Image.open(image) as opened:
        pixels_wide, pixels_high = opened.size
    target = fit_rect(slot, aspect=pixels_wide / pixels_high)
    picture = get_placeholder(slide, idx=idx).insert_picture(str(image))
    picture.crop_left = picture.crop_right = picture.crop_top = picture.crop_bottom = 0
    place(picture, rect=target)
    return target


def set_text(slide: Slide, *, idx: int, text: str, bulleted: bool = False,
             size_pt: int | None = None) -> None:
    """Fill placeholder ``idx`` with one paragraph; strip the layout bullet unless asked."""
    frame = get_placeholder(slide, idx=idx).text_frame
    frame.text = text
    if size_pt is not None:
        frame.paragraphs[0].runs[0].font.size = Pt(size_pt)
    if not bulleted:
        _remove_bullet(frame.paragraphs[0])


def set_bullets(slide: Slide, *, idx: int, items: Sequence[str], size_pt: int,
                space_before_pt: float = 9) -> None:
    """Fill placeholder ``idx`` with ink-colored bullets styled like the previous deck."""
    frame = get_placeholder(slide, idx=idx).text_frame
    frame.text = items[0]
    for item in items[1:]:
        frame.add_paragraph().text = item
    for position, paragraph in enumerate(frame.paragraphs):
        paragraph.space_before = Pt(0 if position == 0 else space_before_pt)
        _set_bullet(paragraph)
        for run in paragraph.runs:
            run.font.size = Pt(size_pt)
            run.font.name = FONT_NAME
            run.font.color.rgb = INK


def _set_bullet(paragraph) -> None:
    """Small hanging-indent round bullet, as in the previous deck's text boxes."""
    properties = paragraph._p.get_or_add_pPr()
    properties.set("marL", str(BULLET_INDENT_EMU))
    properties.set("indent", str(-BULLET_INDENT_EMU))
    for child in list(properties):
        if child.tag in (qn("a:buChar"), qn("a:buNone"), qn("a:buClr"), qn("a:buFont")):
            properties.remove(child)
    colour = properties.makeelement(qn("a:buClr"), {})
    colour.append(colour.makeelement(qn("a:srgbClr"), {"val": "000000"}))
    properties.append(colour)
    properties.append(properties.makeelement(qn("a:buChar"), {"char": "\u2022"}))


def _remove_bullet(paragraph) -> None:
    """Use the same pPr override as the template's own sample slides: no bullet, no indent."""
    properties = paragraph._p.get_or_add_pPr()
    properties.set("marL", "0")
    properties.set("indent", "0")
    for child in list(properties):
        if child.tag in (qn("a:buChar"), qn("a:buNone"), qn("a:buAutoNum")):
            properties.remove(child)
    properties.append(properties.makeelement(qn("a:buNone"), {}))


def remove_placeholder(slide: Slide, *, idx: int) -> None:
    """Delete an unused placeholder so no 'Click to add text' prompt remains."""
    element = get_placeholder(slide, idx=idx)._element
    element.getparent().remove(element)


def add_picture_in_placeholder(slide: Slide, *, idx: int, image: Path) -> None:
    """Insert ``image`` into picture placeholder ``idx`` (cropped to fit by python-pptx)."""
    get_placeholder(slide, idx=idx).insert_picture(str(image))


def add_picture_at(slide: Slide, *, image: Path, rect: Rect) -> None:
    """Place ``image`` at ``rect`` (EMU); a zero height keeps the aspect ratio."""
    slide.shapes.add_picture(
        str(image), Emu(rect.left), Emu(rect.top), width=Emu(rect.width),
        height=Emu(rect.height) if rect.height else None,
    )


def add_speaker_notes(slide: Slide, text: str) -> None:
    """Attach speaker notes to ``slide``."""
    slide.notes_slide.notes_text_frame.text = text


def _write_lines(frame, *, lines: Sequence[TextLine], align: PP_ALIGN) -> None:
    """Replace the frame's text with one styled paragraph per ``TextLine``."""
    for position, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if position == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.space_before = Pt(line.space_before_pt)
        if line.bullet:
            _set_bullet(paragraph)
        for text, bold in ((line.lead, True), (line.text, line.bold)):
            if not text:
                continue
            run = paragraph.add_run()
            run.text = text
            run.font.size = Pt(line.size_pt)
            run.font.bold = bold
            run.font.italic = line.italic
            run.font.name = FONT_NAME
            run.font.color.rgb = line.color


def _prepare_frame(frame, *, anchor: MSO_ANCHOR, inset_x: int = INSET_X_EMU) -> None:
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = Emu(inset_x)
    frame.margin_top = frame.margin_bottom = Emu(INSET_Y_EMU)


def add_box(slide: Slide, *, rect: Rect, fill: RGBColor, lines: Sequence[TextLine],
            shape: MSO_SHAPE = MSO_SHAPE.ROUNDED_RECTANGLE, align: PP_ALIGN = PP_ALIGN.CENTER,
            anchor: MSO_ANCHOR = MSO_ANCHOR.MIDDLE, inset_x: int = INSET_X_EMU,
            outline: RGBColor | None = None):
    """Add a filled shape (thin ``outline`` if given) holding styled text, one shape."""
    box = slide.shapes.add_shape(shape, Emu(rect.left), Emu(rect.top),
                                 Emu(rect.width), Emu(rect.height))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        box.adjustments[0] = CORNER_RADIUS
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if outline is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = outline
        box.line.width = Pt(OUTLINE_WIDTH_PT)
    box.shadow.inherit = False
    _prepare_frame(box.text_frame, anchor=anchor, inset_x=inset_x)
    _write_lines(box.text_frame, lines=lines, align=align)
    return box


def add_label(slide: Slide, *, rect: Rect, lines: Sequence[TextLine],
              align: PP_ALIGN = PP_ALIGN.LEFT, anchor: MSO_ANCHOR = MSO_ANCHOR.MIDDLE):
    """Add a plain text box (band labels, captions, footers)."""
    label = slide.shapes.add_textbox(Emu(rect.left), Emu(rect.top),
                                     Emu(rect.width), Emu(rect.height))
    _prepare_frame(label.text_frame, anchor=anchor)
    label.text_frame.margin_top = label.text_frame.margin_bottom = Emu(0)
    label.text_frame.margin_left = label.text_frame.margin_right = Emu(0)
    _write_lines(label.text_frame, lines=lines, align=align)
    return label


def add_line(slide: Slide, *, start: tuple[int, int], end: tuple[int, int],
             color: RGBColor = MAROON, head: bool = False, tail: bool = True):
    """Add a straight connector; ``tail`` puts an arrowhead at ``end``, ``head`` at ``start``."""
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(start[0]), Emu(start[1]),
                                           Emu(end[0]), Emu(end[1]))
    connector.line.width = Pt(ARROW_WIDTH_PT)
    connector.line.color.rgb = color
    outline = connector.line._get_or_add_ln()
    for flag, tag in ((head, "a:headEnd"), (tail, "a:tailEnd")):
        if flag:
            outline.append(outline.makeelement(qn(tag), {"type": "triangle"}))
    return connector

