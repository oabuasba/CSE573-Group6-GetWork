"""Crop white margins and the footnote line off the regenerated taxonomy cluster map.

Usage: uv run python scripts/python/slides/crop_taxonomy_image.py [SRC.png] [DEST.png]
The footnote is the last text row, below FOOTNOTE_CUTOFF_FRACTION of the image height; the
slide's speaker notes carry that sentence instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops

DEFAULT_SRC = Path("docs/presentation/plot_v7/cluster_map.png")
DEFAULT_DEST = Path("docs/presentation/screenshots/taxonomy.png")
FOOTNOTE_CUTOFF_FRACTION = 0.96
PADDING_PX = 12
WHITE_THRESHOLD = 245


def crop_margins(src: Path, dest: Path) -> tuple[int, int]:
    """Write the cropped image to ``dest`` and return its (width, height)."""
    image = Image.open(src).convert("RGB")
    body = image.crop((0, 0, image.width, int(image.height * FOOTNOTE_CUTOFF_FRACTION)))
    mask = ImageChops.invert(body.convert("L").point(lambda v: 255 if v < WHITE_THRESHOLD else 0))
    box = ImageChops.invert(mask).getbbox()
    left, top, right, bottom = box
    padded = (max(left - PADDING_PX, 0), max(top - PADDING_PX, 0),
              min(right + PADDING_PX, body.width), min(bottom + PADDING_PX, body.height))
    cropped = body.crop(padded)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(dest)
    return cropped.size


if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DEST
    print(crop_margins(source, target))
