"""The index.html that links and embeds every plot, so one file opens the whole story."""

from __future__ import annotations

import html
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexEntry:
    title: str
    how_to_read: str
    image: str | None = None
    interactive: str | None = None


def _entry_html(entry: IndexEntry) -> str:
    parts = [f"<h2>{html.escape(entry.title)}</h2>", f"<p>{html.escape(entry.how_to_read)}</p>"]
    if entry.interactive:
        src = html.escape(entry.interactive, quote=True)
        parts.append(f'<p><a href="{src}">open {src} on its own</a></p>')
        parts.append(f'<iframe src="{src}" width="100%" height="880" loading="lazy"></iframe>')
    if entry.image:
        src = html.escape(entry.image, quote=True)
        parts.append(f'<a href="{src}"><img src="{src}" alt="{html.escape(entry.title)}"></a>')
    return "\n".join(parts)


def build_index(*, title: str, summary: str, entries: list[IndexEntry]) -> str:
    body = "\n".join(_entry_html(entry) for entry in entries)
    return (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>\n"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 1em}"
        "img{max-width:100%;border:1px solid #ddd}</style></head>\n"
        f"<body><h1>{html.escape(title)}</h1><p>{html.escape(summary)}</p>\n{body}\n"
        "</body></html>\n"
    )
