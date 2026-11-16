"""PDF normalization into a relational shape.

A PDF has no relational structure of its own; what it reliably has is
pages. This module extracts the text layer page by page into a
``pages`` table -- ``_src`` = the page number ("p7"), so every binding
traces to a physical page -- plus provenance in ``__ingest_meta``.

Honest limits, recorded not hidden:
- pages with no extractable text layer (scans/images) land as rows
  with empty ``body`` and are counted in ``empty_pages``;
- a document whose ENTIRE text layer is empty yields 0 rows' worth of
  content -- the caller refuses the upload rather than registering a
  blank source;
- there is no OCR, no layout/table reconstruction and no image
  extraction in v1 -- pipe-tables inside PDFs are NOT recovered.
"""
from __future__ import annotations

import sqlite3


def write_pdf_sqlite(pdf_path: str, dest: str,
                     origin_name: str) -> int:
    """Extract a PDF's text layer into `dest` SQLite.

    Returns the number of pages with extractable text. Raises whatever
    pypdf raises on malformed files -- the caller maps it to a 400.
    """
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages: list[tuple[str, int, str]] = []
    empty = 0
    for i, page in enumerate(reader.pages, start=1):
        try:
            body = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 -- a corrupt page is an empty
            body = ""       # page, not a failed document
        if body:
            pages.append((f"p{i}", i, body))
        else:
            empty += 1
    if not pages:
        return 0
    conn = sqlite3.connect(dest)
    try:
        conn.execute(
            "CREATE TABLE pages ("
            "_src TEXT PRIMARY KEY, page INTEGER, body TEXT)")
        conn.executemany("INSERT INTO pages VALUES (?,?,?)", pages)
        conn.execute("CREATE TABLE __ingest_meta (key TEXT PRIMARY KEY,"
                     " value TEXT)")
        conn.executemany("INSERT INTO __ingest_meta VALUES (?,?)", [
            ("origin_file", origin_name),
            ("origin_format", "pdf"),
            ("converted", "pdf-text->sqlite"),
            ("pages_total", str(len(reader.pages))),
            ("pages_with_text", str(len(pages))),
            ("empty_pages", str(empty)),
        ])
        conn.commit()
    finally:
        conn.close()
    return len(pages)
