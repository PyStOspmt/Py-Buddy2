"""Markdown normalization into a relational shape.

A .md upload is a document, not a table -- but its structure is real:
headings define a section hierarchy and pipe-tables are relational.
This module converts that structure into SQLite so every downstream
evidence channel works unchanged:

- each heading opens a row in ``sections`` carrying ``level``, the
  heading ``title``, the heading ancestry ``path`` ("Головне > Підрозділ"),
  the section ``body`` text, and ``_src`` = the line range the section
  spans ("L12-L40") -- provenance to the exact lines;
- pipe-tables become real tables ``table_N`` with typed columns and a
  ``section_id`` FK back to the section that contains them;
- text before the first heading lands in a level-0 preamble section;
- ``__ingest_meta`` records the conversion and counts.

Lists, links and emphasis stay in ``body`` text for v1 -- searchable
through the entity-column channel, honestly not structured.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

_HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
_TROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TSEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


@dataclass
class Section:
    level: int
    title: str
    path: str
    body: list[str] = field(default_factory=list)
    start: int = 0
    end: int = 0


def normalize_markdown(text: str) -> tuple[list[Section],
                                           list[tuple[str, list[str],
                                                      list[list[str]],
                                                      str]]]:
    """Split markdown into a section tree and extracted pipe-tables.

    Returns (sections, tables) where each table is
    (name, headers, rows, src_line_range)."""
    lines = text.splitlines()
    sections: list[Section] = []
    md_tables: list[tuple[str, list[str], list[list[str]]]] = []
    stack: list[Section] = []
    cur = Section(level=0, title="(preamble)", path="", start=1)
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEAD.match(line)
        if m:
            cur.end = i
            sections.append(cur)
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1].level >= level:
                stack.pop()
            path = " > ".join([s.title for s in stack] + [title])
            cur = Section(level=level, title=title, path=path,
                          start=i + 1)
            stack.append(cur)
            i += 1
            continue
        if _TROW.match(line):
            tstart = i
            block = []
            while i < len(lines) and _TROW.match(lines[i]):
                if not _TSEP.match(lines[i]):
                    block.append(lines[i])
                i += 1
            if len(block) >= 2:
                headers = [c.strip() or f"col_{j}"
                           for j, c in enumerate(
                               block[0].strip("|").split("|"))]
                rows = [[c.strip() for c in r.strip("|").split("|")]
                        for r in block[1:]]
                md_tables.append(
                    (f"table_{len(md_tables) + 1}", headers, rows,
                     f"L{tstart + 1}-L{i}"))
                cur.body.append(f"[table_{len(md_tables)}]")
            continue
        cur.body.append(line)
        i += 1
    cur.end = len(lines)
    sections.append(cur)
    return sections, md_tables


def write_markdown_sqlite(text: str, dest: str,
                          origin_name: str) -> int:
    """Write a normalized markdown document to `dest`. Returns the
    section count (0 = empty document, honestly refused upstream)."""
    sections, md_tables = normalize_markdown(text)
    sections = [s for s in sections
                if s.title != "(preamble)" or any(
                    l.strip() for l in s.body)]
    if not sections:
        return 0
    conn = sqlite3.connect(dest)
    try:
        conn.execute(
            "CREATE TABLE sections ("
            "_src TEXT PRIMARY KEY, "
            "level INTEGER, title TEXT, path TEXT, body TEXT)")
        conn.executemany(
            "INSERT INTO sections VALUES (?,?,?,?,?)",
            [(f"L{s.start}-L{s.end}", s.level, s.title, s.path,
              "\n".join(s.body).strip())
             for s in sections])
        for name, headers, rows, tsrc in md_tables:
            cols = ", ".join(
                f"\"{h.replace(chr(34), chr(34)*2)}\" TEXT"
                for h in headers)
            sec_src = None
            # the table belongs to the section whose body referenced it
            for s in sections:
                if f"[{name}]" in "\n".join(s.body):
                    sec_src = f"L{s.start}-L{s.end}"
                    break
            conn.execute(
                f"CREATE TABLE \"{name}\" (_src TEXT, "
                f"section_id TEXT REFERENCES sections(_src), {cols})")
            conn.executemany(
                f"INSERT INTO \"{name}\" VALUES "
                f"({','.join(['?'] * (len(headers) + 2))})",
                [[tsrc, sec_src] + r for r in rows])
        conn.execute("CREATE TABLE __ingest_meta (key TEXT PRIMARY KEY,"
                     " value TEXT)")
        conn.executemany("INSERT INTO __ingest_meta VALUES (?,?)", [
            ("origin_file", origin_name),
            ("origin_format", "markdown"),
            ("converted", "markdown->sqlite"),
            ("sections", str(len(sections))),
            ("tables", str(len(md_tables))),
        ])
        conn.commit()
    finally:
        conn.close()
    return len(sections)
