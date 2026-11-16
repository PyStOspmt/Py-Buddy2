"""Deep JSON normalization into a relational shape.

A .json/.jsonl upload is not a flat table by nature: values nest. The
old ingest path bound raw dicts/lists into a single `records` table and
crashed on the first nested value. This module normalizes instead:

- a top-level ``{key: [objects]}`` becomes one table per array key;
- a top-level ``[objects]`` becomes the ``records`` table;
- nested dicts flatten into ``parent.child`` columns at any depth --
  the budget that binds is the column count per table, not the nesting
  level; keys dropped past the budget are counted in ``__ingest_meta``;
- arrays of objects inside a row become child tables
  ``parent_key`` whose ``_parent`` column is a DECLARED foreign key to
  the parent's ``_src`` -- so every downstream evidence channel
  (fk_dimension, multi_hop, junction) traverses the JSON structure
  exactly like a real schema;
- arrays of scalars are stored as compact JSON text;
- every row carries ``_src`` = its JSON path ("$.items[3]"), so cell
  lineage is reconstructable, and ``_src`` is the parent's PK.

The output is plain SQLite -- the projection engine needs no special
casing for JSON-origin sources beyond what a normal schema provides.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

MAX_COLS_PER_TABLE = 500  # column-explosion budget; past it subtrees
                          # serialize to JSON text (disclosed in meta)
MAX_TABLES = 64           # pathological nesting budget
_IDENT_SAFE = re.compile(r"[^A-Za-z0-9_]+")


@dataclass
class TableData:
    name: str
    headers: list[str] = field(default_factory=list)
    kinds: dict[str, str] = field(default_factory=dict)   # col -> REAL|TEXT
    rows: list[list] = field(default_factory=list)
    fks: list[tuple[str, str, str]] = field(default_factory=list)  # (col, parent_table, parent_col)
    truncated_keys: int = 0  # keys dropped by the column budget


def _ident(name: Any, used: set[str]) -> str:
    """Sanitize an arbitrary JSON key into a SQLite identifier; dedupe
    against `used` so colliding keys ('a-b', 'a b') stay separate."""
    base = _IDENT_SAFE.sub("_", str(name)).strip("_") or "col"
    if base[0].isdigit():
        base = "_" + base
    cand, i = base, 2
    while cand in used:
        cand = f"{base}_{i}"
        i += 1
    used.add(cand)
    return cand


def _infer_kinds(headers: list[str], rows: list[list]) -> dict[str, str]:
    """A column is REAL only when every non-null seen value parses as a
    number -- the same honest typing the flat ingest used."""
    kinds: dict[str, str] = {}
    probe = rows[:1000]
    for i, h in enumerate(headers):
        vals = [r[i] for r in probe
                if i < len(r) and r[i] is not None and str(r[i]).strip() != ""]
        def _num(v) -> bool:
            try:
                float(v)
                return True
            except (TypeError, ValueError):
                return False
        kinds[h] = "REAL" if vals and all(_num(v) for v in vals) else "TEXT"
    return kinds


def normalize_json(parsed: Any, base_name: str = "records",
                   max_tables: int = MAX_TABLES) -> dict[str, TableData]:
    """Normalize a parsed JSON document into named table sets.

    Raises ValueError for un-tabularizable roots (bare scalar).
    """
    tables: dict[str, TableData] = {}
    used_names: set[str] = set()
    # key -> column is stable per TABLE (row N must land in the same
    # column as row 1); only genuinely colliding keys get suffixed.
    keymap_by_table: dict[str, dict[str, str]] = {}
    child_by_key: dict[tuple[str, str], TableData] = {}

    def _table_for(name_hint: str) -> TableData:
        tname = _ident(name_hint, used_names)
        td = TableData(name=tname)
        tables[tname] = td
        return td

    def _emit_row(td: TableData, obj: dict[str, Any], src: str,
                  parent_src: str | None, depth: int) -> None:
        keymap = keymap_by_table.setdefault(td.name, {})
        vals: dict[str, Any] = {}
        if parent_src is not None:
            keymap.setdefault("_parent", "_parent")
            vals["_parent"] = parent_src

        def _col(key: str) -> str | None:
            """Resolve a JSON key to its stable column; None when the
            table's column budget is spent (caller serializes to text)."""
            if key in keymap:
                return keymap[key]
            if len(keymap) >= MAX_COLS_PER_TABLE:
                return None
            keymap[key] = _ident(key, set(keymap.values()))
            return keymap[key]

        queue = [(k, v) for k, v in obj.items()]
        while queue:
            key, val = queue.pop(0)
            if isinstance(val, dict):
                if _col(key) is None:
                    td.truncated_keys += len(val)
                    continue  # column budget spent -> counted, disclosed
                for sk, sv in val.items():
                    queue.append((f"{key}.{sk}", sv))
            elif isinstance(val, list):
                if all(isinstance(x, dict) for x in val) and val \
                        and len(tables) < max_tables:
                    leaf = key.split(".")[-1]
                    # one child table per (parent_table, json key) --
                    # rows of every parent row land together
                    child = child_by_key.get((td.name, leaf))
                    if child is None:
                        child = _table_for(
                            f"{td.name}_{_ident(leaf, set())}")
                        child.fks.append(("_parent", td.name, "_src"))
                        child_by_key[(td.name, leaf)] = child
                    for ci, cobj in enumerate(val):
                        _emit_row(child, cobj, f"{src}.{leaf}[{ci}]",
                                  src, depth + 1)
                elif _col(key) is not None:
                    vals[keymap[key]] = json.dumps(val,
                                                   ensure_ascii=False)
                else:
                    td.truncated_keys += 1
            else:
                c = _col(key)
                if c is None:
                    td.truncated_keys += 1
                else:
                    vals[c] = val
        for c in vals:
            if c not in td.headers:
                td.headers.append(c)
        if "_src" not in td.headers:
            td.headers.insert(0, "_src")
        row = [src] + [None] * (len(td.headers) - 1)
        for c, v in vals.items():
            row[td.headers.index(c)] = v
        td.rows.append(row)

    if isinstance(parsed, list):
        objs = [x if isinstance(x, dict) else {"value": x}
                for x in parsed]
        if not objs:
            raise ValueError("JSON array is empty")
        td = _table_for(base_name)
        for i, obj in enumerate(objs):
            _emit_row(td, obj, f"$[{i}]", None, 0)
    elif isinstance(parsed, dict):
        array_keys = [k for k, v in parsed.items()
                      if isinstance(v, list) and v
                      and all(isinstance(x, dict) for x in v)]
        scalars = {k: v for k, v in parsed.items() if k not in array_keys}
        if array_keys:
            for k in array_keys:
                td = _table_for(k)
                for i, obj in enumerate(parsed[k]):
                    _emit_row(td, obj, f"$.{k}[{i}]", None, 0)
            if scalars:
                td = _table_for("root")
                _emit_row(td, scalars, "$", None, 0)
        else:
            td = _table_for(base_name)
            _emit_row(td, parsed, "$", None, 0)
    else:
        raise TypeError(f"JSON root of type {type(parsed).__name__} "
                        "is not tabularizable")

    # Second pass: child tables may have grown columns after their first
    # rows were appended -> re-pad rows to header width.
    for td in tables.values():
        for row in td.rows:
            while len(row) < len(td.headers):
                row.append(None)
        td.kinds = _infer_kinds(td.headers[1:], [r[1:] for r in td.rows])
        td.kinds["_src"] = "TEXT"
    return tables


def write_json_sqlite(tables: dict[str, TableData], dest: str,
                      origin_name: str, origin_format: str) -> int:
    """Write normalized tables to `dest` as real SQLite with declared
    FKs and an ingest_meta provenance log. Returns total data rows."""
    from source_projection import quote_ident
    conn = sqlite3.connect(dest)
    try:
        total = 0
        for td in tables.values():
            coldefs = ", ".join(
                f"{quote_ident(h)} "
                + ("TEXT PRIMARY KEY" if h == "_src"
                   else td.kinds.get(h, "TEXT"))
                for h in td.headers)
            for col, ptab, pcol in td.fks:
                # A REAL declared FK: the projection's link registry and
                # every downstream channel see the JSON parent-child
                # relation exactly like a written-down schema relation.
                coldefs += (f", FOREIGN KEY ({quote_ident(col)}) "
                            f"REFERENCES {quote_ident(ptab)}"
                            f"({quote_ident(pcol)})")
            conn.execute(f"CREATE TABLE {quote_ident(td.name)} ({coldefs})")
            for col, _, _ in td.fks:
                conn.execute(
                    f"CREATE INDEX {quote_ident(td.name + '_' + col)} "
                    f"ON {quote_ident(td.name)} ({quote_ident(col)})")
            conn.executemany(
                f"INSERT INTO {quote_ident(td.name)} VALUES "
                f"({','.join('?' * len(td.headers))})",
                td.rows)
            total += len(td.rows)
        conn.execute("CREATE TABLE __ingest_meta (key TEXT PRIMARY KEY, "
                     "value TEXT)")
        conn.executemany("INSERT INTO __ingest_meta VALUES (?,?)", [
            ("origin_file", origin_name),
            ("origin_format", origin_format),
            ("converted", "json-normalized->sqlite"),
            ("tables", ",".join(tables)),
            ("rows", str(total)),
            ("truncated_keys", json.dumps(
                {td.name: td.truncated_keys for td in tables.values()
                 if td.truncated_keys}, ensure_ascii=False)),
        ])
        conn.commit()
        # FKs are real (declared in CREATE TABLE); a machine-readable
        # copy stays in ingest_meta for inspection tooling.
        conn.execute(
            "INSERT INTO __ingest_meta VALUES ('child_links', ?)",
            (json.dumps({td.name: td.fks for td in tables.values()
                         if td.fks}, ensure_ascii=False),))
        conn.commit()
    finally:
        conn.close()
    return total
