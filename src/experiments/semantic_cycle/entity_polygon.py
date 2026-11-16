"""Entity-table detection polygon (C1): 11 tiny SYNTHETIC SQLite
shapes with labelled expectations, generated deterministically.

Every shape uses opaque non-commerce names except
`canonical_commerce_mini`, which reuses the real project's canonical
DDL (sqlite_master text of data/real_off.db tables -- schema text
only, never its rows).

run_polygon() -> list[dict] per shape:
    {shape, chosen, entity_ambiguous, alternatives,
     expect_table, expect_ambiguous, passed, note}
CLI:  python entity_polygon.py --out docs/polygon.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))

# Canonical DDL copied from `sqlite_master` of data/real_off.db
# (schema text only -- no data ever read from that source).
CANONICAL_DDL = [
    ('CREATE TABLE "products" ('
    'id TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,'
    ' brand TEXT NOT NULL, price REAL, merchant_id TEXT NOT NULL,'
    ' stock_quantity INTEGER NOT NULL, performance_score REAL NOT NULL,'
    ' created_at TEXT DEFAULT CURRENT_TIMESTAMP)'),
    ('CREATE TABLE merchants ('
    'id TEXT PRIMARY KEY, name TEXT NOT NULL, rating REAL NOT NULL,'
    ' delivery_days INTEGER NOT NULL, is_official INTEGER NOT NULL)'),
    ('CREATE TABLE commercial ('
    'product_id TEXT PRIMARY KEY, sponsored INTEGER NOT NULL DEFAULT 0,'
    ' commission_pct REAL NOT NULL DEFAULT 0.0)'),
    ('CREATE TABLE product_eav_meta ('
    'meta_id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT NOT NULL,'
    ' meta_key TEXT NOT NULL, meta_value TEXT NOT NULL)'),
]


def _db(tmp: str, name: str, ddl: list[str], rows) -> str:
    path = os.path.join(tmp, name + ".db")
    conn = sqlite3.connect(path)
    conn.executescript(";".join(ddl))
    rows(conn)
    conn.commit()
    conn.close()
    return path


def _flat(tmp):
    def rows(c):
        for i in range(1, 61):
            c.execute("INSERT INTO dossiers VALUES (?,?,?,?)",
                      (i, f"dossier headline {i}",
                       (f"a summary paragraph for dossier {i} with "
                       f"several descriptive words inside"), i * 1.5))
    return _db(tmp, "flat",
               [("CREATE TABLE dossiers(did INTEGER PRIMARY KEY,"
                " headline TEXT, summary TEXT, weight REAL)")], rows)


def _one_to_many_numeric_prefix(tmp):
    def rows(c):
        for i in range(1, 7):
            c.execute("INSERT INTO channels VALUES (?,?)",
                      (i, f"channel label {i}"))
        for i in range(1, 301):
            c.execute("INSERT INTO messages VALUES (?,?,?)",
                      (i, (i % 6) + 1,
                       f"message body {i} with enough text to read"))
    return _db(tmp, "one_to_many_numeric_prefix",
               [("CREATE TABLE channels(cid INTEGER PRIMARY KEY,"
                " cname TEXT)"),
                ("CREATE TABLE messages(mid INTEGER PRIMARY KEY,"
                " cid INTEGER REFERENCES channels(cid), body TEXT)")],
               rows)


def _m2m_ddl(declared: bool) -> list:
    fk = (" REFERENCES volumes(vid)" if declared else "",
          " REFERENCES creators(cid)" if declared else "",
          " REFERENCES topics(tid)" if declared else "",
          " REFERENCES shelves(sid)" if declared else "")
    return [
        ("CREATE TABLE volumes(vid INTEGER PRIMARY KEY, vtitle TEXT,"
         " vabstract TEXT)"),
        "CREATE TABLE creators(cid INTEGER PRIMARY KEY, cname TEXT)",
        "CREATE TABLE topics(tid INTEGER PRIMARY KEY, tname TEXT)",
        "CREATE TABLE shelves(sid INTEGER PRIMARY KEY, sname TEXT)",
        (f"CREATE TABLE volume_creators(v INTEGER{fk[0]},"
         f" c INTEGER{fk[1]})"),
        (f"CREATE TABLE volume_topics(v INTEGER{fk[0]},"
         f" t INTEGER{fk[2]})"),
        (f"CREATE TABLE volume_shelves(v INTEGER{fk[0]},"
         f" s INTEGER{fk[3]})"),
    ]


def _m2m_rows(c):
    for i in range(1, 81):
        c.execute("INSERT INTO volumes VALUES (?,?,?)",
                  (i, f"volume title {i}",
                   (f"an abstract for volume {i} containing "
                    f"descriptive sentences about the work")))
    for i in range(1, 91):
        c.execute("INSERT INTO creators VALUES (?,?)",
                  (i, f"creator name {i}"))
    for i in range(1, 121):
        c.execute("INSERT INTO topics VALUES (?,?)",
                  (i, f"topic name {i}"))
    for i in range(1, 7):
        c.execute("INSERT INTO shelves VALUES (?,?)", (i, f"shelf {i}"))
    for i in range(1, 81):
        for j in range(3):
            c.execute("INSERT INTO volume_creators VALUES (?,?)",
                      (i, (i + j - 1) % 90 + 1))
        c.execute("INSERT INTO volume_topics VALUES (?,?)",
                  (i, (i - 1) % 120 + 1))
        c.execute("INSERT INTO volume_shelves VALUES (?,?)",
                  (i, (i - 1) % 6 + 1))


def _m2m(tmp, name: str, declared: bool):
    return _db(tmp, name, _m2m_ddl(declared), _m2m_rows)


def _eav(tmp):
    def rows(c):
        for i in range(1, 51):
            c.execute("INSERT INTO specimens VALUES (?,?,?)",
                      (i, f"specimen label {i}",
                       f"notes about specimen {i} and its origin"))
            for k in ("color", "texture", "origin"):
                c.execute("INSERT INTO specimen_attrs VALUES (?,?,?)",
                          (i, k, f"{k} value {i}"))
    return _db(tmp, "eav",
               [("CREATE TABLE specimens(sid INTEGER PRIMARY KEY,"
                " label TEXT, notes TEXT)"),
                ("CREATE TABLE specimen_attrs(sid INTEGER, axis TEXT,"
                " reading TEXT)")], rows)


def _canonical_commerce_mini(tmp):
    def rows(c):
        for i in range(1, 61):
            c.execute(
                "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,?)",
                (f"p{i:03d}", f"product name {i}", f"cat{i % 7}",
                 f"brand{i % 11}", 10.0 + i, f"m{(i % 3) + 1}",
                 i * 2, 0.5, "2026-01-01"))
        for i in range(1, 4):
            c.execute("INSERT INTO merchants VALUES (?,?,?,?,?)",
                      (f"m{i}", f"merchant {i}", 4.5, 3, 1))
        for i in range(1, 61):
            c.execute("INSERT INTO commercial VALUES (?,?,?)",
                      (f"p{i:03d}", 0, 2.5))
            for k in ("color", "material"):
                c.execute("INSERT INTO product_eav_meta"
                          "(product_id, meta_key, meta_value)"
                          " VALUES (?,?,?)",
                          (f"p{i:03d}", k, f"{k}-{i}"))
    return _db(tmp, "canonical_commerce_mini", CANONICAL_DDL, rows)


def _snowflake_chain(tmp):
    def rows(c):
        for i in range(1, 7):
            c.execute("INSERT INTO desks VALUES (?,?)",
                      (i, f"desk {i}"))
        for i in range(1, 41):
            c.execute("INSERT INTO sections VALUES (?,?,?)",
                      (i, (i - 1) % 6 + 1, f"section {i}"))
        for i in range(1, 301):
            c.execute("INSERT INTO articles VALUES (?,?,?,?)",
                      (i, (i - 1) % 40 + 1, f"article headline {i}",
                       f"the full text of article number {i} goes here"))
    return _db(tmp, "snowflake_chain",
               ["CREATE TABLE desks(did INTEGER PRIMARY KEY, dname TEXT)",
                ("CREATE TABLE sections(sid INTEGER PRIMARY KEY,"
                " did INTEGER REFERENCES desks(did), sname TEXT)"),
                ("CREATE TABLE articles(aid INTEGER PRIMARY KEY,"
                " sid INTEGER REFERENCES sections(sid),"
                " headline TEXT, body TEXT)")], rows)


def _parent_rich_text(tmp):
    def rows(c):
        for i in range(1, 61):
            c.execute("INSERT INTO gadgets VALUES (?,?,?)",
                      (i, f"gadget {i}",
                       (f"an extended narrative description of gadget "
                       f"{i} that easily runs past one hundred and "
                       f"twenty characters of prose to qualify")))
        for i in range(1, 401):
            c.execute("INSERT INTO opinions VALUES (?,?,?)",
                      (i, (i - 1) % 60 + 1,
                       f"opinion body number {i} with commentary"))
    return _db(tmp, "parent_with_rich_text_and_children",
               [("CREATE TABLE gadgets(gid INTEGER PRIMARY KEY,"
                " gname TEXT, description TEXT)"),
                ("CREATE TABLE opinions(oid INTEGER PRIMARY KEY,"
                " gid INTEGER REFERENCES gadgets(gid), body TEXT)")],
               rows)


def _two_record_tables(tmp):
    def rows(c):
        for i in range(1, 121):
            c.execute("INSERT INTO memos VALUES (?,?,?)",
                      (i, f"memo subject {i}",
                       f"memo content {i} with some discussion"))
        for i in range(1, 91):
            c.execute("INSERT INTO faq VALUES (?,?,?)",
                      (i, f"question {i}?",
                       f"answer {i} in a few sentences"))
    return _db(tmp, "two_independent_record_tables",
               [("CREATE TABLE memos(mid INTEGER PRIMARY KEY,"
                " subject TEXT, content TEXT)"),
                ("CREATE TABLE faq(fid INTEGER PRIMARY KEY,"
                " question TEXT, answer TEXT)")], rows)


def _no_pk_log(tmp):
    def rows(c):
        for i in range(1, 80):
            c.execute("INSERT INTO events VALUES (?,?)",
                      (f"2026-01-{(i % 28) + 1:02d}T10:{i % 60:02d}",
                       f"log line {i} of event stream output"))
    return _db(tmp, "no_pk_log",
               ["CREATE TABLE events(ts TEXT, line TEXT)"], rows)


def _json_path_detail(tmp):
    def rows(c):
        for i in range(50):
            c.execute("INSERT INTO docs VALUES (?,?,?)",
                      (f"$.docs[{i}]", f"doc title {i}",
                       f"body of document {i} with real content"))
            for j in range(3):
                c.execute("INSERT INTO variants VALUES (?,?,?)",
                          (f"$.docs[{i}].variants[{j}]",
                           f"$.docs[{i}]",
                           f"variant label {j} of doc {i}"))
    return _db(tmp, "json_path_detail",
               [("CREATE TABLE docs(pk TEXT PRIMARY KEY, title TEXT,"
                " body TEXT)"),
                ("CREATE TABLE variants(pk TEXT PRIMARY KEY,"
                " doc TEXT REFERENCES docs(pk), label TEXT)")], rows)


SHAPES = [
    ("flat", _flat, "dossiers", False, None),
    ("one_to_many_numeric_prefix", _one_to_many_numeric_prefix,
     "messages", False, None),
    ("m2m_big_dimensions",
     lambda t: _m2m(t, "m2m_big_dimensions", True),
     "volumes", False, None),
    ("m2m_undeclared",
     lambda t: _m2m(t, "m2m_undeclared", False),
     "volumes", None,
     "invariant: chosen == volumes or entity_ambiguous"),
    ("eav", _eav, "specimens", False, None),
    ("canonical_commerce_mini", _canonical_commerce_mini,
     "products", False, None),
    ("snowflake_chain", _snowflake_chain, "articles", False, None),
    ("parent_with_rich_text_and_children", _parent_rich_text,
     "opinions", True, "alternatives must contain gadgets"),
    ("two_independent_record_tables", _two_record_tables,
     ("memos", "faq"), True, None),
    ("no_pk_log", _no_pk_log, None, False, None),
    ("json_path_detail", _json_path_detail, "docs", False, None),
]


def _check(chosen, ambiguous, alternatives, expect_table,
           expect_ambiguous, note) -> bool:
    if note and note.startswith("invariant"):
        return bool(chosen == "volumes" or ambiguous)
    ok_table = (chosen in expect_table if isinstance(expect_table, tuple)
                else chosen == expect_table)
    if not ok_table:
        return False
    if expect_ambiguous is not None and ambiguous != expect_ambiguous:
        return False
    if note and "alternatives" in note:
        return "gadgets" in (alternatives or [])
    return True


def run_polygon() -> list[dict]:
    """Build each shape in a temp dir, project it, check the label."""
    from source_projection import build_source_projection

    out = []
    tmp = tempfile.mkdtemp(prefix="entity_polygon_")
    for name, builder, expect_table, expect_amb, note in SHAPES:
        db = builder(tmp)
        proj = build_source_projection(db, fresh=True)
        ambiguous = getattr(proj, "entity_ambiguous", None)
        alternatives = list(getattr(proj, "entity_alternatives",
                                    []) or [])
        out.append({
            "shape": name,
            "chosen": proj.entity_table,
            "entity_ambiguous": ambiguous,
            "alternatives": alternatives,
            "candidates": [c.table for c in proj.entity_candidates],
            "expect_table": expect_table,
            "expect_ambiguous": expect_amb,
            "note": note,
            "passed": _check(proj.entity_table, ambiguous,
                             alternatives, expect_table,
                             expect_amb, note),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = run_polygon()
    for r in rows:
        print(f"{r['shape']:38s} -> {r['chosen']!s:12s} "
              f"ambig={r['entity_ambiguous']} alt={r['alternatives']} "
              f"expect={r['expect_table']}/{r['expect_ambiguous']} "
              f"{'PASS' if r['passed'] else 'FAIL'}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"polygon": rows}, f, indent=2)
        print("saved:", args.out)


if __name__ == "__main__":
    main()
