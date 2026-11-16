"""C1 regression: the 11-shape entity-table polygon asserts the
post-fix expectations (it FAILS on the old largest-table pick for
one_to_many_numeric_prefix, m2m_big_dimensions, snowflake_chain,
parent_with_rich_text_and_children), plus unit tests for the
delimiter-boundary rule and the new projection attributes.

All data is SYNTHETIC, generated deterministically by
entity_polygon.py's own builders into tmp dirs.
"""
from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "experiments",
                                "semantic_cycle"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "core"))
import entity_polygon
from source_projection import build_source_projection


def test_polygon_all_shapes():
    rows = entity_polygon.run_polygon()
    for r in rows:
        assert r["passed"], (
            f"{r['shape']}: chose {r['chosen']} "
            f"(ambig={r['entity_ambiguous']}, alt={r['alternatives']}) "
            f"expected {r['expect_table']}/"
            f"{r['expect_ambiguous']} {r['note'] or ''}")
    assert len(rows) == 11


def _db(tmp_path, ddl, rows):
    p = str(tmp_path / "x.db")
    conn = sqlite3.connect(p)
    conn.executescript(";".join(ddl))
    rows(conn)
    conn.commit()
    conn.close()
    return p


def test_numeric_prefix_is_not_a_detail(tmp_path):
    """'1234' extending '1' is a numeric coincidence -- the old
    raw-startswith rule demoted the real record table (the forum
    defect). The bigger child table must now win outright."""
    def rows(c):
        for i in range(1, 7):
            c.execute("INSERT INTO channels VALUES (?,?)",
                      (i, f"channel {i}"))
        for i in range(1, 301):
            c.execute("INSERT INTO messages VALUES (?,?,?)",
                      (i, (i % 6) + 1, f"body text {i} here"))
    db = _db(tmp_path,
             [("CREATE TABLE channels(cid INTEGER PRIMARY KEY,"
              " cname TEXT)"),
              ("CREATE TABLE messages(mid INTEGER PRIMARY KEY,"
              " cid INTEGER REFERENCES channels(cid), body TEXT)")],
             rows)
    proj = build_source_projection(db, fresh=True)
    assert proj.entity_table == "messages"
    assert proj.entity_ambiguous is False
    kinds = [e["kind"] for e in proj.build_trace]
    assert "entity_detail" not in kinds


def test_path_style_prefix_is_still_a_detail(tmp_path):
    """'$.docs[0].variants[1]' extends '$.docs[0]' at a delimiter --
    the child stays a facet of the parent even with more rows."""
    def rows(c):
        for i in range(60):
            c.execute("INSERT INTO docs VALUES (?,?,?)",
                      (f"$.docs[{i}]", f"title {i}",
                       f"body {i} with content"))
            for j in range(3):
                c.execute("INSERT INTO variants VALUES (?,?,?)",
                          (f"$.docs[{i}].variants[{j}]",
                           f"$.docs[{i}]", f"variant {j}"))
    db = _db(tmp_path,
             [("CREATE TABLE docs(pk TEXT PRIMARY KEY, title TEXT,"
              " body TEXT)"),
              ("CREATE TABLE variants(pk TEXT PRIMARY KEY,"
              " doc TEXT REFERENCES docs(pk), label TEXT)")],
             rows)
    proj = build_source_projection(db, fresh=True)
    assert proj.entity_table == "docs"
    assert proj.entity_ambiguous is False
    kinds = [e["kind"] for e in proj.build_trace]
    assert "entity_detail" in kinds


def test_lookup_fold_makes_rich_parent_an_alternative(tmp_path):
    """A folded lookup whose text column is genuinely free prose
    becomes an alternative and flips ambiguous -- a rich dimension is
    a plausible entity, honestly surfaced rather than dropped."""
    def rows(c):
        for i in range(1, 61):
            c.execute("INSERT INTO gadgets VALUES (?,?,?)",
                      (i, f"gadget {i}",
                       (f"an extended narrative description of gadget "
                       f"{i} that easily runs past one hundred and "
                       f"twenty characters of real prose")))
        for i in range(1, 401):
            c.execute("INSERT INTO opinions VALUES (?,?,?)",
                      (i, (i - 1) % 60 + 1, f"opinion body {i}"))
    db = _db(tmp_path,
             [("CREATE TABLE gadgets(gid INTEGER PRIMARY KEY,"
              " gname TEXT, description TEXT)"),
              ("CREATE TABLE opinions(oid INTEGER PRIMARY KEY,"
              " gid INTEGER REFERENCES gadgets(gid), body TEXT)")],
             rows)
    proj = build_source_projection(db, fresh=True)
    assert proj.entity_table == "opinions"
    assert proj.entity_ambiguous is True
    assert "gadgets" in proj.entity_alternatives


def test_no_entity_table_when_no_pk(tmp_path):
    def rows(c):
        for i in range(50):
            c.execute("INSERT INTO events VALUES (?,?)",
                      (f"ts-{i}", f"log line {i}"))
    db = _db(tmp_path,
             ["CREATE TABLE events(ts TEXT, line TEXT)"], rows)
    proj = build_source_projection(db, fresh=True)
    assert proj.entity_table is None
    assert proj.entity_ambiguous is False
    # and plan_documents honestly returns None downstream
    from record_documents import plan_documents
    assert plan_documents(proj, db) is None
