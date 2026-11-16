"""Handoff-2 record_documents tests: dirty merge-stream rows and the
identifier-like refinement. Split out of test_record_documents.py for
the 350-line bound; helpers are imported from there.

All databases are built in-test -- SYNTHETIC, labelled test data.
"""

from __future__ import annotations

import sqlite3

from record_documents import (
    RecordDoc,
    RenderStats,
    iter_record_documents,
    plan_documents,
)
from semantic_fixtures import (
    make_eav_db,
    make_normalized_db,
    projection,
)
from test_record_documents import _docs, _excluded_reasons


def test_dirty_reference_rows_skipped_and_counted(tmp_path):
    """SYNTHETIC dirty merge-stream rows: a NULL or orphan entity ref
    must be skipped and counted, never stall the merge nor change a
    single rendered document."""
    db = make_eav_db(str(tmp_path / "e.db"), n=8)
    clean = {d.entity_id: d.text for d in iter_record_documents(
        db, plan_documents(projection(db), db))
        if isinstance(d, RecordDoc)}
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO attr_rows VALUES (NULL,'x','no owner')")
    conn.execute("INSERT INTO attr_rows VALUES ('0','x','before')")
    conn.execute("INSERT INTO attr_rows VALUES ('4x','x','middle')")
    conn.execute("INSERT INTO attr_rows VALUES ('999','x','trailing')")
    conn.commit()
    conn.close()
    stats = RenderStats()
    dirty = {d.entity_id: d.text for d in iter_record_documents(
        db, plan_documents(projection(db), db), stats=stats)
        if isinstance(d, RecordDoc)}
    assert dirty == clean
    assert stats.null_ref_rows == 1 and stats.orphan_rows == 3

    db2 = make_normalized_db(str(tmp_path / "n.db"))
    clean2 = {d.entity_id: d.text for d in iter_record_documents(
        db2, plan_documents(projection(db2), db2))
        if isinstance(d, RecordDoc)}
    conn = sqlite3.connect(db2)
    conn.execute("INSERT INTO entry_labels VALUES (NULL, 1)")
    conn.execute("INSERT INTO entry_labels VALUES (0, 2)")
    conn.execute("INSERT INTO entry_labels VALUES (9998, 1)")
    conn.execute("INSERT INTO entry_labels VALUES (9999, 3)")
    conn.commit()
    conn.close()
    stats2 = RenderStats()
    dirty2 = {d.entity_id: d.text for d in iter_record_documents(
        db2, plan_documents(projection(db2), db2), stats=stats2)
        if isinstance(d, RecordDoc)}
    assert dirty2 == clean2
    assert stats2.null_ref_rows == 1 and stats2.orphan_rows == 3


def _make_singleword_label_db(path: str) -> str:
    """SYNTHETIC: unique single-word lookup labels on a dimension and
    on a bridge's other table (the digit-less label shape)."""
    conn = sqlite3.connect(path)
    conn.executescript("""
    CREATE TABLE topics(tid INTEGER PRIMARY KEY, tname TEXT);
    INSERT INTO topics VALUES
        (1,'astronomy'), (2,'cooking'), (3,'sailing');
    CREATE TABLE gloss(gid INTEGER PRIMARY KEY, gword TEXT);
    INSERT INTO gloss VALUES (1,'anchor'), (2,'rudder'), (3,'mast');
    CREATE TABLE volumes(vid INTEGER PRIMARY KEY, title TEXT,
                         blurb TEXT,
                         tid INTEGER REFERENCES topics(tid));
    CREATE TABLE volume_gloss(volume INTEGER REFERENCES volumes(vid),
                              term INTEGER REFERENCES gloss(gid));
    """)
    for i in range(1, 13):
        conn.execute(
            "INSERT INTO volumes VALUES (?,?,?,?)",
            (i, f"chapter {i} long narrative passage",
             f"more descriptive prose number {i}", ((i - 1) % 3) + 1))
        conn.execute("INSERT INTO volume_gloss VALUES (?,?)",
                     (i, ((i - 1) % 3) + 1))
    conn.commit()
    conn.close()
    return path


def test_unique_singleword_labels_fold(tmp_path):
    """Lookup-table labels are unique by design: single-word names
    fold through dimensions AND bridges (no identifier veto there)."""
    db = _make_singleword_label_db(str(tmp_path / "l.db"))
    plan = plan_documents(projection(db), db)
    assert plan is not None and plan.entity_table == "volumes"
    assert any("tname" in d.label_cols for d in plan.dimensions)
    assert any("gword" in b.label_cols for b in plan.bridges)
    docs = {d.entity_id: d for d in _docs(db, plan)}
    assert "topics: astronomy" in docs["1"].text
    assert "gloss: anchor" in docs["1"].text


def test_code_like_columns_stay_identifier_like(tmp_path):
    """Digit-bearing unique tokens (codes, UUIDs, ISBNs) remain
    excluded; a pure-alpha unique token column now folds."""
    db = str(tmp_path / "c.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE items(iid INTEGER PRIMARY KEY, "
                 "descr TEXT, ref1 TEXT, ref2 TEXT, ref3 TEXT, "
                 "ref4 TEXT)")
    vals = [
        (i + 1, f"a descriptive text body for item {i + 1}",
         f"SKU-{i + 1:05d}",
         f"a94a8fe{i}-ccb1-42ba-9c4e-4f1d3b2a1c0e",
         f"978-3-16-1484{i}-X",
         f"XYZZQWVA{i // 26 if i >= 26 else ''}{chr(65 + i)}UNIQ")
        for i in range(8)]
    conn.executemany("INSERT INTO items VALUES (?,?,?,?,?,?)", vals)
    conn.commit()
    conn.close()
    plan = plan_documents(projection(db), db)
    reasons = _excluded_reasons(plan)
    assert reasons["items.ref1"] == "identifier_like"
    assert reasons["items.ref2"] == "identifier_like"
    assert reasons["items.ref3"] == "identifier_like"
    assert "ref4" in plan.own_columns    # digit-free unique tokens fold


def test_unique_singleword_own_column_folds(tmp_path):
    """Glossary shape: a unique single-word `term` column is the
    entity's name -- excluding it would leave every entity no_text."""
    db = str(tmp_path / "g.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE terms(tid INTEGER PRIMARY KEY, "
                 "term TEXT, gloss TEXT)")
    for i, (t, g) in enumerate([
            ("photosynthesis", "process using light energy"),
            ("metamorphosis", "a change of form"),
            ("equilibrium", "a balanced stable state")], 1):
        conn.execute("INSERT INTO terms VALUES (?,?,?)", (i, t, g))
    conn.commit()
    conn.close()
    plan = plan_documents(projection(db), db)
    assert "term" in plan.own_columns
    docs = _docs(db, plan)
    assert all(isinstance(d, RecordDoc) for d in docs)
    assert "photosynthesis" in docs[0].text


def test_parent_key_not_unique_skips_dimension(tmp_path):
    """A dim join on a non-unique parent key would duplicate entity
    rows -- the link is excluded, entities still reconcile 1:1."""
    db = str(tmp_path / "p.db")
    conn = sqlite3.connect(db)
    conn.executescript("""
    CREATE TABLE groups(gid INTEGER PRIMARY KEY, gcode TEXT,
                        gname TEXT);
    INSERT INTO groups VALUES (1,'shared','warm gear'),
        (2,'shared','cold gear'), (3,'solo','wet gear');
    CREATE TABLE items(iid INTEGER PRIMARY KEY, name TEXT,
                       descr TEXT, gc TEXT REFERENCES groups(gcode));
    """)
    for i in range(6):
        conn.execute(
            "INSERT INTO items VALUES (?,?,?,?)",
            (i + 1, f"item {i + 1} short name",
             f"item {i + 1} longer descriptive text",
             "shared" if i % 2 else "solo"))
    conn.commit()
    conn.close()
    plan = plan_documents(projection(db), db)
    assert plan.dimensions == ()
    reasons = _excluded_reasons(plan)
    assert reasons["groups.gcode"] == "parent_key_not_unique"
    docs = _docs(db, plan)
    assert len(docs) == 6
    assert all(isinstance(d, RecordDoc) for d in docs)
