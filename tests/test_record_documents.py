"""Tests for record_documents (Retrieval Fabric Phase 1).

All databases are built by `semantic_fixtures` -- SYNTHETIC, labelled
test data with opaque-but-plausible names and no commerce assumptions.
"""

from __future__ import annotations

import sqlite3
import tracemalloc

import pytest
from record_documents import (
    DocPlan,
    ExcludedRecord,
    RecordDoc,
    iter_record_documents,
    plan_documents,
)
from semantic_fixtures import (
    make_eav_db,
    make_flat_db,
    make_normalized_db,
    make_null_pk_and_empty_db,
    projection,
)


def _docs(db_path, plan, **kw):
    return list(iter_record_documents(db_path, plan, **kw))


def _excluded_reasons(plan):
    return dict(plan.excluded)


def test_flat_shape_folds_text_and_excludes_rest(tmp_path):
    db = make_flat_db(str(tmp_path / "f.db"))
    plan = plan_documents(projection(db), db)
    assert plan is not None and plan.entity_table == "entries"
    assert set(plan.own_columns) == {"title", "body"}
    reasons = _excluded_reasons(plan)
    assert reasons["entries.eid"] == "id_column"
    assert reasons["entries.refcode"] == "identifier_like"
    assert reasons["entries.amount"] == "numeric"
    assert reasons["entries.stamp"] == "temporal"
    assert reasons["entries.note"] == "sparse"
    docs = _docs(db, plan)
    assert all(isinstance(d, RecordDoc) for d in docs)
    assert len(docs) == 8
    for d in docs:
        assert "title:" in d.text and "body:" in d.text
        for bad in ("CODE-", "refcode", "amount", "stamp", "note"):
            assert bad not in d.text
        assert not d.truncated and len(d.doc_hash) == 32
    # rd/2: own columns ordered by distinctiveness; on tied ratios the
    # shorter column still leads -- title before body
    assert docs[0].text.index("title:") < docs[0].text.index("body:")
    assert plan.display_column == "title"


def test_normalized_shape_folds_dim_and_bridge_labels(tmp_path):
    db = make_normalized_db(str(tmp_path / "n.db"))
    plan = plan_documents(projection(db), db)
    assert len(plan.dimensions) == 1
    assert plan.dimensions[0].parent_table == "groups"
    assert len(plan.bridges) == 1
    assert plan.bridges[0].other_table == "labels"
    docs = {d.entity_id: d for d in _docs(db, plan)}
    assert len(docs) == 12
    d = docs["1"]                       # grp=1, labs {1,2}
    assert "groups: warm weather gear" in d.text
    # bridge labels sorted ascending, joined "; "
    assert "labels: alpha tag; zeta tag" in d.text


def test_eav_shape_folds_axis_values(tmp_path):
    db = make_eav_db(str(tmp_path / "e.db"))
    plan = plan_documents(projection(db), db)
    assert len(plan.eav) == 1
    spec = plan.eav[0]
    assert (spec.entity_col, spec.key_col, spec.value_col) == \
        ("owner", "axis", "reading")
    docs = {d.entity_id: d for d in _docs(db, plan)}
    assert "thing 1" in docs["1"].text
    assert "hue: hue value 1" in docs["2"].text
    # entities 1 and 4 carry a pure-numeric hue value -> pair skipped
    assert "hue:" not in docs["1"].text
    assert "hue:" not in docs["4"].text
    assert "origin:" in docs["4"].text


def test_determinism_byte_identical(tmp_path):
    db = make_normalized_db(str(tmp_path / "n.db"))
    plan = plan_documents(projection(db), db)
    a = [d for d in _docs(db, plan)]
    b = [d for d in _docs(db, plan)]
    assert [d.doc_hash for d in a] == [d.doc_hash for d in b]
    assert [d.text for d in a] == [d.text for d in b]
    assert [d.entity_id for d in a] == [d.entity_id for d in b]


def test_every_entity_yields_exactly_one_outcome(tmp_path):
    db = make_null_pk_and_empty_db(str(tmp_path / "x.db"))
    plan = plan_documents(projection(db), db)
    out = _docs(db, plan)
    assert len(out) == 3           # 3 rows -> 3 outcomes, no drops
    docs = [o for o in out if isinstance(o, RecordDoc)]
    excl = [o for o in out if isinstance(o, ExcludedRecord)]
    # The NULL-bearing PK is not a valid id column: the plan falls
    # back to rowid, so rows are identified 1..3 and only the all-NULL
    # row is excluded.
    assert plan.id_column == "rowid"
    assert dict(plan.excluded).get("rows_x.xid") == "id_not_unique"
    assert sorted(d.entity_id for d in docs) == ["1", "3"]
    assert [o.reason for o in excl] == ["no_text"]


def test_truncation_flag_and_cut(tmp_path):
    db = make_flat_db(str(tmp_path / "f.db"))
    plan = plan_documents(projection(db), db)
    docs = _docs(db, plan, max_chars=40)
    assert all(d.truncated and len(d.text) == 40 for d in docs)


def _folded_values(doc):
    """Values without their source-name labels."""
    return [line.partition(": ")[2] for line in doc.text.split("\n")]


def test_rename_control_same_values(tmp_path):
    """RENAMED to opaque tokens, the same entities fold the same VALUES
    -- nothing in the renderer reads identifier spellings."""
    db = make_normalized_db(str(tmp_path / "n.db"))
    base = _docs(db, plan_documents(projection(db), db))
    conn = sqlite3.connect(db)
    for old, new in [("groups", "t1"), ("labels", "t2"),
                     ("entries", "t3"), ("entry_labels", "t4")]:
        conn.execute(f"ALTER TABLE {old} RENAME TO {new}")
    for t, pairs in [
            ("t1", [("gid", "c1"), ("gname", "c2")]),
            ("t2", [("lid", "c3"), ("lname", "c4")]),
            ("t3", [("eid", "c5"), ("title", "c6"), ("body", "c7"),
                    ("grp", "c8"), ("amount", "c9"), ("code", "c10")]),
            ("t4", [("entry", "c11"), ("lab", "c12")])]:
        for old, new in pairs:
            conn.execute(f"ALTER TABLE {t} RENAME COLUMN {old} TO {new}")
    conn.commit()
    conn.close()
    plan2 = plan_documents(projection(db), db)
    assert plan2 is not None and plan2.entity_table == "t3"
    assert plan2.dimensions and plan2.bridges
    renamed = _docs(db, plan2)
    assert sorted(d.entity_id for d in base) == \
        sorted(d.entity_id for d in renamed)
    vals = lambda ds: {d.entity_id: sorted(_folded_values(d))
                       for d in ds}
    assert vals(base) == vals(renamed)


def test_plan_json_roundtrip(tmp_path):
    db = make_normalized_db(str(tmp_path / "n.db"))
    plan = plan_documents(projection(db), db)
    again = DocPlan.from_dict(
        __import__("json").loads(plan.to_json()))
    assert again == plan
    docs_a = _docs(db, plan)
    docs_b = _docs(db, again)
    assert [d.doc_hash for d in docs_a] == [d.doc_hash for d in docs_b]


def test_streaming_memory_bounded(tmp_path):
    """50k-entity EAV source: iterating stays per-entity bounded."""
    db = make_eav_db(str(tmp_path / "big.db"), n=50000)
    plan = plan_documents(projection(db), db)
    tracemalloc.start()
    try:
        count = 0
        for _d in iter_record_documents(db, plan):
            count += 1
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == 50000
    assert peak < 64 * 1024 * 1024


def test_readonly_connection_cannot_write(tmp_path):
    from record_documents import _ro_connect
    db = make_flat_db(str(tmp_path / "f.db"))
    conn = _ro_connect(db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE nope(x)")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO entries DEFAULT VALUES")
    finally:
        conn.close()


def test_no_entity_table_returns_none(tmp_path):
    """A source with no entity-table candidate yields plan None -- the
    caller reports 'unavailable', nothing is invented."""
    db = str(tmp_path / "p.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE z(a, b)")
    conn.commit()
    conn.close()
    assert plan_documents(projection(db), db) is None



def _low_card_db(path):
    """SYNTHETIC library-like table: a constant `lang`, a low-cardinality
    `loc_class`, unique `title`, long `summary` (rd/2 ordering fixture)."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE books(bid INTEGER PRIMARY KEY, "
                 "lang TEXT, loc_class TEXT, title TEXT, summary TEXT)")
    for i in range(6):
        conn.execute(
            "INSERT INTO books VALUES (?, 'en', ?, ?, ?)",
            (i + 1, "PZ" if i % 2 else "Q",
             f"Book Title Number {i + 1}",
             f"A longer summary passage of book {i + 1}"))
    conn.commit()
    conn.close()
    return path


def test_rd2_ordering_constant_and_display_column(tmp_path):
    db = _low_card_db(str(tmp_path / "lc.db"))
    plan = plan_documents(projection(db), db)
    assert plan is not None
    # constant column is excluded entirely -- no discriminative info
    reasons = dict(plan.excluded)
    assert reasons["books.lang"] == "constant"
    # most identifying own column leads; low-cardinality trails it
    assert plan.own_columns[0] == "title"
    assert plan.display_column == "title"
    docs = list(iter_record_documents(db, plan))
    assert docs[0].text.startswith("title:")
    assert "lang:" not in docs[0].text


def test_display_column_none_when_low_cardinality_only(tmp_path):
    db = str(tmp_path / "body.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE posts(pid INTEGER PRIMARY KEY, "
                 "board TEXT, body TEXT)")
    for i in range(6):
        conn.execute("INSERT INTO posts VALUES (?, ?, ?)",
                     (i + 1, f"b{i % 3}",
                     f"post body {i} shared-phrase post body {i}"))
    conn.commit()
    conn.close()
    plan = plan_documents(projection(db), db)
    assert plan is not None
    assert plan.display_column is None \
        or plan.display_column in plan.own_columns
    # DocPlan JSON round-trip preserves display_column
    assert plan.from_dict(plan.to_dict()).display_column \
        == plan.display_column
