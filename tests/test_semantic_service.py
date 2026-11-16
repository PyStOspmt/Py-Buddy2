"""SemanticIndex lifecycle service tests (Handoff 3).

All fixtures are SYNTHETIC dbs from `semantic_fixtures`; embedding is
the offline HashEmbedder -- no network, no model weights, no paid
calls. The gated live test at the bottom uses the real local
e5-small only when PYBUDDY_LIVE_LOCAL_MODEL=1.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time

import pytest
from embedders import EmbedderUnavailable
from semantic_fixtures import HashEmbedder, make_flat_db
from semantic_service import MISMATCH_REASON, SemanticIndex
from source_lifecycle import source_stamp
from vector_store import StoreMismatch  # noqa: F401  (state evidence)


def _factory(**kw):
    return lambda: HashEmbedder(**kw)


def _mk(tmp_path, db=None, index_dir=None, **kw):
    db = db or make_flat_db(str(tmp_path / "src.db"))
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory(**kw),
                        index_dir=index_dir)
    return db, idx


def _wait(idx, states, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = idx.status()["state"]
        if st in states:
            return st
        time.sleep(0.05)
    raise AssertionError(f"state stayed {idx.status()['state']!r}")


def _append_row(db):
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO entries(title, body, refcode) "
                 "VALUES ('new entry', 'fresh body', 'CODE-9999')")
    conn.commit()
    conn.close()


def _library_db(path):
    """SYNTHETIC library-like source: a constant `lang` column, a
    low-cardinality class column, a unique title -- the rd/2 ordering
    and display_column case."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE books(bid INTEGER PRIMARY KEY, "
                 "lang TEXT, loc_class TEXT, title TEXT, summary TEXT)")
    for i in range(6):
        conn.execute(
            "INSERT INTO books VALUES (?, 'en', ?, ?, ?)",
            (i + 1, "PZ" if i % 2 else "Q",
             f"Book Title Number {i + 1}",
             f"A longer summary of book {i + 1} for searching"))
    conn.commit()
    conn.close()
    return path


# -- state machine ----------------------------------------------------
def test_absent_then_ready_and_fresh(tmp_path):
    _db, idx = _mk(tmp_path)
    assert idx.status()["state"] == "absent"
    rep = idx.sync_blocking(reason="test")
    assert rep.complete, rep.error
    st = idx.status()
    assert st["state"] == "ready" and st["fresh"] is True
    assert st["coverage"]["indexed"] == 8
    assert st["entity"]["table"] == "entries"
    assert st["source_stamp_indexed"] == st["source_stamp_now"]
    idx.close()


def test_status_exact_keys(tmp_path):
    _db, idx = _mk(tmp_path)
    idx.sync_blocking(reason="test")
    st = idx.status()
    assert set(st) == {"state", "reason", "progress", "index",
                       "coverage", "fresh", "source_stamp_indexed",
                       "source_stamp_now", "last_sync", "entity"}
    assert set(st["progress"]) == {"processed", "total", "embedded"}
    assert set(st["index"]) == {"generation", "live", "model_id",
                                "dim", "disk_bytes"}
    assert set(st["entity"]) == {"table", "ambiguous", "alternatives"}
    assert set(st["coverage"]) == {
        "source_rows", "indexed", "excluded", "truncated_docs",
        "model_truncated_docs", "max_tokens", "null_ref_rows",
        "orphan_rows"}
    idx.close()


def test_source_stamp_detects_wal_only_change(tmp_path):
    db = make_flat_db(str(tmp_path / "w.db"))
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    before = source_stamp(db)
    conn.execute("INSERT INTO entries(title, body, refcode) "
                 "VALUES ('wal row', 'wal body', 'CODE-1')")
    conn.commit()           # WAL grows; db file itself untouched
    after = source_stamp(db)
    conn.close()
    assert after != before


def test_stale_then_ensure_fresh_syncs(tmp_path):
    db, idx = _mk(tmp_path)
    idx.sync_blocking(reason="test")
    _append_row(db)
    assert idx.status()["state"] == "stale"
    assert idx.ensure_fresh() is False          # refresh started
    assert _wait(idx, {"ready"}) == "ready"
    assert idx.status()["coverage"]["indexed"] == 9
    idx.close()


def test_search_during_building_not_ready(tmp_path):
    gate = threading.Event()

    class Gated(HashEmbedder):
        def embed_documents(self, texts):
            gate.wait(15)
            return super().embed_documents(texts)

    db = make_flat_db(str(tmp_path / "g.db"))
    idx = SemanticIndex("dom_t", db,
                        embedder_factory=lambda: Gated())
    res = idx.start_sync(reason="test")
    assert res["started"] and _wait(idx, {"building"}) == "building"
    out = idx.search("entry")
    assert out["status"] == "not_ready" and out["hits"] == []
    gate.set()
    _wait(idx, {"ready"})
    idx.close()


def test_single_flight_start_sync(tmp_path):
    gate = threading.Event()

    class Gated(HashEmbedder):
        def embed_documents(self, texts):
            gate.wait(15)
            return super().embed_documents(texts)

    db = make_flat_db(str(tmp_path / "sf.db"))
    idx = SemanticIndex("dom_t", db, embedder_factory=lambda: Gated())
    r1 = idx.start_sync(reason="a")
    _wait(idx, {"building"})
    r2 = idx.start_sync(reason="b")
    gate.set()
    _wait(idx, {"ready"})
    assert r1["started"] and not r2["started"]
    idx.close()


def test_lazy_search_triggers_sync(tmp_path):
    _db, idx = _mk(tmp_path)
    out = idx.search("entry")
    assert out["status"] == "not_ready"
    _wait(idx, {"ready"})
    idx.close()


def test_embedder_failure_is_failed_not_hits(tmp_path):
    class Broken(HashEmbedder):
        def embed_documents(self, texts):
            raise EmbedderUnavailable("offline test failure")

    db = make_flat_db(str(tmp_path / "b.db"))
    idx = SemanticIndex("dom_t", db,
                        embedder_factory=lambda: Broken())
    rep = idx.sync_blocking(reason="test")
    assert not rep.complete and rep.error
    assert idx.status()["state"] == "failed"
    out = idx.search("entry")
    assert out["status"] == "failed" and out["hits"] == []
    idx.close()


def test_mismatch_then_rebuild_recovers(tmp_path):
    db = make_flat_db(str(tmp_path / "m.db"))
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory(dim=64))
    idx.sync_blocking(reason="test")
    idx.close()
    sentinel = tmp_path / "keep.me"
    sentinel.write_text("x")
    # same .arf dir, different embedding space -> typed mismatch
    idx2 = SemanticIndex("dom_t", db,
                         embedder_factory=_factory(dim=32))
    st = idx2.status()
    assert st["state"] == "failed" and st["reason"] == MISMATCH_REASON
    res = idx2.start_sync(reason="rebuild", rebuild=True)
    assert res["started"]
    _wait(idx2, {"ready"})
    assert idx2.status()["index"]["dim"] == 32
    assert sentinel.exists()           # only the .arf dir was deleted
    idx2.close()


def test_unavailable_no_entity_table(tmp_path):
    db = str(tmp_path / "ev.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE events(ts TEXT, line TEXT)")
    conn.execute("INSERT INTO events VALUES ('t', 'x')")
    conn.commit()
    conn.close()
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory())
    st = idx.status()
    assert st["state"] == "unavailable" and "entity" in st["reason"]
    idx.close()


def test_unavailable_no_db_path(tmp_path):
    idx = SemanticIndex("dom_t", None, embedder_factory=_factory())
    assert idx.status()["state"] == "unavailable"


def test_cancel_keeps_batches_resume_completes(tmp_path):
    seen = {"n": 0}

    def cancel():
        return seen["n"] >= 1          # after the first committed batch

    def progress(_ev):
        seen["n"] += 1

    db = make_flat_db(str(tmp_path / "c.db"), n=200)  # > batch_size
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory())
    rep = idx.sync_blocking(reason="t", should_cancel=cancel,
                            progress=progress)
    assert not rep.complete
    st = idx.status()
    assert st["state"] in ("absent", "stale")
    assert idx._ensure_store().live_count() >= 64  # batches kept
    rep2 = idx.sync_blocking(reason="t")
    assert rep2.complete
    assert idx.status()["state"] == "ready"
    assert idx.status()["coverage"]["indexed"] == 200
    idx.close()


def test_mid_sync_source_edit_stays_stale(tmp_path):
    """A2: the stored stamp is the PRE-pass stamp; an edit landing
    mid-sync leaves the index honestly stale."""
    edited = {"done": False}

    def progress(_ev):
        if not edited["done"]:
            edited["done"] = True
            _append_row(db)

    db = make_flat_db(str(tmp_path / "m2.db"), n=200)  # progress fires
    # WAL: a writer coexists with the streaming read cursor
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory())
    rep = idx.sync_blocking(reason="t", progress=progress)
    assert rep.complete
    st = idx.status()
    assert st["state"] == "stale" and st["fresh"] is False
    idx.close()


def test_search_stale_serves_old_generation_with_note(tmp_path):
    db, idx = _mk(tmp_path)
    idx.sync_blocking(reason="t")
    _append_row(db)
    out = idx.search("entry number 3")
    assert out["status"] == "ok"
    assert out["index"]["fresh"] is False
    assert any("stale" in n for n in out["notes"])
    _wait(idx, {"ready"})               # refresh was started
    idx.close()


def test_search_keys_and_name_from_display_column(tmp_path):
    db = _library_db(str(tmp_path / "lib.db"))
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory())
    idx.sync_blocking(reason="t")
    out = idx.search("book title", k=3)
    assert set(out) == {"status", "query", "mode", "k", "hits",
                        "confidence", "index", "notes"}
    assert out["status"] == "ok" and out["hits"]
    h = out["hits"][0]
    assert set(h) == {"entity_id", "name", "score", "dense_rank",
                      "dense_sim", "lexical_rank", "lexical_score",
                      "snippet", "provenance"}
    assert h["name"].startswith("Book Title Number")  # not "en"/"PZ"
    assert h["provenance"]
    idx.close()


def test_ambiguous_entity_note(tmp_path):
    db = str(tmp_path / "amb.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE memos(mid INTEGER PRIMARY KEY, "
                 "subject TEXT, content TEXT)")
    conn.execute("CREATE TABLE faq(fid INTEGER PRIMARY KEY, "
                 "question TEXT, answer TEXT)")
    for i in range(50):
        conn.execute("INSERT INTO memos VALUES (?, ?, ?)",
                     (i, f"memo subject {i}", f"memo body {i}"))
        conn.execute("INSERT INTO faq VALUES (?, ?, ?)",
                     (i, f"question {i}", f"answer {i}"))
    conn.commit()
    conn.close()
    idx = SemanticIndex("dom_t", db, embedder_factory=_factory())
    idx.sync_blocking(reason="t")
    st = idx.status()
    # facet-count tie -> ambiguous; alternatives may legitimately be
    # empty (they hold only free-text folded lookups)
    assert st["entity"]["ambiguous"]
    out = idx.search("memo")
    assert any("ambiguous" in n for n in out["notes"])
    idx.close()


# -- gated live test (real e5-small, offline cache) --------------------
@pytest.mark.live_model
@pytest.mark.skipif((os.environ.get("PYBUDDY_LIVE_LOCAL_MODEL") or os.environ.get("AINSA_LIVE_LOCAL_MODEL")) != "1",
                    reason="set PYBUDDY_LIVE_LOCAL_MODEL=1 to run")
def test_live_e5_crosslingual(tmp_path):
    from embedders import default_embedder

    db = str(tmp_path / "live.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE docs(did INTEGER PRIMARY KEY, "
                 "title TEXT, body TEXT)")
    conn.execute("INSERT INTO docs VALUES (1, 'Coffee brewing guide', "
                 "'How to brew filter coffee with fresh beans')")
    conn.execute("INSERT INTO docs VALUES (2, 'Tax filing notes', "
                 "'Annual income tax return checklist')")
    conn.commit()
    conn.close()
    idx = SemanticIndex("dom_t", db,
                        embedder_factory=default_embedder)
    idx.sync_blocking(reason="live")
    out = idx.search("як заварити каву", k=2)
    idx.close()
    assert out["status"] == "ok"
    assert out["hits"][0]["name"] == "Coffee brewing guide"
