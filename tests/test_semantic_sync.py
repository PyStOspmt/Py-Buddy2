"""Tests for semantic_sync (Retrieval Fabric Phase 1).

`HashEmbedder`/`CountingEmbedder` are TEST DOUBLES: they prove which
rows get re-embedded and how failures surface, not semantic quality.
Source DBs are synthetic fixtures.
"""

from __future__ import annotations

import os
import sqlite3

from embedders import CountingEmbedder, EmbedderUnavailable
from semantic_fixtures import HashEmbedder, make_flat_db, projection
from semantic_sync import sync_source
from vector_store import VectorStore


def _db(tmp_path, n=8):
    return make_flat_db(str(tmp_path / "s.db"), n=n)


def _store(tmp_path):
    d = tmp_path / "idx"
    return VectorStore.open(str(d), model_id="test:hash", dim=32)


def _counting():
    return CountingEmbedder(HashEmbedder(dim=32, model_id="test:hash"))


def _exec(db, sql, args=()):
    conn = sqlite3.connect(db)
    try:
        conn.execute(sql, args)
        conn.commit()
    finally:
        conn.close()


def _invariant(rep):
    assert rep.source_rows == (
        rep.inserted + rep.updated + rep.unchanged
        + sum(rep.excluded.values()))


def test_initial_sync_embeds_all_docs(tmp_path):
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete and rep.source_rows == 8
    assert rep.inserted == 8 and rep.embedded_texts == 8
    assert rep.error is None and rep.generation > 0
    _invariant(rep)
    assert len(dict(store.hashes())) == 8
    assert store.get_meta("plan_json")
    assert store.get_meta("renderer_version") == "rd/2"
    assert store.get_meta("sync_report")
    store.close()


def test_resync_embeds_nothing_bytes_unchanged(tmp_path):
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    sync_source(db, projection(db), store, emb)
    vpath = os.path.join(str(tmp_path / "idx"), "vectors.bin")
    with open(vpath, "rb") as f:
        before = f.read()
    emb2 = _counting()
    rep = sync_source(db, projection(db), store, emb2)
    assert rep.complete and rep.embedded_texts == 0
    assert rep.unchanged == 8 and rep.inserted == rep.updated == 0
    with open(vpath, "rb") as f:
        assert f.read() == before
    _invariant(rep)
    store.close()


def test_point_update_embeds_one_row(tmp_path):
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    sync_source(db, projection(db), store, emb)
    _exec(db, "UPDATE entries SET title='changed title nine nine' "
              "WHERE eid=1")
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete and rep.updated == 1
    assert rep.embedded_texts == 1 and rep.unchanged == 7
    _invariant(rep)
    store.close()


def test_point_delete_no_embeds(tmp_path):
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    sync_source(db, projection(db), store, emb)
    _exec(db, "DELETE FROM entries WHERE eid=2")
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete and rep.deleted == 1
    assert rep.embedded_texts == 0 and rep.unchanged == 7
    assert store.get("2") is None
    _invariant(rep)
    store.close()


def test_point_insert_embeds_one(tmp_path):
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    sync_source(db, projection(db), store, emb)
    _exec(db, "INSERT INTO entries VALUES (9,'new entry nine',"
              "'a fresh description body for nine','CODE-0009',31.5,"
              "'2026-02-01 09:00',NULL)")
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete and rep.inserted == 1
    assert rep.embedded_texts == 1 and rep.unchanged == 8
    _invariant(rep)
    store.close()


def test_cancel_keeps_commits_no_deletes(tmp_path):
    db, store, emb = _db(tmp_path, n=10), _store(tmp_path), _counting()
    calls = [0]
    bs = 3

    def cancel():
        calls[0] += 1
        return calls[0] > 2 * bs

    rep = sync_source(db, projection(db), store, emb,
                      batch_size=bs, should_cancel=cancel)
    assert not rep.complete and rep.error is None
    assert rep.deleted == 0 and rep.inserted == 2 * bs
    live = len(dict(store.hashes()))
    assert live == 2 * bs
    emb2 = _counting()
    rep2 = sync_source(db, projection(db), store, emb2, batch_size=bs)
    assert rep2.complete
    assert rep2.inserted == 10 - 2 * bs      # only the remainder
    assert emb2.doc_texts == 10 - 2 * bs
    _invariant(rep2)
    store.close()


def test_embedder_failure_typed_and_consistent(tmp_path):
    db, store = _db(tmp_path), _store(tmp_path)
    inner = _counting()

    class Flaky:
        model_id = inner.model_id
        dim = inner.dim

        def __init__(self):
            self.n = 0

        def embed_documents(self, texts):
            self.n += 1
            if self.n >= 2:
                raise EmbedderUnavailable("simulated outage")
            return inner.embed_documents(texts)

        def embed_queries(self, texts):
            return inner.embed_queries(texts)

    rep = sync_source(db, projection(db), store, Flaky(), batch_size=4)
    assert not rep.complete and rep.error
    assert "outage" in rep.error
    assert rep.inserted == 4                 # first batch committed
    store.close()
    store = VectorStore.open(str(tmp_path / "idx"),
                             model_id="test:hash", dim=32)
    assert len(dict(store.hashes())) == 4    # consistent, reopenable
    rep2 = sync_source(db, projection(db), store, _counting())
    assert rep2.complete and rep2.inserted == 4
    store.close()


def test_plan_change_reembeds_affected_docs(tmp_path):
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    sync_source(db, projection(db), store, emb)
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE entries ADD COLUMN extra TEXT")
    for i in range(8):
        conn.execute("UPDATE entries SET extra=? WHERE eid=?",
                     (f"extra words for entry {i + 1}", i + 1))
    conn.commit()
    conn.close()
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete and rep.updated == 8
    assert rep.embedded_texts == 8
    _invariant(rep)
    store.close()


def test_noop_resync_does_not_bump_generation(tmp_path):
    """A no-op re-sync must leave generation and vector bytes
    untouched -- it is not a record change."""
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    rep1 = sync_source(db, projection(db), store, emb)
    rep2 = sync_source(db, projection(db), store, emb)
    assert rep2.complete and rep2.embedded_texts == 0
    assert rep2.generation == rep1.generation
    _invariant(rep2)
    store.close()


def test_dirty_ref_rows_counted_not_merged(tmp_path):
    """NULL/orphan EAV refs: documents unaffected, counters exact."""
    from semantic_fixtures import make_eav_db
    db = make_eav_db(str(tmp_path / "e.db"), n=8)
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO attr_rows VALUES (NULL,'x','no owner')")
    conn.execute("INSERT INTO attr_rows VALUES ('999','x','orphan')")
    conn.commit()
    conn.close()
    store = _store(tmp_path)
    rep = sync_source(db, projection(db), store, _counting())
    assert rep.complete and rep.inserted == 8
    assert rep.null_ref_rows == 1 and rep.orphan_rows == 1
    _invariant(rep)
    store.close()


def test_model_token_truncation_counted(tmp_path):
    """A doc over the embedder's token budget is reported as
    model-truncated -- the char-cap flag alone cannot see it."""
    db, store, emb = _db(tmp_path), _store(tmp_path), _counting()
    _exec(db, "UPDATE entries SET body=? WHERE eid=1",
          (" ".join(f"word{i}" for i in range(80)),))
    rep = sync_source(db, projection(db), store, emb)
    assert rep.complete
    assert rep.max_tokens == 64
    assert rep.model_truncated_docs == 1
    store.close()


def test_tokenless_embedder_reports_unknown(tmp_path):
    """Without count_tokens the report fields stay None -- unknown,
    never a fabricated 0."""
    db, store = _db(tmp_path), _store(tmp_path)
    inner = _counting()

    class NoTok:
        model_id = inner.model_id
        dim = inner.dim

        def embed_documents(self, texts):
            return inner.embed_documents(texts)

        def embed_queries(self, texts):
            return inner.embed_queries(texts)

    rep = sync_source(db, projection(db), store, NoTok())
    assert rep.complete
    assert rep.max_tokens is None
    assert rep.model_truncated_docs is None
    store.close()


def test_no_entity_table_reports_unavailable(tmp_path):
    db = str(tmp_path / "p.db")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE z(a, b)")
    conn.commit()
    conn.close()
    store = _store(tmp_path)
    rep = sync_source(db, projection(db), store, _counting())
    assert not rep.complete and rep.error == "unavailable: no entity table"
    store.close()
