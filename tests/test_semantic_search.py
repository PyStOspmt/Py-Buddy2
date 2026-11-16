"""Tests for semantic_search (Retrieval Fabric Phase 1).

Embeddings are scripted synthetic vectors (HashEmbedder test double);
they exercise fusion math and channel plumbing, not semantic quality.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from semantic_fixtures import HashEmbedder
from semantic_search import search
from vector_store import StoredRecord, VectorStore

DIM = 8


def _store_with(tmp_path, docs, vecs):
    store = VectorStore.open(str(tmp_path / "idx"),
                             model_id="test:hash", dim=DIM)
    recs = [StoredRecord(eid, f"h{eid}", text,
                         np.asarray(v, np.float32), "prov")
            for (eid, text), v in zip(docs, vecs)]
    store.apply(recs, [])
    return store


def _basis(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[i % (DIM - 1) + 1] = 0.8, 0.6
    return v


def test_rrf_matches_hand_computed(tmp_path):
    docs = [("e1", "alpha one"), ("e2", "alpha beta two"),
            ("e3", "alpha alpha alpha three")]
    vecs = [np.array([1, 0, 0, 0, 0, 0, 0, 0], np.float32),
            np.array([0.9, 0.4359, 0, 0, 0, 0, 0, 0], np.float32),
            np.array([0.7, 0.7141, 0, 0, 0, 0, 0, 0], np.float32)]
    store = _store_with(tmp_path, docs, vecs)
    q = np.array([1, 0, 0, 0, 0, 0, 0, 0], np.float32)

    dense = store.search_dense(q, 50)
    lex = store.search_lexical("alpha", 50)
    exp = {}
    for r, eid in enumerate(dense.ids, 1):
        exp[eid] = exp.get(eid, 0.0) + 1.0 / (60 + r)
    for r, (eid, _s) in enumerate(lex, 1):
        exp[eid] = exp.get(eid, 0.0) + 1.0 / (60 + r)
    want = sorted(exp.items(), key=lambda kv: (-kv[1], kv[0]))

    # scripted query vector -> deterministic dense ranks
    class QVec:
        model_id = "test:hash"
        dim = DIM

        def embed_queries(self, texts):
            return np.vstack([q] * len(texts))

        def embed_documents(self, texts):
            return np.vstack([q] * len(texts))

    res = search(store, QVec(), "alpha", k=3, mode="hybrid")
    assert [h.entity_id for h in res.hits] == [e for e, _s in want]
    for h, (_e, s) in zip(res.hits, want):
        assert h.score == pytest.approx(s, abs=1e-9)
    store.close()


def test_modes_dense_lexical_hybrid(tmp_path):
    docs = [("e1", "alpha one"), ("e2", "beta two")]
    vecs = [_basis(0), _basis(1)]
    store = _store_with(tmp_path, docs, vecs)
    emb = HashEmbedder(dim=DIM)
    rd = search(store, emb, "alpha", k=2, mode="dense")
    assert all(h.dense_rank is not None and h.lexical_rank is None
               for h in rd.hits)
    rl = search(store, emb, "alpha", k=2, mode="lexical")
    assert all(h.lexical_rank is not None and h.dense_rank is None
               for h in rl.hits)
    rh = search(store, emb, "alpha", k=2, mode="hybrid")
    assert rh.hits
    store.close()


def test_top1z_and_margin_match_numpy(tmp_path):
    rng = np.random.RandomState(0)
    docs = [(f"e{i}", f"text {i}") for i in range(10)]
    vecs = [rng.standard_normal(DIM) for _ in range(10)]
    vecs = [v / np.linalg.norm(v) for v in vecs]
    store = _store_with(tmp_path, docs, vecs)
    q = vecs[0] + 0.01 * rng.standard_normal(DIM)
    q = q / np.linalg.norm(q)

    class QVec:
        model_id = "test:hash"
        dim = DIM

        def embed_queries(self, texts):
            return np.vstack([q] * len(texts))

        def embed_documents(self, texts):
            return np.vstack([q] * len(texts))

    res = search(store, QVec(), "text", k=3, mode="hybrid")
    scores = np.array([float(np.dot(q.astype(np.float32),
                              np.asarray(v, np.float16)
                              .astype(np.float32)))
                       for v in vecs])
    exp_z = (scores.max() - scores.mean()) / scores.std()
    srt = np.sort(scores)[::-1]
    exp_margin = srt[0] - srt[min(3, len(srt)) - 1]
    assert res.top1_z == pytest.approx(exp_z, abs=1e-4)
    assert res.margin == pytest.approx(exp_margin, abs=1e-4)
    store.close()


def test_empty_index_and_empty_query_notes(tmp_path):
    store = VectorStore.open(str(tmp_path / "idx"),
                             model_id="test:hash", dim=DIM)
    emb = HashEmbedder(dim=DIM)
    res = search(store, emb, "alpha")
    assert res.hits == [] and res.notes
    res2 = search(store, emb, "   ")
    assert res2.hits == [] and res2.notes
    store.close()


def test_k_larger_than_index(tmp_path):
    docs = [("e1", "alpha one"), ("e2", "beta two")]
    store = _store_with(tmp_path, docs, [_basis(0), _basis(1)])
    res = search(store, HashEmbedder(dim=DIM), "alpha", k=50)
    assert len(res.hits) <= 2
    store.close()


def test_prefix_truncate_finds_inflection(tmp_path):
    docs = [("e1", "trail running shoes"),
            ("e2", "completely unrelated text here")]
    store = _store_with(tmp_path, docs, [_basis(0), _basis(1)])
    on = store.search_lexical("runner", 5, prefix_truncate=True)
    off = store.search_lexical("runner", 5, prefix_truncate=False)
    assert [e for e, _s in on] == ["e1"]
    assert off == []
    store.close()


def test_store_corrupt_propagates(tmp_path):
    """A missing vector file raises typed corruption through search --
    it is not an empty result set."""
    from vector_store import StoreCorrupt
    docs = [("e1", "alpha one"), ("e2", "beta two")]
    store = _store_with(tmp_path, docs, [_basis(0), _basis(1)])
    os.remove(os.path.join(str(tmp_path / "idx"), "vectors.bin"))
    with pytest.raises(StoreCorrupt):
        search(store, HashEmbedder(dim=DIM), "alpha", mode="dense")
    with pytest.raises(StoreCorrupt):
        search(store, HashEmbedder(dim=DIM), "alpha", mode="hybrid")
    store.close()


def test_hits_hydrated_in_one_query(tmp_path):
    """Bulk hydration: exactly one get_many call per search, hits
    identical to per-id lookups."""
    docs = [(f"e{i}", f"alpha text {i}") for i in range(6)]
    store = _store_with(tmp_path, docs,
                        [_basis(i) for i in range(6)])
    calls = [0]
    orig = store.get_many

    def spy(ids):
        calls[0] += 1
        return orig(ids)

    store.get_many = spy
    res = search(store, HashEmbedder(dim=DIM), "alpha", k=3)
    assert calls[0] == 1
    assert len(res.hits) == 3
    assert res.hits[0].snippet.startswith("alpha")
    store.close()


def test_deleted_ids_never_surface(tmp_path):
    docs = [("e1", "alpha one"), ("e2", "alpha two")]
    store = _store_with(tmp_path, docs, [_basis(0), _basis(1)])
    store.apply([], ["e2"])
    res = search(store, HashEmbedder(dim=DIM), "alpha", k=10)
    assert [h.entity_id for h in res.hits] == ["e1"]
    assert res.indexed == 1
    store.close()


def test_weighted_rrf_knob(tmp_path):
    """A4: dense_weight/lexical_weight scale each channel's RRF share.
    1.0/1.0 must equal the production default; 1.0/0.0 == dense order;
    0.0/1.0 == lexical order."""
    docs = [("e1", "alpha one"), ("e2", "alpha beta two"),
            ("e3", "alpha alpha alpha three")]
    vecs = [_basis(0), _basis(1), _basis(2)]
    store = _store_with(tmp_path, docs, vecs)
    emb = HashEmbedder(dim=DIM)
    base = search(store, emb, "alpha", k=3, mode="hybrid")
    w11 = search(store, emb, "alpha", k=3, mode="hybrid",
                 dense_weight=1.0, lexical_weight=1.0)
    assert [h.entity_id for h in base.hits] \
        == [h.entity_id for h in w11.hits]
    for h1, h2 in zip(base.hits, w11.hits):
        assert h1.score == pytest.approx(h2.score, abs=1e-9)

    wd = search(store, emb, "alpha", k=3, mode="hybrid",
                dense_weight=1.0, lexical_weight=0.0)
    rd = search(store, emb, "alpha", k=3, mode="dense")
    assert [h.entity_id for h in wd.hits] \
        == [h.entity_id for h in rd.hits]

    wl = search(store, emb, "alpha", k=3, mode="hybrid",
                dense_weight=0.0, lexical_weight=1.0)
    lex = store.search_lexical("alpha", 50)
    assert [h.entity_id for h in wl.hits] == [e for e, _s in lex][:3]

    # 1.0/0.3: hand-computed weighted expectation
    dense = store.search_dense(
        emb.embed_queries(["alpha"])[0].astype(np.float32), 50)
    exp = {}
    for r, eid in enumerate(dense.ids, 1):
        exp[eid] = exp.get(eid, 0.0) + 1.0 / (60 + r)
    for r, (eid, _s) in enumerate(lex, 1):
        exp[eid] = exp.get(eid, 0.0) + 0.3 / (60 + r)
    want = sorted(exp.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    w = search(store, emb, "alpha", k=3, mode="hybrid",
               dense_weight=1.0, lexical_weight=0.3)
    assert [h.entity_id for h in w.hits] == [e for e, _s in want]
    for h, (_e, s) in zip(w.hits, want):
        assert h.score == pytest.approx(s, abs=1e-9)
    store.close()


def test_repeated_search_no_per_query_count(tmp_path):
    """A1: search() must not pay the O(N) stats() count per query --
    the live count rides the cached snapshot bitmap."""
    docs = [("e1", "alpha one"), ("e2", "beta two")]
    vecs = [_basis(0), _basis(1)]
    store = _store_with(tmp_path, docs, vecs)
    emb = HashEmbedder(dim=DIM)
    calls = {"n": 0}
    orig_stats = store.stats

    def counted():
        calls["n"] += 1
        return orig_stats()

    store.stats = counted
    search(store, emb, "alpha", k=2, mode="hybrid")
    search(store, emb, "alpha", k=2, mode="hybrid")
    assert calls["n"] == 0                      # stats never called
    assert store._snap_rebuilds == 1            # bitmap reused
    store.close()
