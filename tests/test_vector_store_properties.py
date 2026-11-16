"""Property tests for vector_store (Retrieval Fabric Phase 1).

All vectors/texts here are SYNTHETIC and generated in-test; nothing is
evidence about semantic quality -- only about storage, atomicity, and
scan correctness.
"""

from __future__ import annotations

import os
import random
import threading
import tracemalloc

import numpy as np
import pytest
from vector_store import StoredRecord, StoreMismatch, VectorStore

MODEL = "test:store@unit"
DIM = 8


def _f16(v):
    return np.asarray(v, np.float16).astype(np.float32)


def _rec(eid, vec, text, doc_hash):
    return StoredRecord(eid, doc_hash, text,
                        np.asarray(vec, np.float32), "prov")


def _open(path):
    return VectorStore.open(str(path), model_id=MODEL, dim=DIM)


def _unit(rng, dim=DIM):
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _oracle_topk(oracle, q, k):
    scored = [(float(np.dot(np.asarray(q, np.float32), _f16(v))),
               eid) for eid, (v, _t, _h) in oracle.items()]
    scored.sort(key=lambda p: (-p[0], p[1]))
    return scored[:k]


def test_randomized_ops_match_dict_oracle(tmp_path):
    """Seeded random op streams vs a brute-force dict oracle."""
    vocab = ["alpha", "beta", "gamma", "delta"]
    ops = ["upsert", "upsert", "delete", "compact", "reopen"]
    for seed in range(150):
        rng, nrng = random.Random(seed), np.random.RandomState(seed)
        d = tmp_path / f"s{seed}"
        store = _open(d)
        oracle = {}
        pool = [f"e{i}" for i in range(16)]
        for _ in range(rng.randint(8, 25)):
            op = rng.choice(ops)
            if op == "reopen":
                store.close()
                store = _open(d)
                continue
            if op == "compact":
                store.compact()
                continue
            eid = rng.choice(pool)
            if op == "delete":
                store.apply([], [eid])
                oracle.pop(eid, None)
            else:
                vec, h = _unit(nrng), f"h{seed}:{eid}:{rng.random()}"
                text = f"{rng.choice(vocab)} {rng.choice(vocab)} {eid}"
                store.apply([_rec(eid, vec, text, h)], [])
                oracle[eid] = (vec, text, h)
        assert dict(store.hashes()) == \
            {e: h for e, (_v, _t, h) in oracle.items()}
        for _ in range(3):
            q = _unit(nrng)
            res = store.search_dense(q, 5)
            exp = _oracle_topk(oracle, q, 5)
            assert res.ids == [e for _s, e in exp]
            for got, (es, _e) in zip(res.scores, exp):
                assert got == pytest.approx(es, abs=1e-5)
        hits = store.search_lexical("alpha", 10)
        assert all(e in oracle for e, _s in hits)
        store.close()


def test_reopen_preserves_everything(tmp_path):
    d = tmp_path / "s"
    store = _open(d)
    rng = np.random.RandomState(0)
    vecs = [_unit(rng) for _ in range(6)]
    store.apply([_rec(f"e{i}", vecs[i], f"text alpha {i}", f"h{i}")
                 for i in range(6)], [])
    gen = store.generation
    store.close()
    store = _open(d)
    assert store.generation == gen
    assert dict(store.hashes()) == {f"e{i}": f"h{i}" for i in range(6)}
    r = store.get("e3")
    assert r.text == "text alpha 3"
    assert np.allclose(r.vector, _f16(vecs[3]), atol=1e-7)
    assert store.search_lexical("alpha", 3)
    store.close()


def test_failed_commit_leaves_precall_state(tmp_path):
    """Vectors appended, manifest commit fails -> reopen shows the exact
    pre-call state and truncates the orphan bytes."""
    d = tmp_path / "s"
    store = _open(d)
    rng = np.random.RandomState(1)
    store.apply([_rec(f"e{i}", _unit(rng), f"t{i}", f"h{i}")
                 for i in range(4)], [])
    base = dict(store.hashes())
    vpath = os.path.join(str(d), "vectors.bin")
    base_size = os.path.getsize(vpath)
    with pytest.raises(RuntimeError):
        store._commit = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("simulated crash"))
        store.apply([_rec("e9", _unit(rng), "t9", "h9")], [])
    assert os.path.getsize(vpath) > base_size  # orphan bytes visible
    store.close()
    store = _open(d)
    assert dict(store.hashes()) == base
    assert os.path.getsize(vpath) == base_size
    store.close()


def test_store_mismatch_on_model_and_dim(tmp_path):
    d = tmp_path / "s"
    _open(d).close()
    with pytest.raises(StoreMismatch):
        VectorStore.open(str(d), model_id="test:other", dim=DIM)
    with pytest.raises(StoreMismatch):
        VectorStore.open(str(d), model_id=MODEL, dim=DIM + 1)


def _writer(store, rng, live, deleted_at, errors, lock, stop):
    try:
        for b in range(40):
            ups = []
            for j in range(5):
                eid = f"e{b}_{j}"
                live.append(eid)
                ups.append(_rec(eid, _unit(rng), f"text {eid}",
                                f"h{b}_{j}"))
            dels = []
            if b >= 10 and rng.random() < 0.5:
                dels.append(rng.choice(live[:-5]))
            gen = store.apply(ups, dels)
            with lock:
                for v in dels:
                    deleted_at[v] = gen
                    live.remove(v)
    except Exception as exc:  # noqa: BLE001 -- reported by the test
        errors.append(exc)
    finally:
        stop.set()


def _reader(store, tid, deleted_at, seen_gen, errors, lock, stop):
    qrng = np.random.RandomState(100 + tid)
    try:
        while not stop.is_set():
            res = store.search_dense(_unit(qrng), 5)
            with lock:
                assert res.generation >= seen_gen[tid]
                seen_gen[tid] = res.generation
                for eid in res.ids:
                    assert deleted_at.get(eid, 10**9) > res.generation
    except Exception as exc:  # noqa: BLE001 -- reported by the test
        errors.append(exc)


def test_concurrent_reads_during_writes(tmp_path):
    """A writer applying 40 batches while 4 readers search."""
    store = _open(tmp_path / "s")
    rng = np.random.RandomState(2)
    errors, deleted_at, seen_gen, live = [], {}, [0] * 4, []
    lock, stop = threading.Lock(), threading.Event()
    threads = [threading.Thread(
        target=_reader,
        args=(store, i, deleted_at, seen_gen, errors, lock, stop))
        for i in range(4)]
    for t in threads:
        t.start()
    _writer(store, rng, live, deleted_at, errors, lock, stop)
    for t in threads:
        t.join(30)
    assert not errors
    store.close()


def test_compact_identical_results_less_disk(tmp_path):
    d = tmp_path / "s"
    store = _open(d)
    rng = np.random.RandomState(3)
    vecs = {f"e{i}": _unit(rng) for i in range(30)}
    store.apply([_rec(e, vecs[e], f"text {e}", f"h{e}") for e in vecs],
                [])
    store.apply([], [f"e{i}" for i in range(10)])
    q = _unit(rng)
    before_ids = store.search_dense(q, 8).ids
    before_bytes = store.stats()["disk_bytes"]
    store.compact()
    assert store.search_dense(q, 8).ids == before_ids
    store.close()                      # checkpoint + stale-file cleanup
    store = _open(d)
    assert store.search_dense(q, 8).ids == before_ids
    assert store.stats()["disk_bytes"] < before_bytes
    store.close()


def test_fts_injection_corpus_never_raises(tmp_path):
    d = tmp_path / "s"
    store = _open(d)
    rng = np.random.RandomState(4)
    store.apply([_rec("e1", _unit(rng), "running shoes red", "h1"),
                 _rec("e2", _unit(rng), "walked slowly home", "h2")],
                [])
    corpus = ['"', '*', 'NEAR(', 'AND', 'OR', '-', '()', '(((',
              '"""""', 'a' * 500, '', '   ', '', "'; DROP TABLE x;--",
              'NEAR(a b, 5)', 'running* OR "shoes']
    for q in corpus:
        hits = store.search_lexical(q, 5)  # must not raise
        for eid, _s in hits:
            rec = store.get(eid)
            assert rec is not None and rec.text
    # semantics intact afterwards
    assert store.search_lexical("running", 5)[0][0] == "e1"
    store.close()


def test_serve_time_memory_bounded(tmp_path):
    """200k x 64 synthetic vectors: search stays O(chunk)+bitmap."""
    d = tmp_path / "s"
    store = VectorStore.open(str(d), model_id=MODEL, dim=64)
    rng = np.random.RandomState(5)
    n = 200_000
    for b in range(0, n, 5000):
        vecs = rng.standard_normal((min(5000, n - b), 64))
        recs = [StoredRecord(f"e{b + i}", "h", f"doc {b + i}",
                             vecs[i].astype(np.float32), "")
                for i in range(vecs.shape[0])]
        store.apply(recs, [])
    q = rng.standard_normal(64).astype(np.float32)
    tracemalloc.start()
    try:
        for _ in range(20):
            res = store.search_dense(q, 10)
            assert len(res.ids) == 10 and res.n_scanned == n
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 64 * 1024 * 1024
    store.close()
