"""Concurrency, corruption and snapshot-cache tests for vector_store
(Retrieval Fabric, Handoff 2). Split out of
test_vector_store_properties.py for the 350-line bound; helpers are
imported from there.

All vectors/texts here are SYNTHETIC and generated in-test; nothing is
evidence about semantic quality -- only about storage, atomicity, and
scan correctness.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

import numpy as np
import pytest
import vector_store as vs
import vector_store_io as vio
from test_vector_store_properties import DIM, _open, _rec, _unit
from vector_store import StoreCorrupt


def test_store_corrupt_missing_or_short_vector_file(tmp_path):
    """A missing or truncated vector file is a typed corruption --
    never silently an empty result set."""
    d = tmp_path / "s"
    store = _open(d)
    rng = np.random.RandomState(7)
    store.apply([_rec(f"e{i}", _unit(rng), f"t{i}", f"h{i}")
                 for i in range(4)], [])
    vpath = os.path.join(str(d), "vectors.bin")
    os.remove(vpath)
    with pytest.raises(StoreCorrupt):
        store.search_dense(_unit(rng), 3)
    with open(vpath, "wb") as f:
        f.write(b"\x00" * (4 * DIM * 2 - 2))   # too short for 4 slots
    with pytest.raises(StoreCorrupt):
        store.search_dense(_unit(rng), 3)
    store.close()
    empty = _open(tmp_path / "empty")
    res = empty.search_dense(_unit(rng), 3)
    assert res.n_scanned == 0 and res.ids == []   # not corrupt: no slots
    empty.close()


def test_search_dense_survives_concurrent_compact(tmp_path):
    """REPRO2: a search whose snapshot predates a compact() must still
    return correct ids -- compaction waits for the reader."""
    store = _open(tmp_path / "s")
    rng = np.random.RandomState(8)
    recs = [_rec(f"e{i:03d}", _unit(rng), f"t{i}", f"h{i}")
            for i in range(40)]
    store.apply(recs, [])
    store.apply([], [f"e{i:03d}" for i in range(20)])
    done = threading.Event()

    def hook():
        def _c():
            try:
                store.compact()
            finally:
                done.set()
        threading.Thread(target=_c, daemon=True).start()
        time.sleep(0.5)
        assert not done.is_set()   # compaction must wait for us

    vs._TEST_HOOK_AFTER_SNAPSHOT = hook
    try:
        res = store.search_dense(recs[30].vector, 5)
    finally:
        vs._TEST_HOOK_AFTER_SNAPSHOT = None
    assert done.wait(15)
    assert res.ids[0] == "e030" and res.n_scanned == 20
    store.close()


def _race_writer(store, vecs, errors, stop):
    try:
        for b in range(30):
            for e in ("e000", "e050", "e100", "e150", "e199"):
                store.apply([_rec(e, vecs[e], f"text {e}",
                                  f"h{b}:{e}")], [])
            store.compact()
    except Exception as exc:  # noqa: BLE001 -- reported by test
        errors.append(exc)
    finally:
        stop.set()


def _race_reader(tid, store, vecs, n, errors, stop):
    qrng = np.random.RandomState(50 + tid)
    try:
        while not stop.is_set():
            e = f"e{qrng.randint(n):03d}"
            res = store.search_dense(vecs[e], 1)
            assert res.ids == [e], (e, res.ids)
            assert res.n_scanned == n, res.n_scanned
    except Exception as exc:  # noqa: BLE001 -- reported by test
        errors.append(exc)


def test_compact_readers_race_randomized(tmp_path):
    """Writer loops {re-upsert 5 ids same vectors; compact} x30 while
    4 readers exact-vector query: top-1==that id, n_scanned==200."""
    store = _open(tmp_path / "s")
    rng = np.random.RandomState(9)
    n = 200
    vecs = {f"e{i:03d}": _unit(rng) for i in range(n)}
    store.apply([_rec(e, vecs[e], f"text {e}", f"h0:{e}")
                 for e in vecs], [])
    errors, stop = [], threading.Event()
    ts = [threading.Thread(target=_race_reader,
                           args=(i, store, vecs, n, errors, stop),
                           daemon=True) for i in range(4)]
    for t in ts:
        t.start()
    _race_writer(store, vecs, errors, stop)
    for t in ts:
        t.join(120)
    assert not errors, errors
    store.close()


def test_snapshot_bitmap_cached_per_generation(tmp_path):
    """Second search at the same generation reuses the live bitmap;
    a write or compaction rebuilds it."""
    store = _open(tmp_path / "s")
    rng = np.random.RandomState(10)
    store.apply([_rec(f"e{i}", _unit(rng), f"t{i}", f"h{i}")
                 for i in range(20)], [])
    store.search_dense(_unit(rng), 3)
    builds = store._snap_rebuilds
    store.search_dense(_unit(rng), 3)
    store.search_dense(_unit(rng), 3)
    assert store._snap_rebuilds == builds
    store.apply([_rec("eX", _unit(rng), "tX", "hX")], [])
    res = store.search_dense(_unit(rng), 3)
    assert store._snap_rebuilds == builds + 1
    assert res.n_scanned == 21
    store.compact()
    store.search_dense(_unit(rng), 3)
    assert store._snap_rebuilds == builds + 2
    store.close()


def test_get_many_bulk_hydration(tmp_path):
    """get_many: one pass, live rows only, no vector reads."""
    store = _open(tmp_path / "s")
    rng = np.random.RandomState(11)
    store.apply([_rec(f"e{i}", _unit(rng), f"text alpha {i}", f"h{i}")
                 for i in range(10)], [])
    got = store.get_many(["e3", "e7", "ghost"])
    assert set(got) == {"e3", "e7"}
    assert got["e3"].text == "text alpha 3"
    assert got["e3"].doc_hash == "h3"
    assert got["e7"].provenance == "prov"
    store.apply([], ["e3"])
    assert set(store.get_many(["e3", "e7"])) == {"e7"}
    store.close()


def test_generation_only_on_effective_change(tmp_path):
    """generation bumps iff the live set changed; set_meta never
    bumps; compaction is not a record change."""
    store = _open(tmp_path / "s")
    rng = np.random.RandomState(12)
    g0 = store.generation
    g1 = store.apply([_rec("e1", _unit(rng), "t1", "h1")], [])
    assert g1 == g0 + 1 and store.generation == g1
    assert store.apply([], []) == g1            # nothing changed
    assert store.apply([], ["ghost"]) == g1     # unknown id ignored
    store.set_meta({"note": "hi"})
    assert store.generation == g1
    assert store.get_meta("note") == "hi"
    g2 = store.apply([], ["e1"])
    assert g2 == g1 + 1
    store.compact()
    assert store.generation == g2               # renumber != change
    store.close()


def test_sqlite_reader_conn_is_readonly(tmp_path):
    """Reader-side manifest connection never writes (WAL reader)."""
    d = tmp_path / "s"
    store = _open(d)
    store.close()
    conn = vio.connect(os.path.join(str(d), "manifest.sqlite"))
    try:
        conn.execute("PRAGMA query_only=ON")
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE nope(x)")
    finally:
        conn.close()
