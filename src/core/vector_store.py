"""
Retrieval Fabric: on-disk vector + lexical store.

One directory per index: `manifest.sqlite` (WAL; meta, records, FTS5)
plus a float16 append-only vector file. Dense scan is exact brute
force in float32 over fixed-size chunks (`np.fromfile` reads, no mmap
-- Windows truncation hazards): serve-time memory stays O(chunk) + a
live-slot bitmap, never O(vectors) or O(text). The bitmap is cached
on (vector_file, generation), rebuilt only when the record set or
slot layout changes.

Single writer process, many reader threads. `apply` is one atomic
commit: vector bytes are appended + fsynced FIRST, then a single
manifest transaction flips records/FTS/meta/generation; orphan bytes
of a failed commit are truncated at open and before the next append.
`generation` increments IFF the live set changed. `compact()`
rewrites live vectors into a seq-named file swapped via the manifest;
it renumbers slots, so readers (`search_dense`/`get`/`get_many`) hold
the shared side of an in-process RWLock for their whole duration and
`compact()` holds the exclusive side (vector_store_io.rwlock). A
missing/short vector file is `StoreCorrupt`, never a silent empty
result.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import vector_store_io as vio
from rwlock import RWLock

logger = logging.getLogger(__name__)

_VEC_PREFIX = "vectors"
_MANIFEST = "manifest.sqlite"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS records(
    slot INTEGER PRIMARY KEY, entity_id TEXT, doc_hash TEXT,
    text TEXT, provenance TEXT, live INTEGER NOT NULL DEFAULT 1);
CREATE UNIQUE INDEX IF NOT EXISTS records_live_entity
    ON records(entity_id) WHERE live=1;
CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
    entity_id UNINDEXED, text,
    tokenize='unicode61 remove_diacritics 2');
"""

#: TEST HOOK (None in production): invoked inside `search_dense` right
#: after the consistent snapshot, while the reader lock is held --
#: lets tests drive a concurrent `compact()` into the scan window.
_TEST_HOOK_AFTER_SNAPSHOT: Callable[[], None] | None = None


class StoreMismatch(RuntimeError):
    """The store on disk belongs to a different model/vector space."""


class StoreCorrupt(RuntimeError):
    """The vector file is missing or shorter than the committed slot
    range -- a typed failure, never a silent empty result."""


@dataclass(frozen=True)
class StoredRecord:
    entity_id: str
    doc_hash: str
    text: str
    vector: np.ndarray
    provenance: str


@dataclass(frozen=True)
class StoredText:
    """Bulk-hydration record: no vector read."""
    entity_id: str
    doc_hash: str
    text: str
    provenance: str


@dataclass(frozen=True)
class DenseResult:
    ids: list
    scores: list
    n_scanned: int
    score_mean: float | None
    score_std: float | None
    generation: int


class VectorStore:
    def __init__(self):
        self._dir = ""
        self._conn: sqlite3.Connection | None = None
        self._wlock = threading.Lock()
        self._rwlock = RWLock()
        self._snap_lock = threading.Lock()
        self._snap_cache: tuple | None = None
        self._snap_rebuilds = 0     # diagnostic counter for tests
        self._vector_file = f"{_VEC_PREFIX}.bin"
        self.model_id = ""
        self.dim = 0

    @classmethod
    def open(cls, directory: str, *, model_id: str,
             dim: int) -> VectorStore:
        os.makedirs(directory, exist_ok=True)
        st = cls()
        st._dir, st.model_id, st.dim = directory, model_id, int(dim)
        conn, st._vector_file = vio.open_manifest(
            st._mpath(), model_id, st.dim, _SCHEMA)
        st._conn = conn
        vio.truncate_orphans(st._vpath(st._vector_file),
                             vio.slot_count(conn), st.dim)
        vio.drop_stale_vectors(st._dir, _VEC_PREFIX, st._vector_file)
        return st

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error as exc:
                logger.debug("wal checkpoint skipped: %s", exc)
            self._conn.close()
            self._conn = None

    @property
    def generation(self) -> int:
        return int(self._meta("generation") or 0)

    def _meta(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _vpath(self, name: str) -> str:
        return os.path.join(self._dir, name)

    def _mpath(self) -> str:
        return os.path.join(self._dir, _MANIFEST)

    def apply(self, upserts: Sequence[StoredRecord],
              deletes: Sequence[str], *,
              meta_updates: dict | None = None) -> int:
        """One atomic commit; returns the generation after it (bumped
        iff the live set changed). Duplicate ids: last wins; deletes
        apply after upserts; unknown ids ignored."""
        last = {}
        for i, r in enumerate(upserts):
            last[r.entity_id] = (i, r)
        recs = [r for _i, r in sorted(last.values(), key=lambda p: p[0])]
        dels = list(dict.fromkeys(deletes))
        with self._wlock:
            n = vio.slot_count(self._conn)
            vio.append_vectors(self._vpath(self._vector_file),
                               self.dim, recs, n)
            ops = [("upsert", n + i, r) for i, r in enumerate(recs)]
            ops += [("delete", None, d) for d in dels]
            return self._commit(ops, dict(meta_updates or {}))

    def _commit(self, ops, meta_updates: dict) -> int:
        """Crash-test seam: the transaction body lives in
        `vector_store_io.commit_rows`."""
        return vio.commit_rows(self._conn, ops, meta_updates)

    def set_meta(self, updates: dict) -> None:
        """Meta-only transaction -- report/stamp writes are not record
        changes, so generation never bumps here."""
        with self._wlock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                for k, v in updates.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO meta VALUES (?,?)",
                        (k, str(v)))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def _snapshot(self) -> tuple:
        """(vector_file, generation, live bitmap, n_slots). The bitmap
        is rebuilt only when (vector_file, generation) changed; the
        rebuild runs in one read transaction that re-reads meta, so it
        is correct across writer processes too."""
        with self._snap_lock:
            key = vio.snapshot_key(self._mpath())
            snap = self._snap_cache
            if snap is not None and (snap[0], snap[1]) == key:
                return snap
            self._snap_cache = vio.snapshot(self._mpath())
            self._snap_rebuilds += 1
            return self._snap_cache

    def search_dense(self, qvec: np.ndarray, k: int) -> DenseResult:
        """Exact top-k over a consistent snapshot; deleted slots are
        never returned. Order: score desc, entity_id asc. Holds the
        reader side of the RWLock from snapshot through slot->id
        resolution, so a compact() cannot renumber mid-scan."""
        q = np.asarray(qvec, dtype=np.float32)
        if q.shape != (self.dim,):
            raise ValueError(f"query vector shape {q.shape}")
        with self._rwlock.read():
            vfile, gen, live, n = self._snapshot()
            if _TEST_HOOK_AFTER_SNAPSHOT is not None:
                _TEST_HOOK_AFTER_SNAPSHOT()
            top, cnt, mean, std = vio.dense_scan(
                self._vpath(vfile), self.dim, n, live, q, k)
            ids = vio.slot_entity_ids(
                self._mpath(), [s for _sc, s in top])
        ranked = sorted(((sc, ids[s]) for sc, s in top),
                        key=lambda p: (-p[0], p[1]))
        return DenseResult([e for _s, e in ranked],
                           [s for s, _e in ranked],
                           cnt, mean, std, gen)

    def search_lexical(self, query: str, k: int, *,
                       prefix_truncate: bool = True) -> list:
        """BM25 over live FTS rows; score = -bm25 (higher better)."""
        match = vio.lexical_match(query, prefix_truncate)
        if match is None:
            return []
        return vio.fts_top(self._mpath(), match, k)

    def hashes(self) -> Iterator[tuple]:
        conn = vio.connect(self._mpath())
        try:
            yield from conn.execute(
                "SELECT entity_id, doc_hash FROM records "
                "WHERE live=1 ORDER BY entity_id")
        finally:
            conn.close()

    def get(self, entity_id: str) -> StoredRecord | None:
        with self._rwlock.read():
            conn = vio.connect(self._mpath())
            try:
                row = conn.execute(
                    "SELECT slot, doc_hash, text, provenance FROM "
                    "records WHERE entity_id=? AND live=1",
                    (entity_id,)).fetchone()
                vfile = dict(conn.execute(
                    "SELECT key,value FROM meta")).get(
                        "vector_file", self._vector_file)
            finally:
                conn.close()
            if row is None:
                return None
            vec = np.fromfile(self._vpath(vfile), dtype=np.float16,
                              count=self.dim,
                              offset=row[0] * self.dim * 2)
        return StoredRecord(entity_id, row[1], row[2],
                            vec.astype(np.float32), row[3] or "")

    def get_many(self, ids: Sequence[str]) -> dict:
        """Bulk hydration: live rows only, ONE connection, chunked IN
        queries, no vector reads."""
        uniq = list(dict.fromkeys(ids))
        if not uniq:
            return {}
        out = {}
        with self._rwlock.read():
            conn = vio.connect(self._mpath())
            try:
                for i in range(0, len(uniq), 400):
                    block = uniq[i:i + 400]
                    ph = ",".join("?" * len(block))
                    for r in conn.execute(
                            "SELECT entity_id,doc_hash,text,provenance "
                            f"FROM records WHERE live=1 AND entity_id "
                            f"IN ({ph})", block):
                        out[r[0]] = StoredText(r[0], r[1], r[2],
                                               r[3] or "")
            finally:
                conn.close()
        return out

    def live_count(self) -> int:
        """Live record count from the cached snapshot bitmap -- no
        O(N) SQL per call. Rebuilt only when (vector_file,
        generation) changes, same as the scan snapshot."""
        return int(self._snapshot()[2].sum())

    def get_meta(self, key: str) -> str | None:
        conn = vio.connect(self._mpath())
        try:
            row = conn.execute("SELECT value FROM meta WHERE key=?",
                               (key,)).fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def stats(self) -> dict:
        live, dead = self._conn.execute("SELECT SUM(live=1), SUM(live=0)"
                                        " FROM records").fetchone()
        disk = sum(os.path.getsize(self._vpath(n))
                   for n in os.listdir(self._dir)
                   if os.path.isfile(self._vpath(n)))
        return {"live": int(live or 0), "tombstoned": int(dead or 0),
                "disk_bytes": disk, "generation": self.generation,
                "model_id": self.model_id, "dim": self.dim}

    def compact(self) -> None:
        """Renumber live slots densely and swap the vector file.
        Holds the RWLock exclusively: in-process readers finish before
        the swap, and no snapshot can observe a renumbering in
        progress. Not a record change -> generation is untouched."""
        with self._rwlock.write(), self._wlock:
            vfile, gen, live, _n = vio.snapshot(self._mpath())
            order = np.flatnonzero(live).tolist()
            seq = int(self._meta("vector_seq") or 0) + 1
            new_name = f"{_VEC_PREFIX}_g{gen}_s{seq}.bin"
            vio.write_compacted(self._vpath(vfile), order,
                                self.dim, self._vpath(new_name))
            try:
                vio.compact_commit(
                    self._conn, order,
                    (("vector_file", new_name),
                     ("vector_seq", str(seq))))
            except Exception:
                try:
                    os.remove(self._vpath(new_name))
                except OSError as exc:
                    logger.debug("compact cleanup: %s", exc)
                raise
            self._vector_file = new_name
            try:
                os.remove(self._vpath(vfile))  # old-gen file now orphan
            except OSError as exc:
                logger.debug("old vector file kept: %s", exc)
