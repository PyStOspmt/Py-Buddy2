"""
Retrieval Fabric: scan/snapshot/file helpers for `vector_store`
(private). Kept separate so both modules stay within the project's
350-line bound. Nothing here holds process state -- every function
opens what it needs and closes it.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_VECS = 8192          # dense-scan chunk size (vectors per read)
_TOK_RE = re.compile(r"[^\W_]+", re.UNICODE)


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None,
                           check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def slot_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT MAX(slot) FROM records").fetchone()
    return int(row[0]) + 1 if row and row[0] is not None else 0


def open_manifest(path: str, model_id: str, dim: int,
                  schema: str) -> tuple:
    """Create/open the manifest and verify the embedding-space stamp.
    Returns (conn, vector_file). Model/dim mismatch raises
    StoreMismatch -- embedding spaces are never mixed."""
    from vector_store import StoreMismatch  # deferred: vs imports vio
    conn = connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(schema)
        meta = dict(conn.execute("SELECT key,value FROM meta"))
        if meta.get("model_id") is not None \
                and meta["model_id"] != model_id:
            raise StoreMismatch(
                f"store model_id {meta['model_id']!r} != {model_id!r}")
        if meta.get("dim") is not None and int(meta["dim"]) != int(dim):
            raise StoreMismatch(f"store dim {meta['dim']} != {dim}")
        if meta.get("model_id") is None:
            conn.execute("BEGIN IMMEDIATE")
            for k, v in (("model_id", model_id), ("dim", str(dim)),
                         ("generation", "0"), ("vector_seq", "0"),
                         ("vector_file", "vectors.bin")):
                conn.execute(
                    "INSERT OR REPLACE INTO meta VALUES (?,?)", (k, v))
            conn.execute("COMMIT")
    except Exception:
        conn.close()
        raise
    return conn, meta.get("vector_file", "vectors.bin")


def truncate_orphans(vec_path: str, n_slots: int, dim: int) -> None:
    """Vector bytes beyond max(slot)+1 are a failed commit's orphans,
    removed before any reading handle exists."""
    try:
        size = os.path.getsize(vec_path)
    except OSError:
        return
    if size > n_slots * dim * 2:
        with open(vec_path, "r+b") as f:
            f.truncate(n_slots * dim * 2)


def drop_stale_vectors(directory: str, prefix: str,
                       current: str) -> None:
    """Remove unreferenced vector files (retried at open when a
    reader still holds them)."""
    for name in os.listdir(directory):
        if name.startswith(prefix) and name.endswith(".bin") \
                and name != current:
            try:
                os.remove(os.path.join(directory, name))
            except OSError as exc:
                logger.debug("stale vector file kept: %s", exc)


def snapshot(manifest_path: str) -> tuple[str, int, np.ndarray, int]:
    """Atomic read snapshot under one transaction: (vector_file,
    generation, live-slot bool array, committed slots). Writers
    committing later are invisible to the caller."""
    conn = connect(manifest_path)
    try:
        conn.execute("BEGIN")
        meta = dict(conn.execute("SELECT key,value FROM meta"))
        n = slot_count(conn)
        live = np.zeros(n, dtype=bool)
        for (s,) in conn.execute(
                "SELECT slot FROM records WHERE live=1"):
            if 0 <= s < n:
                live[s] = True
        conn.execute("COMMIT")
    finally:
        conn.close()
    return (meta.get("vector_file", "vectors.bin"),
            int(meta.get("generation", 0)), live, n)


def snapshot_key(manifest_path: str) -> tuple:
    """Cheap (vector_file, generation) probe for the bitmap cache --
    both change only inside the writer's commit transaction, so the
    pair uniquely identifies the live record set + slot layout."""

    conn = connect(manifest_path)
    try:
        meta = dict(conn.execute(
            "SELECT key,value FROM meta WHERE key IN "
            "('vector_file','generation')"))
    finally:
        conn.close()
    return (meta.get("vector_file", "vectors.bin"),
            int(meta.get("generation", 0)))


def slot_entity_ids(manifest_path: str, slots: list) -> dict:
    if not slots:
        return {}
    conn = connect(manifest_path)
    try:
        ph = ",".join("?" * len(slots))
        return {r[0]: r[1] for r in conn.execute(
            f"SELECT slot, entity_id FROM records "
            f"WHERE slot IN ({ph})", slots)}
    finally:
        conn.close()


def lexical_match(query: str, prefix_truncate: bool) -> str | None:
    """Build a safe FTS5 MATCH string. Tokens are whitelisted word
    characters (len>=2), each double-quoted and OR-joined, so FTS syntax
    injection is impossible. With prefix_truncate a token of len>=5
    becomes the quoted prefix of its first max(4, len-2) chars + '*'
    (crude language-agnostic stemming, no dictionary). None = no usable
    token."""
    toks = [t for t in _TOK_RE.findall(query.lower()) if len(t) >= 2]
    if not toks:
        return None
    parts = []
    for t in toks:
        if prefix_truncate and len(t) >= 5:
            parts.append(f'"{t[:max(4, len(t) - 2)]}"*')
        else:
            parts.append(f'"{t}"')
    return " OR ".join(parts)


def fts_top(manifest_path: str, match: str, k: int) -> list:
    conn = connect(manifest_path)
    try:
        rows = conn.execute(
            "SELECT entity_id, bm25(docs_fts) AS b FROM docs_fts "
            "WHERE docs_fts MATCH ? ORDER BY b LIMIT ?",
            (match, k)).fetchall()
    finally:
        conn.close()
    return [(r[0], -float(r[1])) for r in rows]


def _require_vector_file(vec_path: str, n_slots: int,
                         dim: int) -> None:
    """Missing/short vector file = StoreCorrupt, not empty results."""
    from vector_store import StoreCorrupt  # deferred: vs imports vio
    try:
        size = os.path.getsize(vec_path)
    except OSError:
        size = None
    need = n_slots * dim * 2
    if need and (size is None or size < need):
        raise StoreCorrupt(
            f"vector file {vec_path!r}: need {need} bytes, found "
            f"{size}")


def dense_scan(vec_path: str, dim: int, n_slots: int,
               live: np.ndarray, qvec: np.ndarray, k: int):
    """Exact brute-force top-k over float16 vectors, chunked float32.

    Returns (top[(score, slot)] desc, n_live_scored, mean, std) --
    mean/std are population stats over ALL live scores accumulated in
    float64 while streaming. Memory stays O(chunk) + live bitmap."""
    _require_vector_file(vec_path, n_slots, dim)
    n = n_slots
    bitmap = np.zeros(n, dtype=bool)
    m = min(n, len(live))
    bitmap[:m] = live[:m]
    cand: list = []
    ssum = ssq = 0.0
    cnt = 0
    for s0 in range(0, n, CHUNK_VECS):
        m = min(CHUNK_VECS, n - s0)
        arr = np.fromfile(vec_path, dtype=np.float16, count=m * dim,
                          offset=s0 * dim * 2)
        scores = arr.reshape(m, dim).astype(np.float32) @ qvec
        mask = bitmap[s0:s0 + m]
        lv = scores[mask]
        if not lv.size:
            continue
        l64 = lv.astype(np.float64)
        ssum += float(l64.sum())
        ssq += float((l64 * l64).sum())
        cnt += int(lv.size)
        cand += _chunk_topk(scores, mask, s0, k)
    cand.sort(key=lambda p: -p[0])
    mean = ssum / cnt if cnt else None
    var = max(0.0, ssq / cnt - mean * mean) if cnt else None
    return cand[:k], cnt, mean, (var ** 0.5) if var is not None else None


def _chunk_topk(scores: np.ndarray, mask: np.ndarray, s0: int,
                k: int) -> list:
    sm = np.where(mask, scores, -np.inf)
    take = min(k, scores.shape[0])
    idx = np.argpartition(-sm, take - 1)[:take]
    return [(float(sm[i]), s0 + int(i))
            for i in idx if np.isfinite(sm[i])]


def write_compacted(old_path: str, live_order: list, dim: int,
                    new_path: str) -> None:
    """Write live vectors contiguously (in old-slot order) to a new
    file; streamed in chunks, fsynced before returning."""
    with open(new_path, "wb") as out:
        for s0 in range(0, len(live_order), CHUNK_VECS):
            block = live_order[s0:s0 + CHUNK_VECS]
            vecs = np.vstack([
                np.fromfile(old_path, dtype=np.float16, count=dim,
                            offset=s * dim * 2)
                for s in block])
            out.write(vecs.tobytes())
        out.flush()
        os.fsync(out.fileno())


# ---------------------------------------------------------------------
# Write path (used by VectorStore.apply/compact; the transaction body
# lives here so vector_store.py stays within the project's size bound).
# ---------------------------------------------------------------------

def append_vectors(vec_path: str, dim: int, recs, start_slot: int) -> None:
    """Append float16 vector bytes + fsync BEFORE the manifest commit
    (orphan bytes of a failed commit are truncated later)."""
    if not recs:
        return
    with open(vec_path, "ab") as f:
        f.truncate(start_slot * dim * 2)   # drop prior orphans
        for r in recs:
            v = np.asarray(r.vector, dtype=np.float32)
            if v.shape != (dim,):
                raise ValueError(f"vector shape {v.shape} != ({dim},)")
            f.write(v.astype(np.float16).tobytes())
        f.flush()
        os.fsync(f.fileno())


def tombstone(conn: sqlite3.Connection, entity_id: str) -> bool:
    """Tombstone the live row + its FTS row for `entity_id`.
    docs_fts.rowid is bound to records.slot (O(1) delete --
    `entity_id` is UNINDEXED there, a WHERE on it is a full scan).
    Returns True iff a live row was actually removed."""
    row = conn.execute(
        "SELECT slot FROM records WHERE entity_id=? AND live=1",
        (entity_id,)).fetchone()
    if row is None:
        return False
    conn.execute("UPDATE records SET live=0 WHERE slot=?", (row[0],))
    conn.execute("DELETE FROM docs_fts WHERE rowid=?", (row[0],))
    return True


def upsert_row(conn: sqlite3.Connection, slot: int, r) -> None:
    tombstone(conn, r.entity_id)
    prov = r.provenance if isinstance(r.provenance, str) \
        else "|".join(r.provenance)
    conn.execute(
        "INSERT INTO records(slot,entity_id,doc_hash,text,"
        "provenance,live) VALUES (?,?,?,?,?,1)",
        (slot, r.entity_id, r.doc_hash, r.text, prov))
    conn.execute(
        "INSERT INTO docs_fts(rowid,entity_id,text) VALUES (?,?,?)",
        (slot, r.entity_id, r.text))


def commit_rows(conn: sqlite3.Connection, ops, meta_updates: dict,
                ) -> int:
    """Single manifest transaction: upserts, tombstones, meta, and a
    generation bump IFF the live record set effectively changed (an
    upsert or a delete that matched a live row). Returns the
    generation after the commit."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        changed = False
        for kind, slot, payload in ops:
            if kind == "upsert":
                upsert_row(conn, slot, payload)
                changed = True
            elif tombstone(conn, payload):
                changed = True
        for k, v in meta_updates.items():
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                         (k, str(v)))
        row = conn.execute(
            "SELECT value FROM meta WHERE key='generation'").fetchone()
        gen = int(row[0]) if row else 0
        if changed:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                         ("generation", str(gen + 1)))
            gen += 1
        conn.execute("COMMIT")
        return gen
    except Exception:
        conn.execute("ROLLBACK")
        raise


def compact_commit(conn: sqlite3.Connection, order: list,
                   meta_pairs) -> None:
    """Manifest side of compaction in ONE transaction: drop
    tombstones, renumber live slots densely, rebuild FTS rows (their
    rowid is bound to records.slot), write the meta pairs."""
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM records WHERE live=0")
        conn.execute("UPDATE records SET slot=-(slot+1)")
        for i, old in enumerate(order):
            conn.execute("UPDATE records SET slot=? WHERE slot=?",
                         (i, -(old + 1)))
        conn.execute("DELETE FROM docs_fts")
        conn.execute(
            "INSERT INTO docs_fts(rowid,entity_id,text) "
            "SELECT slot,entity_id,text FROM records WHERE live=1")
        for k, v in meta_pairs:
            conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)",
                         (k, v))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
