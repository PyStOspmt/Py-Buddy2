"""Materialized channel index: an FTS5 trigram sidecar that turns the
O(rows) LIKE sweeps of the lexical evidence channels into indexed
substring lookups.

Trigram tokenization preserves LIKE semantics (`'кіра'` matches
`шкіра`), so bindings are identical to the scan path -- only the cost
changes. The index lives in a sidecar file `<source>.ebridx` so the
customer's database is never modified, and it is rebuilt whenever the
source's (mtime, size) stamp moves. Coverage:

- entity-table text columns      -> channel 'entity_column'
- EAV value cells                -> channel 'eav_value' (axis recorded)
- child_mention text columns     -> channel 'child_mention'
  (tables that FK into the entity table -- reviews, doc tables)

Small dimension tables and text-link vocabularies stay on the plain
scan path -- indexing 12-row lookups buys nothing.

A term shorter than 3 chars cannot trigram-match; those go through the
exact-vocabulary path which is untouched by this index.
"""
from __future__ import annotations

import os
import sqlite3
import threading

IDX_SUFFIX = ".ebridx"
MAX_INDEX_ROWS = 5_000_000   # materialization budget
MIN_FTS_TERM = 3             # trigram needs >=3 chars


def _is_text(col) -> bool:
    from source_projection import _is_text as _t
    return _t(col)


def sidecar_path(db_path: str) -> str:
    return os.path.abspath(db_path) + IDX_SUFFIX


def index_valid(db_path: str) -> bool:
    """True when a sidecar exists and matches the source's stamp."""
    side = sidecar_path(db_path)
    if not os.path.exists(side):
        return False
    st = os.stat(os.path.abspath(db_path))
    try:
        conn = sqlite3.connect(f"file:{side}?mode=ro", uri=True)
        row = conn.execute(
            "SELECT value FROM __idx_meta WHERE key='stamp'").fetchone()
        conn.close()
    except sqlite3.Error:
        return False
    return bool(row) and row[0] == f"{st.st_mtime}:{st.st_size}"


def build_channel_index(db_path: str, proj) -> dict:
    """Materialize the index next to the source. Returns stats."""
    from source_projection import quote_ident
    side = sidecar_path(db_path)
    if os.path.exists(side):
        os.remove(side)
    src = sqlite3.connect(
        f"file:{os.path.abspath(db_path)}?mode=ro".replace("\\", "/"),
        uri=True)
    dst = sqlite3.connect(side)
    t0 = __import__("time").time()
    try:
        dst.execute(
            "CREATE VIRTUAL TABLE cells USING fts5("
            "content, entity_id UNINDEXED, channel UNINDEXED, "
            "value UNINDEXED, tokenize='trigram')")
        total = 0
        ent = proj.entity_table
        ident = proj.resolve("identity")
        eav = getattr(proj, "eav_tables", {}) or {}
        with src:
            # entity text columns
            prof = proj.snapshot.tables.get(ent)
            if prof is not None and ident.column:
                for col in prof.columns.values():
                    if not _is_text(col):
                        continue
                    for rid, v in src.execute(
                            f"SELECT {quote_ident(ident.column)}, "
                            f"{quote_ident(col.name)} FROM "
                            f"{quote_ident(ent)} WHERE "
                            f"{quote_ident(col.name)} IS NOT NULL"):
                        if total >= MAX_INDEX_ROWS:
                            break
                        dst.execute(
                            "INSERT INTO cells VALUES (?,?,?,?)",
                            (str(v)[:4000], str(rid),
                             "entity_column", col.name))
                        total += 1
            # EAV values
            for tbl, meta in eav.items():
                ef, kc, vc = (meta.get("entity_fk"),
                              meta.get("key_column"),
                              meta.get("value_column"))
                if not (ef and kc and vc):
                    continue
                for rid, k, v in src.execute(
                        f"SELECT {quote_ident(ef)}, {quote_ident(kc)}, "
                        f"{quote_ident(vc)} FROM {quote_ident(tbl)} "
                        f"WHERE {quote_ident(vc)} IS NOT NULL"):
                    if total >= MAX_INDEX_ROWS:
                        break
                    dst.execute("INSERT INTO cells VALUES (?,?,?,?)",
                                (str(v)[:4000], str(rid),
                                 "eav_value", f"{tbl}.{kc}={k}"))
                    total += 1
            # child mentions: tables FK -> entity
            if ident.column:
                for tname, tprof in proj.snapshot.tables.items():
                    fks = [c.name for c in tprof.columns.values()
                           if c.is_foreign_key
                           and c.fk_target_table == ent]
                    if not fks:
                        continue
                    for fk in fks:
                        for tcol in tprof.columns.values():
                            if not _is_text(tcol) or tcol.name == fk:
                                continue
                            for rid, v in src.execute(
                                    f"SELECT {quote_ident(fk)}, "
                                    f"{quote_ident(tcol.name)} FROM "
                                    f"{quote_ident(tname)} WHERE "
                                    f"{quote_ident(tcol.name)} "
                                    f"IS NOT NULL"):
                                if total >= MAX_INDEX_ROWS:
                                    break
                                dst.execute(
                                    "INSERT INTO cells VALUES "
                                    "(?,?,?,?)",
                                    (str(v)[:4000], str(rid),
                                     "child_mention",
                                     f"{tname}|{fk}|{tcol.name}"))
                                total += 1
        dst.execute("CREATE TABLE __idx_meta (key TEXT PRIMARY KEY,"
                    " value TEXT)")
        st = os.stat(os.path.abspath(db_path))
        dst.executemany("INSERT INTO __idx_meta VALUES (?,?)", [
            ("stamp", f"{st.st_mtime}:{st.st_size}"),
            ("rows", str(total)),
            ("built_s", f"{__import__('time').time() - t0:.2f}"),
        ])
        dst.commit()
        return {"status": "ready", "rows": total,
                "elapsed_s": round(__import__("time").time() - t0, 2),
                "capped": total >= MAX_INDEX_ROWS}
    finally:
        src.close()
        dst.close()


class ChannelIndex:
    """Read-only handle over a valid sidecar; returns binding-shaped
    hits identical to the LIKE scan path."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        # check_same_thread=False: the handle is created in the init-job
        # worker but bound from request threads; reads are serialized by
        # the lock (the sidecar is immutable once written).
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(
            f"file:{sidecar_path(db_path)}?mode=ro".replace("\\", "/"),
            uri=True, check_same_thread=False)

    def bind(self, term: str, limit: int) -> dict:
        """entity_id -> [binding,...] for one term via trigram MATCH."""
        hits: dict[str, list] = {}
        esc = term.replace('"', '""')
        try:
            with self._lock:
                rows = self.conn.execute(
                    "SELECT entity_id, channel, value, content FROM cells "
                    "WHERE cells MATCH ? LIMIT ?",
                    (f'"{esc}"', int(limit))).fetchall()
        except sqlite3.Error:
            return hits
        for rid, channel, where, content in rows:
            # `value` column of the index holds the binding's WHERE
            # (column name / axis descriptor); the cell text is the
            # provenance VALUE, same as the LIKE path records.
            b = {"term": term, "channel": channel,
                 "value": str(content)[:120], "indexed": True}
            if channel == "entity_column":
                b["column"] = where
            elif channel == "eav_value":
                tbl_and_k, axis = (where or ".").split("=", 1) \
                    if "=" in (where or "") else (where, "?")
                b["table"], b["axis"] = tbl_and_k.rsplit(".", 1)[0], axis
            elif channel == "child_mention" and "|" in (where or ""):
                tbl, fkcol, tcol = where.split("|", 2)
                b["table"], b["column"] = tbl, tcol
                b["via"] = f"{tbl}.{fkcol}"
            else:
                b["column"] = where
            hits.setdefault(str(rid), []).append(b)
        return hits

    def close(self) -> None:
        self.conn.close()


def get_index(db_path: str, proj=None) -> ChannelIndex | None:
    """Open the sidecar when valid; build it when missing and a
    projection is available; return None when neither holds."""
    if index_valid(db_path):
        return ChannelIndex(db_path)
    if proj is not None:
        stats = build_channel_index(db_path, proj)
        if stats.get("status") == "ready":
            return ChannelIndex(db_path)
    return None
