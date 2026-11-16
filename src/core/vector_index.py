"""
Vector channel: dense-embedding bindings for paraphrase/semantic terms.

The lexical channels bind a term only when its characters appear in a
cell. Paraphrase breaks that -- "воскресити" never LIKE-matches
"resurrect". This module adds the dense channel as ONE MORE evidence
channel with the same honesty contract as the others:

  * every hit carries provenance (entity row, column, cell text,
    cosine similarity, embedding model);
  * the index is built explicitly and reports exactly how many cells it
    covered -- a sampled index is disclosed, never hidden;
  * a missing API key or a failed embed call marks the channel
    unavailable with a reason; it never fabricates results and never
    breaks the lexical channels;
  * the embedder is a protocol (`.embed(texts) -> vectors`), so tests
    use a scripted deterministic embedder, not a live paid call.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from typing import Any

from source_projection import SourceProjection, _is_text, quote_ident

# Embedding model choices -- surfaced in the UI like the chat model
# selector. The first entry is the default.
EMBEDDING_MODELS = [
    "qwen/qwen3-embedding-8b",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    "nvidia/nemotron-3-embed-1b:free",
    "perplexity/pplx-embed-v1-0.6b",
]

EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_BATCH = 32
MAX_INDEX_CELLS = 1500        # disclosed sampling cap per source
MAX_CELL_CHARS = 240
VEC_THRESHOLD = 0.55          # absolute cosine floor for a binding
VEC_Z = 2.0                   # hit must also clear mean + Z*sd of the
                              # term's own score distribution
VEC_EDGE = 0.03               # accepted cluster must sit this far above
                              # the best rejected score (live run finding:
                              # a real model gave 'титан' top1=0.721 with
                              # margin 0.027 -- high score, no structure;
                              # the paraphrase hit had margin 0.074)
VEC_TOP_K = 250               # per-term hit cap
VEC_DIM_EXPAND = 200          # entity cap when a dimension label binds

CH_VECTOR = "vector"
VEC_SUFFIX = ".vecidx"          # sidecar file next to the source
VEC_META_KEYS = ("stamp", "model", "max_cells", "cells", "built_s")


class OpenRouterEmbedder:
    """Dense embedder over the same OpenRouter key the chat runner uses.

    `.embed` returns a list of vectors, or None on failure -- callers
    distinguish "the channel is down" from "nothing was similar".
    """

    def __init__(self, api_key: str, model: str,
                 timeout: float = 30.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]] | None:
        if not texts:
            return []
        import requests
        try:
            resp = requests.post(
                EMBED_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/PyStOspmt/Py-Buddy2",
                    "X-Title": "Py-Buddy",
                },
                data=json.dumps({"model": self.model, "input": texts}),
                timeout=self.timeout)
            if resp.status_code != 200:
                return None
            data = resp.json().get("data") or []
            out = []
            for item in data:
                v = item.get("embedding")
                if not isinstance(v, list):
                    return None
                out.append(v)
            return out if len(out) == len(texts) else None
        except Exception:  # noqa: BLE001 -- provider failure degrades
            return None


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return num / (na * nb) if na and nb else 0.0


class VectorIndex:
    """Dense index over entity-table text cells.

    Each cell keeps full provenance -- (entity_id, column, text) -- so a
    vector binding is inspectable like any other channel's evidence.
    """

    def __init__(self, model: str):
        self.model = model
        self.cells: list[dict] = []
        self.vectors: list[list[float]] = []
        self.stats: dict = {"model": model, "cells": 0,
                            "sampled": False, "status": "empty"}
        self._term_cache: dict[str, list[float] | None] = {}
        # Dimension-label expansion context: (entity_table, identity
        # column, source path) recorded at build/load so a hit on a
        # dimension row can be resolved to the entities referencing it.
        self._expand_ctx: dict | None = None
        self._src_conn: sqlite3.Connection | None = None
        # read-only expansion conn is shared across request threads
        import threading
        self._src_lock = threading.Lock()

    def build(self, db_path: str, proj: SourceProjection,
              embedder: Any, max_cells: int = MAX_INDEX_CELLS) -> dict:
        """Index text cells of the entity table. Deterministic order,
        disclosed cap -- `stats` records exactly what was covered."""
        t0 = time.time()
        ident = proj.resolve("identity")
        tp = proj.snapshot.tables.get(proj.entity_table or "")
        if not ident.column or tp is None:
            self.stats["status"] = "unavailable"
            self.stats["reason"] = "no entity table / identity role"
            return self.stats

        conn = sqlite3.connect(
            "file:{}?mode=ro".format(
                os.path.abspath(db_path).replace("\\", "/")),
            uri=True)
        try:
            text_cols = [c.name for c in tp.columns.values()
                         if _is_text(c)]
            cells: list[dict] = []
            by_channel = {"entity_column": 0, "eav_value": 0,
                          "child_mention": 0, "fk_dimension": 0}
            # Dimension-label expansion context: a vector hit on a
            # dimension row resolves to entities through the recorded
            # link at bind time.
            self._expand_ctx = {
                "db_path": os.path.abspath(db_path),
                "entity_table": proj.entity_table,
                "ident_col": ident.column}
            for col in text_cols:
                if len(cells) >= max_cells:
                    break
                rows = conn.execute(
                    f"SELECT {quote_ident(ident.column)}, "
                    f"{quote_ident(col)} FROM "
                    f"{quote_ident(proj.entity_table)} "
                    f"WHERE {quote_ident(col)} IS NOT NULL "
                    f"LIMIT {max_cells - len(cells)}").fetchall()
                for rid, v in rows:
                    txt = str(v)[:MAX_CELL_CHARS]
                    if txt.strip():
                        cells.append({"entity_id": str(rid),
                                      "channel": "entity_column",
                                      "column": col, "text": txt})
                        by_channel["entity_column"] += 1
            # EAV value cells: paraphrase must reach attributes too --
            # "для шкіри" lives in the value column, not entity text.
            for tbl, meta in (getattr(proj, "eav_tables", {})
                              or {}).items():
                ef = meta.get("entity_fk") or meta.get("entity_column")
                kc, vc = meta.get("key_column"), meta.get("value_column")
                if not (ef and kc and vc):
                    continue
                try:
                    rows = conn.execute(
                        f"SELECT {quote_ident(ef)}, {quote_ident(kc)}, "
                        f"{quote_ident(vc)} FROM {quote_ident(tbl)} "
                        f"WHERE {quote_ident(vc)} IS NOT NULL "
                        f"LIMIT {max_cells}").fetchall()
                except sqlite3.Error:
                    continue
                for rid, k, v in rows:
                    if len(cells) >= max_cells:
                        break
                    txt = str(v)[:MAX_CELL_CHARS]
                    if txt.strip():
                        cells.append({"entity_id": str(rid),
                                      "channel": "eav_value",
                                      "column": vc, "table": tbl,
                                      "axis": str(k), "text": txt})
                        by_channel["eav_value"] += 1
            # Child mentions: text in tables that FK INTO the entity
            # (reviews etc.) carries an entity reference directly.
            for tname, tprof in proj.snapshot.tables.items():
                fks = [c.name for c in tprof.columns.values()
                       if c.is_foreign_key
                       and c.fk_target_table == proj.entity_table]
                if not fks:
                    continue
                for fk in fks:
                    for tcol in tprof.columns.values():
                        if not _is_text(tcol) or tcol.name == fk:
                            continue
                        try:
                            rows = conn.execute(
                                f"SELECT {quote_ident(fk)}, "
                                f"{quote_ident(tcol.name)} FROM "
                                f"{quote_ident(tname)} WHERE "
                                f"{quote_ident(tcol.name)} IS NOT NULL "
                                f"LIMIT {max_cells}").fetchall()
                        except sqlite3.Error:
                            continue
                        for rid, v in rows:
                            if len(cells) >= max_cells:
                                break
                            txt = str(v)[:MAX_CELL_CHARS]
                            if txt.strip():
                                cells.append({
                                    "entity_id": str(rid),
                                    "channel": "child_mention",
                                    "column": tcol.name,
                                    "table": tname,
                                    "text": txt})
                                by_channel["child_mention"] += 1
            # Dimension labels: text in tables the entity references
            # through declared/inferred links. The cell has no single
            # entity -- it is stored with the link needed to expand
            # (label hit -> every entity whose FK points at the row).
            dim_links = [
                (c.name, c.fk_target_table,
                 c.fk_target_column)
                for c in tp.columns.values()
                if c.is_foreign_key and c.fk_target_table]
            for l in getattr(proj, "inferred_links", []) or []:
                if l.get("child_table") == proj.entity_table:
                    dim_links.append((l["child_column"],
                                      l["parent_table"],
                                      l["parent_column"]))
            for fk_col, ptab, pcol in dim_links:
                if len(cells) >= max_cells:
                    break
                pprof = proj.snapshot.tables.get(ptab)
                if pprof is None or not pcol:
                    continue
                for tcol in pprof.columns.values():
                    if not _is_text(tcol):
                        continue
                    try:
                        rows = conn.execute(
                            f"SELECT {quote_ident(pcol)}, "
                            f"{quote_ident(tcol.name)} FROM "
                            f"{quote_ident(ptab)} WHERE "
                            f"{quote_ident(tcol.name)} IS NOT NULL "
                            f"LIMIT {max_cells}").fetchall()
                    except sqlite3.Error:
                        continue
                    for pk, v in rows:
                        if len(cells) >= max_cells:
                            break
                        txt = str(v)[:MAX_CELL_CHARS]
                        if txt.strip():
                            cells.append({
                                "entity_id": None,
                                "channel": "fk_dimension",
                                "table": ptab, "column": tcol.name,
                                "dim_pk": str(pk), "pk_col": pcol,
                                "expand_col": fk_col, "text": txt})
                            by_channel["fk_dimension"] += 1
            self.stats["sampled"] = sum(
                tp.row_count or 0 for _ in text_cols) > len(cells)
            self.stats["by_channel"] = by_channel
        finally:
            conn.close()

        self.cells = cells
        self.vectors = []
        for i in range(0, len(cells), EMBED_BATCH):
            batch = [c["text"] for c in cells[i:i + EMBED_BATCH]]
            vecs = embedder.embed(batch)
            if vecs is None:
                self.stats["status"] = "unavailable"
                self.stats["reason"] = (
                    f"embed call failed at batch {i // EMBED_BATCH}")
                return self.stats
            self.vectors.extend(vecs)

        self.stats.update({
            "cells": len(cells), "status": "ready",
            "max_cells": max_cells,
            "elapsed_s": round(time.time() - t0, 2)})
        return self.stats

    def bind(self, term: str, embedder: Any,
             threshold: float = VEC_THRESHOLD,
             top_k: int = VEC_TOP_K) -> dict[str, list]:
        """Cosine-bind one query term to indexed cells.

        Returns entity_id -> [binding]. An empty dict is an honest
        'nothing similar'; an unavailable index binds nothing and the
        caller sees stats['reason']."""
        if not self.vectors:
            return {}
        if term not in self._term_cache:
            vecs = embedder.embed([term])
            self._term_cache[term] = vecs[0] if vecs else None
        qv = self._term_cache[term]
        if qv is None:
            return {}
        scored = sorted(
            ((_cosine(qv, v), i) for i, v in enumerate(self.vectors)),
            key=lambda t: -t[0])
        # Outlier gate, not a fixed floor: real models produce model-
        # specific score clouds (qwen3-8b puts unrelated short Cyrillic
        # at ~0.55-0.72). A binding is honest only when it clears BOTH
        # the absolute floor AND stands out against the term's own
        # distribution -- the accepted cluster must be separated from
        # the best rejected score by VEC_EDGE. A uniform cloud (even a
        # high-scoring one) binds nothing: that is the 'титан' case.
        floor = [t for t in scored if t[0] >= threshold]
        if not floor:
            return {}
        mu = sum(s for s, _ in scored) / len(scored)
        var = sum((s - mu) ** 2 for s, _ in scored) / len(scored)
        bar = max(threshold, mu + VEC_Z * math.sqrt(var))
        cand = [t for t in floor if t[0] >= bar]
        rejected = [s for s, _ in floor if s < bar]
        if not cand:
            return {}
        if rejected and min(s for s, _ in cand) - max(rejected) < VEC_EDGE:
            return {}
        hits: dict[str, list] = {}
        for s, i in cand[:top_k]:
            c = self.cells[i]
            b = {"channel": CH_VECTOR, "column": c["column"],
                 "value": c["text"][:120],
                 "similarity": round(s, 4), "model": self.model,
                 "gate": {"bar": round(bar, 4), "mu": round(mu, 4)}}
            # Cell provenance travels with the binding: a vector hit on
            # an EAV value names its axis, on a child row its table.
            if c.get("channel") and c["channel"] != "entity_column":
                b["cell_channel"] = c["channel"]
            for extra in ("axis", "table"):
                if c.get(extra) is not None:
                    b[extra] = c[extra]
            if c.get("entity_id"):
                hits.setdefault(c["entity_id"], []).append(b)
            elif c.get("dim_pk") is not None:
                # Dimension-label hit: expand the dimension row to the
                # entities whose FK references it -- recorded, bounded.
                for rid in self._expand_dim(c):
                    eb = dict(b)
                    eb["expanded_via"] = (
                        f"{self._expand_ctx['entity_table']}."
                        f"{c['expand_col']}={c['dim_pk']}")
                    hits.setdefault(rid, []).append(eb)
        return hits

    def _expand_dim(self, cell: dict) -> list[str]:
        """Resolve a dimension-label cell to entity ids via the link the
        cell carries (entity.fk_col = dim_pk). Bounded; opens the source
        read-only and lazily."""
        ctx = self._expand_ctx
        if ctx is None:
            return []
        if self._src_conn is None:
            try:
                self._src_conn = sqlite3.connect(
                    "file:{}?mode=ro".format(
                        ctx["db_path"].replace("\\", "/")), uri=True,
                    check_same_thread=False)
            except sqlite3.Error:
                return []
        try:
            with self._src_lock:
                return [str(r[0]) for r in self._src_conn.execute(
                    f"SELECT {quote_ident(ctx['ident_col'])} FROM "
                    f"{quote_ident(ctx['entity_table'])} WHERE "
                    f"{quote_ident(cell['expand_col'])} = ? "
                    f"LIMIT {VEC_DIM_EXPAND}",
                    (cell["dim_pk"],))]
        except sqlite3.Error:
            return []

    # -- persistence: <db>.vecidx sidecar --------------------------

    def save(self, db_path: str) -> str:
        """Serialize cells+vectors into a SQLite sidecar keyed by the
        source's (mtime,size) stamp and the model name. A restart that
        finds a matching sidecar skips every paid embed call."""
        import struct
        path = os.path.abspath(db_path) + VEC_SUFFIX
        if os.path.exists(path):
            os.remove(path)
        st = os.stat(os.path.abspath(db_path))
        conn = sqlite3.connect(path)
        try:
            conn.execute("CREATE TABLE __vec_meta "
                         "(key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("CREATE TABLE cells ("
                         "entity_id TEXT, channel TEXT, ref TEXT, "
                         "text TEXT, vec BLOB)")
            with conn:
                conn.executemany(
                    "INSERT INTO __vec_meta VALUES (?,?)", [
                        ("stamp", f"{st.st_mtime}:{st.st_size}"),
                        ("model", self.model),
                        ("max_cells",
                         str(self.stats.get("max_cells")
                             or len(self.cells))),
                        ("cells", str(len(self.cells))),
                        ("built_s", str(self.stats.get("elapsed_s"))),
                        ("entity_table",
                         (self._expand_ctx or {}).get(
                             "entity_table") or ""),
                        ("ident_col",
                         (self._expand_ctx or {}).get(
                             "ident_col") or ""),
                    ])
                for cell, vec in zip(self.cells, self.vectors):
                    ref = json.dumps(
                        {k: cell.get(k) for k in
                         ("column", "axis", "table", "dim_pk",
                          "pk_col", "expand_col")
                         if cell.get(k) is not None},
                        ensure_ascii=False)
                    conn.execute(
                        "INSERT INTO cells VALUES (?,?,?,?,?)",
                        (cell["entity_id"], cell.get("channel"),
                         ref, cell["text"],
                         struct.pack(f"{len(vec)}f", *vec)))
        finally:
            conn.close()
        return path

    @classmethod
    def load(cls, db_path: str, model: str,
             max_cells: int | None = None) -> VectorIndex | None:
        """Open a valid persisted index -- same stamp, same model, same
        cell budget -- without any embed calls. Returns None otherwise."""
        import struct
        ap = os.path.abspath(db_path)
        path = ap + VEC_SUFFIX
        if not os.path.exists(path):
            return None
        st = os.stat(ap)
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro".replace(
                "\\", "/"), uri=True)
            meta = dict(conn.execute(
                "SELECT key, value FROM __vec_meta").fetchall())
            if (meta.get("stamp") != f"{st.st_mtime}:{st.st_size}"
                    or meta.get("model") != model
                    or (max_cells is not None
                        and meta.get("max_cells") != str(max_cells))):
                conn.close()
                return None
            idx = cls(model)
            for rid, chan, ref, text, blob in conn.execute(
                    "SELECT entity_id, channel, ref, text, vec "
                    "FROM cells"):
                n = len(blob) // 4
                vec = list(struct.unpack(f"{n}f", blob))
                cell = {"entity_id": rid, "channel": chan,
                        "text": text}
                cell.update(json.loads(ref or "{}"))
                idx.cells.append(cell)
                idx.vectors.append(vec)
            conn.close()
        except (sqlite3.Error, ValueError, KeyError):
            return None
        if meta.get("entity_table") and meta.get("ident_col"):
            idx._expand_ctx = {
                "db_path": ap,
                "entity_table": meta["entity_table"],
                "ident_col": meta["ident_col"]}
        idx.stats.update({
            "cells": len(idx.cells), "status": "ready",
            "model": model, "loaded_from": "sidecar",
            "built_s": float(meta.get("built_s") or 0)})
        return idx


# Per-source index cache: one index per (db, mtime, model). Rebuilt on
# source change; stats travel with it for inspection.
_INDEX_CACHE: dict[tuple, VectorIndex] = {}


def get_vector_index(db_path: str, proj: SourceProjection,
                     embedder: Any, model: str,
                     max_cells: int = MAX_INDEX_CELLS) -> VectorIndex:
    ap = os.path.abspath(db_path)
    mtime = os.path.getmtime(ap) if os.path.exists(ap) else 0
    key = (ap, mtime, model, max_cells)
    idx = _INDEX_CACHE.get(key)
    if idx is None:
        # Persisted sidecar wins over a fresh paid build: same source
        # stamp + same model means the stored vectors are the answer.
        idx = VectorIndex.load(db_path, model, max_cells)
        if idx is None:
            idx = VectorIndex(model)
            idx.build(db_path, proj, embedder, max_cells)
            if idx.stats.get("status") == "ready":
                try:
                    idx.save(db_path)
                except (OSError, sqlite3.Error):
                    pass  # persistence is an optimization, not a gate
        _INDEX_CACHE.clear()          # one live index per process
        _INDEX_CACHE[key] = idx
    return idx
