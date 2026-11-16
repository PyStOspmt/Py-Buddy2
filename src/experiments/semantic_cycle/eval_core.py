"""Building blocks of the retrieval-level evaluation (docs/18).

Index construction, per-request ranking collection for every compared
system, independent checks (count reconciliation, differential lexical
oracle) and process-memory sampling. Pure evaluator code: it calls the
production API exactly as a client would and never post-processes results.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "src" / "core", ROOT / "src" / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

SOURCES = {
    "library": (ROOT / "data" / "eval_semantic" / "library.db", "books"),
    "forum": (ROOT / "data" / "eval_semantic" / "forum.db", "posts"),
    "food": (ROOT / "data" / "real_off.db", "products"),
}
MODES = ("bm25", "bm25_prefix", "dense", "hybrid")
QTYPES = ("en_natural", "en_strict", "uk")
DEPTH = 50
REQUEST_FILE = ROOT / "docs" / "semantic_queries_20261112.json"


def load_requests() -> dict:
    raw = REQUEST_FILE.read_bytes()
    expected = (REQUEST_FILE.parent / (REQUEST_FILE.name + ".sha256")
                ).read_text().strip()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("request file does not match its frozen sha256")
    return json.loads(raw)


def environment() -> dict:
    import numpy
    import sentence_transformers
    import torch
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--short"],
                           capture_output=True, text=True).stdout.count("\n")
    import sqlite3
    mods = ("embedders", "entity_select", "record_documents",
            "record_docs_plan", "record_docs_render", "semantic_search",
            "semantic_sync", "source_projection", "vector_store",
            "vector_store_io", "rwlock")
    code = {m: hashlib.sha256((ROOT / "src" / "core" / f"{m}.py"
                               ).read_bytes()).hexdigest()[:16]
            for m in mods if (ROOT / "src" / "core" / f"{m}.py").exists()}
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "code_sha256_16": code,
            "cpu_logical": psutil.cpu_count(), "ram_gb": round(
                psutil.virtual_memory().total / 2**30, 1),
            "numpy": numpy.__version__, "torch": torch.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "sqlite": sqlite3.sqlite_version, "git_head": head,
            "git_dirty_entries": dirty}


class PeakMemory:
    """Samples this process' USS/RSS on a background thread."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.proc = interval, psutil.Process()
        self.peak_uss = self.peak_rss = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        mi = self.proc.memory_full_info()
        self.peak_uss = max(self.peak_uss, mi.uss)
        self.peak_rss = max(self.peak_rss, mi.rss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self) -> "PeakMemory":
        self._sample()
        self.start_uss = self.proc.memory_full_info().uss
        self._t.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._t.join()
        self._sample()

    def summary_mb(self) -> dict:
        mb = 1 / 2**20
        return {"start_uss_mb": round(self.start_uss * mb, 1),
                "peak_uss_mb": round(self.peak_uss * mb, 1),
                "peak_rss_mb": round(self.peak_rss * mb, 1),
                "delta_uss_mb": round((self.peak_uss - self.start_uss) * mb, 1)}


def build_index(source: str, index_dir: Path, embedder, progress=None,
                llm=None):
    """Projection + incremental sync through the PRODUCT service
    (SemanticIndex), so the sidecar carries the product's own freshness
    stamp and the same handle can later serve product-path searches."""
    from semantic_service import SemanticIndex
    from source_projection import build_source_projection
    path, _intended = SOURCES[source]
    proj = build_source_projection(str(path), fresh=True)
    gen = (lambda p: llm.complete_text(p)) if llm is not None else None
    kwargs = {"llm_generate": gen} if gen is not None else {}
    idx = SemanticIndex(source, str(path), embedder_factory=lambda: embedder,
                        index_dir=str(index_dir), **kwargs)
    t0 = time.time()
    with PeakMemory() as mem:
        rep = idx.sync_blocking(reason="eval", progress=progress)
    store = idx._ensure_store()
    return proj, store, rep, time.time() - t0, mem.summary_mb(), idx


def check_reconciliation(proj, rep, store) -> dict:
    """Independent count checks (P6): the source COUNT(*) is read by the
    evaluator, not taken from the system."""
    import sqlite3
    out = {"entity_table": proj.entity_table}
    if proj.entity_table is None:
        return {**out, "ok": False, "reason": "no entity table"}
    uri = "file:" + str(proj.db_path).replace("\\", "/") + "?mode=ro"
    c = sqlite3.connect(uri, uri=True)
    n = c.execute(f'SELECT COUNT(*) FROM "{proj.entity_table}"').fetchone()[0]
    c.close()
    parts = rep.inserted + rep.updated + rep.unchanged + sum(
        rep.excluded.values())
    live = store.stats()["live"]
    out.update({"source_count_star": n, "report_source_rows": rep.source_rows,
                "inserted+updated+unchanged+excluded": parts,
                "store_live": live,
                "ok": n == rep.source_rows == parts and rep.complete
                and live == rep.inserted + rep.updated + rep.unchanged})
    return out


def _one_mode(store, emb, query: str, mode: str, raw_ix) -> dict:
    from semantic_search import search
    if mode == "raw_bm25":
        return {"ids": raw_ix.query(query, DEPTH, False)}
    if mode in ("bm25", "bm25_prefix"):
        rows = store.search_lexical(
            query, DEPTH, prefix_truncate=(mode == "bm25_prefix"))
        return {"ids": [r[0] for r in rows],
                "top_score": rows[0][1] if rows else None}
    if mode.startswith("hybrid_l"):
        res = search(store, emb, query, k=DEPTH, mode="hybrid",
                     dense_weight=1.0,
                     lexical_weight=float(mode[len("hybrid_l"):]))
    elif mode == "hybrid":      # the REGISTERED system: RRF 1:1, explicit
        res = search(store, emb, query, k=DEPTH, mode="hybrid",
                     dense_weight=1.0, lexical_weight=1.0)
    else:
        res = search(store, emb, query, k=DEPTH, mode=mode)
    rec = {"ids": [h.entity_id for h in res.hits],
           "top1_z": res.top1_z, "margin": res.margin}
    if mode == "dense":
        rec["top1_sim"] = res.hits[0].dense_sim if res.hits else None
    return rec


def run_modes(store, emb, query: str, raw_ix=None, grid=(),
              hyde=None, product=None, product_hyde: bool = False) -> dict:
    """Ranking (top DEPTH ids) of every compared system for one request.
    `raw_ix` (ablation) is the evaluator-owned BM25 over raw own columns;
    `grid` = lexical weights for the dev-only fusion sweep; `hyde` adds the
    exploratory hypothetical-document modes."""
    out = {}
    modes = (MODES + (("raw_bm25",) if raw_ix is not None else ())
             + tuple(f"hybrid_l{w}" for w in grid))
    for mode in modes:
        t0 = time.perf_counter()
        try:
            rec = _one_mode(store, emb, query, mode, raw_ix)
            rec["error"] = None
        except Exception as exc:  # noqa: BLE001 -- recorded, counted as failure
            rec = {"ids": [], "error": f"{type(exc).__name__}: {exc}"}
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        out[mode] = rec
    if hyde is not None:
        out.update(hyde.run(query))
    if product is not None:
        out.update(product_modes(product, query, product_hyde))
    return out


def product_modes(idx, query: str, hyde: bool) -> dict:
    """Searches through the shipped service layer (default hybrid and,
    when asked, the HyDE mode). A silently degraded HyDE request is
    flagged as an error string so it is counted, never hidden."""
    out = {}
    for name, use_hyde in (("product_default", False),
                           ("product_hyde", True)):
        if use_hyde and not hyde:
            continue
        t0 = time.perf_counter()
        try:
            kw = {"hyde": True} if use_hyde else {}
            res = idx.search(query, k=DEPTH, mode="hybrid", **kw)
            conf = res.get("confidence") or {}
            err = None if res.get("status") == "ok"                 else f"status {res.get('status')}"
            if use_hyde and res.get("requested_mode") == "hyde":
                err = "hyde_degraded: " + "; ".join(res.get("notes") or [])
            rec = {"ids": [h["entity_id"] for h in res.get("hits", [])],
                   "error": err, "top1_sim": conf.get("top1_sim"),
                   "top1_z": conf.get("top1_z"),
                   "margin": conf.get("margin"),
                   "effective_mode": res.get("mode"),
                   "hypothesis": (res.get("hypothesis") or {}).get("text")}
        except Exception as exc:  # noqa: BLE001 -- recorded, counted
            rec = {"ids": [], "error": f"{type(exc).__name__}: {exc}"}
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 2)
        out[name] = rec
    return out


def differential_lexical(store, plan, db_path: str, queries: list[str]
                         ) -> dict:
    """Evaluator-owned BM25 over the SAME documents vs the store's
    lexical channel: identical top-10 lists are expected."""
    from bm25_independent import IndependentBM25
    from record_documents import iter_record_documents
    ix = IndependentBM25()
    n = ix.add_many((d.entity_id, d.text)
                    for d in iter_record_documents(db_path, plan)
                    if hasattr(d, "text"))
    res = {"docs": n, "queries": len(queries), "identical_top10": 0,
           "same_set_top10": 0, "examples": []}
    for q in queries:
        for pref in (False, True):
            mine = ix.query(q, 10, prefix_truncate=pref)
            theirs = [r[0] for r in store.search_lexical(
                q, 10, prefix_truncate=pref)]
            res["identical_top10"] += mine == theirs
            res["same_set_top10"] += set(mine) == set(theirs)
            if mine != theirs and len(res["examples"]) < 3:
                res["examples"].append({"q": q, "prefix": pref,
                                        "independent": mine[:5],
                                        "store": theirs[:5]})
    res["comparisons"] = 2 * len(queries)
    ix.close()
    return res


def raw_columns_baseline(plan, db_path: str):
    """Ablation: BM25 over the entity table's own columns only (no
    folding of EAV / dimension / bridge evidence)."""
    import dataclasses

    from bm25_independent import IndependentBM25
    from record_documents import iter_record_documents
    raw = dataclasses.replace(plan, eav=(), dimensions=(), bridges=())
    ix = IndependentBM25()
    ix.add_many((d.entity_id, d.text)
                for d in iter_record_documents(db_path, raw)
                if hasattr(d, "text"))
    return ix
