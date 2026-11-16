"""Per-source semantic index lifecycle handle (Retrieval Fabric service).

One SemanticIndex per domain source: it owns the ``<db_path>.arf``
VectorStore sidecar, builds it from the domain-independent source
projection, tracks fresh/stale state through a deterministic file
stamp, and serves search without silent fallbacks. Embedding passes
are serialised process-wide (_EMBED_SEM): one CPU-heavy embedder at
a time across all sources. Status/search dict shapes live in
``semantic_service_io``.
"""
import logging
import os
import shutil
import sqlite3
import threading
from collections.abc import Callable

import semantic_hyde as hyde_mod
import semantic_service_io as ssio
from embedders import SemaphoreGatedEmbedder
from semantic_search import PRODUCT_LEXICAL_WEIGHT
from semantic_search import search as semantic_search
from semantic_sync import SyncReport, sync_source
from source_lifecycle import source_stamp
from source_projection import build_source_projection
from vector_store import StoreCorrupt, StoreMismatch, VectorStore

logger = logging.getLogger(__name__)

_EMBED_SEM = threading.Semaphore(1)

MISMATCH_REASON = "embedding model or dimension changed; rebuild required"
_K_MIN, _K_MAX = 1, 50


class SemanticIndex:
    """Lifecycle: absent -> building -> ready|stale|failed, plus
    `unavailable` for sources that can never be indexed."""

    def __init__(self, domain_id: str, db_path: str | None, *,
                 embedder_factory: Callable,
                 index_dir: str | None = None,
                 llm_generate: Callable | None = None):
        self.domain_id = domain_id
        self.db_path = db_path
        self._embedder_factory = embedder_factory
        self.index_dir = index_dir or (f"{db_path}.arf"
                                       if db_path else None)
        # HyDE hypothesis generator (prompt -> raw LLM text); None =
        # the mode degrades visibly to plain hybrid.
        self._llm_generate = llm_generate
        self._exemplar_list: list | None = None
        self._exemplar_gen: int = -1
        self._exemplar_calls = 0      # test-visible: one hashes() per generation
        self._lock = threading.RLock()
        self._store: VectorStore | None = None
        self._open_error: tuple | None = None
        self._embedder = None
        self._embedder_error: str | None = None
        self._proj = None
        self._proj_error: str | None = None
        self._sync_thread: threading.Thread | None = None
        self._sync_reason: str | None = None
        self._progress = {"processed": 0, "total": 0, "embedded": 0}
        self._last_sync: dict | None = None
        self._sync_failed: str | None = None
        self._last_peek: dict | None = None

    # -- lazy dependencies ----------------------------------------------
    def _project(self):
        with self._lock:
            if self._proj is None and self._proj_error is None:
                if not self.db_path or not os.path.exists(self.db_path):
                    self._proj_error = "no source database path"
                else:
                    try:
                        self._proj = build_source_projection(self.db_path)
                    except Exception as exc:  # noqa: BLE001 -- a projection build failure degrades to `unavailable`, never raises
                        self._proj_error = (f"source projection failed: "
                                            f"{type(exc).__name__}: {exc}")
            return self._proj

    def _ensure_embedder(self):
        with self._lock:
            if self._embedder is None and self._embedder_error is None:
                try:
                    emb = self._embedder_factory()
                    _ = emb.dim  # force the lazy weight load now
                    self._embedder = emb
                except Exception as exc:  # noqa: BLE001 -- any load failure is reported as `unavailable`
                    self._embedder_error = (f"embedder cannot load: "
                                            f"{type(exc).__name__}: {exc}")
            return self._embedder

    def _ensure_store(self):
        with self._lock:
            if self._store is None and self._open_error is None:
                emb = self._ensure_embedder()
                if emb is None or self.index_dir is None:
                    return None
                try:
                    self._store = VectorStore.open(
                        self.index_dir, model_id=emb.model_id,
                        dim=emb.dim)
                except StoreMismatch:
                    self._open_error = ("mismatch", MISMATCH_REASON)
                except StoreCorrupt as exc:
                    self._open_error = ("corrupt", f"store corrupt: {exc}")
                except Exception as exc:  # noqa: BLE001 -- store open failure is typed state, not an exception
                    self._open_error = ("error",
                                        f"{type(exc).__name__}: {exc}")
            return self._store

    # -- state ------------------------------------------------------------
    def _running(self) -> bool:
        t = self._sync_thread
        return t is not None and t.is_alive()

    def _has_manifest(self) -> bool:
        return bool(self.index_dir and os.path.isfile(
            os.path.join(self.index_dir, "manifest.sqlite")))

    def _peek_embedder(self):
        with self._lock:
            if self._embedder is not None:
                return self._embedder
            if self._embedder_error is not None:
                return None
            try:
                emb = self._embedder_factory()
                _ = emb.model_id
                if hasattr(emb, "peek_dim"):
                    emb.peek_dim()
                else:
                    _ = emb.dim
                return emb
            except Exception as exc:  # noqa: BLE001 -- embedder init failure degrades to `unavailable`
                self._embedder_error = (f"embedder cannot load: "
                                        f"{type(exc).__name__}: {exc}")
                return None

    def _peek_manifest_status(self) -> dict | None:
        with self._lock:
            if self._store is not None or not self._has_manifest():
                return None
            emb = self._peek_embedder()
            if emb is None:
                return None
            known_dim = (emb.peek_dim() if hasattr(emb, "peek_dim")
                         else emb.dim)
            try:
                snap = ssio.peek_manifest(
                    self.index_dir,
                    expected_model_id=emb.model_id,
                    expected_dim=known_dim,
                )
                self._last_peek = snap
                return snap
            except StoreMismatch:
                self._open_error = ("mismatch", MISMATCH_REASON)
            except StoreCorrupt as exc:
                self._open_error = ("corrupt", f"store corrupt: {exc}")
            except Exception as exc:  # noqa: BLE001 -- manifest inspect failure is typed state
                self._open_error = ("error", f"{type(exc).__name__}: {exc}")
            return None

    def _meta(self, key: str):
        if self._store is None:
            if not self._has_manifest():
                return None
            snap = self._last_peek or self._peek_manifest_status()
            return snap["meta"].get(key) if snap is not None else None
        return self._store.get_meta(key)

    def _state(self) -> tuple:
        """(state, reason) -- derived from stored meta, never cached."""
        self._last_peek = None
        if not self.db_path or not self.index_dir:
            return "unavailable", "no source database path"
        proj = self._project()
        if proj is None:
            return "unavailable", self._proj_error
        if proj.entity_table is None:
            return "unavailable", "no entity table in source"
        if self._embedder_error:
            return "unavailable", self._embedder_error
        if self._running():
            return "building", self._sync_reason or "sync running"
        if self._open_error:
            return "failed", self._open_error[1]
        if self._store is None:
            if not self._has_manifest():
                return "absent", "no complete sync recorded"
            snap = self._peek_manifest_status()
            if snap is None:
                return ("unavailable" if self._embedder_error
                        else "failed",
                        self._embedder_error or self._open_error[1])
            if self._sync_failed:
                return "failed", self._sync_failed
            indexed = snap["meta"].get("source_stamp")
            if indexed is None:
                return "absent", "no complete sync recorded"
            if indexed == source_stamp(self.db_path):
                return "ready", "index up to date"
            return "stale", "source changed since last sync"
        if self._sync_failed:
            return "failed", self._sync_failed
        indexed = self._meta("source_stamp")
        if indexed is None:
            return "absent", "no complete sync recorded"
        if indexed == source_stamp(self.db_path):
            return "ready", "index up to date"
        return "stale", "source changed since last sync"

    def status(self) -> dict:
        return ssio.status_of(self)

    # -- sync --------------------------------------------------------------
    def _rebuild_dir(self) -> None:
        """Derived data only: assert the sidecar shape before any
        removal -- a SemanticIndex may never delete source files."""
        adir = os.path.abspath(self.index_dir)
        adb = os.path.abspath(self.db_path)
        if not adir.endswith(".arf"):
            raise ValueError(f"refusing to delete {adir!r}")
        if os.path.dirname(adir) != os.path.dirname(adb):
            raise ValueError(
                f"index dir {adir!r} is not a sidecar of {adb!r}")
        with self._lock:
            if self._store is not None:
                self._store.close()
                self._store = None
            self._open_error = None
            if os.path.isdir(adir):
                shutil.rmtree(adir)

    def sync_blocking(self, *, reason: str, rebuild: bool = False,
                      should_cancel=None, progress=None) -> SyncReport:
        """One full pass. Failures are recorded in state and in the
        returned report -- they never raise past this boundary."""
        rep = SyncReport()
        try:
            if rebuild:
                self._rebuild_dir()
            proj = self._project()
            if proj is None or proj.entity_table is None:
                rep.error = self._proj_error or "no entity table"
                self._sync_failed = rep.error
                return rep
            stamp = source_stamp(self.db_path)   # A2: BEFORE the pass
            with self._lock:
                self._progress = {"processed": 0,
                                  "total": self._source_rows(proj),
                                  "embedded": 0}
            store = self._ensure_store()
            emb = self._ensure_embedder()
            if store is None or emb is None:
                rep.error = (self._embedder_error
                             or (self._open_error[1] if self._open_error
                                 else "store error"))
                self._sync_failed = rep.error
                return rep
            rep = sync_source(
                self.db_path, proj, store,
                SemaphoreGatedEmbedder(emb, _EMBED_SEM),
                should_cancel=should_cancel,
                progress=self._wrap_progress(progress),
                source_stamp=stamp)
            with self._lock:
                self._last_sync = rep.to_dict()
                # a clean cancel is not a failure: the state derives
                # from stored meta and returns to absent/stale
                self._sync_failed = rep.error if rep.error else None
                if rep.complete:
                    # the last batch commits in _finish without a
                    # progress event -- post the real final counts
                    self._progress["processed"] = rep.source_rows
                    self._progress["embedded"] = (rep.inserted
                                                  + rep.updated)
            return rep
        except StoreMismatch:
            rep.error = MISMATCH_REASON
        except StoreCorrupt as exc:
            rep.error = f"store corrupt: {exc}"
        except Exception as exc:  # noqa: BLE001 -- sync failures land in the report + state, never raise
            rep.error = f"{type(exc).__name__}: {exc}"
        self._sync_failed = rep.error
        return rep

    def _source_rows(self, proj) -> int:
        try:
            conn = sqlite3.connect(
                f"file:{os.path.abspath(self.db_path)}?mode=ro", uri=True)
            try:
                return int(conn.execute(
                    f'SELECT COUNT(*) FROM "{proj.entity_table}"'
                ).fetchone()[0])
            finally:
                conn.close()
        except sqlite3.Error:
            return 0

    def _wrap_progress(self, progress):
        def _cb(ev: dict) -> None:
            with self._lock:
                p = self._progress
                p["processed"] = ev.get("processed", p["processed"])
                p["embedded"] = ((ev.get("inserted") or 0)
                                 + (ev.get("updated") or 0))
                if ev.get("total"):
                    p["total"] = ev["total"]
            if progress:
                progress(ev)
        return _cb

    def start_sync(self, *, reason: str, rebuild: bool = False) -> dict:
        with self._lock:
            if self._running():
                return {"started": False, "state": "building"}
            self._sync_failed = None
            self._sync_reason = reason
            t = threading.Thread(target=self._run_sync,
                                 args=(reason, rebuild), daemon=True)
            self._sync_thread = t
            t.start()
            return {"started": True, "state": "building"}

    def _run_sync(self, reason: str, rebuild: bool) -> None:
        try:
            self.sync_blocking(reason=reason, rebuild=rebuild)
        finally:
            self._sync_reason = None

    def ensure_fresh(self) -> bool:
        state, _ = self._state()
        if state == "stale" and not self._running():
            self.start_sync(reason="source_changed")
        return state == "ready"

    # -- search -----------------------------------------------------------
    def search(self, query: str, *, k: int = 10,
               mode: str = "hybrid", hyde: bool = False) -> dict:
        k = max(_K_MIN, min(_K_MAX, int(k)))
        notes: list[str] = []
        requested_mode = "hyde" if hyde else None
        # the response's index block reflects the state AT SEARCH
        # TIME -- a triggered refresh must not retro-freshen it
        st = self.status()
        state = st["state"]
        if state == "absent":
            self.start_sync(reason="lazy")
        if state in ("absent", "building"):
            return ssio.result_of(self, "not_ready", query, mode, k,
                                  [], notes, st)
        if state in ("unavailable", "failed"):
            return ssio.result_of(self, state, query, mode, k,
                                  [], notes, st)
        if state == "stale":
            self.start_sync(reason="source_changed")
            notes.append("index stale; refresh started, serving "
                         "the previous generation")
        proj = self._project()
        if proj is not None and proj.entity_ambiguous:
            notes.append("entity table is ambiguous; alternatives: "
                         + ", ".join(proj.entity_alternatives))
        if hyde:
            ranking = hyde_mod.try_hyde(self, query, notes)
            if ranking is not None:
                out = ssio.result_of(
                    self, "ok", query, "hyde", k,
                    ssio.hyde_hits(self, ranking, k), notes, st)
                out["requested_mode"] = "hyde"
                out["hypothesis"] = {
                    "text": ranking.hypothesis,
                    "generated_by": "llm",
                    "note": ("model-written hypothesis; steers "
                             "retrieval only, never evidence")}
                return out
        try:
            res = semantic_search(self._ensure_store(),
                                  self._ensure_embedder(), query,
                                  k=k, mode=mode,
                                  lexical_weight=PRODUCT_LEXICAL_WEIGHT)
        except Exception as exc:  # noqa: BLE001 -- a search failure returns status=failed, never raises
            notes.append(f"{type(exc).__name__}: {exc}")
            return ssio.result_of(self, "failed", query, mode, k,
                                  [], notes, st)
        notes.extend(res.notes)
        hits = ssio.named_hits(self, res)
        out = ssio.result_of(self, "ok", query, mode, k, hits, notes,
                             st)
        if requested_mode:
            out["requested_mode"] = requested_mode
        out["confidence"] = {"top1_z": res.top1_z,
                             "margin": res.margin,
                             "top1_sim": res.top1_sim}
        return out

    def close(self) -> None:
        with self._lock:
            if self._store is not None:
                self._store.close()
                self._store = None
