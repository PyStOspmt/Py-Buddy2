"""Init-job lifecycle mixin for SystemContext: queue, run, cancel, observe (F33)."""

from fastapi import HTTPException
from typing import Dict, List, Any
import threading
import time
import uuid
from server_bundle import DomainBundle, _InitCancelled
from source_lifecycle import SourceState, source_version




class InitJobsMixin:
    def start_init_job(self, domain_id: str,
                       reanalyze: Dict[str, Any] | None = None
                       ) -> Dict[str, Any]:
        """Queue a background engine build for a registered domain.

        Returns the job record. Raises 400 for unknown domains, 409 when
        a live job already exists for the same source -- duplicate work
        on one source is a scheduling bug, not parallelism. `reanalyze`
        carries operator-chosen options {"llm_assist", "vector"} and
        forces a full rebuild: the projection cache is invalidated and
        the source-version pin is cleared so _bundle_needs_init cannot
        shortcut to the stale engine."""
        if domain_id not in self.domains:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown domain {domain_id!r}. Available: {', '.join(self.domains.keys())}.")
        bundle = self.domains[domain_id]
        with self._jobs_lock:
            for job in self.init_jobs.values():
                if job["domain_id"] == domain_id and job["status"] in ("queued", "running"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"Init job already {job['status']} for {domain_id!r}: {job['job_id']}")
            job_id = "init_%s" % uuid.uuid4().hex[:12]
            job = {
                "job_id": job_id,
                "domain_id": domain_id,
                "status": "queued",
                "stage": None,
                "created_at": time.time(),
                "finished_at": None,
                "error": None,
                # Live build events: the projection streams each step as
                # it happens so the UI can show the schema being built,
                # not just a finished diagram.
                "trace": [],
                "reanalyze": dict(reanalyze) if reanalyze else None,
            }
            self.init_jobs[job_id] = job
        t = threading.Thread(
            target=self._run_init_job, args=(job_id, bundle), daemon=True)
        t.start()
        return dict(job)

    def cancel_init_job(self, job_id: str) -> Dict[str, Any]:
        """Cooperatively cancel a job: the running stage finishes, the
        build unwinds at the next boundary and the source returns to
        PENDING -- a cancelled init is not a failed one."""
        with self._jobs_lock:
            job = self.init_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {job_id!r}.")
        if job["status"] in ("queued", "running"):
            bundle = self.domains[job["domain_id"]]
            bundle._cancel.set()
        return dict(job)

    def list_init_jobs(self) -> List[Dict[str, Any]]:
        with self._jobs_lock:
            return [dict(j) for j in self.init_jobs.values()]

    def _run_init_job(self, job_id: str, bundle: DomainBundle) -> None:
        with self._jobs_lock:
            job = self.init_jobs[job_id]
            job["status"] = "running"
        stage_cb = lambda s: self._set_job_stage(job_id, s)
        trace_cb = lambda ev: self._append_job_trace(job_id, ev)
        try:
            re_opts = self.init_jobs[job_id].get("reanalyze")
            if re_opts:
                stage_cb("reanalyze: cache invalidated")
                from source_projection import invalidate_cache
                if bundle.db_path:
                    invalidate_cache(bundle.db_path)
                # Force the engine path: a cleared version pin makes
                # _bundle_needs_init true even though a router exists.
                bundle.source_version = None
                if re_opts.get("llm_assist"):
                    bundle.llm_assist = True
            if self._bundle_needs_init(bundle):
                with bundle._init_lock:
                    if self._bundle_needs_init(bundle):
                        self._init_bundle_engine(bundle, progress=stage_cb,
                                                 trace_listener=trace_cb)
            if bundle.semantic is not None:
                stage_cb("semantic_index")
                try:
                    throttle = {"t": 0.0}

                    def _sem_progress(_ev, _th=throttle):
                        now = time.time()
                        if now - _th["t"] >= 2.0:
                            _th["t"] = now
                            p = bundle.semantic.status()["progress"]
                            trace_cb({
                                "kind": "semantic_index",
                                "label": "семантичний індекс",
                                "detail": f"{p['processed']}/{p['total']}"})

                    rep = bundle.semantic.sync_blocking(
                        reason="init",
                        should_cancel=bundle._cancel.is_set,
                        progress=_sem_progress)
                    if not rep.complete:
                        trace_cb({
                            "kind": "semantic_index",
                            "label": "семантичний індекс",
                            "detail": f"incomplete: {rep.error or 'cancelled'}"})
                except Exception as exc:
                    trace_cb({
                        "kind": "semantic_index",
                        "label": "семантичний індекс",
                        "detail": f"failed: {exc}"})
            if re_opts and re_opts.get("vector") and bundle.db_path:
                stage_cb("vector_index")
                try:
                    from vector_index import (OpenRouterEmbedder,
                                              get_vector_index)
                    key = getattr(self.runner, "api_key", None)
                    if key:
                        from source_projection import (
                            build_source_projection)
                        emb = OpenRouterEmbedder(key,
                                                 self.embedding_model)
                        # get_vector_index reuses a valid .vecidx
                        # sidecar -- same source stamp + model means the
                        # stored vectors ARE the answer, no paid calls.
                        idx = get_vector_index(
                            bundle.db_path,
                            build_source_projection(bundle.db_path),
                            emb, self.embedding_model)
                        if idx.stats.get("status") == "ready":
                            bundle.vector_index = idx
                            bundle.vector_embedder = emb
                        self._append_job_trace(job_id, {
                            "kind": "vector_index",
                            "label": "векторний індекс",
                            "detail": str(idx.stats)})
                    else:
                        self._append_job_trace(job_id, {
                            "kind": "vector_index",
                            "label": "векторний індекс",
                            "detail": "unavailable: no api key"})
                except Exception as exc:
                    self._append_job_trace(job_id, {
                        "kind": "vector_index", "label": "векторний "
                        "індекс", "detail": f"failed: {exc}"})
            cancelled = bundle._cancel.is_set()
            if cancelled:
                bundle._cancel.clear()
                # Honest state: a cancelled build produced nothing usable.
                bundle.state = SourceState.PENDING
                bundle.state_reason = "init cancelled before completion"
            with self._jobs_lock:
                job["status"] = "cancelled" if cancelled else "done"
        except _InitCancelled:
            bundle._cancel.clear()
            bundle.state = SourceState.PENDING
            bundle.state_reason = "init cancelled before completion"
            with self._jobs_lock:
                job["status"] = "cancelled"
        except Exception as exc:
            with self._jobs_lock:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            with self._jobs_lock:
                job["finished_at"] = time.time()

    def _set_job_stage(self, job_id: str, stage: str) -> None:
        with self._jobs_lock:
            job = self.init_jobs.get(job_id)
            if job is not None:
                job["stage"] = stage

    def _append_job_trace(self, job_id: str, ev: dict) -> None:
        with self._jobs_lock:
            job = self.init_jobs.get(job_id)
            if job is not None and len(job["trace"]) < 500:
                job["trace"].append(ev)

    @staticmethod
    def _mark_degraded(bundle: DomainBundle, reason: str) -> None:
        """Every failed preparation stage is recorded; the first cause is
        not silently overwritten by a later one."""
        bundle.state = SourceState.DEGRADED
        bundle.state_reason = f"{bundle.state_reason}; {reason}" \
            if bundle.state_reason else reason

    @staticmethod
    def _bundle_needs_init(bundle: DomainBundle) -> bool:
        if bundle.fast_router is None:
            return True
        if bundle.db_path:
            return source_version(bundle.db_path) != bundle.source_version
        return False

    @staticmethod
    def _describe_from_profile(bundle: DomainBundle) -> str:
        """F45: a domain-free description generated from the source's own
        profile -- table and row counts only, no invented domain prose."""
        if bundle.snapshot is None:
            return ""
        tables = bundle.snapshot.tables
        total_rows = sum(t.row_count for t in tables.values())
        return (f"SQLite-джерело з {len(tables)} таблиць, "
                f"{total_rows} рядків загалом "
                f"(опис згенеровано з профілю джерела).")

    @staticmethod
    def _check_cancel(bundle: DomainBundle, stage: str) -> None:
        """Cooperative cancel point between init stages (F33)."""
        if bundle._cancel.is_set():
            raise _InitCancelled(stage)
