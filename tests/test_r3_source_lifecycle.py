"""
Source lifecycle, content stamping, and state-machine regressions.

Failing-before counterexamples the audit described:

- a registered source whose file disappears was silently served from the
  unrelated default PC database (source substitution);
- caches keyed only on the absolute path trusted stale data after the
  database file was replaced;
- initialization had no ready/degraded/failed state and no protection
  against concurrent duplicate builds.
"""

import json
import os
import sqlite3
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))

from fastapi import HTTPException


def _make_db(path, rows):
    # Replace the file wholesale -- the scenario under test is a source
    # swap at the same path, not an in-place migration.
    if os.path.exists(path):
        os.unlink(path)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL)")
    conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def _offline_init_runner(monkeypatch):
    """Init/lazy-build paths must not depend on live provider latency.

    Every test in this file exercises lifecycle logic, not the model:
    with ``api_key`` unset the ontology and question-bank inductors take
    their documented offline branch (generic bank / deterministic
    induce), so an init job cannot stall past its timeout waiting on a
    network call."""
    try:
        from server import ctx
    except Exception:
        return
    monkeypatch.setattr(ctx.runner, "api_key", None)


class TestVersionPinnedCache:
    """source_lifecycle: a cache written for one source version must not
    be trusted after the file changes, and corrupt caches recompute."""

    def test_stale_cache_is_ignored_after_source_change(self, tmp_path):
        from source_lifecycle import (
            read_versioned_cache,
            source_version,
            write_versioned_cache,
        )

        db = tmp_path / "src.db"
        _make_db(db, [("a", "A", "cat1", 1.0)])
        v1 = source_version(str(db))
        cache = tmp_path / "cache.json"
        write_versioned_cache(str(cache), v1, {"marker": "v1"})

        assert read_versioned_cache(str(cache), v1) == {"marker": "v1"}

        # Replace the source: same path, different content.
        _make_db(db, [("a", "A", "cat1", 1.0), ("b", "B", "cat2", 2.0),
                      ("extra padding row", "C", "cat3", 3.0)])
        v2 = source_version(str(db))
        assert v2 != v1
        assert read_versioned_cache(str(cache), v2) is None

    def test_corrupt_cache_is_ignored(self, tmp_path):
        from source_lifecycle import read_versioned_cache, source_version

        db = tmp_path / "src.db"
        _make_db(db, [("a", "A", "cat1", 1.0)])
        cache = tmp_path / "cache.json"
        cache.write_text("{not json", encoding="utf-8")
        assert read_versioned_cache(str(cache), source_version(str(db))) is None

    def test_cache_write_is_atomic_complete_envelope(self, tmp_path):
        from source_lifecycle import write_versioned_cache

        cache = tmp_path / "cache.json"
        write_versioned_cache(str(cache), (123, 456), {"x": 1})
        env = json.loads(cache.read_text(encoding="utf-8"))
        assert env["source_version"] == [123, 456]
        assert env["payload"] == {"x": 1}


class TestOntologyCacheVersioning:
    """The ontology cache used to be keyed on the absolute path only, so a
    replaced database inherited the previous source's induced profile."""

    def test_induce_rejects_cache_from_another_version(self, tmp_path):
        from ontograph import SemanticOntologyInductor
        from source_lifecycle import write_versioned_cache

        cache = tmp_path / "onto.json"
        # A cache written for a different source version carries bogus data.
        write_versioned_cache(str(cache), (111, 111),
                              {"domain_concept": "BOGUS STALE DOMAIN",
                               "macro_departments": {"Stale": ["x"]}})

        profile = SemanticOntologyInductor.induce(
            categories=["alpha", "beta", "gamma"],
            sample_rows_by_cat={},
            discovered_params={},
            runner=None,
            cache_path=str(cache),
            source_version=(222, 222),
        )
        assert profile.domain_concept != "BOGUS STALE DOMAIN"

        # Same version: the cache is honoured (bounded recompute cost).
        write_versioned_cache(str(cache), (333, 333),
                              {"domain_concept": "PINNED DOMAIN",
                               "macro_departments": {"Pinned": ["alpha"]}})
        profile = SemanticOntologyInductor.induce(
            categories=["alpha"],
            sample_rows_by_cat={},
            discovered_params={},
            runner=None,
            cache_path=str(cache),
            source_version=(333, 333),
        )
        assert profile.domain_concept == "PINNED DOMAIN"


class TestDomainBundleLifecycle:
    """server.get_domain_bundle: explicit states, no source substitution,
    version pinning, single concurrent init."""

    @pytest.fixture
    def ctx(self):
        from server import ctx
        return ctx

    def _register(self, ctx, domain_id, db_path):
        from server import DomainBundle
        bundle = DomainBundle(domain_id, f"test {domain_id}", db_path, None, "test")
        ctx.register_domain_bundle(bundle)
        return bundle

    def test_missing_source_fails_closed_no_substitution(self, ctx, tmp_path):
        ghost = str(tmp_path / "does_not_exist.db")
        self._register(ctx, "ghost_source", ghost)
        with pytest.raises(HTTPException) as exc:
            ctx.get_domain_bundle("ghost_source")
        assert exc.value.status_code == 503
        # The bundle is marked failed, never silently repointed at another db.
        bundle = ctx.domains["ghost_source"]
        assert bundle.state == "failed"
        assert bundle.fast_router is None
        ctx.domains.pop("ghost_source")

    def test_db_path_none_bundle_never_substitutes_default_db(self, ctx):
        from server import DB_PATH
        self._register(ctx, "memory_only", None)
        bundle = ctx.get_domain_bundle("memory_only")
        # Graph-only bundle: degraded, and the agent's db is None -- not the
        # unrelated PC database the old effective_db fallback injected.
        assert bundle.state == "degraded"
        assert bundle.agent.db_path is None
        assert bundle.agent.db_path != DB_PATH
        ctx.domains.pop("memory_only")

    def test_source_replacement_rebuilds_engine(self, ctx, tmp_path):
        db = tmp_path / "swap.db"
        _make_db(db, [("p1", "Widget", "cat1", 10.0)])
        bundle = self._register(ctx, "swap_src", str(db))
        ctx.get_domain_bundle("swap_src")
        agent_v1 = bundle.agent
        version_v1 = bundle.source_version
        assert bundle.state == "ready"

        # Replace the file at the same path: same name, new content.
        _make_db(db, [(f"p{i}", f"Widget {i}", "cat2", 10.0 + i)
                      for i in range(50)])
        ctx.get_domain_bundle("swap_src")
        assert bundle.source_version != version_v1
        assert bundle.agent is not agent_v1
        ctx.domains.pop("swap_src")

    def test_concurrent_init_builds_once(self, ctx, tmp_path, monkeypatch):

        db = tmp_path / "conc.db"
        _make_db(db, [("p1", "Widget", "cat1", 10.0)])
        bundle = self._register(ctx, "conc_src", str(db))

        calls = {"n": 0}
        real_init = ctx._init_bundle_engine

        def counting_init(b):
            calls["n"] += 1
            return real_init(b)

        monkeypatch.setattr(ctx, "_init_bundle_engine", counting_init)
        errors = []

        def worker():
            try:
                ctx.get_domain_bundle("conc_src")
            except Exception as e:  # noqa: BLE001 - test captures for assert
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert calls["n"] == 1
        assert bundle.state == "ready"
        ctx.domains.pop("conc_src")

    def test_unreadable_source_is_degraded_not_failed_silently(self, ctx, tmp_path):
        bad = tmp_path / "corrupt.db"
        bad.write_bytes(b"this is not a sqlite database at all")
        self._register(ctx, "corrupt_src", str(bad))
        bundle = ctx.get_domain_bundle("corrupt_src")
        assert bundle.state == "degraded"
        assert bundle.state_reason
        ctx.domains.pop("corrupt_src")

    def test_domains_endpoint_exposes_state(self, ctx, tmp_path):
        ghost = str(tmp_path / "gone.db")
        self._register(ctx, "ghost2", ghost)
        from server import get_domains
        data = get_domains()
        entry = next(d for d in data["domains"] if d["id"] == "ghost2")
        assert entry["state"] in ("pending", "degraded", "failed", "ready")
        ctx.domains.pop("ghost2")


class TestAsyncInitJobs:
    """F33: background init jobs -- observable status, single live job
    per source, cooperative cancellation that unwinds to PENDING."""

    @pytest.fixture
    def ctx(self):
        from server import ctx
        return ctx

    def _register(self, ctx, domain_id, db_path):
        from server import DomainBundle
        bundle = DomainBundle(domain_id, f"test {domain_id}", db_path, None, "")
        ctx.register_domain_bundle(bundle)
        return bundle

    def _wait(self, ctx, job_id, timeout=30.0):
        import time as _t
        deadline = _t.time() + timeout
        while _t.time() < deadline:
            job = ctx.init_jobs.get(job_id)
            if job and job["status"] not in ("queued", "running"):
                return job
            _t.sleep(0.05)
        raise AssertionError(f"job {job_id} did not finish in {timeout}s")

    def test_async_init_job_completes_and_reports(self, ctx, tmp_path):
        db = tmp_path / "async.db"
        _make_db(db, [("p1", "Widget", "cat1", 10.0)])
        bundle = self._register(ctx, "async_src", str(db))
        job = ctx.start_init_job("async_src")
        assert job["status"] in ("queued", "running")
        done = self._wait(ctx, job["job_id"])
        assert done["status"] == "done"
        assert done["finished_at"] is not None
        assert done["error"] is None
        assert bundle.state == "ready"
        assert bundle.fast_router is not None
        assert any(j["job_id"] == job["job_id"] for j in ctx.list_init_jobs())
        ctx.domains.pop("async_src")

    def test_second_live_job_for_same_source_is_rejected(self, ctx, tmp_path):
        db = tmp_path / "dup.db"
        _make_db(db, [("p1", "Widget", "cat1", 10.0)])
        self._register(ctx, "dup_src", str(db))
        import threading as _th

        import server as server_mod
        hold = _th.Event()
        real_init = ctx._init_bundle_engine
        def blocking_init(b, progress=None, trace_listener=None):
            hold.wait(10)
            return real_init(b, progress=progress,
                             trace_listener=trace_listener)
        server_mod.ctx._init_bundle_engine = blocking_init
        try:
            job1 = ctx.start_init_job("dup_src")
            # Wait until the worker is actually inside the build.
            for _ in range(200):
                if ctx.init_jobs[job1["job_id"]]["status"] == "running":
                    break
                import time as _t; _t.sleep(0.01)
            with pytest.raises(HTTPException) as exc:
                ctx.start_init_job("dup_src")
            assert exc.value.status_code == 409
            hold.set()
            done = self._wait(ctx, job1["job_id"])
            assert done["status"] == "done"
        finally:
            hold.set()
            server_mod.ctx._init_bundle_engine = real_init
            ctx.domains.pop("dup_src")

    def test_cancel_unwinds_to_pending_not_failed(self, ctx, tmp_path):
        db = tmp_path / "cancel.db"
        _make_db(db, [("p1", "Widget", "cat1", 10.0)])
        bundle = self._register(ctx, "cancel_src", str(db))
        bundle._cancel.set()
        job = ctx.start_init_job("cancel_src")
        done = self._wait(ctx, job["job_id"])
        assert done["status"] == "cancelled"
        # A cancelled build is not a failed source -- it never ran.
        assert bundle.state == "pending"
        assert "cancelled" in bundle.state_reason
        assert bundle.fast_router is None
        ctx.domains.pop("cancel_src")

    def test_cancel_job_marks_event(self, ctx, tmp_path):
        db = tmp_path / "cjob.db"
        _make_db(db, [("p1", "Widget", "cat1", 10.0)])
        bundle = self._register(ctx, "cjob_src", str(db))
        import threading as _th

        import server as server_mod
        hold = _th.Event()
        real_init = ctx._init_bundle_engine
        def blocking_init(b, progress=None, trace_listener=None):
            hold.wait(10)
            return real_init(b, progress=progress,
                             trace_listener=trace_listener)
        server_mod.ctx._init_bundle_engine = blocking_init
        try:
            job = ctx.start_init_job("cjob_src")
            for _ in range(200):
                if ctx.init_jobs[job["job_id"]]["status"] == "running":
                    break
                import time as _t; _t.sleep(0.01)
            ctx.cancel_init_job(job["job_id"])
            assert bundle._cancel.is_set()
            hold.set()
            done = self._wait(ctx, job["job_id"])
            # The flag landed mid-build; the next boundary unwinds to
            # cancelled (or the build already finished -- both are
            # honest statuses, never a silent half-state).
            assert done["status"] in ("cancelled", "done")
            if done["status"] == "cancelled":
                assert bundle.state == "pending"
        finally:
            hold.set()
            server_mod.ctx._init_bundle_engine = real_init
            ctx.domains.pop("cjob_src")
