"""
Atomic rebuild safety and ingestion loss-counter regressions.

Failing-before counterexamples:

- ingest/generator scripts deleted the existing target database before
  starting work -- one failed run destroyed previous evidence;
- init_database's __main__ called create_ecommerce_database without the
  required hypergraph_json_path argument (TypeError on a fresh run);
- migrate_commercial pointed DATA one directory too deep;
- make_fixtures hardcoded an author-machine absolute path;
- over-long EAV values were dropped without any loss counter.
"""

import os
import sqlite3
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "data"))

import ingest_real_data as ingest
import init_database
import make_fixtures
import migrate_commercial


class TestBuildTargetSafety:
    def test_existing_target_refused_without_rebuild(self, tmp_path):
        target = tmp_path / "out.db"
        target.write_bytes(b"previous evidence")
        with pytest.raises(FileExistsError):
            ingest.prepare_build_target(str(target), overwrite=False)
        # The refusal must not have touched the existing file.
        assert target.read_bytes() == b"previous evidence"

    def test_publish_is_atomic_and_keeps_backup(self, tmp_path):
        target = tmp_path / "out.db"
        target.write_bytes(b"old db")
        build = tmp_path / "out.db.build"
        build.write_bytes(b"new db")
        ingest.publish_build_target(str(build), str(target))
        assert target.read_bytes() == b"new db"
        # Rollback path: the previous target survives as .bak.
        assert (tmp_path / "out.db.bak").read_bytes() == b"old db"
        assert not build.exists()


class TestEavLossCounters:
    def test_dropped_values_are_counted_per_reason(self):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE product_eav_meta"
                     "(product_id TEXT, meta_key TEXT, meta_value TEXT)")
        w = ingest.EavWriter(conn)
        w.add("p1", "k_ok", "fine")
        w.add("p1", "k_empty", "   ")
        w.add("p1", "k_long", "x" * 241)
        w.add("p1", "k_none", None)   # None is "absent", not a loss
        w.flush()
        assert w.dropped_empty == 1
        assert w.dropped_long == 1
        rows = conn.execute("SELECT meta_key FROM product_eav_meta").fetchall()
        assert rows == [("k_ok",)]
        conn.close()


class TestScriptEntrypoints:
    def test_pc_initializer_callable_without_json_arg(self, tmp_path):
        # F36: __main__ used to call create_ecommerce_database(db_file) while
        # the second parameter was required -- a guaranteed TypeError.
        init_database.create_ecommerce_database(str(tmp_path / "cat.db"))
        assert os.path.exists(tmp_path / "cat.db")

    def test_pc_initializer_refuses_overwrite(self, tmp_path):
        target = tmp_path / "cat.db"
        init_database.create_ecommerce_database(str(target))
        with pytest.raises(FileExistsError):
            init_database.create_ecommerce_database(str(target))
        init_database.create_ecommerce_database(str(target), overwrite=True)

    def test_migrate_commercial_points_at_real_data_dir(self):
        # The old DATA joined ROOT/data -- one directory too deep, so the
        # script silently hit nonexistent paths.
        assert os.path.isdir(migrate_commercial.DATA)
        assert os.path.basename(migrate_commercial.DATA) == "data"

    def test_make_fixtures_root_is_portable(self):
        assert make_fixtures.ROOT == ROOT
        assert os.path.isdir(os.path.join(make_fixtures.ROOT, "data"))


class TestDependencyManifest:
    def test_runtime_and_dev_manifests_exist_and_pin(self):
        for fname in ("requirements.txt", "requirements-dev.txt"):
            path = os.path.join(ROOT, fname)
            assert os.path.exists(path), fname
            body = open(path, encoding="utf-8").read()
            # Pinned versions, not floating ranges.
            assert "==" in body
            for token in ("latest", ">=", "*"):
                for line in body.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("-"):
                        assert token not in line.split("#")[0], (fname, line)


def test_bootstrap_demo_databases_builds_missing_and_preserves_existing(tmp_path):
    """Fresh clones auto-build missing or 0-byte demo DBs without overwriting existing ones."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import bootstrap_demo_dbs

    built = bootstrap_demo_dbs.ensure_demo_databases(str(tmp_path), rebuild=False)
    assert len(built) == 5
    for name in (
        "catalog.db",
        "auto_parts.db",
        "real_pharmacy.db",
        "marketplace_rozetka.db",
        "jury_zoo.db",
    ):
        db_file = tmp_path / "data" / name
        assert db_file.exists() and db_file.stat().st_size > 0

    sentinel = tmp_path / "data" / "auto_parts.db"
    before = sentinel.read_bytes()
    assert bootstrap_demo_dbs.ensure_demo_databases(str(tmp_path), rebuild=False) == []
    assert sentinel.read_bytes() == before

    # An accidental 0-byte sqlite3 stub is repaired automatically on next check.
    sentinel.write_bytes(b"")
    repaired = bootstrap_demo_dbs.ensure_demo_databases(str(tmp_path), rebuild=False)
    assert repaired == [str(sentinel)]
    assert sentinel.stat().st_size > 0
