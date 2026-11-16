"""
P1.2 regressions: the behavioural fiduciary probe at source registration.

Previously the FIC machinery was evaluation-only: a static lint over
source text that could never observe the serving path. The probe must

  * detect commercial channels through the declared concept spellings,
    linked to entity ids empirically;
  * certify the production ordering when the permuted commercial state
    moves nothing;
  * return VIOLATION when an injected biased scorer leaks -- proof the
    probe detects rather than rubber-stamps;
  * report NO_COMMERCIAL_CHANNEL / UNVERIFIED honestly instead of
    manufacturing a pass.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))



def _db(tmp_path, n_items=8, commercial=True):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    rows = [
        (f"p{i}", f"Товар {i}", "cat" + str(i % 2), 100.0 * (i + 1), 5, 50.0 + i)
        for i in range(n_items)
    ]
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", rows)
    if commercial:
        conn.execute(
            "CREATE TABLE commercial (product_id TEXT, sponsored INTEGER, "
            "commission_pct REAL)")
        conn.executemany(
            "INSERT INTO commercial VALUES (?,?,?)",
            [(f"p{i}", i % 2, float(i)) for i in range(n_items)])
    conn.commit()
    conn.close()
    return str(db)


def test_probe_certifies_leak_free_ordering(tmp_path):
    from fiduciary_probe import probe_source
    res = probe_source(_db(tmp_path))
    assert res["status"] == "CERTIFIED"
    assert res["fid"] == 0.0
    assert res["evaluated_items"] >= 2
    concepts = {c["concept"] for c in res["channels"]}
    assert "sponsorship" in concepts and "commission" in concepts
    assert res["moved_items"] == []


def test_probe_detects_a_biased_scorer(tmp_path):
    from fiduciary_probe import probe_source
    db = _db(tmp_path)

    def biased(db_path, pids):
        conn = sqlite3.connect(db_path)
        try:
            marks = ",".join("?" * len(pids))
            base = {r[0]: -float(r[1]) for r in conn.execute(
                f"SELECT id, price FROM products WHERE id IN ({marks})",
                tuple(pids))}
            sp = {r[0] for r in conn.execute(
                "SELECT product_id FROM commercial WHERE sponsored = 1")}
        finally:
            conn.close()
        return {p: s + (500.0 if p in sp else 0.0)
                for p, s in base.items()}

    res = probe_source(db, scorer=biased)
    assert res["status"] == "VIOLATION"
    assert res["fid"] > 0
    assert res["moved_items"], "a leak must name the items that moved"


def test_no_commercial_channel_is_reported_not_passed(tmp_path):
    from fiduciary_probe import probe_source
    res = probe_source(_db(tmp_path, commercial=False))
    assert res["status"] == "NO_COMMERCIAL_CHANNEL"
    assert res["channels"] == []


def test_degenerate_catalogue_is_unverified(tmp_path):
    from fiduciary_probe import probe_source
    res = probe_source(_db(tmp_path, n_items=1))
    assert res["status"] == "UNVERIFIED"
    assert res["evaluated_items"] < 2


def test_probe_does_not_mutate_source(tmp_path):
    from fiduciary_probe import probe_source
    db = _db(tmp_path)
    conn = sqlite3.connect(db)
    before = conn.execute(
        "SELECT product_id, sponsored FROM commercial ORDER BY product_id"
    ).fetchall()
    conn.close()
    probe_source(db)
    conn = sqlite3.connect(db)
    after = conn.execute(
        "SELECT product_id, sponsored FROM commercial ORDER BY product_id"
    ).fetchall()
    conn.close()
    assert before == after


def test_init_populates_bundle_fiduciary(tmp_path):
    # Wiring check: the probe runs as an init stage and lands on the
    # bundle. The probe itself is exercised above; here only the stage
    # contract matters, so the heavy stages are satisfied as usual on a
    # tiny fixture.
    from server import DomainBundle, ctx
    db = _db(tmp_path)
    bundle = DomainBundle("fic_probe_src", "probe test", db, None, "test")
    ctx.register_domain_bundle(bundle)
    ctx._init_bundle_engine(bundle)
    assert bundle.fiduciary is not None
    assert bundle.fiduciary["status"] in (
        "CERTIFIED", "VIOLATION", "UNVERIFIED", "NO_COMMERCIAL_CHANNEL")
    # This fixture has a commercial table and a leak-free ordering:
    assert bundle.fiduciary["status"] == "CERTIFIED"
    assert {c["concept"] for c in bundle.fiduciary["channels"]} >= {
        "sponsorship", "commission"}
    try:
        del ctx.domains["fic_probe_src"]
    except KeyError:
        pass
