"""
Tests for the paired scale-envelope protocol (F27/F34).

The properties under test are the protocol guarantees, not the numbers:
nested corpora, an identical query set at every tier, truth transitions
reported rather than absorbed into a drifting denominator, and build
time kept separate from query latency.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from scale_envelope import (
    _permutation_ids,
    make_subset_db,
    run_envelope,
)


def _mini_db(path: str, n: int = 30) -> str:
    """A small but schema-faithful catalogue."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE merchants (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "rating REAL NOT NULL, delivery_days INTEGER NOT NULL, "
        "is_official INTEGER NOT NULL)")
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "category TEXT NOT NULL, brand TEXT NOT NULL, price REAL NOT NULL, "
        "merchant_id TEXT NOT NULL, stock_quantity INTEGER NOT NULL, "
        "performance_score REAL NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)")
    conn.execute(
        "CREATE TABLE product_eav_meta (meta_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "product_id TEXT NOT NULL, meta_key TEXT NOT NULL, meta_value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE commercial (product_id TEXT PRIMARY KEY, "
        "sponsored INTEGER NOT NULL DEFAULT 0, commission_pct REAL NOT NULL DEFAULT 0.0)")
    conn.execute("INSERT INTO merchants VALUES ('m1','Shop',4.5,2,1)")
    cats = ("cpu", "gpu", "psu")
    for i in range(n):
        pid = f"p{i:03d}"
        cat = cats[i % len(cats)]
        conn.execute(
            "INSERT INTO products VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
            (pid, f"Товар {i:03d} {cat}", cat, "Brand",
             float(500 + 100 * i), "m1", 10, 0.5))
        conn.execute("INSERT INTO product_eav_meta(product_id, meta_key, meta_value) "
                     "VALUES (?,?,?)", (pid, "Потужність", f"{100 + i}W"))
        conn.execute("INSERT INTO commercial VALUES (?,?,?)", (pid, 0, 0.0))
    conn.commit()
    conn.close()
    return path


def test_tiers_are_strictly_nested(tmp_path):
    db = _mini_db(str(tmp_path / "full.db"), 30)
    make_subset_db(db, str(tmp_path / "t10.db"), 10)
    make_subset_db(db, str(tmp_path / "t20.db"), 20)

    def ids(p):
        c = sqlite3.connect(p)
        got = {r[0] for r in c.execute("SELECT id FROM products")}
        c.close()
        return got

    t10, t20 = ids(str(tmp_path / "t10.db")), ids(str(tmp_path / "t20.db"))
    assert len(t10) == 10 and len(t20) == 20
    assert t10 < t20                      # strict nesting: a true prefix

    conn = sqlite3.connect(str(tmp_path / "t10.db"))
    eav = conn.execute(
        "SELECT COUNT(DISTINCT product_id) FROM product_eav_meta").fetchone()[0]
    conn.close()
    assert eav == 10                       # child rows follow the subset


def test_permutation_is_deterministic_and_category_spread(tmp_path):
    db = _mini_db(str(tmp_path / "full.db"), 30)
    assert _permutation_ids(db) == _permutation_ids(db)
    first10 = _permutation_ids(db)[:10]
    # the generator writes category-major rows; a rowid prefix would give
    # one category, the permutation must spread them
    conn = sqlite3.connect(db)
    cats = {conn.execute("SELECT category FROM products WHERE id=?",
                         (i,)).fetchone()[0] for i in first10}
    conn.close()
    assert len(cats) > 1


def test_identical_query_set_at_every_tier(tmp_path):
    db = _mini_db(str(tmp_path / "full.db"), 30)
    report = run_envelope(db, [10, 30], repeat=1,
                          workdir=str(tmp_path / "tiers"))
    ids_per_tier = [
        sorted({c["case_id"] for c in t["cases"]}) for t in report["tiers"]
    ]
    assert ids_per_tier[0] == ids_per_tier[1]   # F27: the denominator
    assert report["query_set_version"]
    assert sorted(report["query_set"]) == ids_per_tier[0]
    assert report["method"]["queries"].startswith("8 fixed")


def test_category_cases_drop_out_on_foreign_corpus(tmp_path):
    """On a corpus without cpu/gpu the query set is the price/absent
    core -- and the report says exactly which cases ran."""
    db = _mini_db(str(tmp_path / "full.db"), 30)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE products SET category='Computers_and_Accessories'")
    conn.commit()
    conn.close()
    report = run_envelope(db, [30], repeat=1, workdir=str(tmp_path / "tiers"))
    assert len(report["query_set"]) == 4
    assert all(not cid.startswith(("sq_cheapest", "sq_avail", "sq_budget"))
               for cid in report["query_set"])


def test_truth_transitions_are_reported_not_hidden(tmp_path):
    """A case whose oracle legitimately flips must be listed, not dropped."""
    db = _mini_db(str(tmp_path / "full.db"), 30)
    report = run_envelope(db, [5, 30], repeat=1,
                          workdir=str(tmp_path / "tiers"))
    # at least one of the three named-price cases must flip present/absent
    # over a 5-vs-30 comparison -- the transition table exists to carry it
    names = {tr["case_id"] for tr in report["truth_transitions"]}
    assert names, "expected at least one reported truth transition"
    for tr in report["truth_transitions"]:
        assert len(tr["per_tier"]) == 2


def test_build_and_query_time_are_reported_separately(tmp_path):
    db = _mini_db(str(tmp_path / "full.db"), 30)
    report = run_envelope(db, [30], repeat=2, workdir=str(tmp_path / "tiers"))
    t = report["tiers"][0]
    for key in ("build_s", "agent_init_s"):
        assert key in t and t[key] >= 0.0
    lat = t["latency_s"]
    for key in ("mean", "std", "ci95_halfwidth", "p50", "p95", "max"):
        assert key in lat
    assert t["corpus"]["products"] == 30
    assert t["corpus"]["distinct_names"] == 30
    assert t["corpus"]["eav_rows"] == 30
    assert "environment" in report
    assert report["environment"]["db_bytes"] > 0


def test_no_timeout_and_every_case_scored(tmp_path):
    db = _mini_db(str(tmp_path / "full.db"), 30)
    report = run_envelope(db, [30], repeat=2, workdir=str(tmp_path / "tiers"))
    t = report["tiers"][0]
    assert t["timeouts"] == 0
    assert t["queries"] == 8 * 2
    assert all("correct" in c and "violations" in c for c in t["cases"])
