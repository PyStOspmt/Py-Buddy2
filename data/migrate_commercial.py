"""One-off: add the `commercial` table to the pre-existing catalogues.

The catalogues were generated/ingested before the commercial table existed
in the DDL. Re-generating or re-ingesting would change the product data and
invalidate the measured paper numbers, so the table is added in place:

* real_*.db -- documented neutral constants (sponsored=0, commission_pct=0.0)
  for every product, provenance recorded in ingest_meta;
* marketplace_rozetka.db -- the same deterministic sponsored set the
  generator now produces (a fresh RNG seeded with the product id, ~12%
  sponsored), so a future regeneration stays consistent with this one.

Idempotent: re-running on a migrated database is a no-op.
"""
import os
import random
import sqlite3

# This script already lives inside data/, so the data directory is the
# script's own directory -- the old ROOT/data pointed one level too deep.
DATA = os.path.dirname(os.path.abspath(__file__))

NEUTRAL = ("synthetic-neutral: the source has no sponsor/commission field; "
           "all rows sponsored=0, commission_pct=0.0")


def sponsored_state(prod_id: str):
    """The exact rule generate_marketplace_scale.py now uses."""
    rng = random.Random(prod_id)
    sponsored = 1 if rng.random() < 0.12 else 0
    commission = round(rng.uniform(5.0, 15.0), 1) if sponsored else 0.0
    return sponsored, commission


def migrate(fname: str, dataset: str, sponsored_rule: bool) -> None:
    path = os.path.join(DATA, fname)
    conn = sqlite3.connect(path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    if "commercial" not in tables:
        conn.execute("""
        CREATE TABLE commercial (
            product_id TEXT PRIMARY KEY,
            sponsored INTEGER NOT NULL DEFAULT 0,
            commission_pct REAL NOT NULL DEFAULT 0.0
        )""")
        conn.execute(
            "CREATE INDEX idx_commercial_sponsored ON commercial(sponsored)")
        if sponsored_rule:
            rows = [(pid, *sponsored_state(pid))
                    for (pid,) in conn.execute("SELECT id FROM products")]
            conn.executemany(
                "INSERT INTO commercial (product_id, sponsored, commission_pct) "
                "VALUES (?, ?, ?)", rows)
        else:
            conn.execute(
                "INSERT INTO commercial (product_id, sponsored, commission_pct) "
                "SELECT id, 0, 0.0 FROM products")
        if "ingest_meta" in tables:
            conn.execute(
                "INSERT OR REPLACE INTO ingest_meta(key, value) VALUES (?, ?)",
                (f"commercial_source:{dataset}", NEUTRAL))
        conn.commit()
        print(f"{fname}: commercial table added")
    else:
        print(f"{fname}: commercial table already present, skipped")
    products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    rows = conn.execute("SELECT COUNT(*) FROM commercial").fetchone()[0]
    sponsored = conn.execute(
        "SELECT COUNT(*) FROM commercial WHERE sponsored=1").fetchone()[0]
    print(f"  products={products}, commercial_rows={rows}, sponsored={sponsored}")
    conn.close()


if __name__ == "__main__":
    migrate("real_off.db", "off", sponsored_rule=False)
    migrate("real_prices.db", "prices", sponsored_rule=False)
    migrate("real_wdc.db", "wdc", sponsored_rule=False)
    migrate("marketplace_rozetka.db", "marketplace", sponsored_rule=True)
