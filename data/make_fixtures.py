"""Build small committed fixtures from the full real catalogues.

The full `data/real_*.db` files are 3-26 MB and git-ignored, so the real-data
code path is never exercised in CI: `tests/test_real_data_validation.py` skips
when they are absent. A trimmed copy of each keeps that path honest without
committing megabytes.

The trim is by product id, keeps the schema and `ingest_meta` provenance rows,
and is VACUUMed so the file carries no slack from the rows it dropped.
"""
import os
import sqlite3
import sys

# The repo root is the parent of this script's data/ directory -- an
# author-machine absolute path made the script unportable (F36).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "tests", "fixtures")

#: Products kept per fixture. Open Food Facts is the big one (its EAV rows are
#: 60x its product count), so it is trimmed hardest: 200 products still yields
#: ~15k EAV rows, 100+ keys, unit profiles and algebra gains -- every section of
#: the real-data report has something to say.
TARGETS = {"off": 200, "wdc": 500, "prices": 300}


def trim(name, limit):
    src_path = os.path.join(SRC, f"real_{name}.db")
    if not os.path.exists(src_path):
        print(f"  {name}: source missing, skipped")
        return
    os.makedirs(OUT, exist_ok=True)
    dst_path = os.path.join(OUT, f"real_{name}_fixture.db")

    src = sqlite3.connect(src_path)
    tables = [r[0] for r in src.execute(
        "select name from sqlite_master where type='table'")]
    keep = [r[0] for r in src.execute(
        "select id from products order by id limit ?", (limit,))]
    if not keep:
        print(f"  {name}: no products, skipped")
        src.close()
        return
    placeholders = ",".join("?" * len(keep))

    if os.path.exists(dst_path):
        os.remove(dst_path)
    dst = sqlite3.connect(dst_path)
    src.backup(dst)
    dst.close()

    dst = sqlite3.connect(dst_path)
    for table in tables:
        if table == "products":
            dst.execute(f"delete from products where id not in ({placeholders})", keep)
        elif table == "product_eav_meta":
            columns = [r[1] for r in dst.execute("pragma table_info(product_eav_meta)")]
            key = "product_id" if "product_id" in columns else columns[0]
            dst.execute(
                f"delete from product_eav_meta where {key} not in ({placeholders})", keep)
        elif table == "commercial":
            dst.execute(
                f"delete from commercial where product_id not in ({placeholders})", keep)
    dst.commit()
    dst.execute("vacuum")
    products = dst.execute("select count(*) from products").fetchone()[0]
    eav = 0
    try:
        eav = dst.execute("select count(*) from product_eav_meta").fetchone()[0]
    except sqlite3.Error:
        pass
    dst.close()
    src.close()

    size_kb = os.path.getsize(dst_path) / 1024.0
    print(f"  {name}: {products} products, {eav} EAV rows, {size_kb:.0f} KB")


if __name__ == "__main__":
    print(f"writing fixtures to {OUT}")
    for name, limit in TARGETS.items():
        trim(name, limit)
