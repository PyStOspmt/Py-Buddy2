"""Build missing regenerable demo databases for fresh clones and test runs.

F36-safe: never overwrites an existing non-empty database unless ``--rebuild``
is passed. Empty 0-byte stub files (from prior failed ``sqlite3.connect`` calls)
are treated as missing and replaced cleanly.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _needs_build(path: str, rebuild: bool) -> bool:
    if rebuild:
        return True
    return (not os.path.exists(path)) or os.path.getsize(path) == 0


def _clear_empty_stub(path: str) -> None:
    if os.path.exists(path) and os.path.getsize(path) == 0:
        os.remove(path)


def ensure_demo_databases(root: str = ROOT, rebuild: bool = False) -> List[str]:
    """Ensure all 5 synthetic/demo SQLite catalogues under ``<root>/data`` exist.

    Returns the paths of databases actually created or rebuilt during this call.
    """
    data_dir = os.path.join(os.path.abspath(root), "data")
    os.makedirs(data_dir, exist_ok=True)
    for extra in (
        data_dir,
        os.path.join(os.path.abspath(root), "src", "experiments"),
    ):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    import generate_marketplace_scale
    import init_auto_parts_db
    import init_database
    import init_real_pharmacy_db
    from semantic_cycle import build_jury_demo

    built: List[str] = []

    catalog_db = os.path.join(data_dir, "catalog.db")
    if _needs_build(catalog_db, rebuild):
        _clear_empty_stub(catalog_db)
        init_database.create_ecommerce_database(catalog_db, overwrite=rebuild)
        built.append(catalog_db)

    auto_db = os.path.join(data_dir, "auto_parts.db")
    if _needs_build(auto_db, rebuild):
        _clear_empty_stub(auto_db)
        init_auto_parts_db.init_auto_parts_db(auto_db)
        built.append(auto_db)

    pharmacy_db = os.path.join(data_dir, "real_pharmacy.db")
    if _needs_build(pharmacy_db, rebuild):
        _clear_empty_stub(pharmacy_db)
        init_real_pharmacy_db.build_pharmacy_db(pharmacy_db)
        built.append(pharmacy_db)

    rozetka_db = os.path.join(data_dir, "marketplace_rozetka.db")
    if _needs_build(rozetka_db, rebuild):
        _clear_empty_stub(rozetka_db)
        generate_marketplace_scale.generate_marketplace_database(
            rozetka_db, target_products=10000, overwrite=rebuild, seed=2026
        )
        built.append(rozetka_db)

    jury_db = os.path.join(data_dir, "jury_zoo.db")
    if _needs_build(jury_db, rebuild):
        _clear_empty_stub(jury_db)
        build_jury_demo.build(jury_db)
        built.append(jury_db)

    return built


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rebuild",
        action="store_true",
        help="rebuild existing synthetic demo databases (preserving .bak where supported)",
    )
    args = ap.parse_args()
    created = ensure_demo_databases(ROOT, rebuild=args.rebuild)
    if created:
        print(f"bootstrap_demo_dbs: built {len(created)} database(s): " + ", ".join(created))
    else:
        print("bootstrap_demo_dbs: all synthetic demo databases already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
