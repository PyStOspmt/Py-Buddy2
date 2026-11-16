"""
Q1 sensitivity sweep: how does concept discovery change as the
``min_scope_overlap`` threshold moves?

``discover_concepts`` merges two EAV keys into one logical attribute only
when three structural criteria hold at once: shared value signature,
overlapping *canonical* value sets (``min_value_overlap``), and overlapping
product-category scopes (``min_scope_overlap``). The scope threshold is the
one that is most domain-sensitive: too low and a wheel-hub limit gets merged
with a rim diameter (a containment rule becomes nonsense); too high and
genuinely redundant keys stay split.

This script sweeps ``min_scope_overlap`` over a fixed grid on a committed
catalogue and records, for each threshold, how many keys get merged into
shared concepts versus left as singletons, and the size distribution of the
resulting concepts. It is a measurement, not a claim: the point is to show
the threshold's effect is bounded and stable, not to pick a "better" value.

Usage:
    python src/scope_overlap_sweep.py --db data/real_off.db \
        --out docs/scope_overlap_sweep_20260930.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from typing import Dict, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from attribute_algebra import discover_concepts  # noqa: E402

#: The committed default, plus a grid around it.
DEFAULT_GRID = [0.20, 0.25, 0.30, 0.34, 0.40, 0.50]


def _load_eav(db_path: str):
    """key -> set(values) and key -> set(categories), exactly as load_snapshot."""
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row
    key_values: Dict[str, Set[str]] = {}
    key_scopes: Dict[str, Set[str]] = {}
    for r in conn.execute(
        "SELECT m.product_id, m.meta_key, m.meta_value, p.category "
        "FROM product_eav_meta m JOIN products p ON p.id = m.product_id"
    ):
        key, val = r["meta_key"], r["meta_value"]
        if key == "vendor":
            continue
        key_values.setdefault(key, set()).add(val)
        key_scopes.setdefault(key, set()).add(r["category"])
    conn.close()
    return {k: sorted(v) for k, v in key_values.items()}, key_scopes


def _sweep_one(sorted_kv, key_scopes, min_scope_overlap: float) -> Dict[str, object]:
    grouped = discover_concepts(
        sorted_kv, key_scopes, min_keys=2, min_scope_overlap=min_scope_overlap,
    )
    assigned = {k for c in grouped for k in c.keys}
    leftover = {k: v for k, v in sorted_kv.items() if k not in assigned}
    singletons = discover_concepts(leftover, None, min_keys=1, allow_merge=False)
    sizes = Counter(len(c.keys) for c in grouped)
    return {
        "min_scope_overlap": min_scope_overlap,
        "total_keys": len(sorted_kv),
        "grouped_concepts": len(grouped),
        "grouped_keys": len(assigned),
        "singleton_keys": len(singletons),
        "largest_concept": max((len(c.keys) for c in grouped), default=0),
        "concepts_by_size": {str(k): v for k, v in sorted(sizes.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/real_off.db")
    ap.add_argument("--out", default="docs/scope_overlap_sweep_20260930.json")
    ap.add_argument(
        "--grid", default=",".join(str(g) for g in DEFAULT_GRID),
        help="comma-separated min_scope_overlap values",
    )
    args = ap.parse_args()

    grid = [float(g) for g in args.grid.split(",") if g.strip()]
    sorted_kv, key_scopes = _load_eav(args.db)
    print(f"[sweep] {args.db}: {len(sorted_kv)} EAV keys, "
          f"{len(key_scopes)} scoped")

    rows = [_sweep_one(sorted_kv, key_scopes, g) for g in grid]
    report = {
        "db": args.db,
        "total_keys": len(sorted_kv),
        "note": (
            "Sensitivity of discover_concepts to min_scope_overlap. The scope "
            "threshold separates keys that merely share a numeric vocabulary "
            "(a wheel-hub limit vs a rim diameter) from genuinely redundant "
            "keys. Recorded to show the effect is bounded and stable, not to "
            "pick a 'better' value."
        ),
        "rows": rows,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    print(f"{'threshold':>10} | {'grouped':>8} | {'grouped_keys':>13} | "
          f"{'singletons':>11} | {'largest':>8}")
    for r in rows:
        print(f"{r['min_scope_overlap']:>10.2f} | {r['grouped_concepts']:>8} | "
              f"{r['grouped_keys']:>13} | {r['singleton_keys']:>11} | "
              f"{r['largest_concept']:>8}")
    print(f"\n[sweep] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
