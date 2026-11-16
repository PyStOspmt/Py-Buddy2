#!/usr/bin/env python3
"""
Held-out evaluation: real ingested sources, hand-labelled queries.

Ground truth is produced by ORACLE SQL written against the raw source
tables -- it never consults the projection, so the system's own
inference cannot act as its own oracle.

Sources (all real, none built for this bench):
  real_off      -- Open Food Facts dump, 3000 products, 207k EAV rows
  real_wdc      -- WDC product feed, ~3.7k electronics offers
  real_prices   -- Open Prices observations, 6554 rows; names are GTIN
                   codes -> lexical surface is nearly absent; the honest
                   expectation is partial/degraded, not a win

Methods: fts (conjunctive LIKE over entity text columns) vs ebr
(relax_search, all structural channels). Reports recall, precision,
regret per query. No vector channel here -- it is measured separately
(run_live_vector_bench.py) and index coverage is entity-columns only.

Output: immutable bench/results/heldout_bench_<ts>.json
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "core"))

from associative import MAX_PER_TERM_SCAN, _scan_term, relax_search
from run_recall_bench import _prec, _rec, baseline_fts
from source_projection import build_source_projection

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Each query: (term(s), oracle SQL executed verbatim on the RAW db).
# The oracle is written per-source by hand after inspecting the data;
# the projection never sees it.
SOURCES = [
    {
        "id": "real_off",
        "db": "data/real_off.db",
        "queries": [
            ("chamomile",
             "SELECT id FROM products WHERE name LIKE '%hamomile%'"),
            ("milk",
             "SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_key='allergens' AND meta_value LIKE '%milk%'"),
            ("organic",
             "SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_key='labels' AND meta_value LIKE '%organic%'"),
            ("Lindt",
             "SELECT id FROM products WHERE brand LIKE '%Lindt%'"),
            # multi-term conjunctive: products whose allergen list names
            # milk AND whose labels name organic -- oracle = SQL
            # intersection across the same EAV table
            ("milk organic",
             "SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_key='allergens' AND meta_value LIKE '%milk%' "
             "INTERSECT SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_key='labels' AND meta_value LIKE '%organic%'"),
            ("xylophone",      # absent term -- honest zero expected
             "SELECT id FROM products WHERE name LIKE '%xylophone%'"),
        ],
    },
    {
        "id": "real_wdc",
        "db": "data/real_wdc.db",
        "queries": [
            ("samsung",
             "SELECT id FROM products WHERE name LIKE '%amsung%'"),
            ("DDR4",
             "SELECT id FROM products WHERE name LIKE '%DDR4%'"),
            ("sandisk",
             "SELECT id FROM products WHERE name LIKE '%SanDisk%'"),
            # multi-term: brand word + spec fragment both in the name
            ("kingston ssd",
             "SELECT id FROM products WHERE name LIKE '%KINGSTON%' "
             "AND name LIKE '%SSD%'"),
            ("oboe",           # absent in electronics
             "SELECT id FROM products WHERE name LIKE '%oboe%'"),
        ],
    },
    {
        "id": "real_prices",
        "db": "data/real_prices.db",
        "queries": [
            ("ONLINE",         # EAV location_type value
             "SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_value LIKE '%ONLINE%'"),
            ("experiments",    # EAV source value fragment
             "SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_value LIKE '%experiments%'"),
            ("abracadabra",    # absent
             "SELECT DISTINCT product_id FROM product_eav_meta "
             "WHERE meta_value LIKE '%abracadabra%'"),
        ],
    },
]


def oracle_ids(db_path: str, sql: str) -> set[str]:
    conn = sqlite3.connect(
        "file:{}?mode=ro".format(
            os.path.abspath(db_path).replace("\\", "/")), uri=True)
    try:
        return {str(r[0]) for r in conn.execute(sql)}
    finally:
        conn.close()


def main():
    report = {
        "benchmark": "heldout_v1",
        "protocol": "docs/BENCHMARKS.md",
        "disclosure":
            "real ingested sources; ground truth via hand-written "
            "oracle SQL on raw tables (independent of the projection); "
            "measures recall/precision of channel binding on REAL data, "
            "not ranking quality",
        "sources": {},
    }
    print(f"{'source':<13}{'query':<14}{'truth':>7}{'fts':>6}"
          f"{'cov':>6}{'prec':>6}{'reg':>4}")
    for src in SOURCES:
        db = os.path.join(ROOT, src["db"])
        proj = build_source_projection(db)
        eav_found = sorted(proj.eav_tables)
        print(f"# {src['id']}: entity={proj.entity_table} "
              f"eav={eav_found}")
        rows = []
        for term, oracle_sql in src["queries"]:
            truth = oracle_ids(db, oracle_sql)
            t0 = time.time()
            fts_h = baseline_fts(db, proj, term)
            t_fts = round((time.time() - t0) * 1000, 1)
            t0 = time.time()
            res = relax_search(db, proj, term)
            ebr_h = ({i.entity_id for i in res.items}
                     if res else set())
            reg = res.regret if res else 0
            t_ebr = round((time.time() - t0) * 1000, 1)
            # Coverage recall: union of entity ids bound BEFORE the
            # 40-item display cap and per-term 500-row scan limit --
            # the channel-reachability denominator. Display recall is
            # reported separately; the gap between them is the bounded
            # budget, disclosed not hidden.
            t0 = time.time()
            conn = sqlite3.connect(
                "file:{}?mode=ro".format(
                    os.path.abspath(db).replace("\\", "/")), uri=True)
            try:
                cur = conn.cursor()
                cov = None
                for t in [x for x in term.split() if len(x) >= 3]:
                    ids = set(_scan_term(
                        cur, proj, t, proj.eav_tables,
                        MAX_PER_TERM_SCAN))
                    cov = ids if cov is None else (cov & ids)
            finally:
                conn.close()
            cov = cov or set()
            t_cov = round((time.time() - t0) * 1000, 1)
            row = {
                "query": term, "truth_n": len(truth),
                "truth_sample": sorted(truth)[:5],
                "fts": {"hits_n": len(fts_h),
                        "recall": _rec(fts_h, truth),
                        "precision": _prec(fts_h, truth),
                        "ms": t_fts},
                "ebr": {"hits_n": len(ebr_h),
                        "recall": _rec(ebr_h, truth),
                        "precision": _prec(ebr_h, truth),
                        "regret": reg, "ms": t_ebr,
                        "hits_sample": sorted(ebr_h)[:5]},
                "ebr_coverage": {"hits_n": len(cov),
                                 "recall": _rec(cov, truth),
                                 "precision": _prec(cov, truth),
                                 "ms": t_cov},
            }
            rows.append(row)
            print(f"{src['id']:<13}{term:<14}{len(truth):>7}"
                  f"{row['fts']['recall']:>6.2f}"
                  f"{row['ebr_coverage']['recall']:>6.2f}"
                  f"{row['ebr_coverage']['precision']:>6.2f}{reg:>4}")
        entry = {
            "entity_table": proj.entity_table,
            "eav_tables": eav_found,
            "queries": rows,
        }
        for m in ("fts", "ebr", "ebr_coverage"):
            entry[f"mean_recall_{m}"] = (
                sum(r[m]["recall"] for r in rows) / len(rows))
            entry[f"mean_precision_{m}"] = (
                sum(r[m]["precision"] for r in rows) / len(rows))
        report["sources"][src["id"]] = entry

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(
        out_dir, f"heldout_bench_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nreport: {out}")


if __name__ == "__main__":
    main()
