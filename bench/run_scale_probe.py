#!/usr/bin/env python3
"""Channel-latency probe on the largest real-scale source.

Times each EBR channel separately on marketplace_500k so the scale
claim names WHICH channel is expensive, not just a total. One run,
one machine -- a point measurement, not a throughput claim.
"""
from __future__ import annotations
import json, os, sqlite3, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "core"))
from source_projection import build_source_projection
from associative import _scan_term, MAX_PER_TERM_SCAN

DB = os.path.join(os.path.dirname(__file__), "..", "data",
                  "marketplace_500k.db")

def main():
    t0 = time.time()
    proj = build_source_projection(DB)
    t_proj = time.time() - t0
    conn = sqlite3.connect("file:{}?mode=ro".format(
        os.path.abspath(DB).replace("\\", "/")), uri=True)
    cur = conn.cursor()
    report = {"source": "marketplace_500k",
              "entity": proj.entity_table,
              "eav": sorted(proj.eav_tables),
              "projection_s": round(t_proj, 1),
              "channels": {}}
    term = "ryzen"
    # entity text columns
    t0 = time.time(); cov = _scan_term(cur, proj, term, {}, MAX_PER_TERM_SCAN)
    report["channels"]["entity_only"] = {"s": round(time.time()-t0,2),
                                         "bound": len(cov)}
    t0 = time.time(); cov = _scan_term(cur, proj, term, proj.eav_tables,
                                       MAX_PER_TERM_SCAN)
    report["channels"]["all_lexical"] = {"s": round(time.time()-t0,2),
                                         "bound": len(cov)}
    # Indexed path: same semantics (trigram substring), sidecar FTS5.
    try:
        from channel_index import get_index
        t0 = time.time(); idx = get_index(DB, proj)
        report["index_build_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        cov_idx = _scan_term(cur, proj, term, proj.eav_tables,
                             MAX_PER_TERM_SCAN, idx=idx)
        report["channels"]["all_lexical_indexed"] = {
            "s": round(time.time() - t0, 2), "bound": len(cov_idx)}
        # differential honesty: indexed binding set must equal scan's
        report["bindings_equal"] = set(cov_idx) == set(cov)
        if idx:
            idx.close()
    except Exception as exc:  # noqa: BLE001 -- index absent/failed -> recorded, not hidden
        report["channels"]["all_lexical_indexed"] = {"error": str(exc)}
    conn.close()
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"scale_probe_{int(time.time())}.json")
    json.dump(report, open(out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("report:", out)

if __name__ == "__main__":
    main()
