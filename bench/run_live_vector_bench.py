#!/usr/bin/env python3
"""
Live-vector differential run: the frozen-protocol shapes from
run_recall_bench.py, but with a REAL OpenRouter embedding model instead
of the scripted embedder.

This is the authorized experiment the claim register names: it measures
what an actual embedding model contributes to the vector channel --
not a scripted synonym map. Budget is disclosed: ~120 cells per shape,
batched at 32, plus one embedding per query term.

Model selection: LIVE_EMBED_MODEL env var, default a free-tier model.
Every call is recorded; per-query top similarities are kept so the
threshold (VEC_THRESHOLD=0.55) can be audited against real scores.

Output: immutable bench/results/live_vector_bench_<ts>.json
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "core"))
sys.path.insert(0, os.path.dirname(__file__))

import json

from run_recall_bench import (SHAPES, _prec, _rec, _verify_link, ebr,
                              vec_only)
from source_projection import build_source_projection
from vector_index import VEC_THRESHOLD, OpenRouterEmbedder, VectorIndex

DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b:free"


def _load_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    root = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(root):
        for line in open(root, encoding="utf-8"):
            if line.startswith("OPENROUTER_API_KEY="):
                return line.strip().split("=", 1)[1].strip("\"'")
    raise SystemExit("OPENROUTER_API_KEY not found -- live run blocked")


def main():
    model = os.environ.get("LIVE_EMBED_MODEL", DEFAULT_MODEL)
    emb = OpenRouterEmbedder(_load_key(), model)
    report = {
        "benchmark": "live_vector_v1",
        "protocol": "docs/BENCHMARKS.md",
        "embedder": f"LIVE {model} via OpenRouter (authorized run)",
        "threshold": VEC_THRESHOLD,
        "disclosure":
            "same synthetic fixtures as ebr_recall_v2; one model, one "
            "date -- a point measurement of the channel's live "
            "contribution, NOT a model-general or ranking claim",
        "calls": 0,
        "shapes": {},
    }

    # count calls honestly
    orig = emb.embed
    def counted(texts):
        report["calls"] += 1
        return orig(texts)
    emb.embed = counted

    print(f"live embedder: {model}  threshold={VEC_THRESHOLD}")
    print(f"{'shape':<14}{'query':<16}{'ebr':>6}{'ebr+v':>7}"
          f"{'vec':>6}{'top_sim':>8}")
    for shape, builder in SHAPES:
        tmp = tempfile.mkdtemp()
        db, queries = builder(tmp)
        proj = build_source_projection(db)
        if shape == "inferred":
            _verify_link(db, proj, "products", "suit", "lookup", "id")
        elif shape == "inferred_junc":
            _verify_link(db, proj, "product_tags", "a",
                         "products", "id")
            _verify_link(db, proj, "product_tags", "b", "tags", "id")

        idx = VectorIndex(model)
        idx.stats = idx.build(db, proj, emb)
        vector = (idx, emb) if idx.stats.get("status") == "ready" \
            else None

        shape_res = []
        for q in queries:
            truth = set(q["truth"])
            t0 = time.time()
            e_h, e_r = ebr(db, proj, q["query"])
            t0, m_ebr = time.time(), {
                "hits": sorted(e_h), "recall": _rec(e_h, truth),
                "precision": _prec(e_h, truth), "regret": e_r,
                "ms": round((time.time() - t0) * 1000, 1)}
            t0 = time.time()
            ev_h, ev_r = ebr(db, proj, q["query"], vector=vector)
            m_ev = {"hits": sorted(ev_h), "recall": _rec(ev_h, truth),
                    "precision": _prec(ev_h, truth), "regret": ev_r,
                    "ms": round((time.time() - t0) * 1000, 1)}
            v_h = vec_only(idx, emb, q["query"])
            m_v = {"hits": sorted(v_h), "recall": _rec(v_h, truth),
                   "precision": _prec(v_h, truth)}
            # audit trail: best similarity per query term
            top_sims = {}
            if idx is not None and idx.vectors:
                for t in [x for x in q["query"].split() if len(x) >= 3]:
                    bs = idx.bind(t, emb)
                    best = max((b["similarity"]
                                for c in bs.values() for b in c),
                               default=None)
                    top_sims[t] = best
            row = {"query": q["query"], "truth": sorted(truth),
                   "note": q["note"], "ebr": m_ebr, "ebr_vec": m_ev,
                   "vec_only": m_v, "top_sims": top_sims}
            shape_res.append(row)
            top = max((s or 0) for s in top_sims.values()) \
                if top_sims else 0
            print(f"{shape:<14}{q['query']:<16}"
                  f"{m_ebr['recall']:>6.2f}{m_ev['recall']:>7.2f}"
                  f"{m_v['recall']:>6.2f}{top:>8.2f}")

        entry = {"queries": shape_res, "vector_stats": idx.stats}
        for m in ("ebr", "ebr_vec", "vec_only"):
            entry[f"mean_recall_{m}"] = (
                sum(q[m]["recall"] for q in shape_res) / len(shape_res))
        report["shapes"][shape] = entry

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(
        out_dir, f"live_vector_bench_{int(time.time())}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\ncalls: {report['calls']}\nreport: {out}")


if __name__ == "__main__":
    main()
