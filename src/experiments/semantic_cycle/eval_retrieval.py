"""Retrieval-level evaluation of the semantic cycle (docs/18: P1-P3, P6, P7).

    python eval_retrieval.py --split dev  --tag <name>            # tuning split
    python eval_retrieval.py --split test --tag <name> --confirm-test

Builds (or incrementally syncs) one index per source through the
production API, runs every frozen request against bm25 / bm25_prefix /
dense / hybrid (+ the raw-columns ablation), scores with the
pre-registered metrics and writes immutable result files:
docs/semantic_eval_<tag>_<split>.json (+ .rankings.json.gz). A result
file is never overwritten; `--limit` smoke runs go to the temp dir.
Dev-only extras: `--grid 0.5,0.3` (lexical fusion weights) and `--hyde`
(exploratory hypothetical-document ablation, capped spend).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_core as C  # noqa: E402
import eval_report as R  # noqa: E402

HYDE_GROUPS = ("en_strict", "uk")   # default; --hyde-groups overrides


def _progress(src: str):
    last = [0.0]

    def cb(p: dict) -> None:
        if time.time() - last[0] > 20:
            last[0] = time.time()
            print(f"  [{src}] {p}", flush=True)
    return cb


def _signals(modes: dict) -> dict:
    """Confidence signals of one request (higher = more answerable)."""
    d, lx = modes["dense"], modes.get("bm25_prefix", {})
    return {"top1_z": d.get("top1_z"), "top1_sim": d.get("top1_sim"),
            "margin": d.get("margin"), "lex_top": lx.get("top_score")}


def _answerable_signals(d: dict) -> list[dict]:
    grp = {r["rid"]: r["group"] for r in d["requests"]}
    return [_signals(m) for rid, m in d["results"].items()
            if "unvalidated" not in grp[rid]]


def _unanswerable(store, emb, doc: dict) -> list[dict]:
    out = []
    for u in doc["unanswerable"]:
        for q in (u["en"], u["uk"]):
            out.append(_signals(C.run_modes(store, emb, q)))
    return out


def _serve_memory(store, emb, queries: list[str]) -> dict:
    """USS growth while serving 200 searches with the model already
    loaded (so model weights are excluded from the delta)."""
    from semantic_search import search
    with C.PeakMemory(0.05) as mem:
        for i in range(200):
            search(store, emb, queries[i % len(queries)], k=10)
    return mem.summary_mb()


def _make_hyde(args, store, emb, src: str):
    if not args.hyde:
        return None, None
    import gen_queries as gq
    from eval_hyde import Hyde
    from eval_llm import BudgetedLLM
    p_in, p_out = gq.pricing_for(gq.MODEL)
    log = C.ROOT / "docs" / (f"semantic_hyde_llm_log_{args.tag}_"
                             f"{args.split}.jsonl")
    llm = BudgetedLLM(gq.MODEL, args.hyde_max_usd, log, p_in, p_out)
    cache = C.ROOT / "data" / "eval_semantic" / (
        f"hyde_cache_{args.index_tag}_{src}.jsonl")
    return Hyde(store, emb, llm, cache), llm


def _product_llm(args):
    """Capped, logged LLM behind the PRODUCT HyDE mode (None if off)."""
    if not args.product:
        return None
    import gen_queries as gq
    from eval_llm import BudgetedLLM
    p_in, p_out = gq.pricing_for(gq.MODEL)
    log = C.ROOT / "docs" / (f"semantic_product_hyde_llm_log_{args.tag}_"
                             f"{args.split}.jsonl")
    return BudgetedLLM(gq.MODEL, args.product_max_usd, log, p_in, p_out)


def run_source(src: str, doc: dict, args, emb) -> dict:
    from record_documents import plan_documents
    path, intended = C.SOURCES[src]
    index_dir = C.ROOT / "data" / "eval_semantic" / "index" / \
        args.index_tag / src
    print(f"[{src}] projection + sync -> {index_dir}", flush=True)
    plm = _product_llm(args)
    proj, store, rep, secs, mem, idx = C.build_index(
        src, index_dir, emb, _progress(src), llm=plm)
    plan = plan_documents(proj, str(path))
    out = {"intended_entity_table": intended,
           "chosen_entity_table": proj.entity_table,
           "entity_correct": proj.entity_table == intended,
           "entity_ambiguous": getattr(proj, "entity_ambiguous", None),
           "plan": plan.to_dict() if plan else None,
           "sync": rep.to_dict(), "sync_wall_s": round(secs, 1),
           "sync_memory": mem,
           "reconciliation": C.check_reconciliation(proj, rep, store),
           "embedder_counters": {"doc_texts": emb.doc_texts,
                                 "query_texts": emb.query_texts}}
    reqs = R.request_list(doc, src, args.split, args.limit)
    raw_ix = C.raw_columns_baseline(plan, str(path)) if plan else None
    grid = [float(x) for x in args.grid.split(",") if x]
    hyde, llm = _make_hyde(args, store, emb, src)
    hgroups = tuple(g for g in args.hyde_groups.split(",") if g)
    pgroups = tuple(g for g in args.product_hyde_groups.split(",") if g)
    results = {}
    t0 = time.time()
    for req in reqs:
        use = hyde if req["group"] in hgroups else None
        prod = idx if args.product else None
        results[req["rid"]] = C.run_modes(
            store, emb, req["query"], raw_ix, grid, use, prod,
            req["group"] in pgroups)
    out["query_wall_s"] = round(time.time() - t0, 1)
    out["n_requests"] = len(reqs)
    if llm is not None:
        out["hyde_llm"] = llm.summary()
    if plm is not None:
        out["product_hyde_llm"] = plm.summary()
        out["product_index_status"] = idx.status()
    rng = random.Random(20261112)
    sample = [r["query"] for r in rng.sample(reqs, min(100, len(reqs)))]
    out["lexical_differential"] = C.differential_lexical(
        store, plan, str(path), sample) if plan else None
    out["unanswerable_signals"] = _unanswerable(store, emb, doc)
    out["serve_memory"] = _serve_memory(store, emb, sample or ["test"])
    out["store_stats"] = store.stats()
    idx.close()
    return {"summary": out, "requests": reqs, "results": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test"), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--index-tag", default=None)
    ap.add_argument("--sources", default="library,forum,food")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--confirm-test", action="store_true")
    ap.add_argument("--grid", default="")
    ap.add_argument("--hyde", action="store_true")
    ap.add_argument("--hyde-max-usd", type=float, default=0.2)
    ap.add_argument("--hyde-groups", default="en_strict,uk")
    ap.add_argument("--product", action="store_true",
                    help="also search through the shipped SemanticIndex")
    ap.add_argument("--product-max-usd", type=float, default=0.15)
    ap.add_argument("--product-hyde-groups",
                    default="en_natural,en_strict,uk")
    ap.add_argument("--max-seq", type=int, default=0,
                    help="override the embedder window (experiment only)")
    args = ap.parse_args()
    args.index_tag = args.index_tag or args.tag
    if args.split == "test" and not (args.confirm_test or args.limit):
        raise SystemExit("test split needs --confirm-test (single scored run)")
    base = f"semantic_eval_{args.tag}_{args.split}"
    outdir = C.ROOT / "docs" if not args.limit else Path(os.environ["TEMP"])
    out_json = outdir / f"{base}.json"
    if out_json.exists():
        raise SystemExit(f"{out_json} exists; results are immutable")
    from embedders import (CountingEmbedder, SentenceTransformerEmbedder,
                           default_embedder)
    doc = C.load_requests()
    emb = CountingEmbedder(
        SentenceTransformerEmbedder(max_seq_length=args.max_seq)
        if args.max_seq else default_embedder())
    per_source = {s: run_source(s, doc, args, emb)
                  for s in args.sources.split(",")}
    rows = []
    for s, d in per_source.items():
        rows += R.score_all(d["requests"], d["results"])
    answerable = {s: _answerable_signals(d) for s, d in per_source.items()}
    unans = {s: d["summary"]["unanswerable_signals"]
             for s, d in per_source.items()}
    report = {"protocol": "docs/BENCHMARKS.md",
              "tag": args.tag, "split": args.split, "smoke": bool(args.limit),
              "flags": {"grid": args.grid, "hyde": args.hyde},
              "environment": C.environment(),
              "embedder": {"model_id": emb.model_id, "dim": emb.dim},
              "request_file_sha256": (C.REQUEST_FILE.parent / (
                  C.REQUEST_FILE.name + ".sha256")).read_text().strip(),
              "sources": {s: d["summary"] for s, d in per_source.items()},
              "groups": R.aggregate(rows),
              "endpoints": {**R.endpoint_p1_p2_p3(rows),
                            "P7": R.endpoint_p7(answerable, unans),
                            "exploratory": R.endpoint_exploratory(rows)},
              "latency": R.latency_summary(
                  {s: d["results"] for s, d in per_source.items()})}
    out_json.write_text(json.dumps(report, indent=1, ensure_ascii=False),
                        encoding="utf-8")
    with gzip.open(outdir / f"{base}.rankings.json.gz", "wt",
                   encoding="utf-8") as f:
        json.dump({s: d["results"] for s, d in per_source.items()}, f)
    print(json.dumps(report["endpoints"], indent=1, ensure_ascii=False))
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
