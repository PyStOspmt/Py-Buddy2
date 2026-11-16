"""Scoring, aggregation and endpoint evaluation (docs/18 sections 5-6).

Consumes the raw per-request rankings collected by `eval_core` and the
frozen request file; applies the pre-registered metric definitions and
endpoint rules. No ranking is modified here.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

import retrieval_metrics as M

QTYPES = ("en_natural", "en_strict", "uk")
LABEL_METRICS = ("recall@10", "recall@50", "precision@10", "ndcg@10", "mrr")
RECORD_METRICS = ("hit@1", "hit@5", "hit@10", "hit@50", "mrr")


def request_list(doc: dict, source: str, split: str, limit: int = 0
                 ) -> list[dict]:
    reqs = []
    targets = [t for t in doc["targets"]
               if t["source"] == source and t["split"] == split]
    if limit:
        targets = targets[:limit]
    for t in targets:
        for qtype in QTYPES:
            q = (t["queries"].get(qtype) or "").strip()
            if not q:
                continue
            group = qtype
            if qtype == "en_strict" and not t["flags"]["strict_ok"]:
                group = "en_strict_unvalidated"
            if qtype == "uk" and not t["flags"]["uk_ok"]:
                group = "uk_unvalidated"
            reqs.append({"rid": f"{source}|{t['kind']}|{t['target_id']}|"
                                f"{qtype}",
                         "source": source, "kind": t["kind"],
                         "target_id": str(t["target_id"]), "group": group,
                         "query": q,
                         "relevant": {str(x) for x in t["relevant_ids"]}})
    return reqs


def score_request(req: dict, ids: list[str]) -> dict[str, float]:
    if req["kind"] == "label_set":
        r = req["relevant"]
        return {"recall@10": M.recall_at_k(ids, r, 10),
                "recall@50": M.recall_at_k(ids, r, 50),
                "precision@10": M.precision_at_k(ids, r, 10),
                "ndcg@10": M.ndcg_at_k(ids, r, 10), "mrr": M.mrr(ids, r)}
    t = req["target_id"]
    return {"hit@1": M.hit_at_k(ids, t, 1), "hit@5": M.hit_at_k(ids, t, 5),
            "hit@10": M.hit_at_k(ids, t, 10),
            "hit@50": M.hit_at_k(ids, t, 50),
            "mrr": M.mrr(ids, {t})}


def score_all(requests: list[dict], results: dict) -> list[dict]:
    rows = []
    for req in requests:
        for mode, rec in results[req["rid"]].items():
            rows.append({"source": req["source"], "kind": req["kind"],
                         "group": req["group"], "mode": mode,
                         "rid": req["rid"], "error": rec["error"],
                         "m": score_request(req, rec["ids"])})
    return rows


def aggregate(rows: list[dict], n_boot: int = 10000) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["source"], r["kind"], r["group"], r["mode"])].append(r)
    out = []
    for (src, kind, group, mode), rs in sorted(cells.items()):
        names = LABEL_METRICS if kind == "label_set" else RECORD_METRICS
        cell = {"source": src, "kind": kind, "group": group, "mode": mode,
                "n": len(rs), "failures": sum(1 for r in rs if r["error"])}
        for name in names:
            mean, lo, hi = M.bootstrap_ci_np([r["m"][name] for r in rs],
                                             n=n_boot)
            cell[name] = {"mean": round(mean, 4), "lo": round(lo, 4),
                          "hi": round(hi, 4)}
        out.append(cell)
    return out


def _pooled(rows, kind, groups, metric, mode) -> dict[str, float]:
    return {r["rid"]: r["m"][metric] for r in rows
            if r["kind"] == kind and r["group"] in groups
            and r["mode"] == mode}


def _paired(rows, kind, groups, metric, a, b, n_boot):
    pa = _pooled(rows, kind, groups, metric, a)
    pb = _pooled(rows, kind, groups, metric, b)
    rids = sorted(set(pa) & set(pb))
    if not rids:
        return None
    mean, lo, hi = M.paired_bootstrap_diff_np(
        [pa[x] for x in rids], [pb[x] for x in rids], n=n_boot)
    return {"n": len(rids), "mean_diff": round(mean, 4),
            "lo": round(lo, 4), "hi": round(hi, 4)}


def _best_lexical(rows, kind, groups, metric) -> str:
    means = {}
    for mode in ("bm25", "bm25_prefix"):
        v = list(_pooled(rows, kind, groups, metric, mode).values())
        means[mode] = sum(v) / len(v) if v else -1.0
    return max(means, key=means.get)


def endpoint_p1_p2_p3(rows: list[dict], n_boot: int = 10000) -> dict:
    out = {}
    specs = {"P1": ("label_set", ("en_strict", "uk"), "recall@10", 0.15),
             "P2": ("label_set", ("en_natural",), "recall@10", -0.03),
             "P3": ("record", ("en_strict", "uk"), "hit@10", None)}
    for pid, (kind, groups, metric, thr) in specs.items():
        best = _best_lexical(rows, kind, groups, metric)
        hyb = list(_pooled(rows, kind, groups, metric, "hybrid").values())
        entry = {"kind": kind, "groups": list(groups), "metric": metric,
                 "best_lexical_baseline": best, "n": len(hyb),
                 "hybrid_mean": round(sum(hyb) / len(hyb), 4)
                 if hyb else None}
        for other in (best, "dense", "bm25", "bm25_prefix", "raw_bm25"):
            d = _paired(rows, kind, groups, metric, "hybrid", other, n_boot)
            if d:
                entry[f"hybrid_minus_{other}"] = d
        if pid == "P1":
            d = entry.get(f"hybrid_minus_{best}")
            entry["pass"] = bool(d and d["mean_diff"] >= thr and d["lo"] > 0)
        elif pid == "P2":
            d = entry.get(f"hybrid_minus_{best}")
            entry["pass"] = bool(d and d["mean_diff"] >= thr)
        else:
            entry["pass"] = bool(hyb and entry["hybrid_mean"] >= 0.50)
        out[pid] = entry
    return out


SIGNALS = ("top1_z", "top1_sim", "margin", "lex_top")


def endpoint_p7(answerable: dict, unanswerable: dict) -> dict:
    """AUROC of each confidence signal (answerable vs unanswerable)
    per source. `top1_z` is the pre-registered signal; the others are
    exploratory. The pooled value mixes corpora and is secondary."""
    out = {"target": ">= 0.80 per source on top1_z (proposed)",
           "primary_signal": "top1_z", "signals": {}}
    for sig in SIGNALS:
        per, pa, pn = {}, [], []
        for src in answerable:
            a = [r[sig] for r in answerable[src] if r.get(sig) is not None]
            u = [r[sig] for r in unanswerable.get(src, [])
                 if r.get(sig) is not None]
            per[src] = {"auroc": M.auroc(a, u), "n_answerable": len(a),
                        "n_unanswerable": len(u)}
            pa += a
            pn += u
        out["signals"][sig] = {"per_source": per,
                               "pooled_auroc": M.auroc(pa, pn)}
    return out


def endpoint_exploratory(rows: list[dict], n_boot: int = 10000) -> dict:
    """Paired comparisons of every non-registered system (fusion-weight
    sweep, HyDE) against dense-only and the default hybrid."""
    fixed = {"bm25", "bm25_prefix", "dense", "hybrid", "raw_bm25"}
    extra = sorted({r["mode"] for r in rows} - fixed)
    sets = {"label_strict_uk": ("label_set", ("en_strict", "uk"),
                                "recall@10"),
            "label_natural": ("label_set", ("en_natural",), "recall@10"),
            "record_strict_uk": ("record", ("en_strict", "uk"), "hit@10"),
            "record_natural": ("record", ("en_natural",), "hit@10")}
    out = {}
    for name, (kind, groups, metric) in sets.items():
        entry = {"metric": metric}
        for m in ("dense", "hybrid", "bm25_prefix", *extra):
            v = list(_pooled(rows, kind, groups, metric, m).values())
            if v:
                entry[f"mean_{m}"] = round(sum(v) / len(v), 4)
        bases = ("dense", "hybrid") + (
            ("product_default",) if "product_default" in extra else ())
        for m in ("hybrid", *extra):
            for base in bases:
                if m != base:
                    d = _paired(rows, kind, groups, metric, m, base, n_boot)
                    if d:
                        entry[f"{m}_minus_{base}"] = d
        out[name] = entry
    return out


def latency_summary(results_by_source: dict) -> dict:
    per_mode: dict[str, list[float]] = defaultdict(list)
    for res in results_by_source.values():
        for modes in res.values():
            for mode, rec in modes.items():
                per_mode[mode].append(rec["ms"])
    return {m: {"median_ms": round(statistics.median(v), 2),
                "p95_ms": round(sorted(v)[int(0.95 * (len(v) - 1))], 2),
                "n": len(v)} for m, v in per_mode.items()}
