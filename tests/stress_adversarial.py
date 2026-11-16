"""Adversarial stress harness: run each pathological fixture through the
real production pipeline pieces and print a structured verdict.

Not a pytest module -- a measurement harness. Run:
    python tests/stress_adversarial.py
"""
import json
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))

ADV = os.path.join(ROOT, "data", "adversarial")


def evaluate(db_path: str) -> dict:
    import fiduciary_probe
    from domain_catalog import DomainCatalogRegistry
    from source_projection import build_source_projection

    rep: dict = {"db": os.path.basename(db_path)}
    t0 = time.perf_counter()

    # --- projection
    try:
        proj = build_source_projection(db_path)
        rep["entity_table"] = proj.entity_table if proj else None
        rep["roles"] = {}
        if proj:
            for role in ("identity", "name", "category", "price",
                         "stock", "performance"):
                r = proj.resolve(role)
                rep["roles"][role] = (r.column if r.resolved else None)
            rep["unassigned_numeric"] = list(proj.unassigned_numeric)
    except Exception as exc:
        rep["projection_error"] = f"{type(exc).__name__}: {exc}"

    # --- catalogue build (induction + hypergraph)
    t1 = time.perf_counter()
    try:
        kg = DomainCatalogRegistry.create_database_catalog(db_path,
                                                           seed_limit=120)
        rep["graph_nodes"] = len(kg.nodes)
        rep["constraints"] = len(kg.constraints)
        rep["constraint_kinds"] = {}
        for c in kg.constraints:
            k = type(c).__name__
            rep["constraint_kinds"][k] = rep["constraint_kinds"].get(k, 0) + 1
    except Exception as exc:
        rep["catalog_error"] = f"{type(exc).__name__}: {exc}"
        kg = None
    rep["catalog_seconds"] = round(time.perf_counter() - t1, 2)

    # --- fiduciary probe
    try:
        f = fiduciary_probe.probe_source(db_path)
        rep["fiduciary"] = {k: f.get(k) for k in
                            ("status", "evaluated_items", "reason")}
        rep["fiduciary"]["channels"] = len(f.get("channels") or [])
    except Exception as exc:
        rep["fiduciary"] = {"error": f"{type(exc).__name__}: {exc}"}

    rep["total_seconds"] = round(time.perf_counter() - t0, 2)
    return rep


def main():
    dbs = sorted(
        os.path.join(ADV, f) for f in os.listdir(ADV) if f.endswith(".db"))
    results = []
    for db in dbs:
        try:
            results.append(evaluate(db))
        except Exception:
            results.append({"db": os.path.basename(db),
                            "fatal": traceback.format_exc()[-400:]})
    print(json.dumps(results, indent=1, ensure_ascii=False, default=str))
    out = os.path.join(ROOT, "docs", "stress_adversarial_20261005.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False, default=str)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
