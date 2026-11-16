"""End-to-end adaptive cycle through the REAL HTTP API (docs/18: C1-C7).

    python run_cycle.py --tag <name> [--sources library,forum,food]
                        [--split test] [--limit N] [--chat]

connect (register unseen source) -> automatic preparation (observed
timeline) -> semantic requests -> point mutations of the source file ->
automatic staleness detection + incremental refresh -> requests again ->
independent checks of the sidecar. Raw outputs go to
docs/semantic_cycle_e2e_<tag>.json (never overwritten).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cycle_mutations as CM  # noqa: E402
import eval_report as R  # noqa: E402
from cycle_server import CycleServer, wait_until  # noqa: E402
from eval_core import REQUEST_FILE, SOURCES, ROOT, environment, load_requests  # noqa: E402,E501

SCENARIO = ROOT / "docs" / "semantic_mutations_20261112.json"
ONBOARD_TIMEOUT = 3600.0


def onboard(srv: CycleServer, src: str, db: Path) -> dict:
    t0 = time.time()
    st, body = srv.api("POST", "/api/register_domain", admin=True, json={
        "domain_id": src, "db_path": str(db), "name": src,
        "async_init": True})
    if st not in (200, 202):
        return {"ok": False, "register_status": st, "body": body}
    timeline: list[dict] = []

    def status():
        _s, b = srv.api("GET", f"/api/domains/{src}/semantic")
        return b

    def tick(_last, el):
        if not timeline or el - timeline[-1]["t"] >= 15:
            b = status() or {}
            timeline.append({"t": round(el, 1), "state": b.get("state"),
                             "progress": b.get("progress")})

    done = lambda: (lambda b: b if b.get("state") in (  # noqa: E731
        "ready", "failed", "unavailable") else None)(status() or {})
    try:
        final, _el = wait_until(done, ONBOARD_TIMEOUT, 5.0, tick)
    except TimeoutError as exc:
        return {"ok": False, "error": str(exc), "timeline": timeline}
    return {"ok": final.get("state") == "ready",
            "onboard_wall_s": round(time.time() - t0, 1),
            "final_status": final, "timeline": timeline,
            "reconciliation": reconcile(final, db),
            "server_memory": srv.memory_mb()}


def reconcile(status: dict, db: Path) -> dict:
    """Independent P6 check: COUNT(*) of the entity table is read here,
    not taken from the system's own report."""
    import sqlite3
    table = (status.get("entity") or {}).get("table")
    cov = status.get("coverage") or {}
    if not table or not cov:
        return {"ok": False, "reason": "no entity/coverage in status"}
    c = sqlite3.connect("file:" + str(db).replace("\\", "/") + "?mode=ro",
                        uri=True)
    n = c.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    c.close()
    excl = sum((cov.get("excluded") or {}).values())
    return {"ok": n == cov.get("source_rows") == (cov.get("indexed") or 0)
            + excl, "count_star": n, "reported_source_rows":
            cov.get("source_rows"), "indexed": cov.get("indexed"),
            "excluded": excl}


def run_requests(srv, src: str, reqs: list[dict], mode: str = "hybrid",
                 k: int = 50) -> dict:
    out = {}
    for req in reqs:
        t0 = time.perf_counter()
        st, b = srv.api("GET", f"/api/domains/{src}/search",
                        params={"q": req["query"], "k": k, "mode": mode})
        out[req["rid"]] = {
            "ids": [h["entity_id"] for h in b.get("hits", [])]
            if st == 200 else [],
            "error": None if st == 200 and b.get("status") == "ok"
            else f"http {st} status {b.get('status')}",
            "ms": round((time.perf_counter() - t0) * 1000, 2),
            "top1_z": (b.get("confidence") or {}).get("top1_z"),
            "fresh": (b.get("index") or {}).get("fresh")}
    return {mode: out}


def score(reqs, run) -> dict:
    """Per-request results (mode -> rid -> rec) in the shape that
    eval_report.score_all expects, then aggregated."""
    mode = next(iter(run))
    results = {rid: {mode: rec} for rid, rec in run[mode].items()}
    return {"rows": R.score_all(reqs, results),
            "groups": R.aggregate(R.score_all(reqs, results), n_boot=2000)}


def mutation_phase(srv, src: str, db: Path, base_reqs, base_run) -> dict:
    sc = CM.load_scenario(SCENARIO)
    before = CM.sidecar_snapshot(str(db))
    st0 = srv.api("GET", f"/api/domains/{src}/semantic")[1]
    t_mut = time.time()
    expected = CM.apply_scenario(str(db), sc)
    probe = srv.api("GET", f"/api/domains/{src}/search",
                    params={"q": "probe after change", "k": 5})[1]

    def fresh():
        b = srv.api("GET", f"/api/domains/{src}/semantic")[1]
        return b if b.get("state") == "ready" and b.get("fresh") else None
    final, _ = wait_until(fresh, 1800.0, 2.0)
    t_fresh = time.time() - t_mut
    after = CM.sidecar_snapshot(str(db))
    return {"expected": {k: len(v) for k, v in expected.items()},
            "status_before": st0, "probe_after_change": {
                "status": probe.get("status"),
                "index": probe.get("index"), "notes": probe.get("notes")},
            "time_to_fresh_s": round(t_fresh, 1),
            "status_after": final,
            "independent": CM.compare_snapshots(before, after, expected),
            "expected_ids": expected}


def mutation_requests(sc: dict, doc: dict, expected: dict) -> list[dict]:
    reqs = []
    for tag, items, key in (("ins", sc["inserts"], "record"),
                            ("upd", sc["updates"], "new_content")):
        for it in items:
            tid = str(it[key]["source_book_id"] if tag == "ins"
                      else it["book_id"])
            for q in ("en_strict", "uk", "en_natural"):
                reqs.append({"rid": f"mut|{tag}|{tid}|{q}",
                             "source": "library", "kind": "record",
                             "target_id": tid, "group": f"mut_{tag}_{q}",
                             "query": it["queries"][q], "relevant": {tid}})
    old = {r["target_id"]: r for r in doc["targets"]
           if r["source"] == "library" and r["kind"] == "record"}
    for tag, ids in (("old_upd", expected["updated"]),
                     ("old_del", expected["deleted"])):
        for tid in ids:
            for q in ("en_strict", "uk"):
                reqs.append({"rid": f"mut|{tag}|{tid}|{q}",
                             "source": "library", "kind": "record",
                             "target_id": tid, "group": f"mut_{tag}_{q}",
                             "query": old[tid]["queries"][q],
                             "relevant": {tid}})
    return reqs


def mean_hit(rows, groups, mode="hybrid") -> float | None:
    v = [r["m"]["hit@10"] for r in rows
         if r["group"] in groups and r["mode"] == mode
         and r["kind"] == "record"]
    return round(sum(v) / len(v), 4) if v else None


def p5_summary(mut, post_rows, base_rows, deleted, all_post_run) -> dict:
    ind = mut["independent"]
    sync = (mut["status_after"].get("last_sync") or {})
    viol = sum(1 for rec in all_post_run.values()
               if set(rec["ids"]) & set(deleted))
    base = mean_hit(base_rows, ("en_strict", "uk"))
    return {
        "system_reported_sync": sync,
        "embedded_texts_reported": sync.get("embedded_texts"),
        "independent_appended_slots": ind["appended_vector_slots"],
        "unchanged_vector_mismatches": ind["unchanged_vector_mismatches"],
        "deleted_ids_returned_in_requests": viol,
        "deleted_still_live": ind["deleted_still_live"],
        "baseline_hit10_strict_uk": base,
        "inserted_hit10_strict_uk": mean_hit(
            post_rows, ("mut_ins_en_strict", "mut_ins_uk")),
        "updated_new_hit10_strict_uk": mean_hit(
            post_rows, ("mut_upd_en_strict", "mut_upd_uk")),
        "updated_old_content_hit10_strict_uk": mean_hit(
            post_rows, ("mut_old_upd_en_strict", "mut_old_upd_uk")),
        "time_to_fresh_s": mut["time_to_fresh_s"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--sources", default="library")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chat", action="store_true",
                    help="P9 consultant smoke (cheap model, capped spend)")
    ap.add_argument("--chat-max-usd", type=float, default=0.03)
    a = ap.parse_args()
    out = ROOT / "docs" / f"semantic_cycle_e2e_{a.tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; results are immutable")
    work = ROOT / "data" / "eval_semantic" / f"cycle_{a.tag}"
    if work.exists():
        raise SystemExit(f"{work} exists; pick a new --tag")
    (work / "sources").mkdir(parents=True)
    doc = load_requests()
    srv = CycleServer(work)
    rep: dict = {"protocol": "docs/BENCHMARKS.md",
                 "tag": a.tag, "split": a.split, "environment": environment(),
                 "request_file_sha256": (REQUEST_FILE.parent / (
                     REQUEST_FILE.name + ".sha256")).read_text().strip(),
                 "sources": {}}
    try:
        for src in a.sources.split(","):
            db = work / "sources" / f"{src}.db"
            shutil.copy2(SOURCES[src][0], db)
        rep["startup_s"] = round(srv.start(), 1)
        for src in a.sources.split(","):
            db = work / "sources" / f"{src}.db"
            s: dict = {"onboarding": onboard(srv, src, db)}
            rep["sources"][src] = s
            if not s["onboarding"]["ok"]:
                continue
            reqs = R.request_list(doc, src, a.split, a.limit)
            run = run_requests(srv, src, reqs)
            sc = score(reqs, run)
            s["baseline"] = {"groups": sc["groups"], "n": len(reqs)}
            s["server_memory_after_queries"] = srv.memory_mb()
            if a.chat:
                import chat_probe
                s["chat"] = chat_probe.run(srv, src, str(db), doc,
                                           a.chat_max_usd)
            if src == "library" and not a.limit:
                s["mutation"] = run_library_mutation(
                    srv, src, db, doc, reqs, run, sc)
        rep["server_memory_final"] = srv.memory_mb()
    finally:
        srv.stop()
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                   encoding="utf-8")
    print("wrote", out)
    return 0


def run_library_mutation(srv, src, db, doc, reqs, run, sc) -> dict:
    mut = mutation_phase(srv, src, db, reqs, run)
    scn = CM.load_scenario(SCENARIO)
    mreqs = mutation_requests(scn, doc, mut["expected_ids"])
    post_all = run_requests(srv, src, reqs + mreqs)["hybrid"]
    post_rows = R.score_all(reqs + mreqs, {
        rid: {"hybrid": rec} for rid, rec in post_all.items()})
    mut["summary"] = p5_summary(mut, post_rows, sc["rows"],
                                mut["expected_ids"]["deleted"], post_all)
    st, b = srv.api("POST", f"/api/domains/{src}/refresh", admin=True,
                    json={"rebuild": False})
    noop, _ = wait_until(lambda: (lambda x: x if x.get("state") == "ready"
                                  else None)(srv.api(
                                      "GET", f"/api/domains/{src}/semantic"
                                  )[1]), 600.0, 2.0)
    mut["noop_refresh"] = {"http": st, "last_sync": noop.get("last_sync")}
    return mut


if __name__ == "__main__":
    sys.exit(main())
