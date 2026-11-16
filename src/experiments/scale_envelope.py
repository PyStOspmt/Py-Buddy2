"""
Py-Buddy: paired scale envelope (F27 / F34 residual).

What the audit requires, and what this module does about it
-----------------------------------------------------------
F27: the previously reported "scale comparison" changed the workload
denominator -- the small-corpus run had 39 cases including ten
absent-product cases, the large-corpus run had 29 without them, and the
headline delta was a fraction change (8/39 vs 8/29), not measured
behaviour change. F34: no evidence of search behaviour at arbitrary
scale was ever produced.

The protocol here removes both defects:

  * PAIRED, VERSIONED QUERY SET. One fixed case list (QUERY_SET_VERSION)
    is materialised once from the full catalogue and then run verbatim
    at every corpus size. Cases are never re-generated per tier, so the
    denominator cannot drift.

  * NESTED CORPORA. Each tier is a strict superset of the previous one:
    a deterministic pseudo-random permutation of product rowids
    ((rowid * K) mod M, insertion-order-independent) is taken, and the
    tier-N catalogue is the first N products of that one permutation.
    Corpus growth is therefore the ONLY thing that changes between
    tiers -- not which entities exist, not the query set.

  * TRUTH TRANSITIONS ARE REPORTED, NOT HIDDEN. An oracle computed per
    tier can legitimately differ across tiers (a product name absent at
    500 rows is present at 10 000). Each case records its oracle
    signature at every tier, and every change is listed in
    `truth_transitions` instead of being silently absorbed into a
    changed denominator.

  * BUILD VS QUERY TIME. Snapshot/index construction is timed
    separately from per-query latency (F26 residual): a run whose index
    took minutes but answered in milliseconds does not get to call
    itself "fast".

  * RESOURCES AND ENVIRONMENT ARE DISCLOSED. DB bytes, row counts,
    entity diversity (distinct names/categories), RSS delta, timeouts,
    python/sqlite versions and CPU count are recorded so the envelope
    is a measured claim about THIS hardware, not an "any size" promise.

Measured arm: `pybuddy_offline`, the deterministic core without a model.
The live LLM arm is out of scope (needs an authorised keyed budget);
nothing here claims live-model latency.

Usage:
    python src/experiments/scale_envelope.py \
        --db data/marketplace_rozetka.db \
        --sizes 500,2000,10000 --repeat 3 \
        --out docs/scale_envelope_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sqlite3
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from benchmark_cases import Snapshot, load_snapshot          # noqa: E402
from metrics import Oracle, score_case            # noqa: E402
from offline_agent import OfflineDeterministicAgent          # noqa: E402

QUERY_SET_VERSION = "scale_queries_v1"
TIMEOUT_S = 60.0

# Deterministic pseudo-random permutation of rowids: (rowid * K) mod M.
# Insertion order in the generator is category-major, so a plain rowid
# prefix would build a tier that is ALL cpus and nothing else -- the
# multiplicative permutation spreads every category across every tier
# while keeping tiers strictly nested (tier N is a prefix of tier N+1
# of the SAME permutation).
_PERM_K = 2654435761
_PERM_M = 4294967291


# --------------------------------------------------------------------------
# Nested subset catalogues
# --------------------------------------------------------------------------

def _permutation_ids(src_path: str) -> List[str]:
    conn = sqlite3.connect("file:%s?mode=ro" % src_path.replace("\\", "/"),
                           uri=True)
    try:
        return [r[0] for r in conn.execute(
            "SELECT id FROM products "
            "ORDER BY (rowid * %d) %% %d, id" % (_PERM_K, _PERM_M))]
    finally:
        conn.close()


def make_subset_db(src_path: str, dst_path: str, n: int) -> int:
    """
    Write a tier catalogue: the first n products of the shared
    deterministic permutation, with their EAV rows, commercial rows and
    merchants. Returns the number of products actually written (n may
    exceed the corpus, in which case the tier equals the full set).
    """
    ids = _permutation_ids(src_path)[:n]
    if os.path.exists(dst_path):
        os.remove(dst_path)
    src = sqlite3.connect(src_path)
    dst = sqlite3.connect(dst_path)
    try:
        schema = [r[0] for r in src.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND type IN ('table','index') "
            "ORDER BY type DESC")]
        for ddl in schema:
            if "sqlite_sequence" in ddl:
                continue
            dst.execute(ddl)
        # A tier can hold 10k+ ids, past SQLite's bound-variable limit --
        # stage them in a temp table instead of an IN (...) literal.
        src.execute("CREATE TEMP TABLE _tier_ids(id TEXT PRIMARY KEY)")
        src.executemany(
            "INSERT OR IGNORE INTO _tier_ids VALUES (?)", [(i,) for i in ids])
        in_tier = "IN (SELECT id FROM _tier_ids)"
        # Rows are read from the SOURCE connection and written to the
        # tier file -- an INSERT..SELECT inside the dst connection would
        # silently select from dst's own empty tables.
        dst.executemany(
            "INSERT INTO products VALUES (%s)" % ",".join(
                "?" * len(src.execute(
                    "SELECT * FROM products LIMIT 1").description)),
            src.execute(
                f"SELECT * FROM products WHERE id {in_tier}"))
        dst.executemany(
            "INSERT INTO product_eav_meta(product_id, meta_key, meta_value) "
            "VALUES (?,?,?)",
            src.execute(
                f"SELECT product_id, meta_key, meta_value "
                f"FROM product_eav_meta WHERE product_id {in_tier}"))
        try:
            dst.executemany(
                "INSERT INTO commercial VALUES (?,?,?)",
                src.execute(
                    f"SELECT * FROM commercial WHERE product_id {in_tier}"))
        except sqlite3.OperationalError:
            pass                      # no commercial table: fine
        try:
            dst.executemany(
                "INSERT OR IGNORE INTO merchants VALUES (%s)" % ",".join(
                    "?" * len(src.execute(
                        "SELECT * FROM merchants LIMIT 1").description)),
                src.execute(
                    f"SELECT DISTINCT m.* FROM merchants m JOIN products p "
                    f"ON p.merchant_id = m.id WHERE p.id {in_tier}"))
        except sqlite3.OperationalError:
            pass                      # no merchants table: fine
        dst.commit()
    finally:
        src.close()
        dst.close()
    return len(ids)


def corpus_stats(db_path: str) -> Dict[str, object]:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"),
                           uri=True)
    try:
        row = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT name), "
            "COUNT(DISTINCT category) FROM products").fetchone()
        eav = conn.execute("SELECT COUNT(*) FROM product_eav_meta").fetchone()[0]
    finally:
        conn.close()
    return {
        "products": row[0],
        "distinct_names": row[1],
        "categories": row[2],
        "eav_rows": eav,
        "db_bytes": os.path.getsize(db_path),
    }


# --------------------------------------------------------------------------
# The fixed query set
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ScaleCase:
    case_id: str
    family: str
    query: str
    #: snapshot -> Oracle. Recomputed per tier: the QUERY is fixed, the
    #: truth is whatever that tier's catalogue actually supports.
    oracle: Callable[[Snapshot], Oracle]


def materialize_cases(full: Snapshot) -> List[ScaleCase]:
    """
    Fix the workload once. Product names are chosen by their position in
    the shared permutation (10%, 50%, 95%), so the three named-price
    cases have a designed gradient of presence across tiers -- and any
    absence that appears at a small tier is a reported truth transition,
    not a dropped case.
    """
    order = _permutation_ids(full.db_path)
    names = [full.id_to_name[order[int(len(order) * f)]]
             for f in (0.10, 0.50, 0.95)]

    def price_of(name: str) -> Callable[[Snapshot], Oracle]:
        def build(snap: Snapshot) -> Oracle:
            if name in snap.price_index:
                return Oracle(
                    case_id="", case_type="direct_price",
                    expected_products=[name],
                    expected_prices={name: sorted(snap.price_index[name])},
                    expected_verdict="neutral", checks_verdict=False)
            # Absent in this tier: a truthful answer refuses; expected
            # reject is the *detected absence* verdict (F15-adjacent:
            # whether the agent can tell "unknown" from "absent" is
            # measured, not assumed).
            return Oracle(
                case_id="", case_type="absent_product",
                expected_products=[], expected_prices={},
                expected_verdict="reject", checks_verdict=True)
        return build

    def cheapest(cat: str) -> Callable[[Snapshot], Oracle]:
        def build(snap: Snapshot) -> Oracle:
            rows = [(p["name"], float(p["price"]))
                    for p in snap.products.values()
                    if p.get("category") == cat and p.get("price") is not None]
            if not rows:
                return Oracle(case_id="", case_type="cheapest",
                              expected_products=[], expected_verdict="reject",
                              checks_verdict=True)
            name, price = min(rows, key=lambda t: (t[1], t[0]))
            return Oracle(case_id="", case_type="cheapest",
                          expected_products=[name],
                          expected_prices={name: [price]},
                          expected_verdict="neutral", checks_verdict=False)
        return build

    def availability(cat: str) -> Callable[[Snapshot], Oracle]:
        def build(snap: Snapshot) -> Oracle:
            rows = sorted(
                (p for p in snap.products.values()
                 if p.get("category") == cat
                 and int(p.get("stock_quantity") or 0) > 0),
                key=lambda p: (float(p["price"]), p["name"]))[:5]
            names = sorted({str(p["name"]) for p in rows})
            prices = {str(p["name"]): float(p["price"]) for p in rows}
            return Oracle(case_id="", case_type="availability",
                          expected_products=names, expected_prices=prices,
                          expected_verdict="neutral", checks_verdict=False)
        return build

    def budget(cat: str, amount: float) -> Callable[[Snapshot], Oracle]:
        def build(snap: Snapshot) -> Oracle:
            rows = sorted(
                ((float(p["price"]), str(p["name"]))
                 for p in snap.products.values()
                 if p.get("category") == cat
                 and int(p.get("stock_quantity") or 0) > 0),
                key=lambda t: (t[0], t[1]))
            if not rows or rows[0][0] > amount:
                return Oracle(case_id="", case_type="budget",
                              expected_products=[], expected_verdict="reject",
                              checks_verdict=True)
            return Oracle(case_id="", case_type="budget",
                          expected_products=[rows[0][1]],
                          expected_prices={rows[0][1]: [rows[0][0]]},
                          expected_verdict="neutral", checks_verdict=True)
        return build

    cases = [
        ScaleCase("sq_price_early", "direct_price",
                  f"Скільки коштує {names[0]}?", price_of(names[0])),
        ScaleCase("sq_price_mid", "direct_price",
                  f"Скільки коштує {names[1]}?", price_of(names[1])),
        ScaleCase("sq_price_late", "direct_price",
                  f"Скільки коштує {names[2]}?", price_of(names[2])),
        ScaleCase("sq_absent", "absent_product",
                  "Скільки коштує HyperFlux ZX-9000 Quantum?",
                  price_of("HyperFlux ZX-9000 Quantum")),
    ]
    # Category-bearing cases are included only where the corpus actually
    # carries the category. On a corpus without them the query set is the
    # price/absent core -- including a case the corpus cannot ground would
    # measure the case list, not the system. The realised case ids are
    # recorded in the report, so the denominator is never implicit.
    have = {str(p.get("category")) for p in full.products.values()}
    if "cpu" in have:
        cases.append(ScaleCase("sq_cheapest_cpu", "cheapest",
                               "Який найдешевший процесор у каталозі?",
                               cheapest("cpu")))
        cases.append(ScaleCase("sq_budget_cpu", "budget",
                               "Порадь процесор до 1500 грн",
                               budget("cpu", 1500.0)))
    if "gpu" in have:
        cases.append(ScaleCase("sq_cheapest_gpu", "cheapest",
                               "Який найдешевший відеокарта у каталозі?",
                               cheapest("gpu")))
        cases.append(ScaleCase("sq_avail_gpu", "availability",
                               "Що є в наявності з відеокарт?",
                               availability("gpu")))
    return cases


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def _rss_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def run_tier(db_path: str, cases: Sequence[ScaleCase],
             repeat: int) -> Dict[str, object]:
    """Build the snapshot + agent (timed separately), run the fixed set."""
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    snap = load_snapshot(db_path)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    agent = OfflineDeterministicAgent(snap)
    init_s = time.perf_counter() - t0
    rss1 = _rss_mb()

    case_rows: List[Dict[str, object]] = []
    latencies: List[float] = []
    timeouts = 0
    for rep in range(repeat):
        for case in cases:
            oracle = case.oracle(snap)
            oracle.case_id = case.case_id
            oracle.case_type = case.family
            t0 = time.perf_counter()
            res = agent.answer(case.query)
            lat = time.perf_counter() - t0
            if lat > TIMEOUT_S:
                timeouts += 1
            s = score_case(res["text"], oracle, snap.names(),
                           snap.price_index, latency_s=lat,
                           structured_verdict=res.get("verdict"))
            latencies.append(lat)
            case_rows.append({
                "case_id": case.case_id, "family": case.family,
                "repeat": rep, "latency_s": round(lat, 4),
                "correct": s.is_correct, "violations": s.violations,
                "expected_products": list(oracle.expected_products),
                "expected_verdict": oracle.expected_verdict,
            })
    n = len(latencies)
    mean = statistics.fmean(latencies) if latencies else 0.0
    std = statistics.pstdev(latencies) if n > 1 else 0.0
    return {
        "corpus": corpus_stats(db_path),
        "build_s": round(build_s, 3),
        "agent_init_s": round(init_s, 4),
        "rss_mb_delta": (round(rss1 - rss0, 1)
                         if rss0 is not None and rss1 is not None else None),
        "queries": n,
        "timeouts": timeouts,
        "latency_s": {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci95_halfwidth": round(1.96 * std / (n ** 0.5), 4) if n else 0.0,
            "p50": round(statistics.median(latencies), 4) if latencies else 0.0,
            "p95": round(sorted(latencies)[int(0.95 * (n - 1))], 4)
            if latencies else 0.0,
            "max": round(max(latencies), 4) if latencies else 0.0,
        },
        "cases": case_rows,
    }


def truth_transitions(tiers: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Cases whose oracle changed between tiers -- the quantity F27's bad
    report hid behind a changed denominator.
    """
    out: List[Dict[str, object]] = []
    case_ids = sorted({c["case_id"] for t in tiers for c in t["cases"]})
    for cid in case_ids:
        sigs = []
        for t in tiers:
            row = next(c for c in t["cases"] if c["case_id"] == cid)
            sigs.append((tuple(row["expected_products"]),
                         row["expected_verdict"]))
        if len(set(sigs)) > 1:
            out.append({
                "case_id": cid,
                "per_tier": [
                    {"products": list(s[0]), "verdict": s[1]} for s in sigs
                ],
            })
    return out


def run_envelope(db_path: str, sizes: Sequence[int],
                 repeat: int = 3,
                 workdir: Optional[str] = None,
                 keep_tiers: bool = False) -> Dict[str, object]:
    full = load_snapshot(db_path)
    cases = materialize_cases(full)
    workdir = workdir or os.path.join(
        os.path.dirname(db_path), "_scale_envelope_tmp")
    os.makedirs(workdir, exist_ok=True)

    tiers: List[Dict[str, object]] = []
    written: List[str] = []
    try:
        for n in sorted(set(sizes)):
            sub = os.path.join(workdir, f"tier_{n}.db")
            actual = make_subset_db(db_path, sub, n)
            written.append(sub)
            row = run_tier(sub, cases, repeat)
            row["requested_size"] = n
            row["actual_products"] = actual
            tiers.append(row)
    finally:
        if not keep_tiers:
            for path in written:
                if os.path.exists(path):
                    os.remove(path)

    for t in tiers:
        correct = sum(1 for c in t["cases"] if c["correct"])
        t["execution_accuracy_pct"] = round(
            100.0 * correct / len(t["cases"]), 2) if t["cases"] else 0.0

    return {
        "experiment": "scale_envelope",
        "query_set_version": QUERY_SET_VERSION,
        "query_set": [c.case_id for c in cases],
        "arm": "pybuddy_offline (deterministic core, no LLM)",
        "method": {
            "subsets": "nested prefixes of one deterministic rowid "
                       "permutation (rowid * %d mod %d)" % (_PERM_K, _PERM_M),
            "queries": f"{len(cases)} fixed cases, identical at every tier",
            "repeats": repeat,
            "note": "oracles recomputed per tier; truth changes are "
                    "reported, denominators never change",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "sqlite": sqlite3.sqlite_version,
            "db_bytes": os.path.getsize(db_path),
            "db_mtime": int(os.path.getmtime(db_path)),
        },
        "tiers": tiers,
        "truth_transitions": truth_transitions(tiers),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--sizes", default="500,2000,10000")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--keep-tiers", action="store_true",
                    help="leave the generated tier databases on disk "
                         "(default: remove them; they are deterministic "
                         "and reproducible from --db)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    report = run_envelope(args.db, sizes, args.repeat,
                          keep_tiers=args.keep_tiers)

    print(f"{'tier':>7} {'rows':>6} {'names':>6} {'build_s':>8} "
          f"{'lat mean':>9} {'p95':>8} {'acc%':>6} {'timeouts':>8}")
    for t in report["tiers"]:
        c = t["corpus"]
        lat = t["latency_s"]
        print(f"{t['requested_size']:>7} {c['products']:>6} "
              f"{c['distinct_names']:>6} {t['build_s']:>8} "
              f"{lat['mean']:>9} {lat['p95']:>8} "
              f"{t['execution_accuracy_pct']:>6} {t['timeouts']:>8}")
    if report["truth_transitions"]:
        print("\ntruth transitions (oracle changed across tiers):")
        for tr in report["truth_transitions"]:
            print(f"  {tr['case_id']}: {tr['per_tier']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[scale_envelope] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
