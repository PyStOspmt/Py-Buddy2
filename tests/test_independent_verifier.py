"""
Differential test: oracle verifier vs independent verifier (F26 fix, R0).

The oracle (benchmark_cases.check_bundle, Value-Algebra based) and the
independent verifier (independent_verifier.check_bundle_independent, raw
SQL + own parsing) must AGREE on every compatibility case of the benchmark.
Before F26, the offline agent imported the oracle's own verifier, so
agent-vs-oracle agreement was a tautology of shared code. Now agreement
between two independently written implementations is evidence; disagreement
is surfaced here as a defect of one of them, not hidden.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "core"))

from benchmark_cases import build_cases, check_bundle, load_snapshot
from independent_verifier import check_bundle_independent

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "marketplace_rozetka.db")


def test_independent_verifier_agrees_with_oracle_on_all_compat_cases():
    snap = load_snapshot(DB)
    disagreements = []
    for case in build_cases(snap):
        bundle = case.meta.get("bundle")
        if not bundle:
            continue
        oracle = check_bundle(snap, bundle)
        indep = check_bundle_independent(DB, bundle)
        if oracle.ok != indep.ok:
            disagreements.append({
                "case_id": case.case_id,
                "bundle": bundle,
                "oracle_ok": oracle.ok,
                "oracle_reasons": oracle.reasons,
                "independent_ok": indep.ok,
                "independent_reasons": indep.reasons,
            })
    assert not disagreements, (
        "oracle and independent verifier disagree -- one of them is wrong; "
        f"{len(disagreements)} case(s): {disagreements[:3]}")


def test_independent_verifier_rejects_a_socket_conflict():
    # Synthetic check of the verifier's own logic on real DB rows: pick two
    # CPUs with different sockets and assert a rejection reason appears.
    import sqlite3
    conn = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
    rows = conn.execute(
        "SELECT id FROM products WHERE category='cpu' LIMIT 200"
    ).fetchall()
    conn.close()
    ids = [r[0] for r in rows]
    conflicts = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            v = check_bundle_independent(DB, [ids[i], ids[j]])
            if not v.ok and any("сокет" in r for r in v.reasons):
                conflicts.append((ids[i], ids[j]))
                break
        if conflicts:
            break
    assert conflicts, "expected at least one socket conflict among sampled CPUs"


def test_independent_verifier_accepts_undecidable_bundle():
    # Two SSDs carry no socket/PCD/clearance invariants -> accepted, same
    # semantics as the oracle (absence of evidence is not incompatibility).
    import sqlite3
    conn = sqlite3.connect("file:%s?mode=ro" % DB.replace("\\", "/"), uri=True)
    rows = conn.execute(
        "SELECT id FROM products WHERE category='ssd' LIMIT 2"
    ).fetchall()
    conn.close()
    if len(rows) < 2:
        return
    v = check_bundle_independent(DB, [rows[0][0], rows[1][0]])
    assert v.ok