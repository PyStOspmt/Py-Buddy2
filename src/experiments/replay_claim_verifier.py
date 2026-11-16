"""
Replay measurement: post-generation claim verifier on recorded live answers.

Takes a saved benchmark artifact (per-case query + final answer text +
evaluator flags) and the catalogue DB the run was grounded on, then asks:
for each recorded answer, does the sentence-level verifier flag claims the
independent evaluator already marked ungrounded -- and does it false-flag
the clean cases?

This is a replay over recorded outputs, not a new live run: the numbers
bound how much of the measured free-text hallucination the verifier would
have caught on the same 49 answers, with zero new model calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.claim_verifier import verify_answer_against_rows


def load_catalog_rows(db_path: str) -> List[Dict[str, Any]]:
    """Every product row with its EAV attributes merged in."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        attrs: Dict[int, Dict[str, str]] = {}
        for r in conn.execute(
                "SELECT product_id, meta_key, meta_value FROM product_eav_meta"):
            attrs.setdefault(r["product_id"], {})[r["meta_key"]] = r["meta_value"]
        rows: List[Dict[str, Any]] = []
        for r in conn.execute(
                "SELECT id, name, category, brand, price, stock_quantity "
                "FROM products"):
            row = dict(r)
            row["attributes"] = attrs.get(r["id"], {})
            rows.append(row)
        return rows
    finally:
        conn.close()


def replay(artifact_path: str, db_path: str) -> Dict[str, Any]:
    artifact = json.load(open(artifact_path, encoding="utf-8"))
    run = artifact["runs"][0]
    case_log = run["live_meta"]["case_log"]
    cases = run["cases"]
    rows = load_catalog_rows(db_path)

    per_case: List[Dict[str, Any]] = []
    caught = clean_flagged = clean_passed = missed = 0

    for ev_case, log in zip(cases, case_log):
        evaluator_flagged = bool(
            ev_case.get("ungrounded_products") or ev_case.get("price_errors")
            or ev_case.get("ungrounded_entities"))
        rep = verify_answer_against_rows(
            log.get("text") or "", rows, log.get("query", ""))
        verifier_flagged = rep.n_unsupported > 0

        if evaluator_flagged and verifier_flagged:
            caught += 1
        elif evaluator_flagged and not verifier_flagged:
            missed += 1
        elif not evaluator_flagged and verifier_flagged:
            clean_flagged += 1
        else:
            clean_passed += 1

        per_case.append({
            "case_id": ev_case.get("case_id"),
            "evaluator_flagged": evaluator_flagged,
            "evaluator_ungrounded": ev_case.get("ungrounded_products"),
            "evaluator_price_errors": ev_case.get("price_errors"),
            "verifier_unsupported_sentences": rep.n_unsupported,
            "verifier_unsupported_claims": [
                u for s in rep.sentences for u in s.unsupported],
            "supported_fraction": round(rep.supported_fraction, 4),
        })

    return {
        "artifact": os.path.basename(artifact_path),
        "catalog": os.path.basename(db_path),
        "model": run["live_meta"].get("model"),
        "n_cases": len(cases),
        "confusion": {
            "caught_flagged": caught,
            "missed_flagged": missed,
            "clean_flagged": clean_flagged,
            "clean_passed": clean_passed,
        },
        "coverage_of_evaluator_flags":
            round(caught / (caught + missed), 4) if caught + missed else None,
        "false_flag_rate_on_clean":
            round(clean_flagged / (clean_flagged + clean_passed), 4)
            if clean_flagged + clean_passed else None,
        "per_case": per_case,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = replay(args.artifact, args.db)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
