"""
Py-Buddy: empirical evaluation benchmark suite.

Validates product mentions, per-item price accuracy, compatibility verdicts,
retrieval grounding, and safety-gate outcomes against independent fixtures.

What is measured now, and nothing else:

  execution_accuracy   no false claim AND the verdict matches the oracle
  hallucination_rate   share of answers containing a product or a price
                       that the catalogue does not support
  NVFG                 factual AND informative — the metric that separates
                       "blocked everything" (factuality 100%, usefulness 0%)
  price_error_rate     quoted figures that disagree with the catalogue
  by case type         per-family breakdown, so no family can be averaged
                       away

Usage:
    python -u src/benchmark.py --offline --system both
    python -u src/benchmark.py --repeat 3                 # needs API key
    python -u src/benchmark.py --selftest                 # prove the metric
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

# This module lives in src/experiments/: sibling experiments resolve via this
# directory, product-core modules via ../core (docs/13 C2).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core")))

from benchmark_cases import Snapshot, build_cases, load_snapshot  # noqa: E402
from metrics import CaseScore, aggregate, score_case  # noqa: E402

DEFAULT_DB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "marketplace_rozetka.db")
)


# --------------------------------------------------------------------------
# Self-test: is the metric even satisfiable?
# --------------------------------------------------------------------------


def selftest(snap: Snapshot) -> int:
    """
    Feed the oracle's own answer back through the scorer.

    A metric that a perfect answer cannot satisfy is a broken metric. This
    must print 100% on every axis, and it runs with no API key, so any
    reviewer can reproduce it in one command.
    """
    cases = build_cases(snap)
    scores: List[CaseScore] = []
    for c in cases:
        gold = _gold_answer(c)
        scores.append(score_case(gold, c.oracle, snap.names(), snap.price_index))
    agg = aggregate(scores)

    print("=" * 68)
    print("SELFTEST  (oracle answer scored by the metric)")
    print("=" * 68)
    print(f"  cases                 : {agg.n}")
    print(f"  execution accuracy    : {agg.execution_accuracy_pct}%")
    print(f"  verdict accuracy      : {agg.verdict_accuracy_pct}% "
          f"(з {agg.verdict_cases} кейсів, де вердикт існує)")
    print(f"  hallucination rate    : {agg.hallucination_rate_pct}%")
    print(f"  price error rate      : {agg.price_error_rate_pct}%")
    print(f"  NVFG                  : {agg.nvfg_pct}%")
    print(f"  informative rate      : {agg.informative_rate_pct}%")

    ok = (agg.execution_accuracy_pct == 100.0
          and agg.hallucination_rate_pct == 0.0
          and agg.price_error_rate_pct == 0.0
          and agg.nvfg_pct == 100.0)
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL — the metric is not satisfiable'}")
    for s in scores:
        if s.violations:
            print(f"    {s.case_id} [{s.case_type}] -> {s.violations}")
            print(f"      expected verdict={s.expected_verdict} got={s.verdict}")
    return 0 if ok else 1


def _gold_answer(case) -> str:
    """
    Render the oracle's ground truth as a natural-language answer.

    A verdict sentence is appended ONLY where the question actually asks
    for a compatibility decision. The previous version appended one
    whenever `expected_verdict` was "accept", so the gold answer for "which
    CPU is cheapest?" ended with "Ця конфігурація сумісна і підходить" --
    and the selftest then scored 100% on verdict accuracy because the gold
    answer had been written to satisfy the metric rather than because any
    system had earned it.
    """
    o = case.oracle
    parts: List[str] = []
    for name in o.expected_products:
        price = o.expected_prices.get(name)
        parts.append(f"{name} за {price:.2f} грн" if price is not None else name)
    if o.checks_verdict:
        if o.expected_verdict == "accept":
            parts.append("Ця конфігурація сумісна і підходить.")
        elif o.expected_verdict == "reject":
            parts.append("Така конфігурація несумісна, відхилено.")
        else:
            parts.append("Ось відповідь за вашим запитом.")
    elif case.case_type == "price_trap":
        parts.append("Такого товару в нашому каталозі немає.")
    elif not o.expected_products:
        # A greeting has no verdict either, and a refusal is a wrong answer
        # to "hello". This distinction is the whole point of not inventing
        # one.
        parts.append("Вітаю! Чим допомогти з вибором комплектуючих?")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Systems under test
# --------------------------------------------------------------------------


def run_baseline(snap: Snapshot, case, offline: bool) -> Dict[str, object]:
    from baseline_rag import RetrievalBaseline

    if not hasattr(run_baseline, "_agent"):
        # Index construction is timed separately (F26): a run whose
        # index took minutes but answered in ms does not get to call
        # itself fast, and the build cost must not hide in case latency.
        t_build = time.time()
        run_baseline._agent = RetrievalBaseline(
            snap.db_path, runner=None if offline else _make_runner()
        )
        run_baseline._build_s = time.time() - t_build
    agent = run_baseline._agent
    t0 = time.time()
    text, calls, cost = agent.answer(case.query)
    return {
        "text": text, "steps": calls, "latency_s": time.time() - t0, "cost_usd": cost,
    }


def run_pybuddy(snap: Snapshot, case, offline: bool) -> Dict[str, object]:
    """
    Run the proposed system.

    Offline mode runs the deterministic tool layer without an LLM, which
    measures the *verification* contribution only. It is reported under its
    own name and must never be quoted as a live-LLM result.
    """
    if offline:
        from offline_agent import OfflineDeterministicAgent
        if not hasattr(run_pybuddy, "_agent"):
            t_build = time.time()
            run_pybuddy._agent = OfflineDeterministicAgent(snap)
            run_pybuddy._build_s = time.time() - t_build
        agent = run_pybuddy._agent
    else:
        from live_agent import LivePyBuddyAgent
        if not hasattr(run_pybuddy, "_agent"):
            t_build = time.time()
            run_pybuddy._agent = LivePyBuddyAgent(snap, _make_runner())
            run_pybuddy._build_s = time.time() - t_build
        agent = run_pybuddy._agent

    t0 = time.time()
    out = agent.answer(case.query)
    return {
        "text": out["text"],
        "steps": out.get("steps", 0),
        "latency_s": time.time() - t0,
        "cost_usd": out.get("cost_usd", 0.0),
    }


def _make_runner():
    try:
        from openrouter_client import OpenRouterAgentRunner

        return OpenRouterAgentRunner()
    except Exception as exc:  # pragma: no cover
        print(f"[warn] LLM runner unavailable ({exc}); falling back to offline",
              file=sys.stderr)
        return None


def _live_agent(snap: Snapshot, budget: Optional[Dict[str, object]] = None):
    """
    The live arm, built once per suite run.

    A missing runner raises here instead of degrading to the deterministic
    agent: `--system live` reporting the offline arm would be a fabricated
    measurement, which is precisely what this arm exists to be honest about.
    """
    budget = budget or {}
    if not hasattr(_live_agent, "_agent"):
        from live_agent import LivePyBuddyAgent

        runner = _make_runner()
        if runner is None:
            raise SystemExit(
                "live arm needs OPENROUTER_API_KEY (see .env.example): a live run "
                "without a model is not a cheaper run, it is not a run")
        _live_agent._agent = LivePyBuddyAgent(
            snap, runner,
            max_llm_calls=int(budget.get("max_llm_calls", 100)),
            max_usd=budget.get("max_usd"),
            price_in_per_mtok=float(budget.get("price_in_per_mtok", 0.0)),
            price_out_per_mtok=float(budget.get("price_out_per_mtok", 0.0)),
        )
    return _live_agent._agent


def run_live(snap: Snapshot, case, budget: Optional[Dict[str, object]] = None
             ) -> Dict[str, object]:
    """One case through the live agent; an error is recorded, never invented."""
    agent = _live_agent(snap, budget)
    t0 = time.time()
    out = agent.answer(case.query)
    return {"text": out["text"], "steps": out["steps"], "latency_s": time.time() - t0,
            "cost_usd": out["cost_usd"]}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_suite(snap: Snapshot, system: str, offline: bool,
              budget: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    cases = build_cases(snap)
    scores: List[CaseScore] = []
    latencies: List[float] = []
    verbose = True
    for c in cases:
        if system == "live":
            # A live arm may run out of budget or lose a call. Both are recorded
            # and the case is scored on what actually came back, rather than
            # aborting the suite and losing the cases that did answer.
            try:
                res = run_live(snap, c, budget)
            except SystemExit:
                raise
            except Exception as exc:                      # pragma: no cover - guard
                res = {"text": "", "steps": 0, "latency_s": 0.0, "cost_usd": 0.0,
                       "error": f"{type(exc).__name__}: {exc}"}
        else:
            runner: Callable = run_baseline if system == "baseline" else run_pybuddy
            res = runner(snap, c, offline)
        latencies.append(float(res["latency_s"]))
        s = score_case(
            res["text"], c.oracle, snap.names(), snap.price_index,
            steps=int(res["steps"]), latency_s=float(res["latency_s"]),
            cost_usd=float(res["cost_usd"]),
            structured_verdict=res.get("verdict"),
            user_query=c.query,
        )
        scores.append(s)
        if verbose and not s.is_correct:
            print(f"  [miss] {c.case_id} [{c.case_type}] {s.violations}")
    agg = aggregate(scores)
    out: Dict[str, object] = {
        "system": ("live" if system == "live"
                   else f"{system}{'_offline' if offline else ''}"),
        "aggregate": agg.__dict__,
        "cases": [s.to_dict() for s in scores],
        # F26: build/index time, cold first-query latency and warm mean
        # are reported apart -- an index built slowly but queried fast is
        # measured, not averaged away.
        "timing": {
            "index_build_s": getattr(
                run_baseline if system == "baseline" else run_pybuddy,
                "_build_s", None),
            "cold_latency_s": latencies[0] if latencies else None,
            "warm_mean_s": (round(statistics.fmean(latencies[1:]), 6)
                            if len(latencies) > 1 else None),
            "queries": len(cases),
        },
    }
    if system == "live":
        # The raw responses, the model, the tokens and the caps in force. Without
        # this a non-deterministic arm cannot be reproduced or disputed.
        out["live_meta"] = _live_agent(snap, budget).stats()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--system", choices=("pybuddy", "baseline", "both", "live"),
                    default="both",
                    help="`live` runs the ReAct arm with a real model and "
                         "requires OPENROUTER_API_KEY; it is never implied")
    ap.add_argument("--offline", action="store_true",
                    help="no LLM; measures the deterministic layer only")
    ap.add_argument("--max-llm-calls", type=int, default=60,
                    help="hard cap on model calls for --system live (checked "
                         "before each call, so an over-budget run stops free)")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="USD cap for --system live; armed only when a token "
                         "price is configured, and reported as such")
    ap.add_argument("--token-price-in", type=float, default=0.0,
                    help="USD per 1M prompt tokens; 0 for a :free model")
    ap.add_argument("--token-price-out", type=float, default=0.0,
                    help="USD per 1M completion tokens; 0 for a :free model")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None, help="write full JSON report here")
    args = ap.parse_args(argv)

    snap = load_snapshot(args.db)
    print(f"[benchmark] catalog: {len(snap.products)} products, "
          f"{len(snap.concepts)} discovered concepts", flush=True)

    if args.selftest:
        return selftest(snap)

    systems = ("baseline", "pybuddy") if args.system == "both" else (args.system,)
    budget: Dict[str, object] = {
        "max_llm_calls": args.max_llm_calls,
        "max_usd": args.max_usd,
        "price_in_per_mtok": args.token_price_in,
        "price_out_per_mtok": args.token_price_out,
    }
    report: Dict[str, object] = {"cases_per_run": len(build_cases(snap)), "runs": []}
    if systems == ("live",):
        report["live_budget"] = budget

    for system in systems:
        for rep in range(args.repeat):
            print(f"\n=== {system} (run {rep + 1}/{args.repeat}) ===", flush=True)
            report["runs"].append(run_suite(snap, system, args.offline, budget))
            agg = report["runs"][-1]["aggregate"]  # type: ignore[index]
            print(f"  execution_accuracy : {agg['execution_accuracy_pct']}%")
            print(f"  verdict_accuracy   : {agg['verdict_accuracy_pct']}%")
            print(f"  hallucination_rate : {agg['hallucination_rate_pct']}%")
            print(f"  price_error_rate   : {agg['price_error_rate_pct']}%")
            print(f"  NVFG               : {agg['nvfg_pct']}%")
            print(f"  informative        : {agg['informative_rate_pct']}%")
            print(f"  avg_steps/latency  : {agg['avg_steps']} / {agg['avg_latency_s']}s")
            meta = report["runs"][-1].get("live_meta")  # type: ignore[union-attr]
            if meta:
                print(f"  model              : {meta['model']}")
                print(f"  llm_calls/tokens   : {meta['llm_calls']} "
                      f"({meta['prompt_tokens']}+{meta['completion_tokens']})")
                print(f"  cost               : ${meta['cost_usd']} "
                      f"[{meta['budget_mode']}, cap {meta['max_llm_calls']} calls"
                      f"{'' if meta['max_usd'] is None else ', cap $' + str(meta['max_usd'])}]")
                if meta["truncated"]:
                    print("  WARNING            : budget reached, run truncated — "
                          "the numbers below cover the cases that answered")

    if args.repeat > 1:
        report["stability"] = _stability(report["runs"])
        print("\n=== stability across runs ===")
        print(json.dumps(report["stability"], indent=2, ensure_ascii=False))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[benchmark] full report -> {args.out}")
    return 0


def _stability(runs: List[Dict[str, object]]) -> Dict[str, object]:
    """mean and std of the headline metric across repeated runs."""
    out: Dict[str, object] = {}
    for system in {r["system"] for r in runs}:  # type: ignore[index]
        vals = [
            r["aggregate"]["execution_accuracy_pct"]  # type: ignore[index]
            for r in runs if r["system"] == system  # type: ignore[index]
        ]
        out[str(system)] = {
            "runs": vals,
            "mean": round(statistics.fmean(vals), 2) if vals else 0.0,
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
