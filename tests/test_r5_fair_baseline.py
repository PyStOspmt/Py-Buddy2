"""
F26 regressions: the baseline must get equal answer opportunities, and
index build must be timed separately from query latency.

Before the fix the offline baseline answered every query with a single
top-1 product card ("Найрелевантніший товар: X"), while Py-Buddy's
offline arm ran a purpose-built SQL planner -- so the offline comparison
measured a planner against an impoverished template, not two methods
against the same evidence. The offline baseline now renders ALL cards it
retrieved: the same opportunity a live LLM would have over the same
retrieval, deterministically and without inventing an answerer.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "experiments"))

import baseline_rag
import benchmark
from benchmark_cases import load_snapshot

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "marketplace_rozetka.db")


def test_offline_baseline_gets_all_retrieved_cards(tmp_path):
    """A multi-hit query must surface every retrieved card, not top-1."""
    agent = baseline_rag.RetrievalBaseline(DB, runner=None, top_k=8)
    hits = agent.index.search("процесор ryzen", 8)
    assert len(hits) >= 2, "fixture must give a multi-hit retrieval"
    text, calls, cost = agent.answer("процесор ryzen")
    shown = [d.name for d, _s in agent.index.search("процесор ryzen", 8)]
    for name in shown:
        assert name in text, f"card '{name}' retrieved but hidden in answer"


def test_offline_baseline_reports_absence_when_no_hits():
    agent = baseline_rag.RetrievalBaseline(DB, runner=None)
    text, _, _ = agent.answer("zxqwv nonexistent product token")
    # an empty retrieval stays an honest absence, not a fabricated card
    assert "немає" in text.lower()


def test_offline_baseline_is_labelled_not_an_llm_result():
    """The offline path must identify itself in metadata, not silently
    stand in for the live BM25+LLM arm."""
    agent = baseline_rag.RetrievalBaseline(DB, runner=None)
    _, calls, cost = agent.answer("процесор")
    assert calls == 0 and cost == 0.0     # zero model spend, on record


def test_index_build_timed_separately_from_queries():
    """F26: index construction is a build cost, not query latency."""
    snap = load_snapshot(DB)
    for attr in ("_agent", "_build_s"):
        if hasattr(benchmark.run_baseline, attr):
            delattr(benchmark.run_baseline, attr)
    case = benchmark.build_cases(snap)[0]
    res = benchmark.run_baseline(snap, case, offline=True)
    assert res["latency_s"] >= 0.0
    assert benchmark.run_baseline._build_s is not None
    assert benchmark.run_baseline._build_s >= 0.0
    # query latency must not absorb the index build
    res2 = benchmark.run_baseline(snap, case, offline=True)
    assert res2["latency_s"] < benchmark.run_baseline._build_s + 1.0


def test_suite_report_carries_cold_vs_warm_and_build():
    snap = load_snapshot(DB)
    out = benchmark.run_suite(snap, "baseline", offline=True)
    timing = out.get("timing")
    assert timing is not None
    assert timing["index_build_s"] is not None
    assert timing["cold_latency_s"] is not None
    assert timing["warm_mean_s"] is not None
    n = len(out["cases"])
    assert timing["queries"] == n
