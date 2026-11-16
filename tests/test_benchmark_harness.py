"""
End-to-end check of the benchmark harness itself.

The single most important test in the repository: it proves the metric is
SATISFIABLE. A benchmark that a perfect answer cannot score 100% on is
worse than no benchmark, because every number it reports is a statement
about the metric rather than about the system.

Run directly:  python -u src/benchmark.py --selftest
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from benchmark import DEFAULT_DB, _gold_answer, selftest
from benchmark_cases import build_cases, load_snapshot
from metrics import aggregate, score_case

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_DB),
    reason="marketplace catalog not generated; run data/generate_marketplace_scale.py",
)


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(DEFAULT_DB)


def test_suite_is_large_enough(snap):
    cases = build_cases(snap)
    assert len(cases) >= 40, f"only {len(cases)} cases; the paper must not claim 100"
    families = {c.case_type for c in cases}
    assert len(families) >= 6, f"only {len(families)} case families"
    # Deterministic: same seed -> same suite.
    assert [c.case_id for c in build_cases(snap)] == [c.case_id for c in cases]


def test_every_family_is_represented(snap):
    cases = build_cases(snap)
    for family in ("direct_price", "cheapest_category", "priciest_category",
                   "availability", "compat_accept", "compat_reject",
                   "overconstrained", "price_trap", "conversational"):
        assert any(c.case_type == family for c in cases), f"missing {family}"


def test_oracle_answer_scores_perfectly(snap):
    """The gold answer must score 100% on every axis."""
    cases = build_cases(snap)
    scores = [
        score_case(_gold_answer(c), c.oracle, snap.names(), snap.price_index)
        for c in cases
    ]
    agg = aggregate(scores)
    assert agg.execution_accuracy_pct == 100.0, [
        s.to_dict() for s in scores if s.violations
    ][:3]
    assert agg.hallucination_rate_pct == 0.0
    assert agg.price_error_rate_pct == 0.0
    assert agg.nvfg_pct == 100.0
    assert agg.blocked_rate_pct == 0.0


def test_selftest_entrypoint_passes(snap):
    assert selftest(snap) == 0


def test_price_trap_cases_have_no_expected_product(snap):
    traps = [c for c in build_cases(snap) if c.case_type == "price_trap"]
    assert traps, "no price-trap cases: the suite cannot detect invention"
    for c in traps:
        assert c.oracle.expected_products == []
        assert c.meta["fabricated"] not in snap.price_index


def test_honest_refusal_is_informative_not_vacuous(snap):
    """
    A correct "this product does not exist" must count as informative.

    Otherwise the metric would punish the single most desirable behaviour
    the system has, and NVFG would cap below 100% by construction.
    """
    traps = [c for c in build_cases(snap) if c.case_type == "price_trap"]
    c = traps[0]
    s = score_case(
        "Такого товару в нашому каталозі немає, ціну назвати неможливо.",
        c.oracle, snap.names(), snap.price_index,
    )
    assert s.informative
    assert not s.blocked
    assert not s.violations


def test_blocked_answer_is_not_informative(snap):
    c = [x for x in build_cases(snap) if x.case_type == "direct_price"][0]
    s = score_case(
        "Помилка верифікації фактів. Відповідь заблоковано Zero-Hallucination Gate.",
        c.oracle, snap.names(), snap.price_index,
    )
    assert s.blocked
    assert not s.informative
