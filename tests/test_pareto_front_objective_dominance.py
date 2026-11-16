"""
Pareto-front regression suite: the front is the non-dominated set over price
(minimise) and performance (maximise), and explanations agree with the numbers.

Live reproduction (pc_hardware, POST /api/pareto with budget_max=12000 and
min_performance=80, observed 2026-11-10): the returned front held eight points,
four of them strictly dominated by another returned point. LIVE_ROWS copies the
observed (name, price, performance) triples as a regression fixture only.
"""

import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pareto_relaxer import CandidateSolution, FuzzyParetoRelaxer

LIVE_ROWS = [
    ("Kingston Fury Beast 32GB (2x16) DDR5-6000", 4400.0, 92.0),
    ("Corsair RM750e 750W 80+ Gold ATX 3.0", 4600.0, 88.0),
    ("be quiet! Pure Power 12 M 850W Gold", 5600.0, 93.0),
    ("MSI B650 Gaming Plus WiFi", 6800.0, 85.0),
    ("Seasonic Focus GX-1000 1000W 80+ Gold", 8200.0, 97.0),
    ("ASUS Prime Z790-P", 8400.0, 88.0),
    ("G.Skill Trident Z5 RGB 64GB DDR5-6400", 9200.0, 98.0),
    ("AMD Ryzen 5 7600X", 9500.0, 82.0),
]
LIVE_FRONT = {
    "Kingston Fury Beast 32GB (2x16) DDR5-6000",
    "be quiet! Pure Power 12 M 850W Gold",
    "Seasonic Focus GX-1000 1000W 80+ Gold",
    "G.Skill Trident Z5 RGB 64GB DDR5-6400",
}


def _dominates(a, b):
    (pa, sa), (pb, sb) = a, b
    return pa <= pb and sa >= sb and (pa < pb or sa > sb)


def _oracle_front(rows):
    return {
        name for name, price, perf in rows
        if not any(_dominates((p2, s2), (price, perf))
                   for n2, p2, s2 in rows if n2 != name)
    }


def _front_names(rows, budget, min_perf):
    relaxer = FuzzyParetoRelaxer(budget_max=budget, min_performance=min_perf)
    cands = [CandidateSolution(item_id=n, name=n, price=p, performance_score=s)
             for n, p, s in rows]
    return {c.item_id for c in relaxer.compute_pareto_front(cands)}


def test_live_reproduction_front_holds_only_non_dominated_points():
    assert _oracle_front(LIVE_ROWS) == LIVE_FRONT
    assert _front_names(LIVE_ROWS, 12000.0, 80.0) == LIVE_FRONT


def test_front_matches_brute_force_oracle_on_seeded_random_sets():
    for seed in range(300):
        rng = random.Random(seed)
        rows = [(f"c{i}", float(rng.randint(1, 20) * 500), float(rng.randint(50, 100)))
                for i in range(rng.randint(1, 25))]
        budget = float(rng.choice([5000, 10000, 15000]))
        min_perf = rng.choice([None, 60.0, 80.0])
        assert _front_names(rows, budget, min_perf) == _oracle_front(rows), f"seed={seed}"


def test_equal_objectives_both_stay_on_the_front():
    rows = [("a", 5000.0, 90.0), ("b", 5000.0, 90.0)]
    assert _front_names(rows, 10000.0, 80.0) == {"a", "b"}


def test_same_price_higher_performance_dominates():
    rows = [("a", 5000.0, 80.0), ("b", 5000.0, 90.0)]
    assert _front_names(rows, 10000.0, 80.0) == {"b"}


def test_item_silent_on_performance_cannot_dominate_or_be_dominated_on_it():
    relaxer = FuzzyParetoRelaxer(budget_max=1000.0, min_performance=50.0)
    full = CandidateSolution(item_id="full", name="Full", price=500.0, performance_score=80.0)
    partial = CandidateSolution(item_id="partial", name="Partial", price=100.0,
                                performance_score=None)
    assert {c.item_id for c in relaxer.compute_pareto_front([full, partial])} == {"full", "partial"}


def test_item_with_unknown_delivery_is_not_dominated_through_delivery():
    relaxer = FuzzyParetoRelaxer(budget_max=10000.0, min_performance=80.0)
    a = CandidateSolution(item_id="a", name="A", price=100.0, performance_score=90.0,
                          delivery_days=1)
    b = CandidateSolution(item_id="b", name="B", price=200.0, performance_score=80.0)
    assert {c.item_id for c in relaxer.compute_pareto_front([a, b])} == {"a", "b"}


def test_explanation_does_not_deny_an_existing_exact_match():
    relaxer = FuzzyParetoRelaxer(budget_max=10000.0, min_performance=80.0)
    cands = [
        CandidateSolution(item_id="a", name="Alpha", price=6000.0, performance_score=85.0),
        CandidateSolution(item_id="b", name="Beta", price=7000.0, performance_score=95.0),
        CandidateSolution(item_id="c", name="Gamma", price=12000.0, performance_score=99.0),
    ]
    front = relaxer.compute_pareto_front(cands)
    _, _, explanation = relaxer.select_contrastive_tradeoffs(front)
    assert "не виявлено" not in explanation
    assert "Alpha" in explanation or "Beta" in explanation


def test_explanation_names_each_violated_constraint_when_nothing_fits():
    relaxer = FuzzyParetoRelaxer(budget_max=10000.0, min_performance=90.0)
    cands = [
        CandidateSolution(item_id="x", name="Over", price=11000.0, performance_score=95.0),
        CandidateSolution(item_id="y", name="Weak", price=9000.0, performance_score=80.0),
    ]
    front = relaxer.compute_pareto_front(cands)
    _, _, explanation = relaxer.select_contrastive_tradeoffs(front)
    assert "перевищує бюджет" in explanation
    assert "нижча за мінімальну продуктивність" in explanation
    assert "розширений альтернативний" not in explanation
