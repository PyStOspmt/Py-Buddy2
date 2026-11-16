"""
R2 part 7 regressions: protected constraints, bounded enumeration, Pareto frontier dominance, and elicitation contracts.

F21 remainder:
  P18  kept (non-relaxable) equality constraints were excluded from the
       lattice optimisation: the cheapest relaxation served only items
       violating a kept constraint, and the solver then reported
       UNSATISFIABLE although a more expensive feasible relaxation existed.
  P19  a budget-truncated enumeration was reported as infeasibility rather
       than as an incomplete search.
  P20  reversed Pareto dominance kept a strictly dominated bound on the
       frontier offered to the customer.
  plus the source-supported defects the audit names: tightening infinities
  admitted as relaxations, two encodings of an unchanged bound, and
  unvalidated zero/negative/non-finite regret weights.

F22:
  the quantity called "entropy" is now the differential entropy of the
  Beta posterior over the latent parameter (epistemic uncertainty), while
  the old expected Bernoulli outcome entropy survives under an honest
  name; customer answers fold into a persistent belief carried by the
  session; recorded answers shape the Pareto ranking weights; and a
  missing question bank is reported as missing, not as confidence.
"""
import json
import math
import os
import random
import sqlite3
import sys
from itertools import product

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from active_inference import (
    DEFAULT_QUESTION_BANK,
    ActiveInferenceEngine,
    UserBeliefState,
    _beta_binary_entropy,
    _beta_differential_entropy,
)
from nvfg import Constraint, NVFGStatus, Op, feasible
from react_agent import ReActAgent, SessionState
from relaxation_lattice import RelaxationLattice, solve

CATALOG = [
    {"name": "Бюджетний кулер", "price": 800.0, "power": 45.0, "size": 120.0},
    {"name": "Середній кулер", "price": 1500.0, "power": 105.0, "size": 140.0},
    {"name": "Потужний кулер", "price": 3200.0, "power": 250.0, "size": 160.0},
    {"name": "Тихієнкий кулер", "price": 2600.0, "power": 30.0, "size": 140.0},
]

BUDGET = lambda v: Constraint("бюджет", "price", Op.AT_MOST, v, "грн")
POWER_MIN = lambda v: Constraint("потужність", "power", Op.AT_LEAST, v, "Вт")


def _make_db(tmp_path):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?)",
        [("p1", "Ноутбук Pro 15", "laptops", 25000.0, 5, 80.0),
         ("p2", "Мишка бездротова", "accessories", 500.0, 40, 60.0)])
    conn.commit()
    conn.close()
    return str(db)


class ScriptedRunner:
    """Returns canned ReAct steps in order (test double, never production)."""

    def __init__(self, *steps):
        self.steps = list(steps)

    def generate_react_step(self, **kwargs):
        if self.steps:
            return self.steps.pop(0)
        return {"final_answer": "Немає даних.", "recommended_items": []}


# ---------------------------------------------------------------------------
# P18 · kept constraints are enforced during the optimisation, not after it
# ---------------------------------------------------------------------------

def test_p18_kept_constraints_are_enforced_during_optimisation():
    """Failed before: UNSATISFIABLE.

    The cheapest bound that admits anything under the price constraint
    alone is 150 -- it rescues only the wrong-socket item, so the bridge
    re-checked against the kept equality, saw an empty set and declared
    the query unsatisfiable. The honest answer is the more expensive 250
    relaxation serving the socket-compatible item.
    """
    items = [
        {"name": "Wrong socket, cheap", "socket": 4.0, "price": 150.0},
        {"name": "Right socket, pricey", "socket": 5.0, "price": 250.0},
    ]
    constraints = [
        Constraint("сокет", "socket", Op.EQ, 5.0),
        BUDGET(100.0),
    ]
    r = solve(items, constraints)
    assert r.status == NVFGStatus.RELAXED
    assert r.items == ["Right socket, pricey"]
    assert "250" in r.explain()


def test_p18_a_kept_constraint_that_nothing_meets_stays_unsatisfiable():
    """Hard requirements are not relaxed to manufacture usefulness."""
    items = [{"name": "x", "socket": 4.0, "price": 250.0}]
    r = solve(items, [Constraint("сокет", "socket", Op.EQ, 7.0), BUDGET(100.0)])
    assert r.status == NVFGStatus.UNSATISFIABLE
    assert "сокет" in r.reason


def test_minimal_matches_exhaustive_search_with_kept_constraints():
    """Small-instance oracle: enumerate every candidate relaxation, keep
    those feasible under kept+applied, take the regret-minimal one."""
    rng = random.Random(41)
    for _ in range(60):
        relaxable = tuple(
            (BUDGET(rng.uniform(100, 4000)) if rng.random() < 0.5
             else POWER_MIN(rng.uniform(10, 300)))
            for _ in range(rng.randint(1, 3)))
        kept = ((Constraint("сокет", "socket", Op.EQ, rng.choice([4.0, 5.0])),)
                if rng.random() < 0.5 else ())
        items = CATALOG + [{
            "name": "extra", "socket": 5.0,
            "price": rng.uniform(50, 4000), "power": rng.uniform(20, 300),
        }]
        lattice = RelaxationLattice(relaxable)
        res = lattice.minimal(items, kept=kept)

        best, best_key = None, None
        for combo in product(*lattice.candidates(items)):
            r = tuple(combo)
            if feasible(items, list(kept) + lattice.applied(r)):
                key = (lattice.regret(r), lattice._order_key(r))
                if best_key is None or key < best_key:
                    best, best_key = r, key
        assert res.complete
        assert res.relaxation == best


# ---------------------------------------------------------------------------
# P19 · truncated search is an incomplete search, not infeasibility
# ---------------------------------------------------------------------------

def test_p19_a_truncated_search_reports_incomplete_not_infeasible():
    """Failed before: UNSATISFIABLE after examining 1 of 36 combinations."""
    r = solve(CATALOG, [BUDGET(100.0), POWER_MIN(900.0)], budget=1)
    assert r.status == NVFGStatus.UNKNOWN
    assert r.reason
    assert r.items == []


def test_p19_a_hit_before_the_budget_still_answers_with_a_caveat():
    r = solve(CATALOG, [BUDGET(100.0), POWER_MIN(900.0)], budget=12)
    assert r.status == NVFGStatus.RELAXED
    assert r.items
    assert "обмежено" in r.explain()


def test_p19_an_exhausted_search_reports_true_infeasibility():
    """Exhaustive and empty is real UNSATISFIABLE, not UNKNOWN."""
    rows = [{"name": "тільки AM4", "socket": 4.0, "price": 100.0}]
    r = solve(rows, [Constraint("сокет", "socket", Op.EQ, 5.0), BUDGET(10.0)])
    assert r.status == NVFGStatus.UNSATISFIABLE


# ---------------------------------------------------------------------------
# P20 · dominance on the frontier
# ---------------------------------------------------------------------------

def test_p20_a_dominated_relaxation_leaves_the_frontier():
    """Failed before: (3000,) stayed on the frontier next to (500,).

    Relaxing the budget to 500 rescues "mid" for a movement of 300;
    relaxing it to 3000 rescues both for a movement of 2800. The first
    strictly dominates the second on the only axis that exists, and a
    dominated bound is not a trade-off to offer a customer.
    """
    items = [{"name": "mid", "price": 500.0}, {"name": "top", "price": 3000.0}]
    lattice = RelaxationLattice((BUDGET(200.0),))
    assert lattice.pareto(items) == [(500.0,)]


def test_p20_genuinely_incomparable_options_survive():
    items = [{"name": "A", "price": 3000.0, "power": 300.0},
             {"name": "B", "price": 100.0, "power": 10.0}]
    lattice = RelaxationLattice((BUDGET(200.0), POWER_MIN(250.0)))
    options = lattice.pareto(items)
    assert (3000.0, None) in options
    assert (None, 10.0) in options
    assert len(options) == 2


def test_p20_frontier_matches_pairwise_dominance_on_small_instances():
    """Independent oracle: pairwise dominance over every admissible
    relaxation with a finite cost vector."""
    rng = random.Random(7)
    for _ in range(40):
        cons = tuple(
            (BUDGET(rng.uniform(100, 4000)) if rng.random() < 0.5
             else POWER_MIN(rng.uniform(10, 300)))
            for _ in range(rng.randint(1, 3)))
        lattice = RelaxationLattice(cons)
        admissible, _ = lattice.admissible(CATALOG)
        costs = {r: cv for r in admissible
                 if (cv := lattice._cost_vector(r)) is not None}
        expected = sorted(
            (r for r, cv in costs.items()
             if not any(o != r and lattice._dominates(costs[o], cv)
                        for o in costs)),
            key=lambda r: (lattice.regret(r), lattice._order_key(r)))
        assert lattice.pareto(CATALOG) == expected


# ---------------------------------------------------------------------------
# Additional audit-named defects
# ---------------------------------------------------------------------------

def test_a_tightening_infinity_is_rejected():
    """Only the infinity in the loosening direction drops a bound."""
    le = RelaxationLattice((BUDGET(1000.0),))
    assert le.is_relaxation((math.inf,))           # price <= inf: no bound
    assert not le.is_relaxation((-math.inf,))      # price <= -inf: empty
    ge = RelaxationLattice((POWER_MIN(200.0),))
    assert ge.is_relaxation((-math.inf,))          # power >= -inf: no bound
    assert not ge.is_relaxation((math.inf,))       # power >= inf: empty


def test_an_unchanged_bound_has_one_canonical_encoding():
    """v == c.value and None are the same lattice point; both encodings
    must behave identically everywhere the value is read."""
    lattice = RelaxationLattice((BUDGET(1000.0),))
    assert lattice.canonicalize((1000.0,)) == lattice.bottom()
    assert lattice.moved((1000.0,)) == []
    assert lattice.regret((1000.0,)) == 0.0
    assert lattice.leq(lattice.bottom(), (1000.0,))
    assert lattice.leq((1000.0,), lattice.bottom())


def test_regret_weights_must_be_positive_and_finite():
    lattice = RelaxationLattice((BUDGET(1000.0), POWER_MIN(200.0)))
    r = (2000.0, 100.0)
    for bad in ([0.0, 1.0], [-1.0, 1.0], [math.nan, 1.0],
                [math.inf, 1.0], [0.0, 0.0]):
        with pytest.raises(ValueError):
            lattice.regret(r, weights=bad)
    assert lattice.regret(r, weights=[1.0, 1.0]) == pytest.approx(1.5)


def test_invalid_weights_fail_the_search_early_not_midway():
    lattice = RelaxationLattice((BUDGET(1000.0),))
    with pytest.raises(ValueError):
        lattice.minimal(CATALOG, weights=[-1.0])


# ---------------------------------------------------------------------------
# F22 · epistemic entropy, honest objective
# ---------------------------------------------------------------------------

def _numeric_differential_entropy(alpha, beta, n=40001):
    """-int f(p) ln f(p) dp for f ~ Beta(alpha, beta), by quadrature."""
    log_b = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    total = 0.0
    for i in range(1, n):
        p = i / n
        logf = ((alpha - 1.0) * math.log(p)
                + (beta - 1.0) * math.log1p(-p) - log_b)
        total += -math.exp(logf) * logf
    return total / n


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 2.0), (5.0, 3.0), (10.0, 1.0)])
def test_beta_differential_entropy_matches_quadrature(alpha, beta):
    assert _beta_differential_entropy(alpha, beta) == pytest.approx(
        _numeric_differential_entropy(alpha, beta), abs=2e-3)


def test_the_flat_posterior_has_zero_epistemic_entropy_not_one_bit():
    """Beta(1,1) is uniform over [0,1]: differential entropy 0 nats.
    The old 1 bit was the entropy of a fair coin flip, not of the belief."""
    assert _beta_differential_entropy(1.0, 1.0) == pytest.approx(0.0, abs=1e-9)
    assert _beta_differential_entropy(10.0, 10.0) < _beta_differential_entropy(2.0, 2.0)


def test_dimension_entropy_is_epistemic_and_outcome_entropy_keeps_its_name():
    state = UserBeliefState(dimension_names=["d"])
    assert state.dimension_entropy("d") == pytest.approx(0.0, abs=1e-9)
    state.bayesian_update("d", 0.9)
    assert state.dimension_entropy("d") < 0.0
    alpha, beta = state.to_dict()["betas"]["d"]
    assert state.dimension_outcome_entropy("d") == pytest.approx(
        _beta_binary_entropy(alpha, beta))


def test_expected_information_gain_targets_posterior_uncertainty():
    state = UserBeliefState()
    q = DEFAULT_QUESTION_BANK[0]
    engine = ActiveInferenceEngine()
    first = engine.expected_information_gain(state, q)
    assert first > 0.2
    for _ in range(8):
        state.bayesian_update(q.target_dimension, 0.95)
    # Once the answer is already known the gain collapses toward zero:
    # the residual comes from the small predictive mass on an answer
    # that would loosen the posterior again, not from new uncertainty.
    assert engine.expected_information_gain(state, q) < first * 0.1


# ---------------------------------------------------------------------------
# F22 · answers fold into a persistent belief that feeds ranking
# ---------------------------------------------------------------------------

def test_the_tool_folds_a_customer_answer_into_the_belief(tmp_path):
    agent = ReActAgent(db_path=_make_db(tmp_path))
    state = SessionState(session_id="s1", user_query="порадь щось")
    step = agent.execute_tool(state, "active_inference_step", {})
    assert step["success"] and step["question_id"]
    dim = step["target_dimension"]
    option = next(iter(step["options"]))

    res = agent.execute_tool(
        state, "active_inference_step",
        {"question_id": step["question_id"], "answer": option})
    assert res["success"] is True
    assert res["target_dimension"] == dim
    assert state.belief_state.posterior_mean(dim) != pytest.approx(0.5)
    assert step["question_id"] in state.asked_question_ids


def test_an_unknown_answer_option_is_rejected_not_folded(tmp_path):
    agent = ReActAgent(db_path=_make_db(tmp_path))
    state = SessionState(session_id="s2", user_query="q")
    res = agent.execute_tool(
        state, "active_inference_step",
        {"question_id": "q_budget", "answer": "вигадана опція"})
    assert res["success"] is False
    assert state.belief_state.posterior_mean("price_sensitivity") == pytest.approx(0.5)
    assert state.asked_question_ids == []


def test_a_question_is_not_asked_twice_within_a_session(tmp_path):
    agent = ReActAgent(db_path=_make_db(tmp_path))
    state = SessionState(session_id="s3", user_query="q", max_steps=30)
    first = agent.execute_tool(state, "active_inference_step", {})
    qid = first["question_id"]
    agent.execute_tool(state, "active_inference_step",
                       {"question_id": qid, "answer": next(iter(first["options"]))})
    asked = set()
    for _ in range(6):
        step = agent.execute_tool(state, "active_inference_step", {})
        if not step.get("question_id"):
            break
        asked.add(step["question_id"])
        agent.execute_tool(
            state, "active_inference_step",
            {"question_id": step["question_id"],
             "answer": next(iter(step["options"]))})
    assert qid not in asked


def test_no_question_bank_is_distinguished_from_confidence(tmp_path):
    """Failed before: an empty bank and a confident belief produced the
    same 'sufficient certainty' message."""
    db = _make_db(tmp_path)
    agent = ReActAgent(db_path=db,
                       active_inference=ActiveInferenceEngine([]))
    state = SessionState(session_id="s4", user_query="q")
    res = agent.execute_tool(state, "active_inference_step", {})
    assert res["success"] is True and res["question_id"] is None
    assert res["reason"] == "no_question_bank"

    confident = ReActAgent(db_path=db)
    state2 = SessionState(session_id="s5", user_query="q")
    for dim in state2.belief_state.dimension_names:
        for _ in range(30):
            state2.belief_state.bayesian_update(dim, 0.99)
    res2 = confident.execute_tool(state2, "active_inference_step", {})
    assert res2["success"] is True and res2["question_id"] is None
    assert res2["reason"] == "below_gain_threshold"


def test_asked_questions_and_belief_survive_a_session_roundtrip():
    state = SessionState(session_id="s6", user_query="q")
    state.asked_question_ids.append("q_budget")
    state.belief_state.bayesian_update("price_sensitivity", 0.95)
    restored = SessionState.from_dict(json.loads(state.to_json()))
    assert restored.asked_question_ids == ["q_budget"]
    assert restored.belief_state.posterior_mean("price_sensitivity") > 0.7


def test_a_persisted_belief_is_loaded_into_the_run_not_reset(tmp_path):
    """Failed before: run_live_session always started from a flat prior."""
    agent = ReActAgent(db_path=_make_db(tmp_path))
    prior = UserBeliefState()
    prior.bayesian_update("price_sensitivity", 0.95)
    state = agent.run_live_session(
        "Що порадиш?", ScriptedRunner({"final_answer": "Ок.", "recommended_items": []}),
        schema_prompt="s", session_id="s7",
        belief_state=prior, asked_question_ids=["q_budget"])
    assert state.belief_state is prior
    assert state.belief_state.posterior_mean("price_sensitivity") > 0.7
    assert state.asked_question_ids == ["q_budget"]


def test_recorded_answers_shape_the_pareto_weights():
    """The elicited preference must reach ranking, not die in the belief."""
    state = SessionState(session_id="s8", user_query="q")
    assert ReActAgent._preference_weights(state) is None   # nothing said yet
    state.asked_question_ids.append("q_budget")
    state.belief_state.bayesian_update("price_sensitivity", 0.95)
    weights = ReActAgent._preference_weights(state)
    assert weights is not None
    assert weights["price"] > weights["performance"]
    assert weights["price"] > weights["delivery"]
