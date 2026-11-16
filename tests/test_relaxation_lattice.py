"""
Tests for the relaxation lattice (src/relaxation_lattice.py).

A theorem asserted in a docstring is worth nothing until something checks
it. The order, the join, the existence claim and the "no free lunch" claim
are each verified below, the first three against exhaustive brute force on
randomly generated systems rather than against hand-picked examples --
a lattice law that holds on the three cases you thought of is not a law.
"""
import itertools
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nvfg import Constraint, NVFGStatus, Op
from nvfg import solve as nvfg_solve
from relaxation_lattice import RelaxationLattice, solve

CATALOG = [
    {"name": "Бюджетний кулер", "price": 800.0, "power": 45.0, "size": 120.0},
    {"name": "Середній кулер", "price": 1500.0, "power": 105.0, "size": 140.0},
    {"name": "Потужний кулер", "price": 3200.0, "power": 250.0, "size": 160.0},
    {"name": "Тихієнкий кулер", "price": 2600.0, "power": 30.0, "size": 140.0},
]

BUDGET = lambda v: Constraint("бюджет", "price", Op.AT_MOST, v, "грн")
POWER_MIN = lambda v: Constraint("потужність", "power", Op.AT_LEAST, v, "Вт")
SIZE_MAX = lambda v: Constraint("розмір", "size", Op.AT_MOST, v, "мм")


# --------------------------------------------------------------------------
# The lattice laws
# --------------------------------------------------------------------------

def test_bottom_is_below_everything_and_top_above_everything():
    """`leq(a, b)` reads "a is at least as loose as b"."""
    L = RelaxationLattice((BUDGET(1000), POWER_MIN(200)))
    bottom, top = L.bottom(), L.top()
    for combo in itertools.product(*L.candidates(CATALOG)):
        r = tuple(combo)
        assert L.leq(bottom, r)
        assert L.leq(r, top)
        assert L.leq(r, r)


def test_join_and_meet_are_commutative_idempotent_and_ordered():
    rng = random.Random(20260928)
    for _ in range(200):
        k = rng.randint(1, 3)
        cons = tuple(
            (BUDGET(rng.uniform(100, 4000)) if rng.random() < 0.5
             else POWER_MIN(rng.uniform(10, 300)))
            for _ in range(k)
        )
        L = RelaxationLattice(cons)
        pool = [tuple(c) for c in itertools.islice(
            itertools.product(*L.candidates(CATALOG)), 0, 40)]
        a, b = rng.choice(pool), rng.choice(pool)
        assert L.join(a, a) == a and L.meet(a, a) == a
        assert L.join(a, b) == L.join(b, a)
        assert L.meet(a, b) == L.meet(b, a)
        # join is above both, meet is below both
        assert L.leq(a, L.join(a, b)) and L.leq(b, L.join(a, b))
        assert L.leq(L.meet(a, b), a) and L.leq(L.meet(a, b), b)
        # a Galois pair: meet == a exactly when a is the tighter of the two
        assert L.leq(a, b) == (L.meet(a, b) == a)
        assert L.leq(b, a) == (L.join(a, b) == a)


def test_a_tighter_bound_is_refused_outright():
    """A budget cut is not a relaxation, and the lattice says so loudly."""
    L = RelaxationLattice((BUDGET(1000),))
    assert not L.is_relaxation((500.0,))
    assert L.is_relaxation((2000.0,))
    assert L.is_relaxation(L.bottom())
    assert L.is_relaxation(L.top())
    with pytest.raises(ValueError):
        L.leq((500.0,), L.bottom())


def test_equality_constraints_are_refused():
    """The lattice is defined over inequalities; an `=` has no direction."""
    with pytest.raises(ValueError):
        RelaxationLattice((Constraint("сокет", "socket", Op.EQ, 5.0, "AM5"),))


# --------------------------------------------------------------------------
# Theorem 1: existence
# --------------------------------------------------------------------------

def test_top_is_admissible_whenever_the_catalogue_is_non_empty():
    """Theorem 1(a)-(b). If a category has a product, some answer exists."""
    rng = random.Random(7)
    for _ in range(200):
        cons = tuple(
            (BUDGET(rng.uniform(100, 4000)) if rng.random() < 0.5
             else POWER_MIN(rng.uniform(10, 300)))
            for _ in range(rng.randint(1, 3))
        )
        L = RelaxationLattice(cons)
        assert L.feasible(CATALOG, L.top()) == CATALOG


def test_admissible_relaxations_are_closed_under_join():
    """Theorem 1(c). If two moves work, the looser of them works too."""
    rng = random.Random(11)
    for _ in range(100):
        cons = tuple(
            (BUDGET(rng.uniform(100, 4000)) if rng.random() < 0.5
             else POWER_MIN(rng.uniform(10, 300)))
            for _ in range(rng.randint(2, 3))
        )
        L = RelaxationLattice(cons)
        admissible = [r for r in itertools.product(*L.candidates(CATALOG))
                      if L.feasible(CATALOG, tuple(r))]
        for a, b in itertools.combinations(admissible, 2):
            assert L.feasible(CATALOG, L.join(a, b)), (cons, a, b)


def test_an_empty_catalogue_admits_nothing():
    """The theorem is conditioned on C != {}; without it, it is false."""
    L = RelaxationLattice((BUDGET(1000),))
    assert L.feasible([], L.top()) == []
    assert L.minimal([]).relaxation is None


# --------------------------------------------------------------------------
# Theorem 2: no free lunch
# --------------------------------------------------------------------------

def test_zero_regret_means_nothing_was_moved():
    """Theorem 2. The cheapest answer is the one that changes no bound."""
    L = RelaxationLattice((BUDGET(1000), POWER_MIN(200)))
    assert L.regret(L.bottom()) == 0.0
    assert L.moved(L.bottom()) == []
    for combo in itertools.product(*L.candidates(CATALOG)):
        r = tuple(combo)
        assert (L.regret(r) == 0.0) == (not L.moved(r))


def test_regret_is_the_share_of_each_requirement():
    L = RelaxationLattice((BUDGET(1000), POWER_MIN(200)))
    assert L.regret((2000.0, None)) == pytest.approx(1.0)    # doubled the budget
    assert L.regret((None, 100.0)) == pytest.approx(0.5)     # halved the power
    assert L.regret((2000.0, 100.0)) == pytest.approx(1.5)


def test_weights_change_which_relaxation_is_cheapest():
    """Two admissible moves, one cheap and large in absolute terms.

    A breaks only the budget, B only the power floor, so either bound can
    move alone. Raising the budget 200 -> 3000 gives up 1400% of the
    requirement; dropping the power floor 250 -> 10 gives up 96%. Equal
    weights pick the power floor; weighting the budget heavily flips it.
    """
    items = [{"name": "A", "price": 3000.0, "power": 300.0},
             {"name": "B", "price": 100.0, "power": 10.0}]
    L = RelaxationLattice((BUDGET(200), POWER_MIN(250)))
    assert L.moved(L.minimal(items).relaxation) == [(POWER_MIN(250), 10.0)]
    assert L.moved(L.minimal(items, weights=[0.01, 10.0]).relaxation) == \
        [(BUDGET(200), 3000.0)]


# --------------------------------------------------------------------------
# Theorem 3: the search is complete, and says so
# --------------------------------------------------------------------------

def test_a_truncated_search_reports_that_it_was_truncated():
    found, complete = RelaxationLattice((BUDGET(1000), POWER_MIN(200))).admissible(
        CATALOG, budget=3)
    assert not complete
    assert RelaxationLattice((BUDGET(1000),)).minimal(CATALOG, budget=2).truncated
    assert not RelaxationLattice((BUDGET(1000),)).minimal(CATALOG).truncated


def test_the_case_nvfg_gives_up_on_is_solved_here():
    """No single move rescues {price <= 100, power >= 900}. Two do.

    This is the whole reason the lattice exists: the earlier UNSATISFIABLE
    was a fact about a one-move search, not about the catalogue.
    """
    r = nvfg_solve(CATALOG, [BUDGET(100.0), POWER_MIN(900)])
    assert r.status == NVFGStatus.UNSATISFIABLE
    assert r.items == []

    r2 = solve(CATALOG, [BUDGET(100.0), POWER_MIN(900)])
    assert r2.status == NVFGStatus.RELAXED
    assert r2.items                                  # something real to buy
    assert r2.regret > 0
    assert "бюджет" in r2.explain() and "потужність" in r2.explain()


def test_the_lattice_agrees_with_nvfg_when_a_single_move_suffices():
    """Same decision, and the same cost -- in comparable units.

    `nvfg.regret` is the raw movement in the constraint's own unit (155 W);
    the lattice's regret is the SHARE given up (0.775), because it sums over
    several constraints and watts do not add up to hryvnias. The two are
    reconciled by `relative_regret`, which is why both fields exist.
    """
    a = nvfg_solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    b = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    assert a.status == b.status == NVFGStatus.RELAXED
    assert a.items == b.items
    assert a.relative_regret == pytest.approx(b.relative_regret)


def test_no_item_is_ever_invented_by_the_lattice_either():
    names = {i["name"] for i in CATALOG}
    for constraints in ([BUDGET(100.0), POWER_MIN(900)], [BUDGET(1.0)],
                        [BUDGET(1000), POWER_MIN(200), SIZE_MAX(121.0)]):
        r = solve(CATALOG, constraints)
        assert set(r.items) <= names


def test_a_non_relaxable_constraint_still_blocks():
    rows = [{"name": "Тільки AM5", "socket": 5.0, "price": 100.0}]
    r = solve(rows, [Constraint("сокет", "socket", Op.EQ, 4.0, "AM4"), BUDGET(10.0)])
    assert r.status == NVFGStatus.UNSATISFIABLE
    assert r.items == []
    assert "сокет" in r.reason


def test_pareto_returns_incomparable_options():
    """A breaks the budget, B breaks the power floor: two honest choices.

    Raising the budget to 3000 and dropping the power floor to 10 rescue
    different products and neither is "more correct" than the other. A
    single winner here would be the configurator inventing a preference the
    customer never stated.
    """
    items = [{"name": "A", "price": 3000.0, "power": 300.0},
             {"name": "B", "price": 100.0, "power": 10.0}]
    L = RelaxationLattice((BUDGET(200), POWER_MIN(250)))
    options = L.pareto(items)
    assert len(options) == 2
    assert (3000.0, None) in options
    assert (None, 10.0) in options


def test_determinism():
    L = RelaxationLattice((BUDGET(1000), POWER_MIN(200)))
    assert L.minimal(CATALOG) == L.minimal(CATALOG)
    assert L.describe(L.minimal(CATALOG).relaxation) == \
        L.describe(L.minimal(CATALOG).relaxation)
