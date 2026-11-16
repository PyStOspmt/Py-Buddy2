"""
Tests for the Non-Vacuous Factual Guarantee (src/nvfg.py).

The behaviour under test is the one arXiv:2603.16817 shows is missing from
factual filtering: producing a useful answer *without* inventing anything,
even when the customer's hard constraints are jointly unsatisfiable.

Every number the module emits must be in the constraint's own unit, and
every emitted item must be a row we were given.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from nvfg import (
    Constraint,
    NVFGResult,
    NVFGStatus,
    Op,
    feasible,
    report,
    solve,
)

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
# Constraints
# --------------------------------------------------------------------------

def test_constraint_satisfaction():
    c = BUDGET(2000)
    assert c.satisfied_by(1500.0)
    assert not c.satisfied_by(2500.0)
    assert c.violated_by(2500.0)


def test_missing_attribute_is_not_a_violation():
    """
    F21/P21: unknown data neither satisfies nor violates a hard requirement.
    Failed before: satisfied_by(None) was True, so an item that omitted the
    attribute walked through every hard filter -- including safety and
    identity constraints.
    """
    c = BUDGET(2000)
    assert not c.satisfied_by(None)
    assert not c.violated_by(None)


def test_equality_constraint_is_not_relaxable():
    c = Constraint("сокет", "socket", Op.EQ, 5.0, "AM5")
    assert not c.relaxable
    assert c.satisfied_by(5.0)
    assert c.violated_by(4.0)


def test_protected_inequality_is_not_relaxable():
    """
    Reviewer finding: the existence theorem trades every bound away, but
    safety/identity-style requirements may not be relaxed. `protected`
    freezes an inequality the operator alone would allow moving.
    """
    c = Constraint("доза", "dose", Op.AT_MOST, 100.0, "мг", protected=True)
    assert not c.relaxable
    assert "[захищене]" in str(c)


def test_protected_constraint_never_moves_in_solve():
    """
    A protected bound that nothing satisfies must yield UNSATISFIABLE,
    never a relaxed answer -- the solver may not manufacture usefulness
    by trading a protected requirement away.
    """
    dose_limit = Constraint("доза", "power", Op.AT_MOST, 20.0, "Вт",
                            protected=True)
    # Cheapest item has power 30 -- relaxing the dose cap would 'work'.
    r = solve(CATALOG, [dose_limit])
    assert r.status == NVFGStatus.UNSATISFIABLE
    assert r.items == []
    assert "захищене" in r.reason


def test_protected_relaxed_copy_keeps_flag():
    c = Constraint("доза", "dose", Op.AT_MOST, 100.0, "мг", protected=True)
    # protected constraints are never relaxed, but if rebuilt the flag
    # must survive reconstruction (frozen dataclass positional order).
    clone = Constraint(c.name, c.attribute, c.op, 50.0, c.unit, c.protected)
    assert clone.protected and not clone.relaxable


def test_constraint_renders_in_its_own_unit():
    assert "грн" in str(BUDGET(2000))
    assert "Вт" in str(POWER_MIN(100))


# --------------------------------------------------------------------------
# Feasibility
# --------------------------------------------------------------------------

def test_feasible_set_is_a_pure_predicate():
    got = feasible(CATALOG, [BUDGET(2000), POWER_MIN(50)])
    assert [i["name"] for i in got] == ["Середній кулер"]


def test_feasible_excludes_items_without_the_attribute():
    """
    F21/P21: an item that states no value for a hard requirement is not
    provably feasible. Failed before: it silently satisfied every hard
    filter. It is excluded, but not counted as a violation -- relaxation
    cannot manufacture the missing data.
    """
    rows = [{"name": "A", "price": 10.0}, {"name": "B"}]
    assert [i["name"] for i in feasible(rows, [BUDGET(100)])] == ["A"]


# --------------------------------------------------------------------------
# The three outcomes
# --------------------------------------------------------------------------

def test_satisfiable_query_needs_no_relaxation():
    r = solve(CATALOG, [BUDGET(2000), POWER_MIN(50)])
    assert r.status == NVFGStatus.FEASIBLE
    assert r.items == ["Середній кулер"]
    assert r.relaxation is None
    assert r.regret == 0.0
    assert r.non_vacuous


def test_infeasible_query_produces_a_grounded_usable_answer():
    """Cheap AND powerful cannot both hold. The answer must still be useful."""
    r = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    assert r.status == NVFGStatus.RELAXED
    assert r.grounded
    assert r.informative
    assert r.non_vacuous
    # Only the power constraint had to move, and it moved by the smallest
    # amount that leaves the system satisfiable JOINTLY: 200 -> 45 W, the
    # quietest cooler the 1000 UAH budget can still afford.
    #
    # Not 30 W, though 30 is the quietest cooler in the catalogue: that model
    # costs 2600, so relaxing power to 30 W would satisfy the power bound
    # while emptying the feasible set. And not 150 W, though 250-200 = 50
    # looks like a tidy gap: 250 W already satisfies ">= 200", so nothing
    # needs relaxing for it, and no 150 W cooler exists.
    assert r.relaxation.constraint == "потужність"
    assert r.relaxation.to_value == pytest.approx(45.0)
    assert r.regret == pytest.approx(155.0)
    assert r.items == ["Бюджетний кулер"]     # something buyable, not a guess


def test_relaxation_is_validated_against_the_other_constraints():
    """The isolated minimum is worthless if it empties the feasible set.

    Guards the regression that shipped in the first draft: the gap was read
    off the relaxed constraint alone, so a relaxation that solved nothing
    looked optimal.
    """
    r = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    names = set(r.items)
    for item in CATALOG:
        if item["name"] in names:
            assert item["price"] <= 1000       # budget still holds
            assert item["power"] >= 45         # the relaxed bound really holds


def test_relaxation_picks_the_cheapest_constraint_not_the_first():
    """Cost is the SHARE of the requirement given up, not its raw magnitude.

    Comparing "170 W" with "700 UAH" is meaningless; comparing "we gave up
    78% of the power requirement" with "we gave up 220% of the budget" is
    not. Power is 1st here and 2nd below -- the order in which the customer
    typed the requirements must not decide which one is sacrificed.
    """
    a = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    b = solve(CATALOG, [POWER_MIN(200), BUDGET(1000)])
    assert a.relaxation.constraint == b.relaxation.constraint == "потужність"
    assert a.to_dict() == b.to_dict()


def test_relaxation_prefers_the_smaller_share_not_the_smaller_number():
    """The discriminating case: magnitude and share point opposite ways.

    Dropping the power floor 100 -> 40 W moves the bound by 60, and 60 is
    less than 700, so a raw comparison picks power. But 60 is 60% of a 100 W
    requirement while 700 UAH is 35% of a 2000 UAH one, and only a
    dimensionless ratio is comparable across units -- so the budget moves.

    (Comparing 60 against 700 is not merely a different tie-break. It is a
    comparison of watts against hryvnias, and it is meaningless.)
    """
    rows = [{"name": "Слабкий", "price": 1000.0, "power": 40.0},
            {"name": "Потужний", "price": 2700.0, "power": 100.0},
            {"name": "Найпотужніший", "price": 9000.0, "power": 500.0}]
    r = solve(rows, [BUDGET(2000), POWER_MIN(100)])
    assert r.status == NVFGStatus.RELAXED
    assert r.relaxation.constraint == "бюджет"
    assert r.relaxation.direction == "вгору"
    assert r.regret == pytest.approx(700.0)            # the larger magnitude
    assert r.relative_regret == pytest.approx(0.35)    # the smaller sacrifice
    assert r.regret > 60.0                             # magnitude says power
    assert r.items == ["Потужний"]


def test_relaxation_is_reported_in_the_customers_units():
    r = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    text = r.explain()
    assert "Вт" in text                       # not an abstract score
    assert " бюджет " not in text.split("Найменша зміна")[1]  # budget untouched
    assert "вниз" in text                    # AT_LEAST relaxes downward


def test_explanation_names_the_constraint_and_the_margin():
    r = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)])
    text = r.explain()
    assert "послабити потужність" in text
    assert "200" in text and "45" in text      # from -> to, in the real bound


def test_unsatisfiable_is_reported_honestly_not_as_a_recommendation():
    """Nothing here can be relaxed: the requirement is an equality.

    Note the contrast with the case below -- when a budget IS present and
    can be moved, the system answers with a grounded recommendation instead
    of shrugging, which is the entire point of the module.
    """
    rows = [{"name": "Тільки AM5", "socket": 5.0, "price": 100.0}]
    r = solve(rows, [Constraint("сокет", "socket", Op.EQ, 4.0, "AM4")])
    assert r.status == NVFGStatus.UNSATISFIABLE
    assert r.items == []
    assert not r.grounded
    assert r.reason                          # says what it looked for
    assert "сокет" in r.reason
    assert r.informative                     # a reason is still information
    assert not r.non_vacuous                  # but nothing was delivered


def test_a_relaxable_requirement_is_answered_not_refused():
    """The same catalogue, but the one blocking requirement can be moved.

    An assistant that replies "nothing found" here would be factually
    flawless and useless -- exactly the vacuous output NVFG rules out.
    """
    rows = [{"name": "Тільки AM5", "socket": 5.0, "price": 100.0}]
    r = solve(rows, [
        Constraint("сокет", "socket", Op.EQ, 5.0, "AM5"),
        BUDGET(10.0),
    ])
    assert r.status == NVFGStatus.RELAXED
    assert r.items == ["Тільки AM5"]
    assert r.relaxation.constraint == "бюджет"
    assert r.relaxation.from_value == 10.0
    assert r.relaxation.to_value == 100.0
    assert r.non_vacuous


def test_mutually_binding_requirements_are_not_silently_forced():
    """No single relaxation rescues {price <= 100, power >= 900}.

    Every price relaxation leaves a too-weak cooler, and every power
    relaxation leaves one that is too expensive. Reporting a recommendation
    here would mean inventing a product; reporting a relaxation would mean
    lying about which constraint moved.
    """
    r = solve(CATALOG, [BUDGET(100.0), POWER_MIN(900)])
    assert r.status == NVFGStatus.UNSATISFIABLE
    assert r.items == []
    assert r.relaxation is None
    assert "бюджет" in r.reason and "потужність" in r.reason


def test_relaxation_that_stays_infeasible_is_not_reported_as_success():
    """Dropping the cheapest constraint is not enough if others still bind."""
    rows = [{"name": "A", "price": 5000.0, "power": 10.0},
            {"name": "B", "price": 10.0, "power": 10.0}]
    r = solve(rows, [BUDGET(100.0), POWER_MIN(500)])
    # The only item meeting the power bound is also the expensive one, and
    # the power bound is the only relaxable-looking one: the result must not
    # claim a usable set.
    assert r.items == [] or r.status == NVFGStatus.RELAXED
    assert r.non_vacuous == (r.items != [])


def test_empty_catalogue():
    r = solve([], [BUDGET(100)])
    assert r.status == NVFGStatus.EMPTY_QUERY
    assert not r.non_vacuous


def test_no_constraints_means_everything_is_feasible():
    r = solve(CATALOG, [])
    assert r.status == NVFGStatus.FEASIBLE
    assert r.regret == 0.0
    assert len(r.items) == 3          # capped by max_items


# --------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------

def test_no_item_is_ever_invented():
    """Every emitted name must exist in the input catalogue."""
    names = {i["name"] for i in CATALOG}
    for constraints in ([BUDGET(1000)], [BUDGET(1000), POWER_MIN(200)],
                        [SIZE_MAX(121.0), POWER_MIN(240.0)]):
        r = solve(CATALOG, constraints)
        assert set(r.items) <= names


def test_regret_is_zero_exactly_when_nothing_was_relaxed():
    for constraints in ([BUDGET(5000)], [BUDGET(1000), POWER_MIN(200)]):
        r = solve(CATALOG, constraints)
        assert (r.regret == 0.0) == (r.status == NVFGStatus.FEASIBLE)


def test_regret_is_never_negative():
    for budget in (100.0, 1000.0, 5000.0):
        for power in (10.0, 200.0, 900.0):
            r = solve(CATALOG, [BUDGET(budget), POWER_MIN(power)])
            assert r.regret >= 0.0


def test_determinism():
    a = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)]).to_dict()
    b = solve(CATALOG, [BUDGET(1000), POWER_MIN(200)]).to_dict()
    assert a == b


def test_relaxation_is_stable_under_a_tie():
    """Equal shares, equal amounts: the constraint name breaks the tie.

    Neither row is both left-handed and right-handed, so the query is
    infeasible, and relaxing either requirement by 5 mm rescues exactly one
    row at exactly the same price. The choice must be reproducible, not a
    function of the order the customer happened to type the requirements in.
    """
    rows = [{"name": "Ліва", "left": 0.0, "right": 10.0},
            {"name": "Права", "left": 10.0, "right": 0.0}]
    left = Constraint("лівше", "left", Op.AT_LEAST, 5.0, "мм")
    right = Constraint("праве", "right", Op.AT_LEAST, 5.0, "мм")
    a = solve(rows, [left, right])
    b = solve(rows, [right, left])
    assert a.status == NVFGStatus.RELAXED
    assert a.to_dict() == b.to_dict()            # order-independent
    assert a.relaxation.constraint == "лівше"   # alphabetical, not positional


# --------------------------------------------------------------------------
# Reporting — the number the paper leaves open
# --------------------------------------------------------------------------

def test_recovery_rate_measures_recovery_from_the_deadlock():
    results = [
        solve(CATALOG, [BUDGET(5000)]),                      # feasible
        solve(CATALOG, [BUDGET(1000), POWER_MIN(200)]),      # relaxed
        solve(CATALOG, [BUDGET(100.0), POWER_MIN(900)]),    # unsatisfiable
    ]
    rep = report(results)
    assert rep.n == 3
    assert rep.feasible == 1
    assert rep.relaxed == 1
    assert rep.unsatisfiable == 1
    # Half the deadlocks were recovered, and the metric does not round that
    # up to 100% by counting an honest refusal as a rescue.
    assert rep.recovery_rate == 50.0
    assert rep.vacuity_rate == 0.0          # all three said something true
    assert rep.mean_regret > 0
    assert rep.mean_relative_regret > 0


def test_vacuity_rate_counts_useful_refusals_as_informative():
    rows = [{"name": "Тільки AM5", "socket": 5.0, "price": 100.0}]
    results = [
        solve(rows, [Constraint("сокет", "socket", Op.EQ, 4.0, "AM4")]),  # unsat
        solve(CATALOG, [BUDGET(5000)]),                                     # feasible
    ]
    rep = report(results)
    assert rep.unsatisfiable == 1
    assert rep.vacuity_rate == 0.0        # both said something useful
    assert rep.recovery_rate == 0.0       # the unsat one was not recovered


def test_report_of_nothing_is_safe():
    rep = report([])
    assert rep.n == 0 and rep.mean_regret == 0.0


def test_vacuous_output_is_measured_not_assumed():
    """A refusal that names nothing is vacuous; one that names a reason is not.

    Without this distinction the metric can be gamed to zero by returning
    "I cannot help" to every query, which is the failure mode of the
    conformal filtering this work is measured against.
    """
    rows = [{"name": "Тільки AM5", "socket": 5.0, "price": 100.0}]
    informative = solve(rows, [Constraint("сокет", "socket", Op.EQ, 4.0, "AM4")])
    silent = NVFGResult(NVFGStatus.UNSATISFIABLE, reason="")
    assert informative.informative and not silent.informative
    assert report([informative, silent]).vacuity_rate == 50.0


def test_tiny_positive_regret_never_rounds_to_zero():
    """Theorem 2: any non-trivial lattice relaxation keeps strictly positive regret."""
    from relaxation_lattice import solve as lattice_solve
    rows = [{"name": "Tight", "price": 1_000_000.05}]
    r = lattice_solve(rows, [Constraint("бюджет", "price", Op.AT_MOST, 1_000_000.0, "грн")])
    assert r.status == NVFGStatus.RELAXED
    assert r.regret > 0.0
