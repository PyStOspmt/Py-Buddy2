"""
Tests for the F22 elicitation-policy evaluation.

The properties under test are the evaluation's honesty guarantees:
paired users across arms, an explicit no-elicitation floor, budget
curves instead of a single endpoint, seeds on record, and the
simulation labelled as simulation -- never real-user evidence.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "experiments"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "core"))

import elicitation_eval as ee


def test_paired_users_and_products_are_deterministic():
    u1, u2 = ee.make_users(8), ee.make_users(8)
    p1, p2 = ee.make_products(), ee.make_products()
    assert u1 == u2 and p1 == p2          # same seeds -> same fixtures


def test_no_elicitation_floor_is_the_flat_prior():
    state = ee.run_policy("none", {"price_sensitivity": 0.9}, [], 4,
                          __import__("random").Random(0))
    assert all(abs(v - 0.5) < 1e-9 for v in state.dimensions.values())


def test_every_arm_measured_at_every_budget():
    report = ee.run_eval(n_users=12)
    for arm in ("none", "fixed", "random", "eig"):
        ks = [r["budget_k"] for r in report["policies"][arm]["curve"]]
        assert ks == [0, 1, 2, 3, 4]      # the curve is the comparison


def test_more_questions_never_hurts_belief_on_average():
    """Sanity: the last budget is at least as good as the floor."""
    report = ee.run_eval(n_users=24)
    for arm in ("fixed", "random", "eig"):
        curve = report["policies"][arm]["curve"]
        floor = curve[0]["belief_l1"]["mean"]
        full = curve[-1]["belief_l1"]["mean"]
        assert full <= floor + 1e-9, arm


def test_random_arm_is_seeded_per_user():
    """Two calls with the same user index must replay the same shuffle."""
    import random
    user = ee.make_users(4)[0]
    s1 = ee.run_policy("random", user, ee.DEFAULT_QUESTION_BANK, 2,
                       random.Random(ee.RANDOM_POLICY_SEED))
    s2 = ee.run_policy("random", user, ee.DEFAULT_QUESTION_BANK, 2,
                       random.Random(ee.RANDOM_POLICY_SEED))
    assert s1.dimensions == s2.dimensions


def test_report_labels_simulation_and_records_seeds():
    report = ee.run_eval(n_users=8)
    assert "UE-1" in report["assumptions"]
    assert "simulation" in report["assumptions"]["UE-1"]
    assert report["seeds"]["users"] == ee.USER_SEED
    assert set(report["paired_deltas_vs_eig"]) == {"fixed", "random"}
    for arm, rows in report["paired_deltas_vs_eig"].items():
        for r in rows:
            assert r["n"] == 8 and "ci95_regret" in r


def test_regret_is_zero_when_belief_equals_truth():
    """Perfect weights must produce zero regret -- metric satisfiable."""
    from active_inference import UserBeliefState
    true = {"price_sensitivity": 0.2, "performance_focus": 0.9,
            "aesthetic_brand_focus": 0.5, "urgency_delivery": 0.7}
    state = UserBeliefState(dimensions=true)
    assert ee.regret(state, true, ee.make_products()) == 0.0
