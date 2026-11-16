"""
Py-Buddy: elicitation policy evaluation (F22 residual).

What the audit requires
-----------------------
F22: "Compare fewer-question utility against random/fixed-question and
no-elicitation baselines on held-out users." Until now the EIG selector
was unit-tested for mechanism but never measured against a competitor.

What this experiment is
-----------------------
A labelled simulation -- explicitly an evaluation mechanism, not
real-user evidence (docs/2.1): N seeded "users" carry a true latent
preference vector; each answers a question by picking the option whose
asserted value is nearest their truth (the standard rational-respondent
assumption, stated here as assumption UE-1). Users are generated from
their own seed stream -- held out from any tuning.

Three policies and a floor, paired on the same users:

    none    no questions; the flat prior. The denominator.
    fixed   the bank in declaration order.
    random  a per-user seeded shuffle of the bank.
    eig     ActiveInferenceEngine: net expected information gain with
            its own adaptive ordering and stopping rule.

Utility, measured two ways
--------------------------
  belief L1   mean |posterior_mean - true| over the four dimensions.
              What the belief is literally for.
  regret      each product is a score vector on the three axes the
              ranking layer actually reads (price/performance/delivery
              -- the same map react_agent._PREFERENCE_AXIS_MAP uses).
              Regret = utility lost by recommending argmax under the
              ESTIMATED weights instead of the true ones. This is the
              honest downstream check: a belief can be accurate and
              useless, or partially wrong and still rank correctly.

Each arm is evaluated at every question budget k = 0..len(bank), so
"same opportunity" is a measured curve, not a single endpoint. EIG's
reported point is its own stopping point; the curve shows whether its
ordering reaches the same regret with fewer questions.

Usage:
    python src/experiments/elicitation_eval.py --users 200 \
        --out docs/elicitation_eval_YYYYMMDD.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from active_inference import (  # noqa: E402
    DEFAULT_DIMENSIONS, DEFAULT_QUESTION_BANK, ActiveInferenceEngine,
    DiagnosticQuestion, UserBeliefState,
)

#: The axes the Pareto ranking actually consumes (mirrors
#: react_agent._PREFERENCE_AXIS_MAP; aesthetics is asked about but never
#: reaches ranking -- an honest part of what the evaluation shows).
RANKING_AXES = {
    "price_sensitivity": "price",
    "performance_focus": "performance",
    "urgency_delivery": "delivery",
}

USER_SEED = 20261004
PRODUCT_SEED = 777
RANDOM_POLICY_SEED = 4242
N_PRODUCTS = 24


# --------------------------------------------------------------------------
# Simulated users and products (labelled evaluation fixtures)
# --------------------------------------------------------------------------

def make_users(n: int, seed: int = USER_SEED) -> List[Dict[str, float]]:
    """Held-out true preferences, drawn independently of any tuning."""
    rng = random.Random(seed)
    return [
        {dim: rng.random() for dim in DEFAULT_DIMENSIONS}
        for _ in range(n)
    ]


def make_products(n: int = N_PRODUCTS,
                  seed: int = PRODUCT_SEED) -> List[Dict[str, float]]:
    """Candidate score vectors on the ranking axes, fixed for all arms."""
    rng = random.Random(seed)
    axes = sorted(set(RANKING_AXES.values()))
    return [
        {axis: rng.random() for axis in axes}
        for _ in range(n)
    ]


def answer_for(true: Dict[str, float],
               question: DiagnosticQuestion) -> str:
    """The rational respondent: the option nearest the true value."""
    return min(question.options,
               key=lambda opt: abs(true[question.target_dimension]
                                   - question.options[opt]))


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------

def run_policy(policy: str, user: Dict[str, float],
               bank: Sequence[DiagnosticQuestion], k: int,
               rng: random.Random) -> UserBeliefState:
    """Ask up to k questions under `policy`, fold answers into a belief."""
    state = UserBeliefState()
    if policy == "none" or k == 0:
        return state
    if policy == "eig":
        raise ValueError("eig is adaptive; run it through run_eig")

    if policy == "fixed":
        order = list(bank)
    elif policy == "random":
        order = rng.sample(list(bank), len(bank))
    else:
        raise ValueError(policy)
    asked: set = set()
    for q in order[:k]:
        if q.target_dimension not in state.dimension_names:
            continue
        state.bayesian_update(q.target_dimension,
                              q.options[answer_for(user, q)])
        asked.add(q.id)
    return state


def run_eig(user: Dict[str, float], bank: Sequence[DiagnosticQuestion],
            k: int) -> Tuple[UserBeliefState, int]:
    """
    EIG run for real: select, answer, update, repeat -- k is a budget,
    the engine's own stop rule can end it earlier. Returns the belief
    and how many questions were actually asked.
    """
    engine = ActiveInferenceEngine(question_bank=bank)
    state = UserBeliefState()
    asked: set = set()
    n_asked = 0
    for _ in range(k):
        pick = engine.select_best_question(state, exclude=asked)
        if pick is None:
            break
        question, _net = pick
        option = answer_for(user, question)
        engine.record_answer(state, question, option, asked=asked)
        n_asked += 1
    return state, n_asked


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def belief_error(state: UserBeliefState,
                 true: Dict[str, float]) -> float:
    means = state.dimensions
    return (sum(abs(means[d] - true[d]) for d in DEFAULT_DIMENSIONS)
            / len(DEFAULT_DIMENSIONS))


def regret(state: UserBeliefState, true: Dict[str, float],
           products: Sequence[Dict[str, float]]) -> float:
    """Utility lost recommending under estimated vs true weights."""
    est = {axis: state.posterior_mean(dim)
           for dim, axis in RANKING_AXES.items()}
    tru = {axis: true[dim] for dim, axis in RANKING_AXES.items()}

    def utility(p: Dict[str, float], w: Dict[str, float]) -> float:
        return sum(w[a] * p[a] for a in w)

    best_true = max(products, key=lambda p: utility(p, tru))
    picked = max(products, key=lambda p: utility(p, est))
    return max(0.0, utility(best_true, tru) - utility(picked, tru))


def _ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.pstdev(values) / math.sqrt(len(values))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def run_eval(n_users: int = 200,
             bank: Optional[Sequence[DiagnosticQuestion]] = None,
             ) -> Dict[str, object]:
    bank = list(bank or DEFAULT_QUESTION_BANK)
    users = make_users(n_users)
    products = make_products()
    max_k = len(bank)
    policies = ("none", "fixed", "random", "eig")

    # curve[arm][k] -> per-user metric lists, paired across users
    curve: Dict[str, Dict[int, Dict[str, List[float]]]] = {
        arm: {k: {"belief_l1": [], "regret": [], "asked": []}
              for k in range(max_k + 1)}
        for arm in policies
    }

    for i, user in enumerate(users):
        user_rng = random.Random(RANDOM_POLICY_SEED + i)
        for k in range(max_k + 1):
            for arm in policies:
                if arm == "eig":
                    state, asked_n = run_eig(user, bank, k)
                else:
                    asked_n = 0 if (arm == "none" or k == 0) else k
                    state = run_policy(arm, user, bank, k, user_rng)
                row = curve[arm][k]
                row["belief_l1"].append(belief_error(state, user))
                row["regret"].append(regret(state, user, products))
                row["asked"].append(asked_n)

    out: Dict[str, object] = {
        "experiment": "elicitation_eval",
        "users": n_users,
        "products": len(products),
        "seeds": {"users": USER_SEED, "products": PRODUCT_SEED,
                  "random_policy": RANDOM_POLICY_SEED},
        "assumptions": {
            "UE-1": "rational respondent: user picks the option nearest "
                    "their true value; a simulation, not real-user data",
            "UE-2": "regret axes = the dims the ranking layer reads; "
                    "aesthetic_brand_focus is elicited but not ranked",
        },
        "policies": {},
        "paired_deltas_vs_eig": {},
    }

    for arm in policies:
        out["policies"][arm] = {  # type: ignore[index]
            "curve": [
                {
                    "budget_k": k,
                    "mean_asked": round(statistics.fmean(
                        curve[arm][k]["asked"]), 3),
                    "belief_l1": {
                        "mean": round(statistics.fmean(
                            curve[arm][k]["belief_l1"]), 4),
                        "ci95": round(_ci95(curve[arm][k]["belief_l1"]), 4),
                    },
                    "regret": {
                        "mean": round(statistics.fmean(
                            curve[arm][k]["regret"]), 4),
                        "ci95": round(_ci95(curve[arm][k]["regret"]), 4),
                    },
                } for k in range(max_k + 1)
            ]
        }

    # Paired differences EIG - competitor at equal budget (negative = EIG
    # better), on both metrics. Recomputed per user from paired runs.
    deltas: Dict[str, Dict[str, List[float]]] = {
        arm: {"regret": [], "belief_l1": [], "k": []}
        for arm in ("fixed", "random")
    }
    for i, user in enumerate(users):
        user_rng = random.Random(RANDOM_POLICY_SEED + i)
        for arm in ("fixed", "random"):
            for k in range(max_k + 1):
                eig_state, _ = run_eig(user, bank, k)
                other = run_policy(arm, user, bank, k, user_rng)
                deltas[arm]["k"].append(k)
                deltas[arm]["regret"].append(
                    regret(eig_state, user, products)
                    - regret(other, user, products))
                deltas[arm]["belief_l1"].append(
                    belief_error(eig_state, user)
                    - belief_error(other, user))
    for arm, cols in deltas.items():
        by_k: Dict[int, Dict[str, List[float]]] = {}
        for k, dr, dl in zip(cols["k"], cols["regret"], cols["belief_l1"]):
            by_k.setdefault(k, {"regret": [], "belief_l1": []})
            by_k[k]["regret"].append(dr)
            by_k[k]["belief_l1"].append(dl)
        out["paired_deltas_vs_eig"][arm] = [  # type: ignore[index]
            {"budget_k": k,
             "mean_delta_regret": round(statistics.fmean(v["regret"]), 5),
             "ci95_regret": round(_ci95(v["regret"]), 5),
             "mean_delta_belief_l1": round(
                 statistics.fmean(v["belief_l1"]), 5),
             "ci95_belief_l1": round(_ci95(v["belief_l1"]), 5),
             "n": len(v["regret"])}
            for k, v in sorted(by_k.items())
        ]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    report = run_eval(args.users)
    print(f"{'k':>3} {'policy':<8} {'asked':>6} {'belief_L1':>16} "
          f"{'regret':>16}")
    for arm, block in report["policies"].items():
        for row in block["curve"]:
            print(f"{row['budget_k']:>3} {arm:<8} {row['mean_asked']:>6} "
                  f"{row['belief_l1']['mean']:>8}±{row['belief_l1']['ci95']:<7} "
                  f"{row['regret']['mean']:>8}±{row['regret']['ci95']:<7}")
    print("\npaired deltas (eig - other; negative favours eig):")
    for arm, rows in report["paired_deltas_vs_eig"].items():
        line = ", ".join(
            f"k={r['budget_k']}:{r['mean_delta_regret']:+.4f}"
            for r in rows)
        print(f"  vs {arm:<7} regret    {line}")
        line = ", ".join(
            f"k={r['budget_k']}:{r['mean_delta_belief_l1']:+.4f}"
            for r in rows)
        print(f"  vs {arm:<7} belief_L1 {line}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n[elicitation_eval] report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
