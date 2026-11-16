"""
Per-answer NVFG predicate: indicator = grounded AND informative.
Expectations follow the definition NVFG = |{r : G(r) and I(r)}| / |K|, with
G(r) the gate verdict and I(r) a stated product or concrete refusal reason.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))

from answer_nvfg import summarize_answer_nvfg

FEASIBLE = {"nvfg_status": "FEASIBLE", "relaxed": False,
            "option_a": {"name": "A"}, "explanation": "Знайдено товар."}
RELAXED = {"nvfg_status": "RELAXED", "relaxed": True, "option_a": {"name": "B"},
           "relaxation_text": "бюджет 4600 → 5600", "relative_regret": 0.0217,
           "explanation": "Послаблено бюджет."}
UNSAT = {"nvfg_status": "UNSATISFIABLE", "relaxed": False, "option_a": None,
         "explanation": "Жодна релаксація не дає товару."}
UNKNOWN = {"nvfg_status": "UNKNOWN", "relaxed": False, "option_a": None,
           "explanation": "Пошук неповний."}


def _step(result, tool="pareto_relax"):
    return {"tool_name": tool, "result": result}


def test_no_solver_step_means_nvfg_is_not_applicable():
    assert summarize_answer_nvfg([], "SUCCESS_GROUNDED") is None
    assert summarize_answer_nvfg([_step(FEASIBLE, tool="query_db")], "SUCCESS_GROUNDED") is None


def test_grounded_feasible_answer_has_indicator_true():
    out = summarize_answer_nvfg([_step(FEASIBLE)], "SUCCESS_GROUNDED")
    assert out["grounded"] is True
    assert out["informative"] is True
    assert out["indicator"] is True
    assert out["status"] == "FEASIBLE"
    assert out["relaxed"] is False


def test_relaxed_answer_keeps_relaxation_statement_and_regret():
    out = summarize_answer_nvfg([_step(RELAXED)], "SUCCESS_GROUNDED_AFTER_CORRECTION")
    assert out["indicator"] is True
    assert out["relaxed"] is True
    assert out["relaxation_text"] == "бюджет 4600 → 5600"
    assert out["relative_regret"] == 0.0217


def test_reasoned_refusal_is_informative():
    out = summarize_answer_nvfg([_step(UNSAT)], "SUCCESS_GROUNDED")
    assert out["informative"] is True
    assert out["indicator"] is True


def test_refusal_without_a_stated_reason_is_not_informative():
    bare = {"nvfg_status": "UNSATISFIABLE", "option_a": None, "explanation": "  "}
    out = summarize_answer_nvfg([_step(bare)], "SUCCESS_GROUNDED")
    assert out["informative"] is False
    assert out["indicator"] is False


def test_unknown_solver_status_is_not_informative():
    out = summarize_answer_nvfg([_step(UNKNOWN)], "SUCCESS_GROUNDED")
    assert out["informative"] is False
    assert out["indicator"] is False


def test_gate_rejection_zeroes_the_indicator_even_with_a_product():
    out = summarize_answer_nvfg([_step(FEASIBLE)], "HALLUCINATION_GATE_VIOLATION")
    assert out["grounded"] is False
    assert out["indicator"] is False


def test_unestablished_grounding_leaves_the_indicator_undetermined():
    for reason in ("STEP_BUDGET_EXCEEDED", "PROVIDER_ERROR", None):
        out = summarize_answer_nvfg([_step(FEASIBLE)], reason)
        assert out["grounded"] is None, reason
        assert out["indicator"] is None, reason


def test_last_solver_call_wins():
    out = summarize_answer_nvfg([_step(UNSAT), _step(FEASIBLE)], "SUCCESS_GROUNDED")
    assert out["status"] == "FEASIBLE"


def test_malformed_solver_results_are_ignored():
    steps = [_step("oops"), _step({"nvfg_status": "MAYBE"})]
    assert summarize_answer_nvfg(steps, "SUCCESS_GROUNDED") is None
