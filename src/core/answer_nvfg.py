"""
Per-answer NVFG predicate (NVFG = |{r : G(r) and I(r)}| / |K|).

G(r): the Zero-Hallucination Gate accepted the answer (SUCCESS_GROUNDED*).
I(r): the answer states a product or a concrete reason for refusal.
The indicator is None when grounding is not established for the answer.
"""

from typing import Any, Dict, Optional, Sequence

_GROUNDED_PREFIX = "SUCCESS_GROUNDED"
_GATE_REJECTED = "HALLUCINATION_GATE_VIOLATION"
_SOLVER_STATUSES = frozenset({"FEASIBLE", "RELAXED", "UNSATISFIABLE", "EMPTY_QUERY", "UNKNOWN"})
_REASON_STATUSES = frozenset({"UNSATISFIABLE", "EMPTY_QUERY"})


def _grounded(termination_reason: Optional[str]) -> Optional[bool]:
    if termination_reason is None:
        return None
    if termination_reason.startswith(_GROUNDED_PREFIX):
        return True
    if termination_reason == _GATE_REJECTED:
        return False
    return None


def _last_solver_result(steps: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for step in reversed(steps):
        result = step.get("result")
        if (step.get("tool_name") == "pareto_relax" and isinstance(result, dict)
                and result.get("nvfg_status") in _SOLVER_STATUSES):
            return result
    return None


def summarize_answer_nvfg(steps: Sequence[Dict[str, Any]],
                          termination_reason: Optional[str]) -> Optional[Dict[str, Any]]:
    result = _last_solver_result(steps)
    if result is None:
        return None
    status = result["nvfg_status"]
    has_product = result.get("option_a") is not None
    has_reason = bool(str(result.get("explanation") or "").strip())
    informative = status != "UNKNOWN" and (has_product or (status in _REASON_STATUSES and has_reason))
    grounded = _grounded(termination_reason)
    return {
        "status": status,
        "grounded": grounded,
        "informative": informative,
        "indicator": None if grounded is None else grounded and informative,
        "relaxed": bool(result.get("relaxed")),
        "relaxation": result.get("relaxation"),
        "relaxation_text": result.get("relaxation_text"),
        "relative_regret": result.get("relative_regret"),
        "regret": result.get("regret"),
        "protected": list(result.get("protected") or []),
        "explanation": result.get("explanation"),
    }
