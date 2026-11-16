"""
Py-Buddy: model selection, health, and handover.

Why this module was rewritten
-----------------------------
There were two routers doing two different jobs under two names, and each
had grown the other's responsibilities:

  router.py       decided *which model* to call, tracked failures, counted
                  money -- and ALSO classified query complexity, using a
                  hard-coded Ukrainian trigger list.
  fast_router.py  decided *which execution tier* to use and actually ran
                  it, with its own -- and much better -- signal extraction.

The complexity classifier was duplicated, and the copy in `router.py` was
the worse one: a list of phrases ("збірка", "компроміс") against which
queries were matched, while `fast_router` measured entity density and
operational predicates. Two classifiers meant the deployed behaviour
depended on which object you happened to hold.

So the split is now explicit and one-way:

    fast_router.FastCascadedRouter   classifies intent, picks a TIER, runs it
    router.ModelRouter               given a tier, picks a MODEL and keeps
                                     the pool healthy

`ModelRouter` never reads the query. That is the point: a model router that
also classifies intent will inevitably grow a second classifier, and then
the two will disagree.

What changed besides the split
------------------------------
* The default endpoint list named `gemini-1.5-flash` and
  `claude-3-5-sonnet` -- models this project stopped calling when it moved
  to OpenRouter, and which the free-tier budget never paid for. Defaults
  now come from the environment (`OPENROUTER_MODEL`, `OPENROUTER_FALLBACKS`),
  the same source the client uses, so there is one list in the project
  rather than two that drift apart.
* The "baseline" for the savings figure was the literal 0.007 USD per call.
  It is now a parameter with a name, because a number that decides how
  good the economics look should not be an unlabelled constant buried in
  the middle of a function.
* `_select_healthy_endpoint` reactivated every endpoint when the pool
  looked empty, which turned a total outage into a silent retry storm. An
  exhausted pool now raises; recovery is an explicit call.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

#: An endpoint is taken out of rotation after this many failures. Two, not
#: one: a single 429 says "slow down", two say "this model is not serving".
FAILURES_BEFORE_DEGRADED = 2

#: Used only when neither the environment nor `.env` names a model. Stated
#: here as a fallback rather than as a list, because a list here is a second
#: source of truth that will drift from the client's.
_FALLBACK_MODEL = "deepseek/deepseek-v4-flash-0731"


def _env_models() -> List[str]:
    """Model names from the environment, in priority order, de-duplicated."""
    out: List[str] = []
    primary = os.environ.get("OPENROUTER_MODEL", "").strip()
    if primary:
        out.append(primary)
    for extra in os.environ.get("OPENROUTER_FALLBACKS", "").split(","):
        extra = extra.strip()
        if extra and extra not in out:
            out.append(extra)
    if not out:
        out.append(_FALLBACK_MODEL)
    return out


@dataclass
class ModelEndpoint:
    """One callable model, with the two prices the economics depend on."""

    model_name: str
    tier: int = 1                                  # 1 = cheap, 2 = reasoning
    cost_per_1k_input: float = 0.0                 # USD
    cost_per_1k_output: float = 0.0                # USD
    provider_name: str = "openrouter"
    is_active: bool = True
    failure_count: int = 0

    @property
    def label(self) -> str:
        return f"{self.provider_name}/{self.model_name}"


@dataclass
class ModelChoice:
    """What was selected, and what it was expected to cost."""

    tier: int
    endpoint: ModelEndpoint
    estimated_cost_usd: float
    reason: str

    @property
    def model(self) -> str:
        return self.endpoint.label

    def to_dict(self) -> Dict[str, object]:
        return {"tier": self.tier, "model": self.model,
                "estimated_cost_usd": round(self.estimated_cost_usd, 8),
                "reason": self.reason}


class ModelPoolExhausted(RuntimeError):
    """Every endpoint in the pool is degraded. Not a retry, a real outage."""


class ModelRouter:
    """
    Picks a model for a tier, keeps the pool healthy, and counts the money.

    Deliberately ignorant of query text. It answers "given that this must go
    to the reasoning tier, which model?" -- the tier decision belongs to
    `fast_router`, and the two answers must be able to change independently.
    """

    def __init__(
        self,
        endpoints: Optional[Sequence[ModelEndpoint]] = None,
        *,
        reasoner_tier: int = 2,
        default_prompt_tokens: int = 400,
        default_completion_tokens: int = 150,
    ) -> None:
        self.endpoints: List[ModelEndpoint] = list(endpoints) if endpoints else [
            ModelEndpoint(name, tier=1) for name in _env_models()
        ]
        self.reasoner_tier = reasoner_tier
        self.default_prompt_tokens = default_prompt_tokens
        self.default_completion_tokens = default_completion_tokens
        self.total_cost_usd: float = 0.0
        self.calls_by_tier: Dict[int, int] = {}

    # -- selection --------------------------------------------------------

    def select(
        self,
        tier: int,
        *,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        reason: str = "",
    ) -> ModelChoice:
        """
        The healthiest active endpoint for `tier`, and its expected cost.

        Health is failure count, not a score: an endpoint that has failed
        once is still usable, one that has failed twice is not. Anything
        cleverer here would be tuning a heuristic with no ground truth.
        """
        candidates = [e for e in self.endpoints if e.tier == tier and e.is_active]
        if not candidates:
            candidates = [e for e in self.endpoints if e.is_active]
        if not candidates:
            raise ModelPoolExhausted(
                "Усі моделі позначені як недоступні; викликайте recover() "
                "після перевірки стану провайдера."
            )
        endpoint = min(candidates, key=lambda e: (e.failure_count, e.model_name))
        p = self.default_prompt_tokens if prompt_tokens is None else prompt_tokens
        c = (self.default_completion_tokens if completion_tokens is None
             else completion_tokens)
        cost = (p / 1000.0) * endpoint.cost_per_1k_input \
            + (c / 1000.0) * endpoint.cost_per_1k_output
        return ModelChoice(tier, endpoint, cost,
                           reason or f"Найздоровіша модель для tier {tier}")

    def note_call(self, choice: ModelChoice) -> None:
        """Record a call that actually went out, for the economics report."""
        self.total_cost_usd += choice.estimated_cost_usd
        self.calls_by_tier[choice.tier] = self.calls_by_tier.get(choice.tier, 0) + 1

    # -- health and handover ---------------------------------------------

    def report_failure(self, failed_model: str, status_code: int = 0) -> Optional[ModelEndpoint]:
        """
        Degrade a failing endpoint and return the next one to try.

        Returns None when nothing healthy is left, which the caller must
        treat as an outage -- not as "try the same one again".
        """
        for ep in self.endpoints:
            if ep.model_name in failed_model or failed_model in ep.model_name:
                ep.failure_count += 1
                if ep.failure_count >= FAILURES_BEFORE_DEGRADED:
                    ep.is_active = False
        healthy = [e for e in self.endpoints
                   if e.is_active and e.model_name not in failed_model]
        if not healthy:
            return None
        return min(healthy, key=lambda e: (e.failure_count, e.model_name))

    def recover(self) -> List[str]:
        """
        Put every endpoint back into rotation, and say which ones.

        Explicit on purpose. The previous version did this silently inside
        selection, so an outage looked like a hiccup and a customer waited
        through a retry storm without anyone recording that it happened.
        """
        restored = []
        for ep in self.endpoints:
            if not ep.is_active:
                ep.is_active = True
                ep.failure_count = 0
                restored.append(ep.label)
            elif ep.failure_count:
                ep.failure_count = 0
        return restored

    # -- reporting --------------------------------------------------------

    def economics(self, baseline_cost_per_call_usd: float) -> Dict[str, object]:
        """
        Cost so far against a stated baseline.

        The baseline is a parameter because a savings figure is only as
        honest as the number it is compared against, and a reader cannot
        check a constant they cannot see.
        """
        total = sum(self.calls_by_tier.values())
        if total == 0:
            return {"total_calls": 0, "actual_cost_usd": 0.0,
                    "baseline_cost_usd": 0.0, "savings_percent": 0.0,
                    "calls_by_tier": {}}
        baseline = total * baseline_cost_per_call_usd
        saved = (1.0 - self.total_cost_usd / baseline) if baseline > 0 else 0.0
        return {
            "total_calls": total,
            "actual_cost_usd": round(self.total_cost_usd, 8),
            "baseline_cost_usd": round(baseline, 8),
            "baseline_cost_per_call_usd": baseline_cost_per_call_usd,
            "savings_percent": round(max(0.0, saved) * 100.0, 1),
            "calls_by_tier": dict(self.calls_by_tier),
        }


#: The name the paper and the older tests use. Kept as an alias so the
#: terminology in `docs/02` still matches the code, but there is one class.
CascadedRouter = ModelRouter
