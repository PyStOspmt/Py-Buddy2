"""
Py-Buddy: Fuzzy Pareto Frontier Relaxation Engine
Resolves contradictory constraints and empty-set search results.
"""

from typing import List, Dict, Tuple, Optional
import math
from dataclasses import dataclass, field

from source_currency import format_price_with_currency

_MAXIMISE = frozenset({"performance_score"})


def _price_text(price: Optional[float], currency: Optional[str] = None) -> str:
    """Honest price rendering: an absent price is reported as absent."""
    if price is None:
        return "ціна не вказана"
    return format_price_with_currency(price, currency)


@dataclass
class CandidateSolution:
    item_id: str
    name: str
    # Absent prices stay None so unpriced items never masquerade as free (0.0).
    price: Optional[float] = None
    performance_score: Optional[float] = None
    delivery_days: Optional[int] = None
    membership_vector: Dict[str, float] = field(default_factory=dict)
    distance_to_ideal: float = 0.0


def _objective_values(item: CandidateSolution) -> Dict[str, float]:
    """Objectives the candidate states; an unknown objective is absent, not zero."""
    values: Dict[str, float] = {}
    if item.price is not None:
        values["price"] = item.price
    if item.performance_score is not None:
        values["performance_score"] = item.performance_score
    if item.delivery_days is not None:
        values["delivery_days"] = float(item.delivery_days)
    return values


class FuzzyParetoRelaxer:
    def __init__(
        self,
        budget_max: float,
        min_performance: Optional[float] = None,
        max_delivery_days: int = 5,
        currency: Optional[str] = None,
    ):
        self.budget_max = budget_max
        self.min_performance = min_performance
        self.max_delivery_days = max_delivery_days
        self.currency = currency

    def compute_memberships(self, item: CandidateSolution) -> Dict[str, float]:
        """
        Zadeh fuzzy membership functions mu_j(x) in [0, 1].
        Dynamically adapts to present criteria without fabricating missing dimensions.
        """
        memberships: Dict[str, float] = {}

        # Price: lower is better (monotonically decreasing). F10: an item the
        # source does not price simply has no price membership -- the axis is
        # absent for it, not zero.
        if item.price is not None:
            if item.price <= self.budget_max * 0.8:
                mu_price = 1.0
            elif item.price <= self.budget_max * 1.2:
                mu_price = (self.budget_max * 1.2 - item.price) / (self.budget_max * 0.4)
            else:
                mu_price = 0.0
            memberships["price"] = max(0.0, min(1.0, mu_price))

        # Performance: higher is better (computed only if data exists in domain)
        if item.performance_score is not None and self.min_performance is not None and self.min_performance > 0:
            if item.performance_score >= self.min_performance:
                mu_perf = 1.0
            elif item.performance_score >= self.min_performance * 0.7:
                mu_perf = (item.performance_score - self.min_performance * 0.7) / (self.min_performance * 0.3)
            else:
                mu_perf = 0.0
            memberships["performance"] = max(0.0, min(1.0, mu_perf))

        # Delivery: fewer days is better (computed only if merchant logistics exist)
        if item.delivery_days is not None and item.delivery_days > 0:
            if item.delivery_days <= 1:
                mu_deliv = 1.0
            elif item.delivery_days <= self.max_delivery_days:
                mu_deliv = (self.max_delivery_days - item.delivery_days) / max(1, self.max_delivery_days - 1)
            else:
                mu_deliv = 0.0
            memberships["delivery"] = max(0.0, min(1.0, mu_deliv))

        return memberships

    def compute_pareto_front(self, candidates: List[CandidateSolution]) -> List[CandidateSolution]:
        """
        Computes the non-dominated set (Pareto Frontier) over the raw objectives.
        Membership vectors are stored on every candidate for the trade-off ranking.
        """
        for item in candidates:
            item.membership_vector = self.compute_memberships(item)

        pareto_front = []
        for i, item_a in enumerate(candidates):
            dominated = False
            for j, item_b in enumerate(candidates):
                if i != j and self._dominates(item_b, item_a):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(item_a)
        return pareto_front

    def _dominates(self, a: CandidateSolution, b: CandidateSolution) -> bool:
        """
        Returns True if candidate 'a' Pareto-dominates candidate 'b'.

        Objectives are the raw values: price and delivery_days are minimised,
        performance_score is maximised. Dominance is judged only when both
        candidates state the same objective profile.

        F23: the old loop walked only a's own keys, so an item that was
        silent on an axis (no price, no delivery) could "dominate" on the
        axes it happened to have while ignoring the ones it lacked -- the
        reverse-dominance bug that left genuinely dominated items on the
        front. An axis one candidate leaves unknown cannot establish
        dominance either way.
        """
        vals_a = _objective_values(a)
        vals_b = _objective_values(b)
        if not vals_b or vals_a.keys() != vals_b.keys():
            return False
        strictly_better = False
        for key, value_b in vals_b.items():
            value_a = vals_a[key]
            if key in _MAXIMISE:
                if value_a < value_b:
                    return False
            elif value_a > value_b:
                return False
            if value_a != value_b:
                strictly_better = True
        return strictly_better

    def fits(self, item: CandidateSolution) -> bool:
        if item.price is None or item.price > self.budget_max:
            return False
        if self.min_performance is None:
            return True
        return item.performance_score is not None and item.performance_score >= self.min_performance

    def _violation_text(self, item: CandidateSolution) -> str:
        problems = []
        if item.price is None:
            problems.append("ціну не вказано, бюджет не перевірено")
        elif item.price > self.budget_max:
            delta_text = format_price_with_currency(item.price - self.budget_max, self.currency)
            problems.append(f"перевищує бюджет на {delta_text}")
        if self.min_performance is not None:
            if item.performance_score is None:
                problems.append("продуктивність не вказано")
            elif item.performance_score < self.min_performance:
                problems.append(f"нижча за мінімальну продуктивність "
                                f"({item.performance_score} < {self.min_performance})")
        return "; ".join(problems) or "відхилень немає"

    def _describe(self, item: CandidateSolution) -> str:
        performance = (f", продуктивність {item.performance_score} балів"
                       if item.performance_score is not None else "")
        status = ("задовольняє всі вимоги" if self.fits(item)
                  else f"відхилення: {self._violation_text(item)}")
        return f"'{item.name}' — ціна {_price_text(item.price, self.currency)}{performance}; {status}"

    def select_contrastive_tradeoffs(self, pareto_candidates: List[CandidateSolution], weights: Optional[Dict[str, float]] = None) -> Tuple[CandidateSolution, CandidateSolution, str]:
        """
        Selects two contrastive Pareto-optimal alternatives and generates an explainable natural language trade-off.
        """
        if not weights:
            weights = {"price": 0.5, "performance": 0.3, "delivery": 0.2}

        # Calculate weighted Euclidean distance to ideal point across active criteria
        for item in pareto_candidates:
            active_weights = {k: weights.get(k, 0.33) for k in item.membership_vector}
            w_sum = sum(active_weights.values()) or 1.0
            dist_sq = sum((w / w_sum) * ((1.0 - item.membership_vector[k]) ** 2) for k, w in active_weights.items())
            item.distance_to_ideal = math.sqrt(dist_sq)

        sorted_by_ideal = sorted(pareto_candidates, key=lambda x: x.distance_to_ideal)
        # A fitting candidate, if any exists, has a fitting non-dominated
        # candidate on the front (a dominator of a fitting item also fits).
        exact_found = any(self.fits(c) for c in pareto_candidates)
        if len(sorted_by_ideal) < 2:
            single = sorted_by_ideal[0]
            if exact_found:
                text = f"Єдиний Парето-оптимальний варіант, що задовольняє всі вимоги: {self._describe(single)}."
            else:
                text = (f"Точного збігу за всіма вимогами не знайдено. Єдиний Парето-оптимальний варіант: "
                        f"{self._describe(single)}.")
            return single, single, text

        best_overall = sorted_by_ideal[0]

        # Pick the most contrastive point on the Pareto front relative to best_overall
        contrast_item = max(
            sorted_by_ideal[1:],
            key=lambda x: abs(x.membership_vector.get("price", 0.0) - best_overall.membership_vector.get("price", 0.0))
        )

        # Contrastive explanation synthesis
        if (best_overall.price is not None and contrast_item.price is not None
                and best_overall.price < contrast_item.price):
            cheaper, stronger = best_overall, contrast_item
        else:
            cheaper, stronger = contrast_item, best_overall

        # Heading strictly reflects exact_found rather than asserting constraint satisfaction when relaxed.
        head = ("Є товари, що задовольняють усі вимоги. Два Парето-оптимальні варіанти:" if exact_found
                else "Точного збігу за всіма вимогами не знайдено. Знайдено 2 Парето-оптимальні альтернативи:")
        explanation = f"{head}\n1) {self._describe(cheaper)}.\n2) {self._describe(stronger)}."

        return best_overall, contrast_item, explanation
