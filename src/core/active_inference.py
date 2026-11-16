"""
Py-Buddy: preference elicitation by expected information gain.

What was wrong, and why this file was rewritten
------------------------------------------------
The previous version called itself Bayesian and was not:

    updated = (1 - learning_rate) * prior + learning_rate * observed_value

That is exponential smoothing. It has no likelihood, no prior, and no
posterior -- it just moves a number toward another number by a fraction
that never changes. Two answers, one of them twice, were worth the same as
one answer said twice, which is the one property a belief update must have
and the one property it lacked.

Expected information gain had the matching defect. It weighted every answer
option by `1 / len(options)`, which is a claim that the user is equally
likely to pick each answer regardless of anything known about them. Under
that assumption the expected gain of a question is a property of the
question alone, and asking a user something already answered the only way
to reduce the computed uncertainty.

The model
---------
Each latent preference dimension is a Bernoulli variable with a conjugate
Beta prior. An answer is treated as a noisy observation whose value v in
[0, 1] contributes evidence (v, 1 - v) scaled by a report weight:

    prior      Beta(alpha, beta)
    posterior  Beta(alpha + w*v,  beta + w*(1 - v))

The posterior mean alpha / (alpha + beta) is what the rest of the system
reads as the dimension's value, and the posterior's PARAMETER entropy --
the differential entropy of the Beta distribution, in nats -- is what the
question selector minimises. That quantity is epistemic: it falls as
evidence accumulates. The earlier "expected binary entropy of the
outcome" was aleatoric and could rise while the belief tightened; the
distinction is the F22 correction, and both quantities remain in the file
under their own names so they cannot be conflated again. Because the
update is conjugate, both entropies have closed forms and the whole
selection step is exact -- no sampling, no approximation.

Why this is defensible rather than decorative
----------------------------------------------
Evidence accumulates. Answering the same way twice moves the belief twice
as far, which is what "the user said it again" means. A confident early
answer is not overridden by a weak later one, because the later one adds
proportionally less evidence. Both fall out of the Beta update; neither was
true of convex interpolation.

Assumptions, logged in docs/07_RESEARCH_LOG.md
-----------------------------------------------
  A10. Predictive over answers. Before a question is asked, the probability
       of each option is taken to fall off linearly with its distance from
       the current posterior mean. A user who believes they are
       price-insensitive is more likely to tick "budget is not a
       constraint" than "strict budget", and the selector should know that.
       With a flat belief every option is equally likely and the model
       degrades to the old behaviour, which is the correct answer for a
       system that knows nothing.
  A11. The question bank has two sources with different evidence status.
       The ADAPTED bank is induced per domain by the LLM call in
       `DomainQuestionBankInductor` (or replayed from a source-versioned
       cache of that call); those questions carry `adapted=True`. When no
       model is available the engine falls back to
       `DEFAULT_QUESTION_BANK`: a generic, deliberately unadapted set that
       every question carries `adapted=False` on, so a consumer can never
       mistake fallback vocabulary for domain induction. A bank learned
       from session logs is the obvious next step and is not attempted
       here.
  A12. Report weight. A user answer is treated as worth `w` units of
      evidence, with w = 3. This is a modelling decision with a visible
      consequence, stated here because the number is not arbitrary: at
      w = 1 the engine asks NOTHING, and that is the correct answer
      rather than a defect. One answer worth one pseudo-observation moves
      the posterior mean by at most a quarter and barely narrows the
      distribution, so the expected reduction in uncertainty is smaller
      than the increase in uncertainty of the reported mean, and EIG comes
      out at or below zero for every question. The reading taken is that
      an answer given to a direct question in a live dialogue is a
      considered one, worth more than the flat prior it updates. The
      ordering of questions is stable across w; the stopping point is
      not, and a deployment that wants a different noise model changes
      this number rather than the algorithm.

Related work, which must be cited alongside this module
-------------------------------------------------------
IDSS (arXiv:2603.11399) and arXiv:2606.03135 both use entropy and
information gain to drive an elicitation loop. This module is not a
contribution over them and does not claim to be; what it adds is the
conjugate update and a predictive answer distribution, both of which are
standard, and a catalogue-grounded stop condition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Seed latent dimensions for the belief state when nothing else is
#: configured. These are a labelled generic prior over preference axes,
#: not a claim that every catalogue needs exactly these four; a domain or
#: deployment supplies its own via `dimension_names` (F52).
DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "price_sensitivity",        # 0 = money is no object, 1 = strict budget
    "performance_focus",        # 0 = everyday tasks, 1 = heavy compute
    "aesthetic_brand_focus",    # 0 = pure utility, 1 = brand and looks
    "urgency_delivery",         # 0 = can wait weeks, 1 = needed tomorrow
)

#: Concentration of the uniform prior.
#:
#: The prior on a dimension believed to be at p is Beta(c*p, c*(1-p)), so
#: at p = 0.5 this is Beta(c/2, c/2) and the prior is uniform only when
#: c = 2. The first version used c = 1 and got Beta(0.5, 0.5) -- a U-SHAPED
#: prior asserting that a user we know nothing about is almost certainly at
#: one extreme or the other, which is the exact opposite of what a flat
#: prior means. It still produced sensible-looking numbers, which is why it
#: survived as long as it did.
FLAT_PRIOR = 2.0

#: Evidence one user answer is worth (assumption A12, at the top of this
#: file). Three, and the reason is there; briefly: at 1 the engine correctly
#: concludes that a single weak self-report is not worth a question.
REPORT_WEIGHT = 3.0

_EPS = 1e-9


def _binary_entropy(p: float) -> float:
    """H(p) in bits, safe at the boundaries."""
    q = max(_EPS, min(1.0 - _EPS, p))
    return -(q * math.log2(q) + (1.0 - q) * math.log2(1.0 - q))


def _digamma(x: float) -> float:
    """
    psi(x) = d/dx ln Gamma(x).

    Needed because the expected entropy of a Beta has no expression in
    elementary functions. Standard recurrence down to a region where the
    asymptotic series converges quickly:

        psi(x) = ln x - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    """
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (math.log(x) - 0.5 * inv
               - inv2 * (1.0 / 12.0
                         - inv2 * (1.0 / 120.0
                                   - inv2 * (1.0 / 252.0
                                             - inv2 / 240.0))))
    return result


def _beta_binary_entropy(alpha: float, beta: float) -> float:
    """
    E[ H(p) ] for p ~ Beta(alpha, beta), in bits.

    This is the expected entropy of a Bernoulli OUTCOME -- aleatoric
    uncertainty about what the next answer will be, not epistemic
    uncertainty about the parameter p itself. A belief concentrated
    exactly at p = 0.5 scores almost a full bit here although we are
    certain about the parameter, which is why this quantity cannot drive
    question selection: it measures where the belief sits, not how much
    evidence is behind it. The selector uses `dimension_entropy`
    (posterior parameter entropy); this function survives for reporting
    and for its numerical test only.

    This is NOT H(alpha / (alpha + beta)) either. The entropy of the
    MEAN is always at least the expected entropy of the variable, and
    confusing the two is what made the previous version look like it was
    learning: it always saw the mean's entropy fall, so it always had
    information to gain, and stopped asking only when the threshold
    happened to intervene.

    E[-X ln X] = (alpha / S) * (psi(alpha + 1) - psi(S + 1)),  S = alpha + beta
    E[-(1-X) ln (1-X)] likewise with beta. Sanity check: Beta(1, 1) is
    uniform and gives 0.5 nats = 0.7213 bits, which is integral of the
    binary entropy over [0, 1] -- not 1 bit, as the old constant claimed.
    """
    total = alpha + beta
    ln2 = math.log(2.0)
    d_total = _digamma(total + 1.0)
    e_x = (alpha / total) * (_digamma(alpha + 1.0) - d_total)
    e_rest = (beta / total) * (_digamma(beta + 1.0) - d_total)
    return max(0.0, -(e_x + e_rest) / ln2)


def _beta_differential_entropy(alpha: float, beta: float) -> float:
    """
    h(p) for p ~ Beta(alpha, beta), in nats: the differential entropy of
    the POSTERIOR over the latent parameter -- epistemic uncertainty.

    h = ln B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)

    Sanity anchors: Beta(1, 1) is uniform and gives exactly 0; sharper
    posteriors score below zero (differential entropy is not bounded
    from below, unlike the discrete kind) and each pseudo-observation
    that tightens the belief lowers it. That monotonicity is the
    property a question selector needs and the one the outcome entropy
    above does not have: H(outcome) is largest where the belief is most
    neutral, so it can RISE while evidence accumulates.
    """
    total = alpha + beta
    return ((math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(total))
            - (alpha - 1.0) * _digamma(alpha)
            - (beta - 1.0) * _digamma(beta)
            + (total - 2.0) * _digamma(total))


@dataclass
class DiagnosticQuestion:
    """One elicitation question, addressed to one latent dimension."""

    id: str
    text: str
    target_dimension: str
    #: option text -> the value in [0, 1] that answering it asserts
    options: Dict[str, float]
    #: how much of the user's patience this costs, in the same units as
    #: `lambda_cost`; a question nobody wants to answer should lose to one
    #: they will answer
    cognitive_cost: float = 1.0
    #: F52: was this question induced for THIS domain (LLM call or its
    #: versioned cache), or is it the unadapted generic fallback? The
    #: honest default is False -- provenance is claimed, never assumed.
    adapted: bool = False

    def predicted(self, mean: float) -> Dict[str, float]:
        """
        P(each option), given the current posterior mean (assumption A10).

        Linear falloff in the distance between the option's asserted value
        and what we currently believe. Flat belief, uniform predictive;
        a belief sitting exactly on an option, all the mass there.
        """
        raw = {k: max(0.0, 1.0 - abs(mean - v)) for k, v in self.options.items()}
        total = sum(raw.values())
        if total <= _EPS:
            n = len(self.options) or 1
            return {k: 1.0 / n for k in self.options}
        return {k: v / total for k, v in raw.items()}


class UserBeliefState:
    """
    A conjugate Beta-Bernoulli belief per latent dimension.

    `dimensions` reads and writes the posterior MEAN, because that is what
    the rest of the system wants, but the means are derived from the
    (alpha, beta) pairs and are not the state. Setting a mean rebuilds the
    prior that would produce it.
    """

    def __init__(
        self,
        dimensions: Optional[Dict[str, float]] = None,
        *,
        prior_strength: float = FLAT_PRIOR,
        report_weight: float = REPORT_WEIGHT,
        dimension_names: Sequence[str] = DEFAULT_DIMENSIONS,
        entropy: Optional[float] = None,
    ) -> None:
        # `entropy` is accepted for constructor compatibility and derived dynamically from betas.
        self.prior_strength = prior_strength
        self.report_weight = report_weight
        self.dimension_names: Tuple[str, ...] = tuple(
            (dimensions or {}).keys() or dimension_names)
        self._betas: Dict[str, Tuple[float, float]] = {}
        for name in self.dimension_names:
            mean = 0.5 if dimensions is None else float(dimensions.get(name, 0.5))
            self._betas[name] = self._prior_for(mean)
        self.entropy: float = 0.0
        self.calculate_entropy()

    def _prior_for(self, mean: float) -> Tuple[float, float]:
        p = max(_EPS, min(1.0 - _EPS, mean))
        return (self.prior_strength * p, self.prior_strength * (1.0 - p))

    # -- views ------------------------------------------------------------

    @property
    def dimensions(self) -> Dict[str, float]:
        return {name: self.posterior_mean(name) for name in self._betas}

    @dimensions.setter
    def dimensions(self, values: Dict[str, float]) -> None:
        for name, value in values.items():
            self._betas[name] = self._prior_for(float(value))
        self.calculate_entropy()

    def posterior_mean(self, dimension: str) -> float:
        alpha, beta = self._betas[dimension]
        return alpha / (alpha + beta)

    def dimension_entropy(self, dimension: str) -> float:
        """
        Posterior parameter entropy, in nats: epistemic uncertainty over
        the latent preference on this dimension. Decreases as evidence
        accumulates; the question selector minimises its expectation.
        """
        return _beta_differential_entropy(*self._betas[dimension])

    def dimension_outcome_entropy(self, dimension: str) -> float:
        """
        Expected entropy of the NEXT ANSWER, in bits: aleatoric
        uncertainty, not epistemic. Kept for reporting and tests; it is
        deliberately NOT what the selector optimises (see
        `_beta_binary_entropy` for the failure mode it would cause).
        """
        return _beta_binary_entropy(*self._betas[dimension])

    # -- the update -------------------------------------------------------

    def bayesian_update(
        self,
        dimension: str,
        observed_value: float,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Fold one answer into the belief.

        `learning_rate` is kept for call compatibility and is NOT a
        smoothing factor: it scales the report weight, i.e. how much
        evidence this answer is taken to be. The old meaning -- "move this
        fraction of the way toward the answer" -- is the interpolation this
        function exists to replace, and accepting both would be ambiguous.
        """
        if dimension not in self._betas:
            return
        v = max(0.0, min(1.0, float(observed_value)))
        w = self.report_weight if learning_rate is None else \
            self.report_weight * float(learning_rate)
        alpha, beta = self._betas[dimension]
        self._betas[dimension] = (alpha + w * v, beta + w * (1.0 - v))
        self.calculate_entropy()

    def copy(self) -> "UserBeliefState":
        clone = UserBeliefState(
            prior_strength=self.prior_strength,
            report_weight=self.report_weight,
            dimension_names=self.dimension_names,
        )
        clone._betas = dict(self._betas)
        clone.entropy = self.entropy
        return clone

    def calculate_entropy(self) -> float:
        """Total posterior parameter entropy, in nats, over the dimensions."""
        self.entropy = sum(self.dimension_entropy(d) for d in self._betas)
        return self.entropy

    def to_dict(self) -> Dict[str, object]:
        return {
            "posterior_means": self.dimensions,
            "betas": {k: list(v) for k, v in self._betas.items()},
            # Differential entropy (nats) over the latent Beta posteriors.
            "posterior_entropy_nats": round(self.entropy, 6),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "UserBeliefState":
        """
        Rebuild a belief from a persisted session.

        The Beta parameters are restored when present, because restoring
        only the posterior MEAN would rebuild a flat prior and silently
        discard every answer the user had already given -- the session
        would appear to remember the conclusion and forget the evidence,
        and the engine would start asking the same question again.

        Older payloads carry `dimensions` only, and are restored as means.
        """
        data = data or {}
        betas = data.get("betas") or {}
        if betas:
            state = cls(dimension_names=tuple(betas.keys()))
            for name, pair in betas.items():
                alpha, beta = float(pair[0]), float(pair[1])
                state._betas[name] = (alpha, beta)
            state.calculate_entropy()
            return state
        means = data.get("dimensions") or data.get("posterior_means") or {}
        return cls(dimensions={k: float(v) for k, v in means.items()} or None)


#: Generic fallback bank, used ONLY when no domain-adapted bank could be
#: induced. Every question here carries the default `adapted=False`; it is
#: a labelled seed, not universal retail knowledge (F52).
DEFAULT_QUESTION_BANK: List[DiagnosticQuestion] = [
    DiagnosticQuestion(
        id="q_budget",
        text="Що для вас важливіше: суворий бюджет чи максимальні характеристики та преміум-якість?",
        target_dimension="price_sensitivity",
        options={"Суворий бюджет": 0.95, "Бюджет не обмежений": 0.05},
        cognitive_cost=0.8,
    ),
    DiagnosticQuestion(
        id="q_performance",
        text="Для яких завдань планується покупка: базове повсякденне використання чи підвищені професійні навантаження?",
        target_dimension="performance_focus",
        options={"Базові завдання": 0.05, "Підвищені навантаження": 0.95},
        cognitive_cost=0.8,
    ),
    DiagnosticQuestion(
        id="q_aesthetic",
        text="Наскільки важливий конкретний бренд, преміальне виконання або естетика товару?",
        target_dimension="aesthetic_brand_focus",
        options={"Важливо": 0.85, "Не важливо": 0.15},
        cognitive_cost=1.8,
    ),
    DiagnosticQuestion(
        id="q_urgency",
        text="Наскільки терміново потрібна покупка?",
        target_dimension="urgency_delivery",
        options={"Потрібно якнайшвидше": 0.9, "Можу зачекати": 0.1},
        cognitive_cost=0.6,
    ),
]


class DomainQuestionBankInductor:
    """
    Induces domain-adapted Diagnostic Questions for Active Inference (ZCCS = 0).
    Dynamically generates questions grounded in domain vocabulary via an
    OpenRouter LLM call, preserving conjugate update dimensions and
    predictive distributions. With no model available it returns the
    generic bank -- adaptation happens through the model or not at all;
    no hand-written domain dictionaries hide inside this class.
    """
    @classmethod
    def induce_question_bank(
        cls,
        domain_name: str,
        categories: Sequence[str],
        parameters: Sequence[str],
        runner: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        db_path: Optional[str] = None
    ) -> List[DiagnosticQuestion]:
        import hashlib, json, os, re
        from source_lifecycle import (
            read_versioned_cache, source_version, write_versioned_cache)

        clean_name = str(domain_name or "general").strip().lower()
        # F33: when the caller identifies the source, the cache key binds to
        # that source's identity -- two different databases sharing a domain
        # name and category labels must not inherit each other's bank.
        identity = os.path.abspath(db_path) if db_path else ""
        h = hashlib.md5(
            f"{clean_name}_{'_'.join(sorted(categories)[:10])}_{identity}".encode("utf-8")
        ).hexdigest()[:10]
        # This module lives in src/core/, so the repository data/ dir is three
        # levels up (docs/13 C2).
        data_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        cache_path = os.path.join(data_dir, f".question_bank_{h}.json")
        version = source_version(db_path)

        raw_qs = read_versioned_cache(cache_path, version) if os.path.exists(cache_path) else None
        if isinstance(raw_qs, list) and len(raw_qs) >= 2:
            return [
                DiagnosticQuestion(
                    id=q["id"],
                    text=q["text"],
                    target_dimension=q["target_dimension"],
                    options=q["options"],
                    cognitive_cost=float(q.get("cognitive_cost", 1.0)),
                    adapted=True
                ) for q in raw_qs
            ]

        if runner and getattr(runner, "api_key", None):
            try:
                prompt = f"""Ти — експерт із проектування діалогових систем активного виведення (Active Inference) в електронній комерції.
Перед тобою предметна область каталогу: "{domain_name}".
Приклади категорій товарів: {json.dumps(list(categories[:15]), ensure_ascii=False)}
Ключові атрибути товарів: {json.dumps(list(parameters[:10]), ensure_ascii=False)}

Згенеруй рівно 4 діагностичних запитання українською мовою для уточнення вподобань користувача, адаптованих до термінології цієї предметної області:
1. Цільовий вимір "price_sensitivity" (бюджетні рішення vs преміум-сегмент без обмежень)
2. Цільовий вимір "performance_focus" (базові повсякденні потреби vs високі навантаження/професійний рівень)
3. Цільовий вимір "aesthetic_brand_focus" (чи важливий оригінальний відомий бренд/вигляд чи функціональний аналог)
4. Цільовий вимір "urgency_delivery" (термінова доставка чи можна планово зачекати)

Поверни ВИКЛЮЧНО валідний JSON-список із 4 об'єктів формату:
[
  {{
    "id": "q_budget",
    "text": "Текст запитання про бюджет у термінах домену?",
    "target_dimension": "price_sensitivity",
    "options": {{"Опція 1": 0.95, "Опція 2": 0.05}},
    "cognitive_cost": 0.8
  }},
  ...
]
Жодного супровідного тексту поза JSON.
"""
                raw = runner.generate(prompt, temperature=0.0)
                m = re.search(r"\[.*\]", raw, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    if isinstance(data, list) and len(data) >= 3:
                        bank = []
                        valid_dims = set(DEFAULT_DIMENSIONS)
                        for q in data:
                            dim = q.get("target_dimension")
                            if dim in valid_dims and isinstance(q.get("options"), dict):
                                bank.append(DiagnosticQuestion(
                                    id=str(q.get("id") or f"q_{dim}"),
                                    text=str(q.get("text")),
                                    target_dimension=dim,
                                    options={str(k): float(v) for k, v in q["options"].items()},
                                    cognitive_cost=float(q.get("cognitive_cost", 1.0)),
                                    adapted=True
                                ))
                        if len(bank) >= 3:
                            write_versioned_cache(cache_path, version, [{
                                "id": b.id,
                                "text": b.text,
                                "target_dimension": b.target_dimension,
                                "options": b.options,
                                "cognitive_cost": b.cognitive_cost
                            } for b in bank])
                            return bank
            except Exception:
                # A failed LLM probe yields no cached bank; the generic
                # fallback below covers the source.
                pass

        # Deterministic generic fallback when neither LLM induction nor cache is available.
        return list(DEFAULT_QUESTION_BANK)


class ActiveInferenceEngine:
    """
    Picks the next question by net expected information gain.

        EIG(q) = h(d) - E_a[ h(d | q, a) ]      nats, posterior entropy
        net(q) = EIG(q) - lambda * C(q)

    and asks nothing when the best net gain falls below a threshold -- which
    is the whole point of computing it, since a selector that always asks
    another question is indistinguishable from one that never learned
    anything. With a true Bernoulli likelihood the expectation is the
    Bayesian information gain; under the fractional pseudo-count update
    used here it is a heuristic estimate of it, stated as such.
    """

    def __init__(
        self,
        question_bank: Optional[Sequence[DiagnosticQuestion]] = None,
        lambda_cost: float = 0.01,
        min_gain_threshold: float = 0.005,
    ) -> None:
        self.question_bank: List[DiagnosticQuestion] = list(
            DEFAULT_QUESTION_BANK if question_bank is None else question_bank)
        self.lambda_cost = lambda_cost
        self.min_gain_threshold = min_gain_threshold
        self.asked_question_ids: set = set()

    def expected_information_gain(
        self, state: UserBeliefState, question: DiagnosticQuestion
    ) -> float:
        """
        Nats of posterior parameter entropy this question is expected to
        remove, before cost.

        The objective is h(prior) - E_answer[ h(posterior | answer) ] --
        the expected reduction in epistemic uncertainty, evaluated under
        the pseudo-likelihood the fractional Beta update implies. With a
        true Bernoulli likelihood this expectation is the Bayesian
        information gain; here it is a heuristic estimate of it, named
        and evaluated as such rather than as an exact quantity.

        The expectation is over `question.predicted(state)`, not over a flat
        1/n -- that difference is the whole correction described at the top
        of this file, and it is what makes an already-known answer stop
        being worth asking.
        """
        if question.target_dimension not in state.dimension_names:
            return 0.0
        prior_entropy = state.dimension_entropy(question.target_dimension)
        mean = state.posterior_mean(question.target_dimension)
        posterior = 0.0
        for option, probability in question.predicted(mean).items():
            probe = state.copy()
            probe.bayesian_update(question.target_dimension, question.options[option])
            posterior += probability * probe.dimension_entropy(question.target_dimension)
        return max(0.0, prior_entropy - posterior)

    def select_best_question(
        self,
        current_state: UserBeliefState,
        exclude: Optional[Sequence[str]] = None,
    ) -> Optional[Tuple[DiagnosticQuestion, float]]:
        """
        The question with the highest NET gain, or None when none is worth
        the customer's patience.

        `exclude` is the set of question ids already answered in THIS
        session. When it is not supplied the engine's own bookkeeping is
        used (standalone use in tests); production passes the session's
        set so one customer cannot exhaust another's questions through a
        shared engine.
        """
        asked = self.asked_question_ids if exclude is None else set(exclude)
        best: Optional[DiagnosticQuestion] = None
        best_net = -math.inf

        for q in self.question_bank:
            if q.id in asked:
                continue
            if q.target_dimension not in current_state.dimension_names:
                continue
            net = (self.expected_information_gain(current_state, q)
                   - self.lambda_cost * q.cognitive_cost)
            if net > best_net:
                best, best_net = q, net

        if best is not None and best_net >= self.min_gain_threshold:
            return best, best_net
        return None

    def record_answer(
        self,
        current_state: UserBeliefState,
        question: DiagnosticQuestion,
        chosen_option: str,
        asked: Optional[set] = None,
    ) -> bool:
        """
        Fold the customer's answer into the belief. Returns False when the
        option is not one the question offers -- a non-answer must not
        become evidence.

        `asked`, when provided, is the session's own set of answered
        question ids (production path); without it the engine's
        bookkeeping is updated, which keeps standalone use working.
        """
        if chosen_option not in question.options:
            return False
        (self.asked_question_ids if asked is None else asked).add(question.id)
        current_state.bayesian_update(
            question.target_dimension, question.options[chosen_option])
        return True

    def reset(self) -> None:
        """Forget what was asked. The belief is untouched, on purpose."""
        self.asked_question_ids.clear()


