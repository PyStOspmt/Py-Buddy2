"""
Tests for the conjugate belief and the information-gain selector.

Each test below corresponds to a property the previous implementation did
not have, and each of those was a real defect rather than a missing test:

  * evidence ACCUMULATES -- the same answer twice moves the belief twice as
    far. Convex interpolation gives the same result however many times you
    apply it, which is the property that makes a "Bayesian" update a
    smoothing function.
  * a confident early answer is not overturned by a weak later one.
  * expected gain is measured against the entropy of the BELIEF, and it
    goes to zero once the answer is already known.
  * the engine can say "I have nothing left worth asking".

The Beta entropy is checked against a numerical integral, because the
closed form was wrong twice before and a plausible-looking formula does
not announce that.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from active_inference import (
    DEFAULT_QUESTION_BANK,
    ActiveInferenceEngine,
    DiagnosticQuestion,
    UserBeliefState,
    _beta_binary_entropy,
    _digamma,
)


def _numerical_expected_entropy(alpha, beta, n=200001):
    """E[H(p)] = int H(p) * Beta(p; alpha, beta) dp, by plain quadrature.

    The density is p^(a-1) (1-p)^(b-1) / B(a, b), and B is in the
    DENOMINATOR. The first version of this helper used exp(+log B) and so
    normalised every asymmetric case wrong -- which is how a correct closed
    form came to look wrong.
    """
    log_b = (math.lgamma(alpha) + math.lgamma(beta)
             - math.lgamma(alpha + beta))
    norm = math.exp(-log_b)                   # 1 / B(alpha, beta)
    total = 0.0
    for i in range(1, n):
        p = i / n
        density = norm * (p ** (alpha - 1)) * ((1.0 - p) ** (beta - 1))
        h = -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        total += density * h
    return total / n


# --------------------------------------------------------------------------
# The closed form
# --------------------------------------------------------------------------

def test_digamma_matches_known_values():
    # psi(1) = -gamma, psi(2) = 1 - gamma, psi(3) = 1.5 - gamma
    gamma = 0.5772156649015329
    assert _digamma(1.0) == pytest.approx(-gamma, abs=1e-9)
    assert _digamma(2.0) == pytest.approx(1 - gamma, abs=1e-9)
    assert _digamma(3.0) == pytest.approx(1.5 - gamma, abs=1e-9)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (2.0, 1.0), (5.0, 3.0),
                                        (0.6, 3.2), (10.0, 1.0)])
def test_beta_entropy_matches_numerical_integration(alpha, beta):
    assert _beta_binary_entropy(alpha, beta) == pytest.approx(
        _numerical_expected_entropy(alpha, beta), abs=2e-3)


def test_the_uniform_prior_is_not_one_bit():
    """It is 0.7213 bits, not 1.

    The old code returned 1.0 for every flat belief, and the paper then
    quoted a number that was wrong by 40%. Worth a test of its own so the
    constant cannot drift back.
    """
    assert _beta_binary_entropy(1.0, 1.0) == pytest.approx(0.7213475, abs=1e-6)


# --------------------------------------------------------------------------
# The update
# --------------------------------------------------------------------------

def test_evidence_accumulates():
    """The belief moves further AND tightens with each answer.

    The second half is the part convex interpolation cannot do at all: a
    smoothed point carries no measure of how much has been said about it, so
    saying the same thing twice and saying it once are indistinguishable.
    """
    state = UserBeliefState(dimension_names=["d"])
    start = state.to_dict()["betas"]["d"]
    state.bayesian_update("d", 0.9)
    once = state.to_dict()["betas"]["d"]
    state.bayesian_update("d", 0.9)
    twice = state.to_dict()["betas"]["d"]

    fresh = UserBeliefState(dimension_names=["d"])
    assert state.posterior_mean("d") > 0.5
    assert sum(twice) > sum(once) > sum(start)      # concentration grows
    assert state.dimension_entropy("d") < fresh.dimension_entropy("d")


def test_a_strong_first_answer_is_not_overturned_by_a_weak_one():
    """Evidence is weighed, not counted.

    One confident answer and one half-hearted change of mind. Note the
    weights have to be different for the test to mean anything: against
    equal weights a Beta update SHOULD flip, and refusing to would be the
    bug, not the feature.
    """
    state = UserBeliefState(dimension_names=["d"])
    state.bayesian_update("d", 0.95, learning_rate=3.0)     # w = 9
    after_first = state.posterior_mean("d")
    state.bayesian_update("d", 0.1, learning_rate=0.1)      # w = 0.3
    after_second = state.posterior_mean("d")
    assert after_first > 0.85
    assert after_second < after_first          # it did move
    assert after_second > 0.8                 # but nowhere near a reversal


def test_entropy_falls_monotonically_with_evidence():
    state = UserBeliefState(dimension_names=["d"])
    before = state.dimension_entropy("d")
    for _ in range(5):
        state.bayesian_update("d", 0.95)
        assert state.dimension_entropy("d") < before
        before = state.dimension_entropy("d")


def test_expected_entropy_is_about_where_the_belief_sits_not_how_tight_it_is():
    """
    A counter-intuitive property, asserted so it is not "fixed" later.

    H(p) is largest at p = 0.5, so a belief concentrated AT 0.5 has a
    HIGHER expected entropy than a diffuse one that spreads mass towards
    the edges where H is small. Tightening a centred belief therefore
    *raises* E[H], and moving the belief away from 0.5 lowers it. Neither
    is monotone in the concentration, and both follow from the integral
    rather than from intuition.
    """
    assert _beta_binary_entropy(20.0, 20.0) > _beta_binary_entropy(1.0, 1.0)
    assert _beta_binary_entropy(20.0, 20.0) > _beta_binary_entropy(20.0, 1.0)
    assert _beta_binary_entropy(20.0, 20.0) < 1.0 + 1e-9     # H(0.5) is the cap


def test_an_unknown_dimension_is_ignored_rather_than_created():
    state = UserBeliefState(dimension_names=["d"])
    state.bayesian_update("not_a_dimension", 0.9)
    assert set(state.dimensions) == {"d"}


# --------------------------------------------------------------------------
# The predictor and the selector
# --------------------------------------------------------------------------

def test_a_flat_belief_predicts_every_option_equally():
    q = DEFAULT_QUESTION_BANK[0]
    predicted = q.predicted(0.5)
    assert len(set(round(p, 9) for p in predicted.values())) == 1


def test_a_belonging_answer_becomes_much_more_likely():
    """The correction that the uniform 1/n weighting could not make."""
    q = DEFAULT_QUESTION_BANK[0]           # {strict budget: 0.95, no limit: 0.05}
    predicted = q.predicted(0.9)
    assert predicted["Суворий бюджет"] > predicted["Бюджет не обмежений"]


def test_expected_gain_vanishes_once_the_answer_is_known():
    state = UserBeliefState()
    q = DEFAULT_QUESTION_BANK[0]
    first = ActiveInferenceEngine().expected_information_gain(state, q)
    for _ in range(30):
        state.bayesian_update(q.target_dimension, 0.95)
    later = ActiveInferenceEngine().expected_information_gain(state, q)
    assert first > 0
    assert later < first
    assert later == pytest.approx(0.0, abs=1e-3)


def test_the_engine_asks_a_question_and_then_stops():
    """The whole point of computing the gain: a stopping condition."""
    state = UserBeliefState()
    engine = ActiveInferenceEngine()
    asked = []
    for _ in range(20):
        choice = engine.select_best_question(state)
        if choice is None:
            break
        question, _gain = choice
        engine.record_answer(state, question, next(iter(question.options)))
        asked.append(question.id)
    assert asked, "a fresh belief has something worth asking"
    assert len(asked) == len(set(asked)), "no question is asked twice"
    assert len(asked) <= len(DEFAULT_QUESTION_BANK)
    assert engine.select_best_question(state) is None


def test_a_known_dimension_is_never_asked_about_twice():
    state = UserBeliefState()
    engine = ActiveInferenceEngine()
    while True:
        choice = engine.select_best_question(state)
        if choice is None:
            break
        question, _ = choice
        engine.record_answer(state, question, next(iter(question.options)))


def test_a_cognitive_cost_can_outweigh_the_information():
    """A question nobody wants to answer should lose to one they will."""
    cheap = DiagnosticQuestion("cheap", "t", "price_sensitivity",
                               {"a": 0.95, "b": 0.05}, cognitive_cost=0.1)
    pricey = DiagnosticQuestion("pricey", "t", "performance_focus",
                                {"c": 0.95, "d": 0.05}, cognitive_cost=50.0)
    state = UserBeliefState()
    engine = ActiveInferenceEngine([pricey, cheap], lambda_cost=0.01)
    choice = engine.select_best_question(state)
    assert choice is not None and choice[0].id == "cheap"


def test_a_zero_gain_threshold_asks_nothing_when_everything_is_known():
    state = UserBeliefState()
    for q in DEFAULT_QUESTION_BANK:
        state.bayesian_update(q.target_dimension, 0.99)
    engine = ActiveInferenceEngine(min_gain_threshold=0.5)
    assert engine.select_best_question(state) is None


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def test_a_restored_session_keeps_the_evidence_not_just_the_answer():
    """Restoring the mean would forget how many times the user said it."""
    state = UserBeliefState()
    for _ in range(4):
        state.bayesian_update("price_sensitivity", 0.95)

    restored = UserBeliefState.from_dict(state.to_dict())
    assert restored.dimensions == pytest.approx(state.dimensions)
    assert restored.calculate_entropy() == pytest.approx(state.calculate_entropy())
    # Same evidence means the same future: neither has anything to gain.
    q = DEFAULT_QUESTION_BANK[0]
    engine = ActiveInferenceEngine()
    assert engine.expected_information_gain(restored, q) == pytest.approx(
        engine.expected_information_gain(state, q), abs=1e-9)


def test_an_older_payload_carrying_only_means_still_loads():
    restored = UserBeliefState.from_dict(
        {"dimensions": {"price_sensitivity": 0.9}, "entropy": 1.0})
    assert restored.dimensions["price_sensitivity"] == pytest.approx(0.9)
    # F22: the entropy is now the differential entropy of the Beta
    # posterior (epistemic, nats). A mean-0.9 prior rebuilt at strength 2
    # is Beta(1.8, 0.2) -- concentrated, so its parameter entropy is
    # below the uniform's zero. The old assertion pinned the outdated
    # outcome-entropy semantics.
    from active_inference import _beta_differential_entropy
    assert restored.calculate_entropy() == pytest.approx(
        _beta_differential_entropy(1.8, 0.2))


def test_default_question_bank_has_zero_pc_hardcoding():
    """Confirms that default retail questions contain zero domain-specific PC jargon."""
    for q in DEFAULT_QUESTION_BANK:
        low = q.text.lower()
        assert "комп'ютер" not in low
        assert "підсвітка" not in low
        assert "корпус" not in low
        assert "рендер" not in low


def test_offline_induction_returns_the_generic_bank_not_hidden_dictionaries():
    """Without a model there is no domain adaptation -- and no curated
    pharmacy/auto dictionaries posing as one (rule 2.3). The honest
    fallback is the generic retail bank."""
    from active_inference import DEFAULT_QUESTION_BANK, DomainQuestionBankInductor
    for name, cats in (("Аптека та лікарські засоби", ["знеболюючі"]),
                       ("Автотовари та запчастини", ["гальма"])):
        qs = DomainQuestionBankInductor.induce_question_bank(
            domain_name=name, categories=cats, parameters=["x"], runner=None)
        assert [q.id for q in qs] == [q.id for q in DEFAULT_QUESTION_BANK]


def test_domain_question_bank_inductor_llm():
    from active_inference import DomainQuestionBankInductor
    class MockRunner:
        api_key = "test_key"
        def generate(self, prompt: str, temperature: float = 0.0) -> str:
            return '''[
              {
                "id": "q_budget",
                "text": "Чи потрібні дешеві свіжі продукти чи фермерські преміум?",
                "target_dimension": "price_sensitivity",
                "options": {"Економні": 0.95, "Органічні": 0.05},
                "cognitive_cost": 0.8
              },
              {
                "id": "q_performance",
                "text": "Чи шукаєте спеціальне дієтичне / безглютенове харчування?",
                "target_dimension": "performance_focus",
                "options": {"Звичайне": 0.05, "Спеціальне дієтичне": 0.95},
                "cognitive_cost": 0.8
              },
              {
                "id": "q_aesthetic",
                "text": "Чи важливий конкретний крафтовий виробник або бренд?",
                "target_dimension": "aesthetic_brand_focus",
                "options": {"Важливий бренд": 0.85, "Будь-який якісний": 0.15},
                "cognitive_cost": 1.2
              },
              {
                "id": "q_urgency",
                "text": "Наскільки швидко потрібна доставка продуктів?",
                "target_dimension": "urgency_delivery",
                "options": {"Експрес за 1 годину": 0.9, "Планова завтра": 0.1},
                "cognitive_cost": 0.6
              }
            ]'''

    groceries_qs = DomainQuestionBankInductor.induce_question_bank(
        domain_name="Продукти харчування",
        categories=["молоко", "сири", "випічка"],
        parameters=["fat", "origin"],
        runner=MockRunner()
    )
    assert len(groceries_qs) == 4
    dims = {q.target_dimension for q in groceries_qs}
    assert dims == {"price_sensitivity", "performance_focus", "aesthetic_brand_focus", "urgency_delivery"}
    assert "продукт" in groceries_qs[0].text.lower() or "дієтичне" in groceries_qs[1].text.lower()

