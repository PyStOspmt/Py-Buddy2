"""
AUD-32 regression: the deterministic agent's router must actually reach the
tool that answers the question.

The verdict axis of the benchmark read 0,00% for the deterministic arm while
`selftest` read 100,0% on the same cases. Nothing was wrong with the metric:
the planner asked for `_COMPAT` AND the literal words "сумісн"/"не підійд",
so a case phrased "Чи можна встановити X разом з Y?" matched no branch and
fell into "Уточніть, будь ласка" — the agent never called `check_bundle`,
so there was no verdict to score. On top of that `_answer_compat` read
`BundleVerdict.detail`, a field that does not exist (the real field is
`reasons`), and the `overconstrained` family had no branch at all.

These tests pin the three defects individually, and then prove the whole
offline arm is clean on every case, so a router that quietly stops
answering cannot be mistaken for a system that answers correctly.

Run directly:  python -u src/benchmark.py --offline --system pybuddy
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from benchmark import DEFAULT_DB
from benchmark_cases import VERDICT_FAMILIES, build_cases, load_snapshot
from metrics import aggregate, extract_verdict, score_case
from offline_agent import build

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_DB),
    reason="marketplace catalog not generated; run data/generate_marketplace_scale.py",
)

#: The wording that used to miss the router entirely. None of these contains
#: "сумісн" or "не підійд" — that is exactly why the old gate failed. The two
#: product names are filled in from a real case, because a question with no
#: names in it *should* be asked back.
INSTALL_TEMPLATES = [
    "Чи можна встановити {a} та {b} разом?",
    "Чи можна поставити {a} у одну збірку з {b}?",
]


@pytest.fixture(scope="module")
def snap():
    return load_snapshot(DEFAULT_DB)


@pytest.fixture(scope="module")
def agent(snap):
    return build(snap)


@pytest.fixture(scope="module")
def cases(snap):
    return build_cases(snap)


def _verdict_cases(cases):
    return [c for c in cases if c.case_type in VERDICT_FAMILIES]


def test_install_wording_is_routed_to_the_bundle_check(agent, cases):
    """The router must not require the word it is being asked about."""
    pair = next(c for c in cases if c.case_type == "compat_accept")
    a, b = pair.oracle.expected_products[:2]
    for template in INSTALL_TEMPLATES:
        query = template.format(a=a, b=b)
        text = agent.answer(query)["text"]
        assert "Уточніть" not in text, (
            f"'{query}' впала в гілку уточнення замість перевірки сумісності: {text}"
        )
        assert extract_verdict(text) != "neutral", f"{query}: без вердикта — {text}"


def test_every_verdict_case_produces_a_verdict(agent, cases):
    """No case whose question asks for a decision may come back neutral."""
    silent = []
    for case in _verdict_cases(cases):
        text = agent.answer(case.query)["text"]
        if extract_verdict(text) == "neutral":
            silent.append((case.case_id, case.case_type, text[:80]))
    assert not silent, f"кейси без вердикта: {silent[:5]}"


def test_every_verdict_case_produces_the_right_verdict(agent, cases, snap):
    wrong = []
    for case in _verdict_cases(cases):
        score = score_case(
            agent.answer(case.query)["text"], case.oracle,
            snap.names(), snap.price_index,
        )
        if not score.verdict_correct:
            wrong.append((case.case_id, case.case_type,
                          score.verdict, score.expected_verdict))
    assert not wrong, f"хибні вердикти: {wrong[:5]}"


def test_overconstrained_budget_is_answered(agent, cases):
    """`overconstrained` had no planner branch at all before the fix."""
    over = [c for c in cases if c.case_type == "overconstrained"]
    assert over, "benchmark lost the overconstrained family"
    for case in over:
        text = agent.answer(case.query)["text"]
        assert "Уточніть, будь ласка, які саме два товари" not in text
        assert text.strip(), f"{case.case_id}: порожня відповідь"


def test_compat_reasons_are_carried_into_the_answer(agent, cases):
    """A rejected bundle must say why, using `BundleVerdict.reasons`.

    `v.detail` never existed; reading it silently dropped every reason, and
    an unexplained "Ні" is not something a buyer can act on.
    """
    rejected = [c for c in cases if c.case_type == "compat_reject"]
    assert rejected, "benchmark lost the compat_reject family"
    for case in rejected:
        assert case.oracle.expected_verdict == "reject", case.case_id
        text = agent.answer(case.query)["text"]
        assert "Ні" in text, f"{case.case_id}: {text[:80]}"


def test_offline_arm_scores_clean_on_the_whole_suite(agent, cases, snap):
    """The aggregate the paper quotes, recomputed here on every case."""
    scores = [
        score_case(agent.answer(c.query)["text"], c.oracle,
                   snap.names(), snap.price_index)
        for c in cases
    ]
    agg = aggregate(scores)
    assert agg.execution_accuracy_pct == 100.0, [
        s.to_dict() for s in scores if s.violations
    ][:3]
    assert agg.verdict_accuracy_pct == 100.0, agg.verdict_cases
    assert agg.verdict_cases == len(_verdict_cases(cases))
    assert agg.hallucination_rate_pct == 0.0
    assert agg.price_error_rate_pct == 0.0


def test_answers_are_reproducible(agent, cases):
    """Same question twice -> the same string, including the chosen pair."""
    for case in _verdict_cases(cases)[:6]:
        assert agent.answer(case.query)["text"] == agent.answer(case.query)["text"]
