"""Evaluator-mutation checks for the semantic-cycle metrics.

The metrics are hand-computed here on tiny rankings; each test would fail
under the obvious wrong implementation (mutation) named in its docstring.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "experiments", "semantic_cycle"))

import eval_text as et
import retrieval_metrics as m

R = {"a", "b", "c"}


def test_recall_caps_at_k_not_set_size():
    """Mutation: dividing by |R| would give 2/3 here, but 2 of the best
    possible min(|R|,2)=2 means complete."""
    assert m.recall_at_k(["a", "x", "b"], R, 2) == 0.5
    assert m.recall_at_k(["a", "b", "x"], R, 2) == 1.0
    assert m.recall_at_k(["x", "y"], R, 2) == 0.0


def test_recall_requires_nonempty_relevant_set():
    with pytest.raises(ValueError):
        m.recall_at_k(["a"], set(), 1)


def test_precision_uses_k_not_returned_length():
    """Mutation: dividing by len(top) would score a short list as 1.0."""
    assert m.precision_at_k(["a"], R, 4) == 0.25


def test_ndcg_hand_computed():
    got = m.ndcg_at_k(["x", "a", "b"], R, 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert got == pytest.approx(dcg / ideal)
    assert m.ndcg_at_k(["a", "b", "c"], R, 3) == pytest.approx(1.0)
    assert m.ndcg_at_k(["x", "y", "z"], R, 3) == 0.0


def test_mrr_and_hit():
    assert m.mrr(["x", "y", "b"], R) == pytest.approx(1 / 3)
    assert m.mrr(["x"], R) == 0.0
    assert m.hit_at_k(["x", "t"], "t", 2) == 1.0
    assert m.hit_at_k(["x", "t"], "t", 1) == 0.0


def test_bootstrap_interval_brackets_mean_and_is_deterministic():
    vals = [0.0] * 20 + [1.0] * 20
    a = m.bootstrap_ci(vals, n=2000)
    assert a == m.bootstrap_ci(vals, n=2000)
    mean, lo, hi = a
    assert mean == 0.5 and lo < 0.5 < hi
    assert all(math.isnan(x) for x in m.bootstrap_ci([]))


def test_paired_diff_detects_constant_gap():
    a = [0.8, 0.9, 0.7, 0.85]
    b = [x - 0.2 for x in a]
    mean, lo, hi = m.paired_bootstrap_diff(a, b, n=2000)
    assert mean == pytest.approx(0.2) and lo > 0.19 and hi < 0.21
    with pytest.raises(ValueError):
        m.paired_bootstrap_diff([1.0], [1.0, 2.0])


def test_numpy_bootstrap_is_deterministic_and_brackets_mean():
    vals = [0.0] * 30 + [1.0] * 30
    a = m.bootstrap_ci_np(vals, n=3000)
    assert a == m.bootstrap_ci_np(vals, n=3000)
    mean, lo, hi = a
    assert mean == 0.5 and lo < 0.5 < hi and hi - lo < 0.4
    assert all(math.isnan(x) for x in m.bootstrap_ci_np([]))


def test_numpy_paired_diff_matches_constant_gap_and_validates_lengths():
    a = [0.9, 0.8, 0.7, 0.6]
    mean, lo, hi = m.paired_bootstrap_diff_np(a, [x - 0.1 for x in a], 2000)
    assert mean == pytest.approx(0.1) and lo > 0.09 and hi < 0.11
    with pytest.raises(ValueError):
        m.paired_bootstrap_diff_np([1.0], [])


def test_independent_bm25_grammar_and_ranking():
    """The evaluator-owned lexical baseline: quoting defeats FTS syntax,
    prefix truncation matches inflections, OR semantics rank by BM25."""
    import bm25_independent as b
    assert b.match_expression("a", False) is None
    assert b.match_expression('xy"z NEAR(', False) == '"xy" OR "near"'
    assert b.match_expression("pirates", True) == '"pirat"*'
    ix = b.IndependentBM25()
    ix.add_many([("1", "pirate ship treasure"), ("2", "garden flowers"),
                 ("3", "treasure map")])
    assert ix.query("treasure", 5)[0] in {"1", "3"}
    assert ix.query("pirates", 5) == []
    assert ix.query("pirates", 5, prefix_truncate=True) == ["1"]
    assert ix.query("", 5) == []
    ix.close()


def test_auroc_hand_computed():
    assert m.auroc([3, 4], [1, 2]) == 1.0
    assert m.auroc([1, 2], [3, 4]) == 0.0
    assert m.auroc([1, 3], [2]) == 0.5
    assert m.auroc([2], [2]) == 0.5
    assert math.isnan(m.auroc([], [1]))


def test_overlap_validator_catches_inflections_and_ignores_stopwords():
    """Mutation: exact-token matching would miss 'pirates' vs 'pirate'."""
    assert et.has_overlap("books about pirates", "Pirate -- Fiction")
    assert not et.has_overlap("stories of sea robbers", "Pirate -- Fiction")
    assert not et.has_overlap("the of and", "the of and")


def test_ukrainian_check_rejects_latin_and_short_text():
    assert et.is_ukrainian("Шукаю дитячі вигадані історії про політ")
    assert not et.is_ukrainian("Шукаю books про політ")
    assert not et.is_ukrainian("так")
