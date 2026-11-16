"""
Tests for the Constraint Induction Precision/Recall benchmark.

The point of these tests is not to pin down today's numbers -- pinning them
would only make the benchmark a rubber stamp. It is to keep three things
true, each of which was false at some point during development:

  * the ground truth is derivable from the data, not declared by fiat;
  * induction is driven by the *values*, so renaming every attribute key
    changes nothing for the structural inducer and breaks the keyword one;
  * the known precision ceiling stays visible in the suite rather than in a
    comment nobody re-reads.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from cip_benchmark import (
    EQ,
    LE,
    InducedRule,
    _covers,
    _measure,
    compare,
    induce_structural,
    rules_from_baseline,
    score,
    verify,
)
from hypergraph import ProductNode

ALL = [__import__("cip_benchmark").pc_build(),
       __import__("cip_benchmark").auto_parts(),
       __import__("cip_benchmark").pharma()]


# --------------------------------------------------------------------------
# The ground truth must be a fact about the data
# --------------------------------------------------------------------------

def test_every_dataset_has_a_derivable_ground_truth():
    for d in ALL:
        assert verify(d) == [], f"{d.name}: {verify(d)}"


def test_verify_rejects_a_rule_the_catalogue_does_not_support():
    """A declared rule that the data does not entail is a rigged benchmark."""
    d = ALL[1]
    rigged = InducedRule(LE, ("Діаметр диска",), ("max_diameter",))
    rigged = InducedRule(LE, ("max_diameter",), ("Діаметр",))  # backwards
    broken = type(d)(d.name, d.products, (rigged,))
    assert verify(broken), "an inverted containment rule must be rejected"


def test_verify_rejects_an_unknown_key():
    d = ALL[2]
    fake = InducedRule(EQ, ("немає_такого",), ("немає_такого",))
    broken = type(d)(d.name, d.products, (fake,))
    assert any("unknown key" in p for p in verify(broken))


# --------------------------------------------------------------------------
# The comparison itself
# --------------------------------------------------------------------------

def test_structural_induction_beats_keyword_matching():
    report = compare()
    assert report["proposed"]["f1"] > report["baseline"]["f1"]
    assert report["proposed"]["recall"] > report["baseline"]["recall"]


def test_every_dataset_improves_on_at_least_one_axis():
    for row in compare()["rows"]:
        assert (row["proposed"]["f1"] > row["baseline"]["f1"]
                or row["proposed"]["recall"] > row["baseline"]["recall"]), row


def test_scoring_is_exact_on_the_group_not_the_string():
    truth = [InducedRule(EQ, ("a", "b"), ("a", "b"))]
    half = [InducedRule(EQ, ("a",), ("a",))]
    assert score(half, truth).tp == 0        # half the matching key is wrong
    assert score(half, truth).fp == 1
    assert score(half, truth).fn == 1
    assert score(truth, truth).f1 == 100.0


def test_scoring_of_nothing_is_zero_not_a_crash():
    assert score([], []).f1 == 0.0
    assert score([], []).precision == 0.0


# --------------------------------------------------------------------------
# Induction is driven by values, not by names
# --------------------------------------------------------------------------

def _renamed(products):
    """Rename every attribute key to an opaque token, keeping the values."""
    keys = sorted({k for p in products for k in p.attributes})
    mapping = {k: f"ключ_{i:02d}" for i, k in enumerate(keys)}
    return [ProductNode(p.id, p.name, p.category, p.price,
                        {mapping[k]: v for k, v in p.attributes.items()})
            for p in products], mapping


def _relabel(rules, mapping):
    return {InducedRule(r.kind,
                        tuple(sorted(mapping[k] for k in r.subject)),
                        tuple(sorted(mapping[k] for k in r.obj)))
            for r in rules}


def test_renaming_every_key_changes_nothing_for_the_structural_inducer():
    """The strongest available check that no domain vocabulary leaked in.

    If any part of `induce_structural` keyed off a substring of an attribute
    name, replacing every name with an opaque token would change its output.
    It does not.
    """
    for d in ALL:
        renamed, mapping = _renamed(d.products)
        assert _relabel(induce_structural(d.products), mapping) == \
            set(induce_structural(renamed)), d.name


def test_the_same_renaming_breaks_the_keyword_matcher():
    """And the counterpart: the baseline is a name matcher, and shows it.

    Compared without relabelling -- the point is that the rules it emits
    change, not that they change in a particular direction.
    """
    changed = 0
    for d in ALL:
        renamed, _ = _renamed(d.products)
        before = {(r.kind, r.subject, r.obj) for r in rules_from_baseline(d.products)}
        after = {(r.kind, r.subject, r.obj) for r in rules_from_baseline(renamed)}
        if before != after:
            changed += 1
    assert changed == len(ALL), "the baseline survived a total renaming"


# --------------------------------------------------------------------------
# The criteria themselves
# --------------------------------------------------------------------------

def test_one_matching_key_is_emitted_per_logical_attribute():
    rules = induce_structural(ALL[0].products)
    sockets = [r for r in rules if r.kind == EQ and "Сокет" in r.subject]
    assert len(sockets) == 1
    assert set(sockets[0].subject) == {"Сокет", "socket"}


def test_the_keyword_matcher_finds_no_containment_rule_at_all():
    """Why the baseline scores what it scores.

    Its demand and capacity word lists are English; the catalogues are not.
    Not one containment constraint in any of the three datasets is found,
    and the two socket spellings come back as two separate matching keys
    rather than one. That is the measured cost of matching on names.
    """
    for d in ALL:
        assert not [r for r in rules_from_baseline(d.products) if r.kind == LE], d.name

    legacy = {(r.kind, r.subject) for r in rules_from_baseline(ALL[0].products)}
    # `socket` (two spellings' values) is emitted; `Сокет` takes a single
    # value in this catalogue so the structural zero-variance filter keeps
    # it out of induction entirely. Either way the two spellings are never
    # unified into one matching key -- the measured cost stands.
    assert (EQ, ("socket",)) in legacy
    assert (EQ, ("Сокет", "socket")) not in legacy


def test_both_energy_spellings_share_one_containment_rule():
    """TDP, tdp and 'Споживана потужність' are one attribute, not three."""
    rules = [r for r in induce_structural(ALL[0].products) if r.kind == LE]
    cpu = [r for r in rules if r.obj == ("Потужність БЖ",)
           and set(r.subject) == {"TDP", "tdp", "Споживана потужність"}]
    assert len(cpu) == 1
    # The graphics card is a separate consumer of the same supply, and must
    # not be folded into the CPU's group: different attribute, same ceiling.
    assert any(r.subject == ("Потужність",) and r.obj == ("Потужність БЖ",)
               for r in rules)


def test_headroom_is_measured_from_the_catalogue_not_assumed():
    """The baseline hardcodes 0.9; the structural inducer derives it."""
    rules = [r for r in induce_structural(ALL[0].products)
             if r.kind == LE and r.obj == ("Потужність БЖ",)
             and r.subject == ("Потужність",)]
    assert rules
    assert rules[0].headroom == pytest.approx(320.0 / 850.0, abs=1e-6)


def test_a_measurement_without_a_unit_is_not_compared():
    # A9: a bare number has no unit, hence no magnitude, and a rule must not be
    # induced from it. This is the case the assumption is actually about.
    assert _measure("205") is None
    assert _measure("5x112") is None
    assert _measure("105 Вт") == (105.0, "P")
    assert _measure("R19") == (19.0 * 25.4, "L")


def test_an_unknown_unit_is_measured_in_its_own_dimension_not_discarded():
    """A unit the table lacks is a dimension of its own, not a lost value.

    `7.0J` used to yield None, because the trailing token was dropped and the
    value degenerated to a bare number. Dropping it was safe but blind: two
    values in joules were never compared even with each other. The unit table
    now decides only how much coverage a catalogue gets; soundness comes from
    the dimension being distinct, which is what the second assert pins down.
    """
    assert _measure("7.0J") == (7.0, "u:j")
    assert _measure("7.0J")[1] != _measure("7.0W")[1]


def test_covers_requires_every_provider_to_carry_every_consumer():
    assert _covers([850.0, 650.0], [320.0, 285.0, 170.0])
    assert not _covers([150.0, 130.0], [320.0, 285.0, 170.0])
    assert not _covers([], [1.0])
    assert not _covers([100.0, 50.0], [100.0, 50.0])   # same range, not a rule


def test_determinism():
    for d in ALL:
        assert induce_structural(d.products) == induce_structural(d.products)


# --------------------------------------------------------------------------
# Negative controls and identifier anonymisation (F12)
# --------------------------------------------------------------------------

def test_declared_negative_controls_are_reported_not_hidden():
    """Observed correlation must not be promoted silently to a rule.

    Every dataset declares rules that *look* structurally valid but are
    not compatibility constraints. The report names which system emits
    which negative, so a regression that starts emitting one is visible
    and an improvement that stops emitting one is measured.
    """
    rows = {r["dataset"]: r for r in compare()["rows"]}
    for row in rows.values():
        assert row["negative_controls"] > 0, row["dataset"]

    # The baseline emits a rule for the shared identifier "Ширина" even
    # though rim width in J and tyre width in mm are different attributes.
    # The proposed grouping requires a common canonical vocabulary and
    # does not fall for it.
    assert rows["auto_parts"]["baseline_negative_hits"] == ["eq(Ширина)"]
    assert "eq(Ширина)" not in rows["auto_parts"]["proposed_negative_hits"]

    # The proposed system still promotes colour agreement and
    # demand-vs-demand coverage; that failure is now named, not buried.
    assert "eq(Колір)" in rows["pharma"]["proposed_negative_hits"]
    assert "eq(Колір)" in rows["pc_build"]["proposed_negative_hits"]
    assert rows["auto_parts"]["proposed_negative_hits"] == \
        ["le(Діаметр диска <= max_diameter)"]


def test_structural_induction_is_invariant_to_anonymised_identifiers():
    """Keys, categories and product names replaced by opaque tokens must
    change nothing for a method that claims to read structure, not names."""
    for row in compare()["rows"]:
        assert row["anonymized"]["proposed_invariant"], row["dataset"]


def test_anonymisation_cannot_improve_the_name_matcher():
    """The baseline's residual competence lives in its word lists; on
    opaque identifiers it cannot do better than on real names."""
    for row in compare()["rows"]:
        assert row["anonymized"]["baseline"]["f1"] \
            <= row["baseline"]["f1"] + 1e-9, row["dataset"]


def test_empty_catalogue_is_safe():
    assert induce_structural([]) == []
    assert verify(type(ALL[0])("empty", (), ())) == []


# --------------------------------------------------------------------------
# The known ceiling -- asserted so it cannot be quietly forgotten
# --------------------------------------------------------------------------

def test_two_consumers_of_one_provider_are_mistaken_for_a_pair():
    """The precision ceiling, stated as a test.

    Worst-case coverage cannot tell a *limit* from another *demand*: give it
    a board's CPU ceiling (130..150 W) and a graphics card's draw
    (170..320 W) and the ranges are disjoint, so it concludes the card
    "covers" the board. They are siblings, both consumers of the same
    supply, and the rule is meaningless.

    This is why the benchmark reports precision as well as recall, and why
    the keyword lists this method replaced were not a solution but a
    different guess. Closing it needs a signal neither criterion has --
    the relation between categories, not just between numbers.
    """
    products = [
        ProductNode("a", "board", "board", 100.0, {"limit": "130W"}),
        ProductNode("b", "board", "board", 120.0, {"limit": "150W"}),
        ProductNode("c", "card", "card", 200.0, {"draw": "170W"}),
        ProductNode("d", "card", "card", 300.0, {"draw": "320W"}),
    ]
    rules = induce_structural(products)
    assert InducedRule(LE, ("limit",), ("draw",)) in rules
