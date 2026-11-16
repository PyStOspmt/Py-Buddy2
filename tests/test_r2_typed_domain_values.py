"""
Regression suite: typed universal format parsing and domain-hint isolation invariants.

Every test pins behaviour using the seed-free Value Algebra (`attribute_algebra`)
and `ContractInductor` directly (zero hand-written domain key lists).
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from attribute_algebra import Shape, canonical_form, fold_all, magnitude, spec_of
from contract_inductor import ContractInductor
from hypergraph import (
    CumulativeCapacityConstraint,
    ProductNode,
)

# ---------------------------------------------------------------------------
# Power: the value must carry a recognized power unit, never any arbitrary digit
# ---------------------------------------------------------------------------

def test_f11_power_value_must_be_a_measured_quantity_not_any_digit():
    """
    Failed before: grabbing the first digit run turned 'Socket 1200' into a
    1200.0-watt fact. Value Algebra separates TOKEN/FREE from measured P.
    """
    assert spec_of("Socket 1200").shape is not Shape.MEASURE
    assert spec_of("червоний").shape is not Shape.MEASURE
    assert spec_of("120W").dimension == "P"
    assert canonical_form("120W") == "120W"
    assert canonical_form("120 Вт") == "120W"
    assert spec_of("750").dimension is None


def test_f11_kilowatts_are_converted_not_reinterpreted():
    """
    Failed before: "1.2 kW" became 1.2 watts. Now units convert via ISO table.
    """
    assert canonical_form("1.2 kW") == "1200W"
    assert canonical_form("1,2 кВт") == "1200W"


def test_f11_power_key_with_unparseable_value_stays_out_of_measurements():
    """
    Unmeasured values under any key never participate in capacity induction.
    """
    nodes = [
        ProductNode("c1", "CPU", "cpu", 1.0, {"k": "Socket 1200"}),
        ProductNode("p1", "PSU", "psu", 1.0, {"cap": "750W"}),
    ]
    caps = [c for c in ContractInductor().induce_contracts_from_products(nodes)
            if isinstance(c, CumulativeCapacityConstraint)]
    assert caps == []


# ---------------------------------------------------------------------------
# Diameter: the value must be a length or the universal R<n> notation
# ---------------------------------------------------------------------------

def test_f11_diameter_value_must_be_a_length():
    """Bare numbers carry no inferred unit; explicit length units canonicalize to mm."""
    assert spec_of("червоний").dimension is None
    assert spec_of("Socket 1200").dimension is None
    assert spec_of("17").dimension is None
    assert magnitude("R18")[0] == pytest.approx(18.0 * 25.4)
    assert magnitude('18"')[0] == pytest.approx(18.0 * 25.4)
    assert spec_of("R18").dimension == "L"


def test_f11_length_units_are_converted_to_the_canonical_dimension():
    assert magnitude("457.2 mm")[0] == pytest.approx(457.2)
    assert magnitude("45,72 см")[0] == pytest.approx(457.2)
    assert canonical_form("457.2 mm") == canonical_form("R18")


# ---------------------------------------------------------------------------
# Socket / token: never invents a family prefix on bare integers
# ---------------------------------------------------------------------------

def test_f11_bare_socket_number_gets_no_invented_family():
    """An unstated number stays a bare value without an invented 'LGA' prefix."""
    assert canonical_form("1700", key="socket") == "1700"
    assert canonical_form("LGA 1700", key="socket") == "lga 1700"
    assert canonical_form("Socket AM5", key="socket") == "am5"
    assert canonical_form("sAM5") == "am5"


# ---------------------------------------------------------------------------
# Memory & PCD: distinct tokens do not merge; duplex converts cleanly
# ---------------------------------------------------------------------------

def test_f11_memory_type_distinguishes_distinct_tokens():
    assert canonical_form("GDDR6") != canonical_form("DDR5")
    assert canonical_form("DDR5") == "ddr5"


def test_f11_pcd_duplex_pattern_canonicalizes():
    assert spec_of("сірий").shape is not Shape.DUPLEX
    assert canonical_form("5*112") == "5x112mm"
    assert fold_all(["5*112", "5x112 мм"]) == "5x112mm"


# ---------------------------------------------------------------------------
# Inductor: a capacity/demand pairing must be supported by the sample
# ---------------------------------------------------------------------------

def _node(pid, category, attrs):
    return ProductNode(pid, pid, category, 1.0, attrs)


def test_f11_inductor_capacity_pairing_needs_sample_evidence():
    """
    A capacity rule is induced only from measured quantities where provider
    values strictly dominate consumer values (F11), regardless of key names:
    a key named 'cap' with smaller values than 'draw' NEVER becomes provider.
    """
    inductor = ContractInductor()

    reversed_sample = [
        _node("p1", "psu", {"cap": "100W"}),
        _node("p2", "cpu", {"draw": "500W"}),
        _node("p3", "cpu", {"draw": "600W"}),
    ]
    rules = [c for c in inductor.induce_contracts_from_products(reversed_sample)
             if isinstance(c, CumulativeCapacityConstraint)]
    assert not any(r.capacity_attr == "cap" for r in rules)

    interleaved = [
        _node("p1", "psu", {"cap": "100W"}),
        _node("p2", "psu", {"cap": "750W"}),
        _node("p3", "cpu", {"draw": "200W"}),
        _node("p4", "cpu", {"draw": "500W"}),
    ]
    assert [
        c for c in inductor.induce_contracts_from_products(interleaved)
        if isinstance(c, CumulativeCapacityConstraint)
    ] == []

    supported = [
        _node("p1", "psu", {"cap": "750W"}),
        _node("p2", "cpu", {"draw": "120W"}),
        _node("p3", "cpu", {"draw": "250W"}),
    ]
    rules2 = [c for c in inductor.induce_contracts_from_products(supported)
              if isinstance(c, CumulativeCapacityConstraint)]
    assert len(rules2) == 1
    assert rules2[0].capacity_attr == "cap"
    assert rules2[0].status == "candidate"