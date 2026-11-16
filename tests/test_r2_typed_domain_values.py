"""
Regression suite: typed universal format parsing and domain-hint isolation invariants.

F11's rule: a domain key list is a HINT. A canonical commercial fact is
emitted only when the value parses as the corresponding typed universal
format; anything else falls through to the generic pass-through under its
own key instead of becoming a fabricated fact under a canonical name.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "core"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from contract_inductor import ContractInductor
from hypergraph import (
    CumulativeCapacityConstraint,
    ProductNode,
)
from marketplace_normalizer import MarketplaceEAVNormalizer

# ---------------------------------------------------------------------------
# Power: the value must be a number with an optional power unit
# ---------------------------------------------------------------------------

def test_f11_power_value_must_be_a_number_not_any_digit():
    """
    Failed before: `re.search(r"(\\d+)")` grabbed the first digit run of any
    value, so a mislabelled "TDP: Socket 1200" became a 1200.0-watt fact
    that then ranked and filtered products.
    """
    norm = MarketplaceEAVNormalizer.normalize_power_watts
    assert norm("Socket 1200") is None
    assert norm("червоний") is None
    assert norm("120W") == 120.0
    assert norm("120 Вт") == 120.0
    assert norm("750") == 750.0


def test_f11_kilowatts_are_converted_not_reinterpreted():
    """
    Failed before: "1.2 kW" became 1.2 watts (the number was grabbed, the
    unit ignored), a 1000x error. Now units are standardized with an
    explicit conversion table.
    """
    norm = MarketplaceEAVNormalizer.normalize_power_watts
    assert norm("1.2 kW") == 1200.0
    assert norm("1,2 кВт") == 1200.0


def test_f11_power_key_with_unparseable_value_stays_absent():
    """
    Failed before: canonicalize_eav emitted canonical["tdp_watts"] = 0.0 for
    any unparseable value under a power-ish key. Now nothing is emitted and
    the raw entry passes through generically under its own key.
    """
    canon = MarketplaceEAVNormalizer.canonicalize_eav({"TDP": "Socket 1200"}, category="cpu")
    # No canonical watts are emitted -- and the stated value is preserved
    # verbatim under its own key, not silently reinterpreted by the
    # generic pass-through ("Socket 1200" -> "1200" would be a fabricated
    # numeric fact; dropping it entirely loses a stated fact).
    assert "tdp_watts" not in canon
    assert canon["TDP"] == "Socket 1200"


# ---------------------------------------------------------------------------
# Diameter: the value must be a length or the universal R<n> notation
# ---------------------------------------------------------------------------

def test_f11_diameter_value_must_be_a_length():
    """
    Failed before: any digit run became a diameter, and a non-numeric value
    became 0.0 -- indistinguishable from a real zero.
    """
    norm = MarketplaceEAVNormalizer.normalize_diameter
    assert norm("червоний") is None
    assert norm("Socket 1200") is None
    assert norm("R18") == 18.0
    assert norm('18"') == 18.0
    # F39: a bare number has no stated unit -- unknown, not inches.
    assert norm(17) is None
    assert norm("17") is None


def test_f11_length_units_are_converted_to_the_canonical_dimension():
    """Standardized conversion with an explicit dimension context."""
    norm = MarketplaceEAVNormalizer.normalize_diameter
    assert norm("457.2 mm") == pytest.approx(18.0)
    assert norm("45,72 см") == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# Socket: the alias registry maps stated families, never invents them
# ---------------------------------------------------------------------------

def test_f11_bare_socket_number_gets_no_invented_family():
    """
    Failed before: normalize_socket("1700") returned "LGA1700" -- the LGA
    prefix was invented from a word list, not stated by the source. A stated
    "LGA 1700" still canonicalizes; an unstated number stays a number.
    """
    norm = MarketplaceEAVNormalizer.normalize_socket
    assert norm("1700") == "1700"
    assert norm("LGA 1700") == "LGA1700"
    assert norm("Socket AM5") == "AM5"
    assert norm("AMD Socket AM5") == "AM5"
    assert norm("sAM5") == "AM5"


# ---------------------------------------------------------------------------
# Memory: only the universal DDR format canonicalizes
# ---------------------------------------------------------------------------

def test_f11_memory_type_requires_the_ddr_format():
    """
    Failed before: any value under a memory-ish key became canonical
    ["memory_type"] (e.g. "GDDR6" or a free-text note), letting the
    equivalence machinery treat vendor noise as a matching key.
    """
    canon = MarketplaceEAVNormalizer.canonicalize_eav({"RAM_type": "GDDR6"}, category="gpu")
    assert "memory_type" not in canon
    # Failed typed parses keep the stated value verbatim under its own
    # key, not a canonical-form reinterpretation of it.
    assert canon.get("RAM_type") == "GDDR6"

    canon2 = MarketplaceEAVNormalizer.canonicalize_eav({"RAM_type": "DDR-5"}, category="gpu")
    assert canon2["memory_type"] == "DDR5"


# ---------------------------------------------------------------------------
# PCD: only the universal duplex pattern canonicalizes
# ---------------------------------------------------------------------------

def test_f11_pcd_requires_the_duplex_pattern():
    """Failed before: a non-pattern value was lower-cased into canonical["pcd"]."""
    canon = MarketplaceEAVNormalizer.canonicalize_eav({"PCD": "сірий"}, category="wheels")
    # no canonical pcd is emitted; the stated value is preserved verbatim
    # under its own key rather than dropped or reinterpreted.
    assert "pcd" not in canon
    assert canon == {"PCD": "сірий"}

    canon2 = MarketplaceEAVNormalizer.canonicalize_eav({"PCD": "5*112"}, category="wheels")
    assert canon2["pcd"] == "5x112"


# ---------------------------------------------------------------------------
# Inductor: a capacity/demand pairing must be supported by the sample
# ---------------------------------------------------------------------------

def _node(pid, category, attrs):
    return ProductNode(pid, pid, category, 1.0, attrs)


def test_f11_inductor_capacity_pairing_needs_sample_evidence():
    """
    Failed before: the demand/capacity word lists paired any two numeric
    attributes whose names matched the families, even when the sample's
    numbers refuted the direction -- a "capacity" smaller than its "demand"
    induced a rule that flags every bundle as a conflict.
    """
    inductor = ContractInductor()

    refuted = [
        _node("p1", "psu", {"psu_wattage": 100.0}),
        _node("p2", "cpu", {"tdp_watts": 500.0}),
        _node("p3", "cpu", {"tdp_watts": 600.0}),
    ]
    rules = [c for c in inductor.induce_contracts_from_products(refuted)
             if isinstance(c, CumulativeCapacityConstraint)]
    assert rules == []

    supported = [
        _node("p1", "psu", {"psu_wattage": 750.0}),
        _node("p2", "cpu", {"tdp_watts": 120.0}),
        _node("p3", "cpu", {"tdp_watts": 250.0}),
    ]
    rules2 = [c for c in inductor.induce_contracts_from_products(supported)
              if isinstance(c, CumulativeCapacityConstraint)]
    assert len(rules2) == 1
    # F12: everything induced at runtime stays a candidate.
    assert rules2[0].status == "candidate"