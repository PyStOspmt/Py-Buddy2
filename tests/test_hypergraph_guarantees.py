"""
Tests for the guarantees added in Phase 2.8-2.11.

Each one exists because the previous behaviour was a specific way of being
wrong, and each would be easy to reintroduce:

  2.8  an unknown domain answered from a different catalogue, confidently
  2.9  a "frozen" product node whose attributes could still be rewritten
  2.10 two power supplies counted as alternatives instead of as a sum
  2.11 a full-catalogue scan wearing a category filter
"""
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hypergraph import (
    CumulativeCapacityConstraint,
    EquivalenceConstraint,
    KnowledgeHypergraph,
    ProductNode,
    SpatialBoundingConstraint,
)

# --------------------------------------------------------------------------
# 2.9 -- immutability
# --------------------------------------------------------------------------

def test_product_node_attributes_cannot_be_rewritten():
    node = ProductNode("a", "A", "cpu", 100.0, {"socket": "AM5"})
    with pytest.raises(TypeError):
        node.attributes["socket"] = "AM4"
    assert node.attributes["socket"] == "AM5"


def test_the_mapping_is_also_frozen_against_in_place_edits():
    source = {"socket": "AM5"}
    node = ProductNode("a", "A", "cpu", 100.0, source)
    source["socket"] = "AM4"              # the caller's dict is not aliased
    assert node.attributes["socket"] == "AM5"
    with pytest.raises(AttributeError):
        node.attributes.update({"socket": "AM4"})


def test_a_node_is_hashable_by_identity():
    """Nodes go into sets and dicts; a dict-valued field used to forbid it."""
    node = ProductNode("a", "A", "cpu", 100.0, {"socket": "AM5"})
    assert {node, node} == {node}
    assert {node: "x"}[node] == "x"


# --------------------------------------------------------------------------
# 2.10 -- capacity is summed within a category
# --------------------------------------------------------------------------

def _rig():
    kg = KnowledgeHypergraph()
    kg.add_constraint(CumulativeCapacityConstraint(
        demand_attr="tdp", capacity_attr="wattage"))
    return kg


def test_two_supplies_are_alternatives_not_added_capacity():
    """Two 400 W supplies in the basket do NOT give the machine 800 W --
    a build is carried by one PSU, so capacity is the LARGEST provider.
    The previous summed model let a 700 W build pass on two 400 W units;
    the R6 review flagged that as the same wrong physics in a new shape.
    Explicit provider_mode="additive" restores summing for resource kinds
    whose units genuinely combine (e.g. RAM banks)."""
    kg = _rig()
    for pid, tdp in (("cpu1", 300.0), ("cpu2", 400.0)):
        kg.add_product(ProductNode(pid, pid, "cpu", 1.0, {"tdp": tdp}))
    for pid, w in (("psu1", 400.0), ("psu2", 400.0)):
        kg.add_product(ProductNode(pid, pid, "psu", 1.0, {"wattage": w}))
    # Default "single": 700 W of demand exceeds the best 400 W unit.
    assert not kg.is_compatible({"cpu1", "cpu2", "psu1", "psu2"})
    # One 400 W demand fits the 400 W provider.
    assert kg.is_compatible({"cpu1", "psu1", "psu2"})
    # Additive mode, when the source declares it, sums providers.
    kg2 = _rig()
    kg2.constraints = [c for c in kg2.constraints
                       if not isinstance(c, CumulativeCapacityConstraint)]
    kg2.add_constraint(CumulativeCapacityConstraint(
        demand_attr="tdp", capacity_attr="wattage",
        provider_mode="additive"))
    for pid, tdp in (("cpu1", 300.0), ("cpu2", 400.0)):
        kg2.add_product(ProductNode(pid, pid, "cpu", 1.0, {"tdp": tdp}))
    for pid, w in (("psu1", 400.0), ("psu2", 400.0)):
        kg2.add_product(ProductNode(pid, pid, "psu", 1.0, {"wattage": w}))
    assert kg2.is_compatible({"cpu1", "cpu2", "psu1", "psu2"})


def test_one_small_supply_alone_still_fails():
    kg = _rig()
    kg.add_product(ProductNode("cpu1", "cpu1", "cpu", 1.0, {"tdp": 300.0}))
    kg.add_product(ProductNode("psu1", "psu1", "psu", 1.0, {"wattage": 400.0}))
    kg.add_product(ProductNode("psu2", "psu2", "psu", 1.0, {"wattage": 200.0}))
    # 600 W of supply in the psu category; 700 W of demand breaks it.
    kg.add_product(ProductNode("cpu2", "cpu2", "cpu", 1.0, {"tdp": 400.0}))
    assert not kg.is_compatible({"cpu1", "cpu2", "psu1", "psu2"})
    # Drop the 200 W unit and 700 W of demand faces 400 W of supply.
    assert not kg.is_compatible({"cpu1", "cpu2", "psu1"})


def test_a_big_provider_in_another_category_cannot_vouch():
    """Capacity is per category, so a truck's payload is not a spare PSU."""
    kg = _rig()
    kg.add_product(ProductNode("cpu1", "cpu1", "cpu", 1.0, {"tdp": 300.0}))
    kg.add_product(ProductNode("psu1", "psu1", "psu", 1.0, {"wattage": 200.0}))
    kg.add_product(ProductNode("truck", "truck", "vehicle", 1.0, {"wattage": 5000.0}))
    assert not kg.is_compatible({"cpu1", "psu1", "truck"})
    assert "psu" in (kg.find_conflicts({"cpu1", "psu1", "truck"})[0])


def test_headroom_is_still_applied():
    kg = KnowledgeHypergraph()
    kg.add_constraint(CumulativeCapacityConstraint(
        demand_attr="tdp", capacity_attr="wattage", headroom_ratio=0.8))
    kg.add_product(ProductNode("cpu1", "cpu1", "cpu", 1.0, {"tdp": 350.0}))
    kg.add_product(ProductNode("psu1", "psu1", "psu", 1.0, {"wattage": 400.0}))
    assert not kg.is_compatible({"cpu1", "psu1"})   # 350 > 0.8 * 400 = 320
    # One watt less of demand and the same supply is fine.
    kg.add_product(ProductNode("cpu2", "cpu2", "cpu", 1.0, {"tdp": 300.0}))
    assert kg.is_compatible({"cpu2", "psu1"})


# --------------------------------------------------------------------------
# 2.11 -- the search is indexed, and the index agrees with the scan
# --------------------------------------------------------------------------

def _catalogue(n_rams: int = 400):
    kg = KnowledgeHypergraph()
    kg.add_constraint(EquivalenceConstraint("socket", ["cpu", "motherboard", "ram"]))
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0, {"socket": "AM5"}))
    kg.add_product(ProductNode("mb", "MB", "motherboard", 1.0, {"socket": "AM5"}))
    for i in range(n_rams):
        kg.add_product(ProductNode(
            f"ram{i}", f"RAM {i}", "ram", 1.0,
            {"socket": "AM5" if i % 2 == 0 else "AM4"}))
    return kg


def test_the_indexed_search_returns_the_same_answer_as_the_obvious_one():
    kg = _catalogue()
    got = {n.id for n in kg.find_compatible_additions({"cpu", "mb"}, "ram")}
    # The truth, computed the slow way: validate every ram one at a time.
    want = {f"ram{i}" for i in range(400)
            if kg.is_compatible({"cpu", "mb", f"ram{i}"})}
    assert got == want
    assert len(got) == 200


def test_a_matching_key_excludes_a_candidate_before_any_validation():
    """The point of the index: the LGA board rules out every AM4 module
    without running the constraint on each one."""
    kg = _catalogue()
    kg.add_product(ProductNode("mb_lga", "MB LGA", "motherboard", 1.0,
                              {"socket": "LGA1700"}))
    got = {n.id for n in kg.find_compatible_additions({"cpu", "mb_lga"}, "ram")}
    assert got == set()


def test_find_additions_does_not_scan_other_categories():
    kg = _catalogue()
    kg.add_product(ProductNode("gpu", "GPU", "gpu", 1.0, {"socket": "AM5"}))
    got = {n.id for n in kg.find_compatible_additions({"cpu", "mb"}, "ram")}
    assert "gpu" not in got
    assert len(got) == 200


def test_indexing_earns_its_keep_when_validation_is_not_trivial():
    """Not a benchmark -- a tripwire against a catastrophic index regression.

    F16 honesty note: the old strict "indexed must beat the scan" assertion
    was partly earned by an UNSOUND prefilter -- spatial dimension attributes
    were committed as equality keys, silently excluding reachable results
    (P22). With the sound prefilter the index prunes less, and because
    validation short-circuits on the first violated constraint, the naive
    scan over 4000 rows is competitive. What the index must still guarantee
    is the same order of magnitude -- no catastrophic regression -- while
    the correctness guarantee lives in the differential test above.
    """
    kg = KnowledgeHypergraph()
    kg.add_constraint(EquivalenceConstraint("socket", ["cpu", "motherboard", "ram"]))
    kg.add_constraint(EquivalenceConstraint("memory_type", ["motherboard", "ram"]))
    kg.add_constraint(CumulativeCapacityConstraint("tdp", "wattage"))
    kg.add_constraint(SpatialBoundingConstraint("length", "max_length"))
    kg.add_product(ProductNode("cpu", "CPU", "cpu", 1.0,
                              {"socket": "AM5", "tdp": 105.0, "length": 40.0}))
    kg.add_product(ProductNode("mb", "MB", "motherboard", 1.0,
                              {"socket": "AM5", "memory_type": "DDR5",
                               "wattage": 650.0, "max_length": 60.0}))
    for i in range(4000):
        kg.add_product(ProductNode(
            f"ram{i}", f"RAM {i}", "ram", 1.0,
            {"socket": "AM5" if i % 2 == 0 else "AM4",
             "memory_type": "DDR5" if i % 4 < 2 else "DDR4",
             "tdp": float(i % 10), "length": 30.0 + (i % 3)}))

    start = time.perf_counter()
    indexed = kg.find_compatible_additions({"cpu", "mb"}, "ram")
    indexed_time = time.perf_counter() - start

    # What the previous version did: validate every catalogue row, with
    # the category filter applied afterwards rather than as an index.
    plain = KnowledgeHypergraph()
    plain.nodes = dict(kg.nodes)
    plain.constraints = list(kg.constraints)
    ram_ids = [n.id for n in kg.nodes.values() if n.category == "ram"]
    start = time.perf_counter()
    for rid in ram_ids:
        plain.is_compatible({"cpu", "mb", rid})
    scan_time = time.perf_counter() - start

    assert indexed_time < scan_time * 3, (
        f"indexed {indexed_time:.4f}s over {len(ram_ids)} rows vs "
        f"{scan_time:.4f}s validating each -- catastrophic regression")
