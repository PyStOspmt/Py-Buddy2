"""
Cross-Domain Empirical Verification for Py-Buddy Virtual Hypergraph & ODCI.
Proves that the system works across multiple completely different business domains
(PC hardware vs Auto Parts / Tires) without hardcoding domain-specific rules.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from contract_inductor import ContractInductor
from hypergraph import (
    CumulativeCapacityConstraint,
    EquivalenceConstraint,
    KnowledgeHypergraph,
    ProductNode,
    SpatialBoundingConstraint,
)


def test_virtual_hypergraph_direct_invariants():
    """Validates the 4 Universal Meta-Constraints dynamically without static JSON."""
    kg = KnowledgeHypergraph()

    cpu_am5 = ProductNode("cpu_1", "AMD Ryzen 7 7800X3D", "cpu", 16500, {"socket": "AM5", "tdp_watts": 120})
    mb_am5 = ProductNode("mb_1", "MSI B650 Gaming", "motherboard", 6800, {"socket": "AM5"})
    mb_intel = ProductNode("mb_2", "ASUS Prime Z790", "motherboard", 8400, {"socket": "LGA1700"})
    gpu_4080 = ProductNode("gpu_1", "RTX 4080 Super", "gpu", 44000, {"tdp_watts": 350, "length_mm": 336})
    psu_500 = ProductNode("psu_1", "DeepCool 500W", "psu", 1700, {"psu_wattage": 500})
    psu_850 = ProductNode("psu_2", "be quiet! 850W", "psu", 5600, {"psu_wattage": 850})

    for p in [cpu_am5, mb_am5, mb_intel, gpu_4080, psu_500, psu_850]:
        kg.add_product(p)

    # Add Universal Meta-Constraints
    kg.add_constraint(EquivalenceConstraint("socket", categories=["cpu", "motherboard"], description="Сокет процесора та плати"))
    kg.add_constraint(CumulativeCapacityConstraint("tdp_watts", "psu_wattage", headroom_ratio=0.9, description="Баланс живлення БЖ"))

    # 1. Socket Equivalence Check
    assert kg.is_compatible({"cpu_1", "mb_1"}) is True
    assert kg.is_compatible({"cpu_1", "mb_2"}) is False
    
    conflicts = kg.find_conflicts({"cpu_1", "mb_2"})
    assert len(conflicts) >= 1
    assert "AM5" in conflicts[0] and "LGA1700" in conflicts[0]

    # 2. Cumulative Power Check: 120W (CPU) + 350W (GPU) = 470W
    # 470W > 0.9 * 500W (450W limit) -> DeepCool 500W is rejected
    # 470W <= 0.9 * 850W (765W limit) -> be quiet! 850W is accepted
    assert kg.is_compatible({"cpu_1", "gpu_1", "psu_1"}) is False
    assert kg.is_compatible({"cpu_1", "gpu_1", "psu_2"}) is True
    
    power_conflicts = kg.find_conflicts({"cpu_1", "gpu_1", "psu_1"})
    assert any("470" in c or "500" in c for c in power_conflicts)

def test_cross_domain_proof_auto_parts():
    """
    CRITICAL DECISIVE PROOF:
    Runs on an AUTO PARTS catalog (Wheels, Rims, Hubs) with ZERO modifications to the core engine.
    Demonstrates that the Virtual Hypergraph removes all rigid domain frames.
    """
    kg = KnowledgeHypergraph()

    hub_vag = ProductNode("hub_1", "VW Golf 7 Wheel Hub", "wheel_hub", 3500, {
        "pcd": "5x112", 
        "center_bore": 57.1, 
        "max_diameter": 19
    })
    
    rim_oem = ProductNode("rim_1", "BBS CH-R 18-inch Rim", "rim", 12000, {
        "pcd": "5x112", 
        "diameter": 18, 
        "weight_kg": 9.5
    })
    
    rim_bmw = ProductNode("rim_2", "BMW Styling 19-inch Rim", "rim", 14000, {
        "pcd": "5x120",  # Incompatible bolt pattern
        "diameter": 19, 
        "weight_kg": 11.0
    })
    
    rim_oversized = ProductNode("rim_3", "Mansory 21-inch Rim", "rim", 32000, {
        "pcd": "5x112", 
        "diameter": 21,  # Exceeds max_diameter 19
        "weight_kg": 14.0
    })

    for p in [hub_vag, rim_oem, rim_bmw, rim_oversized]:
        kg.add_product(p)

    # Universal constraints applied to Auto domain
    kg.add_constraint(EquivalenceConstraint("pcd", description="Розболтовка колеса (PCD)"))
    kg.add_constraint(SpatialBoundingConstraint("diameter", "max_diameter", description="Максимальний діаметр колісної арки"))

    # VW Golf + BBS (5x112, 18 <= 19) -> COMPATIBLE
    assert kg.is_compatible({"hub_1", "rim_1"}) is True

    # VW Golf + BMW Rim (5x112 != 5x120) -> INCOMPATIBLE (PCD mismatch)
    assert kg.is_compatible({"hub_1", "rim_2"}) is False
    pcd_conflicts = kg.find_conflicts({"hub_1", "rim_2"})
    assert any("5x112" in c and "5x120" in c for c in pcd_conflicts)

    # VW Golf + Mansory Rim (21 > 19) -> INCOMPATIBLE (Diameter overflow)
    assert kg.is_compatible({"hub_1", "rim_3"}) is False
    size_conflicts = kg.find_conflicts({"hub_1", "rim_3"})
    assert any("21" in c and "19" in c for c in size_conflicts)

    # Dynamic candidate discovery:
    compatible_rims = kg.find_compatible_additions({"hub_1"}, target_category="rim")
    assert len(compatible_rims) == 1
    assert compatible_rims[0].id == "rim_1"

def test_contract_inductor_automated_discovery():
    """
    Validates that ContractInductor automatically discovers invariants from raw products
    without any manual intervention.
    """
    inductor = ContractInductor()

    raw_products = [
        ProductNode("p1", "Winter Parka", "clothing", 4500, {"size_standard": "EU", "gender": "unisex"}),
        ProductNode("p2", "Wool Scarf", "accessory", 1200, {"size_standard": "EU"}),
        ProductNode("p3", "US Gloves", "accessory", 800, {"size_standard": "US"})
    ]

    induced_kg = inductor.build_virtual_hypergraph(raw_products, use_llm=False)
    
    assert len(induced_kg.constraints) >= 1
    # Check that EU Parka + EU Scarf is compatible, but EU Parka + US Gloves is incompatible
    assert induced_kg.is_compatible({"p1", "p2"}) is True
    assert induced_kg.is_compatible({"p1", "p3"}) is False


def test_unified_structural_inductor_alias_and_unit_conversion():
    """ContractInductor handles alias keys + unit-converted capacity and status promotion."""
    from contract_inductor import ContractInductor

    items = [
        ProductNode("c1", "Module A", "cpu", 5000.0, {"socket": "AM5", "weight": "400 g"}),
        ProductNode("m1", "Carrier Y", "motherboard", 6000.0, {"Сокет": "AM5", "max_load": "1.0 kg"}),
        ProductNode("c2", "Module B", "cpu", 5500.0, {"socket": "LGA1700", "weight": "600 g"}),
        ProductNode("g1", "Addon Z", "gpu", 12000.0, {"addon_weight": "0.5 kg"}),
    ]
    corpus = items + [
        ProductNode("c3", "Module A2", "cpu", 5200.0, {"socket": "AM5", "weight": "350 g"}),
        ProductNode("m2", "Carrier Y2", "motherboard", 6100.0, {"socket": "AM5", "max_load": "1.2 kg"}),
        ProductNode("c4", "Module B2", "cpu", 5600.0, {"socket": "LGA1700", "weight": "550 g"}),
        ProductNode("m3", "Carrier Intel", "motherboard", 6400.0, {"socket": "LGA1700", "max_load": "1.1 kg"}),
    ]
    inductor = ContractInductor()
    kg = KnowledgeHypergraph()
    for p in corpus:
        kg.add_product(p)
    rules = inductor.induce_contracts_from_products(corpus)
    for r in rules:
        kg.add_constraint(r)

    assert len(rules) >= 2
    assert any(isinstance(r, EquivalenceConstraint) for r in rules)
    assert any(isinstance(r, CumulativeCapacityConstraint) for r in rules)
    assert all(r.status == "candidate" for r in rules)
    # Candidate rule conservatively blocks mismatch (c2 LGA1700 + m1 AM5)
    assert kg.find_conflicts({"c2", "m1"})
    # Confirming all candidate rules makes clean AM5 build strictly compatible
    for r in rules:
        assert kg.set_constraint_status(r.rule_id(), "verified") is not None
    assert kg.is_compatible({"c1", "m1"}) is True
    assert kg.is_compatible({"c2", "m1"}) is False
    # Cumulative unit capacity: 400 g + 0.5 kg = 0.9 kg <= 1.0 kg (compatible);
    # 600 g + 0.5 kg = 1.1 kg > 1.0 kg (conflict!).
    cap = CumulativeCapacityConstraint(
        demand_attr="weight",
        capacity_attr="max_load",
        headroom_ratio=1.0,
        demand_attrs=("weight", "addon_weight"),
        description="Total weight <= carrier max_load",
    )
    kg_cap = KnowledgeHypergraph()
    for p in [items[0], items[1], items[2], items[3]]:
        kg_cap.add_product(p)
    kg_cap.add_constraint(cap)
    assert kg_cap.is_compatible({"c1", "g1", "m1"}) is True
    assert kg_cap.is_compatible({"c2", "g1", "m1"}) is False
