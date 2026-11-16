"""
TDD Suite for Seed-Free Value Algebra & Hypergraph EAV Equivalence.
Verifies canonical resolution of noisy, multilingual, multi-vendor EAV data
without any domain key/vocabulary dictionaries.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from attribute_algebra import canonical_form, spec_of
from hypergraph import (
    EquivalenceConstraint,
    KnowledgeHypergraph,
    ProductNode,
    SpatialBoundingConstraint,
)


def test_socket_canonical_equivalence_without_seed_dict():
    assert canonical_form("Socket AM5", key="socket") == "am5"
    assert canonical_form("sAM5", key="гніздо") == "am5"
    assert canonical_form("Socket LGA1700", key="Socket Type") == "lga1700"
    assert canonical_form("AM4", key="Сокет") == "am4"

    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("c1", "CPU AM5", "cpu", 100.0, {"Сокет": "Socket AM5"}))
    kg.add_product(ProductNode("m1", "MB AM5", "motherboard", 100.0, {"гніздо": "sAM5"}))
    kg.add_product(ProductNode("m2", "MB LGA", "motherboard", 100.0, {"Socket Type": "LGA1700"}))
    kg.add_constraint(EquivalenceConstraint("Сокет", alias_attrs=("гніздо", "Socket Type", "socket")))

    assert kg.is_compatible({"c1", "m1"}) is True
    assert kg.is_compatible({"c1", "m2"}) is False


def test_power_watts_canonical_conversion():
    assert canonical_form("120W") == "120W"
    assert canonical_form("120 Вт") == "120W"
    assert canonical_form("120 watts") == "120W"
    assert canonical_form("1.2 kW") == "1200W"
    assert canonical_form("253.5 Вт") == "253.5W"
    # Bare number stays unitless (F39: no unit guess)
    assert spec_of("750").dimension is None


def test_pcd_wheel_canonical_equivalence():
    assert canonical_form("5x112") == "5x112mm"
    assert canonical_form("5*112 мм") == "5x112mm"
    assert canonical_form("5x120") == "5x120mm"
    assert canonical_form("4*100") == "4x100mm"


def test_spatial_bounding_with_surface_units():
    assert canonical_form("R18") == "457.2mm"
    assert canonical_form('18"') == "457.2mm"
    assert canonical_form("45.72 см") == "457.2mm"
    assert spec_of("17").dimension is None

    kg = KnowledgeHypergraph()
    kg.add_product(ProductNode("hub", "Маточина", "wheel_hub", 100.0, {"max_d": '19.0 дюймів'}))
    kg.add_product(ProductNode("rim_ok", "Диск 18", "rim", 100.0, {"dim": "R18"}))
    kg.add_product(ProductNode("rim_big", "Диск 21", "rim", 100.0, {"dim": "R21"}))
    kg.add_constraint(SpatialBoundingConstraint("dim", "max_d"))

    assert kg.is_compatible({"hub", "rim_ok"}) is True
    assert kg.is_compatible({"hub", "rim_big"}) is False


def test_raw_eav_node_preserves_multivendor_surface_values():
    from domain_catalog import DomainCatalogRegistry
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'marketplace_rozetka.db'))
    if not os.path.exists(db_path):
        pytest.skip("marketplace_rozetka.db not present")
    hg = DomainCatalogRegistry.create_database_catalog(db_path, seed_limit=120)
    assert len(hg.nodes) == 120
    # Preserves raw surface values rather than forcing domain-renamed keys
    cpu = next(n for n in hg.nodes.values() if n.category == "cpu")
    assert any(k in cpu.attributes for k in ("TDP", "Wattage", "tdp_watts", "Потужність", "Споживання", "Тепловиділення"))
    assert len(hg.constraints) >= 3
