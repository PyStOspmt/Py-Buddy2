"""
Unit tests for Dynamic Domain Taxonomy Tree (OntoGraph) (src/ontograph.py).
Validates multi-level hierarchy generation, parameter discovery, and drill-down navigation.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import sqlite3
import tempfile

from ontograph import OntoGraph


@pytest.fixture
def multi_domain_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE products (
            id TEXT PRIMARY KEY,
            name TEXT,
            category TEXT,
            brand TEXT,
            price REAL,
            vram_gb INTEGER,
            tdp_watts REAL
        )
    """)
    # GPUs
    cur.execute("INSERT INTO products VALUES ('g1', 'Gigabyte RTX 4060 Windforce', 'gpu', 'Gigabyte', 13200.0, 8, 115.0)")
    cur.execute("INSERT INTO products VALUES ('g2', 'Gigabyte RTX 4070 Eagle', 'gpu', 'Gigabyte', 27000.0, 12, 200.0)")
    cur.execute("INSERT INTO products VALUES ('g3', 'MSI RTX 4080 Super', 'gpu', 'MSI', 44000.0, 16, 320.0)")
    # CPUs
    cur.execute("INSERT INTO products VALUES ('c1', 'AMD Ryzen 7 7800X3D', 'cpu', 'AMD', 16500.0, NULL, 120.0)")
    cur.execute("INSERT INTO products VALUES ('c2', 'Intel Core i7-14700K', 'cpu', 'Intel', 17200.0, NULL, 253.0)")
    conn.commit()
    conn.close()

    yield path

    if os.path.exists(path):
        os.remove(path)

def test_ontograph_hierarchy_levels(multi_domain_db):
    graph = OntoGraph(multi_domain_db)
    tree = graph.to_tree_dict()

    # Root Level 0
    assert tree["level"] == 0
    assert tree["item_count"] == 5
    assert len(tree["children"]) == 2  # gpu, cpu

    # Level 1: Categories
    cat_names = [c["name"].lower() for c in tree["children"]]
    assert "gpu" in cat_names
    assert "cpu" in cat_names

    gpu_node = [c for c in tree["children"] if c["name"].lower() == "gpu"][0]
    assert gpu_node["level"] == 1
    assert gpu_node["item_count"] == 3
    assert "vram_gb" in gpu_node["parameters"]

    # Level 2: Brands inside GPU
    brand_names = [b["name"] for b in gpu_node["children"]]
    assert "Gigabyte" in brand_names
    assert "MSI" in brand_names

    gigabyte_node = [b for b in gpu_node["children"] if b["name"] == "Gigabyte"][0]
    assert gigabyte_node["level"] == 2
    assert gigabyte_node["item_count"] == 2
    assert gigabyte_node["price_min"] == 13200.0
    assert gigabyte_node["price_max"] == 27000.0

def test_ontograph_drilldown_navigation(multi_domain_db):
    graph = OntoGraph(multi_domain_db)

    # Drill down with partial words
    res = graph.navigate("відеокарта gigabyte")
    assert res["found"] is True
    assert res["level"] >= 2
    assert "Gigabyte" in res["name"]
    assert "g1" in res["sample_ids"] or "g2" in res["sample_ids"]


def test_semantic_ontology_inductor_llm():
    from ontograph import SemanticOntologyInductor
    class MockRunner:
        api_key = "test_key"
        def generate(self, prompt: str, temperature: float = 0.0) -> str:
            return '''{
              "domain_concept": "Медичні засоби та фармація",
              "macro_departments": {
                "Знеболювальні засоби": ["analgesics", "antipyretics"],
                "Антибіотики": ["antibiotics"]
              },
              "decision_facets": {
                "Знеболювальні засоби": ["діюча речовина", "форма випуску"]
              },
              "routing_synonyms": {
                "знеболююче": "Знеболювальні засоби",
                "проти температури": "antipyretics"
              }
            }'''

    profile = SemanticOntologyInductor.induce(
        categories=["analgesics", "antipyretics", "antibiotics", "vitamins"],
        sample_rows_by_cat={},
        discovered_params={"analgesics": ["dosage"]},
        runner=MockRunner()
    )

    assert profile.domain_concept == "Медичні засоби та фармація"
    assert "Знеболювальні засоби" in profile.macro_departments
    assert "analgesics" in profile.macro_departments["Знеболювальні засоби"]
    # Invariant: unassigned 'vitamins' is preserved
    assert "Інше / Різне" in profile.macro_departments
    assert "vitamins" in profile.macro_departments["Інше / Різне"]
    assert profile.routing_synonyms.get("знеболююче") == "Знеболювальні засоби"


def test_semantic_ontology_fallback_no_single_token_clipping():
    """Verifies that categories like 'turkey-cutlets' or 'country-terrines' are NOT grouped as 'Turkey' or 'Country'."""
    from ontograph import SemanticOntologyInductor
    cats = [f"grocery/cat_{i}" for i in range(15)] + ["turkey-cutlets", "country-terrines"]
    profile = SemanticOntologyInductor.induce(
        categories=cats,
        sample_rows_by_cat={},
        discovered_params={},
        runner=None
    )
    # Shouldn't invent single-word 'Turkey' or 'Country' as macro departments
    assert "Turkey" not in profile.macro_departments
    assert "Country" not in profile.macro_departments

