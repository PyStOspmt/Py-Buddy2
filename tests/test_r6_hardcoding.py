"""
R6 regressions for doc-15 hardcoding findings.

F41: seed prototypes in the fuzzy router must not carry brand names in
their SQL. A catalogue without AMD/Intel still answered those queries --
through a LIKE '%amd%' filter baked into production code. Brand
resolution lives in catalog_query.build_extremum_sql, which reads the
brands the source actually carries.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "core"))


def test_no_brand_literals_in_any_seeded_sql():
    """Every SQL the router can emit is brand-free; brands come from data."""
    from fuzzy_ngram_router import FuzzyNgramRouter
    router = FuzzyNgramRouter()
    store = router.store
    entries = getattr(store, "_entries", None) or getattr(store, "entries", [])
    sqls = [e.target_sql for e in entries.values()] if isinstance(
        entries, dict) else [e.target_sql for e in entries]
    assert sqls, "expected seeded prototypes to inspect"
    for sql in sqls:
        low = sql.lower()
        for brand in ("amd", "ryzen", "intel", "nvidia", "geforce"):
            assert f"'%{brand}%'" not in low, sql


def test_brand_query_falls_through_to_deep_path():
    """'cheapest AMD processor' must not hit a baked brand filter."""
    from fuzzy_ngram_router import FuzzyNgramRouter
    router = FuzzyNgramRouter()
    for q in ("Яка точна ціна найдешевшого процесора AMD",
              "Знайди найбюджетніший камінь Ryzen",
              "Яка точна ціна найдешевшого процесора Intel"):
        res = router.route_query(q)
        if res is not None:
            _name, sql, _sim = res
            assert "%amd%" not in sql and "%ryzen%" not in sql \
                and "%intel%" not in sql, (q, sql)


def test_generic_extrema_prototypes_survive():
    """The brand-free prototypes keep working -- coverage, not deletion."""
    from fuzzy_ngram_router import FuzzyNgramRouter
    router = FuzzyNgramRouter()
    res = router.route_query("Який найдешевший товар у магазині")
    assert res is not None
    assert "ORDER BY price ASC" in res[1]


# ---------------------------------------------------------------------------
# F39: a bare number is not inches
# ---------------------------------------------------------------------------

def test_bare_diameter_number_is_unit_unknown_not_inches():
    """F39: '8' without a unit is unknown in Value Algebra, never inches."""
    from attribute_algebra import canonical_form, magnitude, spec_of
    assert spec_of("17").dimension is None
    assert spec_of("8").dimension is None
    # explicit units and the universal R<n> rim marking carry dimension L
    assert spec_of("R17").dimension == "L"
    assert magnitude("R17")[0] == pytest.approx(17.0 * 25.4)
    assert magnitude('17"')[0] == pytest.approx(17.0 * 25.4)
    assert magnitude("457.2 mm")[0] == pytest.approx(18.0 * 25.4)
    assert canonical_form("457.2 mm") == canonical_form('18"')


def test_bare_diameter_falls_through_untyped():
    """An unlabelled bare number does not participate in unit-based induction."""
    from contract_inductor import induce_structural_rules
    from hypergraph import ProductNode
    prods = [
        ProductNode("r1", "r1", "rim", 1.0, {"diameter": "8"}),
        ProductNode("h1", "h1", "hub", 1.0, {"max_diameter": "19"}),
    ]
    assert induce_structural_rules(prods) == []


import pytest

# ---------------------------------------------------------------------------
# F55: Discord embed colour comes from the structured outcome, not prose
# ---------------------------------------------------------------------------

def _discord_gateway():
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "src", "gateways"))
    from discord_adapter import DiscordGateway
    return DiscordGateway(agent=None)


def test_discord_color_uses_structured_verdict_not_prose():
    """An incompatible verdict in English prose must still colour red --
    the cue words are Ukrainian-only, the verdict field is language-free."""
    gw = _discord_gateway()
    result = {
        "final_answer": "These parts cannot be installed together.",
        "steps": [{"thought": "checking", "verdict": "incompatible"}],
        "termination_reason": "SUCCESS_GROUNDED",
    }
    embed = gw.format_response(result)["embeds"][0]
    assert embed["color"] == 0xEF4444


def test_discord_no_verdict_no_fabricated_verified_badge():
    """No evaluated evidence -> no 'Перевірено' claim in the embed."""
    gw = _discord_gateway()
    result = {"final_answer": "Довільна відповідь без вердикта.",
              "steps": [], "termination_reason": None}
    embed = gw.format_response(result)["embeds"][0]
    assert "Перевірено" not in embed["title"]
    assert embed["color"] != 0x10B981


def test_discord_failed_execution_is_not_green():
    gw = _discord_gateway()
    result = {"final_answer": "Будь-який текст.",
              "steps": [],
              "termination_reason": "HALLUCINATION_GATE_VIOLATION"}
    embed = gw.format_response(result)["embeds"][0]
    assert embed["color"] == 0xEF4444


# ---------------------------------------------------------------------------
# F49: extremum fast path requires full query coverage, not a cue blocklist
# ---------------------------------------------------------------------------

_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "catalog.db")


def test_extremum_gate_declines_reasoning_phrasings_it_never_listed():
    """The old _SPEC_OR_REASONING_RE named specific Ukrainian cues; a
    reasoning question phrased differently slipped to the fast path.
    Now any unexplained token declines -- no cue list needed."""
    from catalog_query import build_extremum_sql
    for q in (
        "Чи станеться найдешевший процесор на мою плату",   # compatibility
        "Найдешевша материнська плата",                      # unresolved selector
        "Найдешевший варіант до 5000 грн",                   # budget constraint
        "Найдешевший блок з 16 гб пам'яті",                  # spec constraint
    ):
        assert build_extremum_sql(_DB, q) is None, q


def test_extremum_gate_accepts_fully_explained_queries():
    """Coverage preserved: superlative + catalogue value + function words."""
    from catalog_query import build_extremum_sql
    for q in (
        "Яка найдешевша деталь у наявності?",
        "Яка точна ціна найдешевшого AMD?",
    ):
        assert build_extremum_sql(_DB, q) is not None, q


# ---------------------------------------------------------------------------
# F48: verdict classification prefers a structured verdict over the lexicon
# ---------------------------------------------------------------------------

def test_structured_verdict_beats_prose_language():
    """A system that emits a verdict field is classified by it, whatever
    language its prose is written in -- the cue lexicon never runs."""
    from metrics import extract_verdict
    # English prose the Ukrainian lexicon cannot read at all:
    assert extract_verdict("These parts cannot be installed together.",
                           structured_verdict="incompatible") == "reject"
    assert extract_verdict("Fits perfectly.",
                           structured_verdict="compatible") == "accept"
    # The structured field wins even over contradictory prose cues:
    assert extract_verdict("сумісно", structured_verdict="incompatible") == "reject"


def test_lexicon_remains_labelled_fallback_without_structure():
    """Without a verdict field the lexicon still classifies -- honestly
    neutral where it cannot read the language."""
    from metrics import extract_verdict
    assert extract_verdict("Ця конфігурація сумісна і підходить.") == "accept"
    assert extract_verdict("Така конфігурація несумісна, відхилено.") == "reject"
    # Unreadable language -> neutral, not a guessed accept/reject:
    assert extract_verdict("These parts fit together.") == "neutral"


# ---------------------------------------------------------------------------
# F40/F47: zero domain seed dictionaries in production induction + full invariance
# ---------------------------------------------------------------------------

def _products():
    from hypergraph import ProductNode
    mk = lambda pid, cat, attrs: ProductNode(
        id=pid, name=pid, category=cat, price=1.0, attributes=attrs)
    return [
        mk("a1", "cpu",    {"socket": "AM5", "TDP": "105W",
                            "shipping_cost": "5", "store_note": "x"}),
        mk("a2", "motherboard", {"socket": "AM5", "shipping_cost": "9",
                                 "store_note": "x"}),
        mk("a3", "cpu",    {"socket": "LGA1700", "TDP": "181W",
                            "shipping_cost": "5", "store_note": "x"}),
        mk("a4", "motherboard", {"socket": "LGA1700", "shipping_cost": "9",
                                 "store_note": "x"}),
        mk("b1", "psu",    {"psu_wattage": "650W", "shipping_cost": "5",
                            "store_note": "x"}),
        mk("b2", "case",   {"max_diameter": '20"', "shipping_cost": "9",
                            "store_note": "x"}),
        mk("c1", "rim",    {"diameter": "R18", "shipping_cost": "5",
                            "store_note": "x"}),
    ]


def test_single_valued_attr_never_induces_equivalence():
    """F40: a constant attribute cannot discriminate across categories."""
    from contract_inductor import ContractInductor
    rules = ContractInductor().induce_contracts_from_products(_products())
    assert not any(getattr(r, "attr_name", None) == "store_note"
                   for r in rules)


def test_zero_domain_seed_vocabularies_in_production_inductor_and_catalog():
    """Production induction carries NO domain key/attribute seed dictionaries."""
    import pathlib
    import contract_inductor
    root = pathlib.Path(__file__).resolve().parents[1]
    assert not (root / "src" / "core" / "marketplace_normalizer.py").exists()
    for banned in ("NON_MATCHING_SEED_KEYS", "DEMAND_SEED_TERMS",
                   "CAPACITY_SEED_TERMS", "DIMENSION_SEED_TERMS",
                   "MAX_DIMENSION_SEED_TERMS"):
        assert not hasattr(contract_inductor, banned), f"banned seed constant remains: {banned}"
    assert not hasattr(contract_inductor.ContractInductor, "_induce_deterministic")


def test_full_pipeline_schema_renaming_invariance(tmp_path):
    """Total schema obfuscation (tables, columns, categories, EAV keys, seller axis) preserves induced graph 1:1."""
    import sqlite3
    from domain_catalog import DomainCatalogRegistry
    from hypergraph import CumulativeCapacityConstraint, EquivalenceConstraint

    orig_db = str(tmp_path / "orig.db")
    obf_db = str(tmp_path / "obf.db")

    # Enough distinct rows per category (18 rows across 3 categories) so purely
    # structural category detection fires without ANY hint column names.
    rows = [
        ("c1", "CPU-A", "cpu", 100.0, {"socket": "AM5", "power": "105W", "seller_tag": "shop_1"}),
        ("c2", "CPU-A", "cpu", 105.0, {"socket": "Socket AM5", "power": "105 Вт", "seller_tag": "shop_2"}),
        ("c3", "CPU-B", "cpu", 120.0, {"гніздо": "sAM5", "power": "120W", "seller_tag": "shop_1"}),
        ("c4", "CPU-C", "cpu", 130.0, {"socket": "LGA1700", "power": "180W", "seller_tag": "shop_1"}),
        ("c5", "CPU-D", "cpu", 140.0, {"гніздо": "LGA1700", "power": "253W", "seller_tag": "shop_2"}),
        ("c6", "CPU-E", "cpu", 110.0, {"Socket Type": "AM4", "power": "65W", "seller_tag": "shop_1"}),
        ("m1", "MB-A", "motherboard", 90.0, {"socket": "AM5", "seller_tag": "shop_1"}),
        ("m2", "MB-B", "motherboard", 92.0, {"гніздо": "sAM5", "seller_tag": "shop_2"}),
        ("m3", "MB-C", "motherboard", 95.0, {"Socket Type": "LGA1700", "seller_tag": "shop_1"}),
        ("m4", "MB-D", "motherboard", 96.0, {"socket": "LGA1700", "seller_tag": "shop_2"}),
        ("m5", "MB-E", "motherboard", 85.0, {"гніздо": "AM4", "seller_tag": "shop_1"}),
        ("m6", "MB-F", "motherboard", 88.0, {"Socket Type": "AM4", "seller_tag": "shop_2"}),
        ("p1", "PSU-1", "psu", 80.0, {"power": "650W", "seller_tag": "shop_1"}),
        ("p2", "PSU-2", "psu", 95.0, {"power": "850W", "seller_tag": "shop_2"}),
        ("p3", "PSU-3", "psu", 75.0, {"power": "550W", "seller_tag": "shop_1"}),
        ("p4", "PSU-4", "psu", 105.0, {"power": "1000W", "seller_tag": "shop_2"}),
        ("p5", "PSU-5", "psu", 110.0, {"power": "1200W", "seller_tag": "shop_1"}),
        ("p6", "PSU-6", "psu", 85.0, {"power": "750W", "seller_tag": "shop_2"}),
    ]
    cat_map = {"cpu": "cat_x1", "motherboard": "cat_x2", "psu": "cat_x3"}
    key_map = {
        "socket": "k_a1", "гніздо": "k_a2", "Socket Type": "k_a3",
        "power": "k_b1", "seller_tag": "k_c1",
    }

    c1 = sqlite3.connect(orig_db)
    c1.executescript(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL);"
        "CREATE TABLE product_eav_meta (product_id TEXT REFERENCES products(id), meta_key TEXT, meta_value TEXT);"
    )
    for pid, nm, cat, pr, meta in rows:
        c1.execute("INSERT INTO products VALUES (?,?,?,?)", (pid, nm, cat, pr))
        for k, v in meta.items():
            c1.execute("INSERT INTO product_eav_meta VALUES (?,?,?)", (pid, k, v))
    c1.commit()
    c1.close()

    c2 = sqlite3.connect(obf_db)
    c2.executescript(
        "CREATE TABLE t_ent (c_id TEXT PRIMARY KEY, c_nm TEXT, c_grp TEXT, c_amt REAL);"
        "CREATE TABLE t_attr (c_ref TEXT REFERENCES t_ent(c_id), c_k TEXT, c_v TEXT);"
    )
    for pid, nm, cat, pr, meta in rows:
        c2.execute("INSERT INTO t_ent VALUES (?,?,?,?)", (pid, f"X_{nm}", cat_map[cat], pr))
        for k, v in meta.items():
            v2 = "AM5" if v == "Socket AM5" else v
            c2.execute("INSERT INTO t_attr VALUES (?,?,?)", (pid, key_map[k], v2))
    c2.commit()
    c2.close()

    hg_orig = DomainCatalogRegistry.create_database_catalog(orig_db)
    hg_obf = DomainCatalogRegistry.create_database_catalog(obf_db)

    assert len(hg_orig.constraints) == len(hg_obf.constraints) >= 2
    assert any(isinstance(c, EquivalenceConstraint) for c in hg_obf.constraints)
    assert any(isinstance(c, CumulativeCapacityConstraint) for c in hg_obf.constraints)
    # Seller tag excluded identically without naming "seller_tag" or "k_c1"
    for c in hg_orig.constraints:
        assert getattr(c, "attr_name", None) != "seller_tag"
    for c in hg_obf.constraints:
        assert getattr(c, "attr_name", None) != "k_c1"
    # Compatibility verdicts match on AM5+AM5 vs AM5+LGA1700 despite total obfuscation
    assert hg_obf.compatibility_verdict({"c1", "m2"})["verdict"] == "unknown"
    assert hg_obf.compatibility_verdict({"c1", "m3"})["verdict"] == "candidate_conflict"


def test_domain_description_generated_from_profile_not_prose():
    """F45: a source without a configured description gets one generated
    from its own profiled schema -- counts, not invented domain prose."""
    from server import SystemContext
    ctx = SystemContext()
    bundle = ctx.get_domain_bundle("wdc_products")
    assert bundle.description
    assert "SQLite-джерело" in bundle.description
    for prose in ("електроніка", "лік", "продукти", "інструменти"):
        assert prose not in bundle.description.lower()


def test_bootstrap_registry_is_config_not_code(tmp_path):
    """F45: the default domain set is data/sources.demo.json -- entries it
    does not name are not registered, and a source-registered domain needs
    no code path."""
    from server import SystemContext
    ctx = SystemContext()
    assert "pc_hardware" in ctx.domains
    # A domain id that exists nowhere in the registry is not conjured up:
    assert "electronics_ua" not in ctx.domains


def test_no_fabricated_headroom_margin():
    """F47: the induced capacity rule asserts 1.0 unless caller states a margin."""
    from contract_inductor import ContractInductor
    ci = ContractInductor()
    rules = ci.induce_contracts_from_products(_products())
    caps = [r for r in rules if hasattr(r, "headroom_ratio")]
    assert caps
    for r in caps:
        assert r.headroom_ratio == 1.0
    rules2 = ci.induce_contracts_from_products(_products(), headroom_ratio=0.85)
    assert any(getattr(r, "headroom_ratio", None) == 0.85 for r in rules2)


def test_fallback_bank_is_labelled_unadapted():
    """F52: without a model the inductor returns the generic bank, and
    every question in it is honestly labelled unadapted -- fallback
    vocabulary cannot masquerade as domain induction."""
    from active_inference import DEFAULT_QUESTION_BANK, DomainQuestionBankInductor
    bank = DomainQuestionBankInductor.induce_question_bank(
        domain_name="anything", categories=["x"], parameters=["y"],
        runner=None, cache_dir="nonexistent_dir_xyz")
    assert len(bank) == len(DEFAULT_QUESTION_BANK)
    assert all(q.adapted is False for q in bank)


def test_cached_bank_marks_questions_adapted(tmp_path):
    """F52: a bank replayed from the versioned cache is the adapted one."""
    import hashlib
    import json
    import os

    from active_inference import DomainQuestionBankInductor
    # Seed the cache the way the LLM path would have written it.
    identity = os.path.abspath(str(tmp_path / "d.db"))
    h = hashlib.md5(f"pharmacy_{'a'}_{identity}".encode()).hexdigest()[:10]
    # Envelope format of write_versioned_cache; the db file does not exist
    # so its source version is None.
    (tmp_path / f".question_bank_{h}.json").write_text(json.dumps({
        "source_version": None,
        "payload": [{
            "id": "q1", "text": "?", "target_dimension": "price_sensitivity",
            "options": {"a": 0.9, "b": 0.1}, "cognitive_cost": 1.0},
            {"id": "q2", "text": "?", "target_dimension": "urgency_delivery",
             "options": {"a": 0.9, "b": 0.1}, "cognitive_cost": 1.0}]}))
    bank = DomainQuestionBankInductor.induce_question_bank(
        domain_name="pharmacy", categories=["a"], parameters=[],
        runner=None, cache_dir=str(tmp_path),
        db_path=str(tmp_path / "d.db"))
    assert all(q.adapted for q in bank)


def test_web_ui_ships_no_fixture_vocabulary():
    """F53: the production UI contains no fixture product names, brands,
    domain ids or demo product ids -- everything is fetched or derived."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    app_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    index = (root / "web" / "index.html").read_text(encoding="utf-8")
    for text in (app_js, index):
        for token in ("Ryzen", "ASUS", "MSI", "Samsung", "Seasonic",
                      "аспірин", "ібупрофен", "5x112", "hub_golf",
                      "rim_bbs", "mk_cpu_1", "med_01"):
            assert token not in text
    # The domain selector must be populated from the backend registry,
    # not carry hardcoded options.
    assert "<option" not in index.split('id="domain-select"')[1].split("</select>")[0]
    assert "/api/domains" in app_js


def test_category_color_is_derived_not_lookup():
    """F53: category colouring hashes the category string -- an unseen
    category cannot collide into another category's colour by substring."""
    import pathlib
    import re
    root = pathlib.Path(__file__).resolve().parents[1]
    app_js = (root / "web" / "app.js").read_text(encoding="utf-8")
    assert "CATEGORY_COLORS" not in app_js
    assert re.search(r"indexOf\(k\)", app_js) is None
