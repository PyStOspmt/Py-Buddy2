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
    """F39: '8' near a length is not '8 inches' -- it is unknown."""
    from marketplace_normalizer import MarketplaceEAVNormalizer
    norm = MarketplaceEAVNormalizer.normalize_diameter
    assert norm(17) is None          # bare float: no stated unit
    assert norm("8") is None         # bare string: no stated unit
    # explicit units and the universal R<n> rim marking still parse
    assert norm("R17") == 17.0
    assert norm('17"') == 17.0
    assert norm("457.2 mm") == pytest.approx(18.0)


def test_bare_diameter_falls_through_to_raw_key():
    """An unparseable-unit diameter must not become a canonical inch fact."""
    from marketplace_normalizer import MarketplaceEAVNormalizer
    canon = MarketplaceEAVNormalizer.canonicalize_eav(
        {"diameter": "8"}, category="misc")
    # the raw string survives under its own key -- stated, not
    # reinterpreted; a float 8.0 would mean "8 inches" was invented
    assert canon.get("diameter") == "8"
    assert not any(isinstance(v, float) for v in canon.values())


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
# F40/F47: inductor seeds are labelled data + structural filters do the work
# ---------------------------------------------------------------------------

def _products():
    from hypergraph import ProductNode
    mk = lambda pid, cat, attrs: ProductNode(
        id=pid, name=pid, category=cat, price=1.0, attributes=attrs)
    return [
        mk("a1", "cpu",    {"socket": "AM5", "TDP": 105.0,
                            "shipping_cost": 5.0, "store_note": "x"}),
        mk("a2", "motherboard", {"socket": "AM5", "shipping_cost": 9.0,
                                 "store_note": "x"}),
        mk("b1", "psu",    {"psu_wattage": 650.0, "shipping_cost": 5.0,
                            "store_note": "x"}),
        mk("b2", "case",   {"max_diameter": 20.0, "shipping_cost": 9.0,
                            "store_note": "x"}),
        mk("c1", "rim",    {"diameter": 18.0, "shipping_cost": 5.0,
                            "store_note": "x"}),
    ]


def test_single_valued_attr_never_induces_equivalence():
    """F40: a constant attribute cannot discriminate -- structural, so it
    holds even for names no seed list mentions."""
    from contract_inductor import ContractInductor
    rules = ContractInductor()._induce_deterministic(_products())
    assert not any(getattr(r, "attr_name", None) == "store_note"
                   for r in rules)


def test_seed_lists_are_overridable_data_not_hidden_logic():
    """F40/F47: empty seeds -> 'socket' still induces (it is not in the
    seeds anyway), and 'brand' becomes inducible when the seed is removed --
    proving the seeds are data the caller controls."""
    from contract_inductor import ContractInductor
    from hypergraph import ProductNode
    prods = _products() + [
        ProductNode(id="d1", name="d1", category="gpu", price=1.0,
                    attributes={"brand": "X", "warranty": "y"}),
        ProductNode(id="d2", name="d2", category="cpu", price=1.0,
                    attributes={"brand": "Y"}),
        ProductNode(id="d3", name="d3", category="psu", price=1.0,
                    attributes={"brand": "Z"}),
    ]
    rules = ContractInductor()._induce_deterministic(prods,
                                                   non_matching_keys=set())
    assert any(getattr(r, "attr_name", None) == "brand" for r in rules)


def test_normalizer_key_hints_are_overridable():
    """F42: a source with its own key profile replaces the seeds -- the
    lists are data, not hidden logic. A schema that writes 'Fassung' for
    socket keys is served without editing the module."""
    from marketplace_normalizer import MarketplaceEAVNormalizer
    canon = MarketplaceEAVNormalizer.canonicalize_eav(
        {"Fassung": "AM5"}, category="cpu",
        key_hints={"socket": ["fassung"]})
    assert canon.get("socket") == "AM5"
    # ...and by default the same key is an unknown key, passed through
    canon2 = MarketplaceEAVNormalizer.canonicalize_eav(
        {"Fassung": "AM5"}, category="cpu")
    assert "socket" not in canon2
    assert canon2.get("Fassung") == "am5"


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
    """F47: the induced capacity rule asserts the exact bound unless the
    caller states a margin -- the old code baked in 0.9 silently."""
    from contract_inductor import ContractInductor
    ci = ContractInductor()
    rules = ci._induce_deterministic(_products())
    caps = [r for r in rules if hasattr(r, "headroom_ratio")]
    for r in caps:
        assert r.headroom_ratio == 1.0
    rules2 = ci._induce_deterministic(_products(), headroom_ratio=0.85)
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
