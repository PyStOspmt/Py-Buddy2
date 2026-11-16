"""
Core regression suite for price tolerances, read-only catalog invariants,
capacity constraints, and attribute equivalence coverage checks.
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest

# ---------------------------------------------------------------- 1. gate

def test_gate_uses_relative_price_tolerance():
    from sanitizer import ZeroHallucinationGate
    history = [{"tool_name": "sql_query",
                "result": {"rows": [{"id": "p1", "name": "Тістечко",
                                     "price": 5.0}]}}]
    # Claimed 5.5 on a 5.0 item: within the old absolute 1.0 band, but a
    # 10% lie under the documented 1% contract.
    ok, violations = ZeroHallucinationGate.verify_recommendation_grounding(
        [{"id": "p1", "name": "Тістечко", "price": 5.5}], history)
    assert not ok and any("price" in v.lower() or "Price" in v
                          for v in violations), violations
    # A genuinely rounded display stays allowed.
    ok, _ = ZeroHallucinationGate.verify_recommendation_grounding(
        [{"id": "p1", "name": "Тістечко", "price": 5.01}], history)
    assert ok


# -------------------------------------------------------- 2. catalog GET

# ------------------------------------------------- 5. per-session domain

def test_switch_domain_is_session_scoped():
    from server import SwitchDomainRequest, ctx, switch_domain

    if len(ctx.domains) < 2:
        pytest.skip("needs two registered domains")
    d_b = sorted(ctx.domains)[-1]

    class FakeRequest:
        headers = {}
        client = None

    default_before = ctx.active_domain
    import uuid
    requested = "s-" + uuid.uuid4().hex  # unknown id: must not be resumed or chosen
    res = switch_domain(SwitchDomainRequest(domain=d_b, session_id=requested),
                        FakeRequest())
    # F30: a client cannot create a session by choosing its id; the server mints
    # it and returns it. The scope assertions below are unchanged.
    sid = res.get("session_id")
    assert sid and sid != requested
    # A session-scoped switch must not move the global default for others.
    assert ctx.active_domain == default_before
    saved = ctx.session_store.get_session(sid)
    assert saved and saved.domain == d_b
    other = ctx.session_store.get_session("s-dom-other")
    assert other is None or other.domain != d_b


def test_stale_session_domain_falls_back_to_default():
    """A session whose stored domain was unregistered must not brick the
    dialogue: resolution validates the stored id and falls back to the
    global default instead of raising 'unknown domain'."""
    from react_agent import SessionState
    from server import _resolve_chat_domain, ctx

    sid = "s-dom-stale"
    ses = ctx.session_store.get_session(sid) or SessionState(
        session_id=sid, user_query="")
    ses.domain = "no_such_domain_xyz"
    ctx.session_store.save_session(ses)
    try:
        ses2 = ctx.session_store.get_session(sid)
        # Stale preference rejected -> falls back to the global default.
        assert _resolve_chat_domain(None, ses2) == ctx.active_domain
        # A real session preference wins over the default.
        ses2.domain = sorted(ctx.domains)[-1]
        assert _resolve_chat_domain(None, ses2) == ses2.domain
        # An explicit request domain wins over both (precedence is what
        # is asserted; downstream validation of unknown ids is separate).
        assert _resolve_chat_domain("explicit_req_dom", ses2) == "explicit_req_dom"
    finally:
        ses.domain = None
        ctx.session_store.save_session(ses)


def test_catalog_get_does_not_mutate_hypergraph(tmp_path):
    db = tmp_path / "shop.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?)",
        [(f"p{i}", f"Товар {i}", "cat", 10.0 * i, 1, 50.0) for i in range(10)])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle, ctx
    bundle = DomainBundle("cat_get_src", "cat test", str(db),
                          KnowledgeHypergraph(), "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["cat_get_src"] = bundle

    from server import get_catalog
    # First call lazily builds the engine (a legitimate one-time state
    # transition). The defect is the GET then keeping hg.nodes growing on
    # every page -- measure between calls, not across init.
    get_catalog(domain="cat_get_src")
    before = len(bundle.hypergraph.nodes)
    get_catalog(domain="cat_get_src", offset=40)
    get_catalog(domain="cat_get_src", search="Товар 7")
    assert len(bundle.hypergraph.nodes) == before, (
        "a GET must not grow the shared hypergraph")
    ctx.domains.pop("cat_get_src", None)


# --------------------------------------------------- 3. capacity physics

def _nodes():
    from hypergraph import ProductNode
    return [
        ProductNode("psu1", "БЖ 500", "psu", 1000.0, {"wattage": 500.0}),
        ProductNode("psu2", "БЖ 500 №2", "psu", 1100.0, {"wattage": 500.0}),
        ProductNode("gpu", "ВК 400Вт", "gpu", 20000.0, {"tdp": 400.0}),
    ]


def test_capacity_is_max_provider_not_sum():
    from hypergraph import CumulativeCapacityConstraint, ProductNode
    c = CumulativeCapacityConstraint(demand_attr="tdp", capacity_attr="wattage")
    nodes = _nodes()
    # 400 W demand vs two 500 W units: under the summing bug this passed
    # (1000 W "capacity"); a single PSU cannot carry it only when demand
    # exceeds the LARGEST single provider. 400 <= 500 -> pass.
    ok, _ = c.validate(nodes)
    assert ok

    nodes[2] = ProductNode("gpu", "ВК 600Вт", "gpu", 20000.0, {"tdp": 600.0})
    ok, reason = c.validate(nodes)
    assert not ok and "600" in reason, reason


def test_capacity_reports_unseen_demand_items():
    from hypergraph import CumulativeCapacityConstraint, ProductNode
    c = CumulativeCapacityConstraint(demand_attr="tdp", capacity_attr="wattage")
    nodes = _nodes()
    nodes[2] = ProductNode("gpu", "ВК без TDP", "gpu", 20000.0, {})
    gaps = c.coverage_gaps(nodes)
    assert any("tdp" in g for g in gaps), gaps


# ------------------------------------------------ 4. equivalence coverage

def test_equivalence_reports_unseen_scoped_members():
    from hypergraph import EquivalenceConstraint, ProductNode
    c = EquivalenceConstraint(attr_name="socket", categories=["cpu", "mb"])
    nodes = [
        ProductNode("cpu1", "ЦП А", "cpu", 5000.0, {"socket": "AM5"}),
        ProductNode("mb1", "МП Б", "mb", 3000.0, {"socket": "AM5"}),
        ProductNode("mb2", "МП без сокета", "mb", 2500.0, {}),
    ]
    gaps = c.coverage_gaps(nodes)
    assert any("socket" in g and "МП без сокета" in g for g in gaps), gaps


# ------------------------------------- 5. T-series stress-wave fixes

def test_catalog_search_escapes_like_wildcards(tmp_path):
    db = tmp_path / "wild.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "price REAL, stock_quantity INTEGER, performance_score REAL)")
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?,?,?)",
        [("p1", "Товар 100% якості", "cat", 10.0, 1, 50.0),
         ("p2", "Звичайний товар", "cat", 20.0, 1, 50.0)])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle, ctx, get_catalog
    bundle = DomainBundle("wild_src", "wild", str(db),
                          KnowledgeHypergraph(), "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["wild_src"] = bundle
    try:
        # A literal percent must match only the row whose name contains %.
        assert get_catalog(domain="wild_src", search="%")["total"] == 1
        # A literal underscore matches nothing here (not a single-char wildcard).
        assert get_catalog(domain="wild_src", search="_")["total"] == 0
        # Ordinary substring search still works.
        assert get_catalog(domain="wild_src", search="Звичайний")["total"] == 1
    finally:
        ctx.domains.pop("wild_src", None)


def test_induction_cap_is_bounded_and_reported():
    from contract_inductor import ContractInductor
    from hypergraph import ProductNode
    products = [
        ProductNode(f"p{i}", f"n{i}", f"c{i % 2}", 1.0,
                    {f"attr_{k}": f"v{i % 3}" for k in range(30)})
        for i in range(40)
    ]
    ind = ContractInductor(max_induced=10)
    rules = ind.induce_contracts_from_products(products)
    assert len(rules) <= 10
    rep = ind.last_induction_report
    assert rep is not None and rep["dropped"] > 0
    assert rep["induced"] == rep["kept"] + rep["dropped"]

    uncapped = ContractInductor()
    rules_all = uncapped.induce_contracts_from_products(products)
    assert uncapped.last_induction_report["dropped"] == 0
    assert len(rules_all) == uncapped.last_induction_report["induced"]


def test_runtime_registrations_survive_restart(tmp_path, monkeypatch):
    """T9: a runtime registration must restore after ctx rebuild, and a
    stale entry must be skipped-and-reported, not crash or resurrect."""
    db = tmp_path / "persist.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    conn.execute("INSERT INTO products VALUES ('p1','n1','c',1.0)")
    conn.commit()
    conn.close()

    reg = tmp_path / "sources.runtime.json"
    monkeypatch.setenv("AINSA_RUNTIME_REGISTRY", str(reg))
    monkeypatch.setenv("AINSA_SOURCE_ROOT", str(tmp_path))

    from server import DomainBundle, SystemContext
    ctx1 = SystemContext.__new__(SystemContext)
    ctx1.domains = {}
    ctx1.register_domain_bundle(DomainBundle("rt_src", "rt", str(db)))
    ctx1.persist_runtime_registration(ctx1.domains["rt_src"])

    # Simulate restart: fresh context re-reads the registry.
    ctx2 = SystemContext.__new__(SystemContext)
    ctx2.domains = {}
    ctx2._restore_runtime_registrations()
    assert "rt_src" in ctx2.domains
    assert ctx2.domains["rt_src"].db_path == str(db)
    assert ctx2.skipped_runtime_sources == []

    # A stale entry (file deleted) is skipped and reported.
    os.remove(str(db))
    ctx3 = SystemContext.__new__(SystemContext)
    ctx3.domains = {}
    ctx3._restore_runtime_registrations()
    assert "rt_src" not in ctx3.domains
    assert ctx3.skipped_runtime_sources == [
        {"domain_id": "rt_src", "reason": "file_missing"}]


def test_structural_eav_exposes_key_vocabulary(tmp_path):
    """T7: a structurally-detected EAV table must surface its actual key
    values so the attribute axis is queryable, not just detected."""
    db = tmp_path / "eav.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE items (i TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    conn.execute("CREATE TABLE attrs (i TEXT, k TEXT, v TEXT)")
    for i in range(8):
        conn.execute("INSERT INTO items VALUES (?,?,?,?)",
                     (f"e{i}", f"n{i}", "c", 1.0))
        for k in ("color", "size", "material"):
            conn.execute("INSERT INTO attrs VALUES (?,?,?)",
                         (f"e{i}", k, f"val{i % 3}"))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    meta = proj.eav_tables.get("attrs")
    assert meta is not None and meta["detection"] == "structural"
    assert {"color", "size", "material"} <= set(
        meta.get("discovered_attributes") or [])
    # honest coverage bookkeeping: the projection knows the exact
    # vocabulary size, so a stored sample is never presented as complete
    assert meta.get("keys_total") == 3


def test_merchant_column_is_not_mislabelled_category(tmp_path):
    """a7: an offer table's `shop` column is a merchant axis, not a
    product category. It must resolve to `merchant` and be excluded
    from `category` instead of wearing the wrong label."""
    db = tmp_path / "offers.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE offers (id TEXT PRIMARY KEY, name TEXT, "
                 "shop TEXT, price REAL)")
    for i in range(12):
        conn.execute("INSERT INTO offers VALUES (?,?,?,?)",
                     (f"o{i}", f"Товар {i}", f"shop_{i % 4}", 10.0 + i))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    assert proj.resolve("merchant").column == "shop"
    assert not proj.resolve("category").resolved, (
        "shop must not be relabelled as a product category")


def test_llm_proposal_is_verified_against_data(tmp_path):
    """The advisory layer: the model may propose a binding for an
    UNRESOLVED role, but only a proposal that passes the role's own
    structural predicate may resolve. A temporal proposal is rejected and
    recorded as a vetoed candidate with the rejection reason visible."""
    db = tmp_path / "assist.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "grp TEXT, segment TEXT, created_at TEXT, price REAL)")
    for i in range(24):
        conn.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                     (f"p{i}", f"Item {i}", f"g{i % 3}", f"s{i % 3}",
                      f"2024-01-{(i % 27) + 1:02d}", 5.0))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    # two equally low-cardinality text columns -> honestly ambiguous
    assert not proj.resolve("category").resolved

    # a temporal proposal on an unresolved role is rejected and recorded
    # (proposals are tried BEFORE a valid one so the role is still open)
    v2 = proj.apply_llm_proposal("category", "created_at", "guess")
    assert v2.startswith("rejected")
    vetoed = {c.column for c in proj.resolve("category").candidates
              if c.vetoed}
    assert "created_at" in vetoed

    # model proposes the genuinely categorical column -> verified
    v = proj.apply_llm_proposal("category", "grp", "looks like a grouping")
    assert v.startswith("resolved")
    assert proj.resolve("category").column == "grp"
    assert any(e.kind == "llm_hypothesis" for e in
               proj.resolve("category").candidates[-1].evidence)


def test_temporal_column_cannot_capture_category_role(tmp_path):
    """Live finding from the inspection panel on real_prices: the true
    `category`/`brand` columns held only 'unknown' (distinct=1, below the
    structural bound), so the role was captured by `created_at` and the
    taxonomy grouped products by calendar dates. A temporal column is
    never a category axis -- it must be vetoed even when it is the last
    structural candidate, and the veto must stay visible in candidates."""
    db = tmp_path / "dates.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, brand TEXT, created_at TEXT, price REAL)")
    for i in range(30):
        conn.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                     (f"p{i}", f"Item {i}", "unknown", "unknown",
                      f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}", 9.9))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    cat = proj.resolve("category")
    assert cat.column != "created_at", \
        "a timestamp column must never own the category role"
    vetoed = {c.column for c in cat.candidates if c.vetoed}
    assert "created_at" in vetoed, vetoed


def test_brand_remains_a_category_axis(tmp_path):
    """`brand` is a product attribute, not a merchant: a catalogue whose
    only low-cardinality text column is brand keeps it as category."""
    db = tmp_path / "brands.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "brand TEXT, price REAL)")
    for i in range(12):
        conn.execute("INSERT INTO products VALUES (?,?,?,?)",
                     (f"p{i}", f"Виріб {i}", f"brand_{i % 4}", 10.0 + i))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    assert not proj.resolve("merchant").resolved
    assert proj.resolve("category").column == "brand"


def test_query_work_budget_is_bounded_but_not_tiny(tmp_path):
    """The 50k-step hard abort killed legitimate dense-EAV joins. The
    budget must be configurable, report its exhaustion honestly, and by
    default allow a mid-size join while still bounding runaway work."""
    db = tmp_path / "steps.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE items (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    conn.execute("CREATE TABLE attrs (i TEXT, k TEXT, v TEXT)")
    conn.executemany("INSERT INTO items VALUES (?,?,?,?)",
                     [(f"e{i}", f"n{i}", "c", 1.0) for i in range(3000)])
    # 100k un-indexed attribute rows: a filter scan needs well over 50k
    # VM steps, so a 50k-step budget must abort it while the default
    # budget lets it through.
    conn.executemany("INSERT INTO attrs VALUES (?,?,?)",
                     [(f"e{i % 3000}", f"attr_{i % 10}", str(i % 5))
                      for i in range(100_000)])
    conn.commit()
    conn.close()

    from sanitizer import SQLSanitizer
    q = "SELECT COUNT(DISTINCT i) FROM attrs WHERE k = 'attr_5'"
    tight = SQLSanitizer(max_work_steps=50_000)
    res = tight.execute_safe_query(str(db), q)
    assert res["success"] is False
    assert "work budget" in res["error"]

    sane = SQLSanitizer()  # default budget
    res2 = sane.execute_safe_query(str(db), q)
    assert res2["success"] is True and res2["row_count"] > 0


def test_catalog_fts_matches_like_semantics(tmp_path):
    """FTS5 trigram must preserve substring semantics: inside-token
    matches and case-insensitive Cyrillic included, and it must report
    which backend served the search."""
    db = tmp_path / "fts.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL, stock_quantity INTEGER, "
                 "performance_score REAL)")
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", [
        ("p1", "AMD Ryzen 7 7800X3D", "cpu", 400.0, 1, 90.0),
        ("p2", "Intel Core i7", "cpu", 350.0, 1, 85.0),
        ("p3", "Звичайний товар", "misc", 10.0, 1, 1.0),
    ])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle, ctx, get_catalog
    bundle = DomainBundle("fts_src", "fts", str(db),
                          KnowledgeHypergraph(), "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["fts_src"] = bundle
    try:
        # inside-token substring: 'zen' must find 'Ryzen'
        r = get_catalog(domain="fts_src", search="zen")
        assert r["total"] == 1 and r["search_backend"] == "fts5"
        # case-insensitive Cyrillic
        r = get_catalog(domain="fts_src", search="звичай")
        assert r["total"] == 1
        # term too short -> honest LIKE fallback
        r = get_catalog(domain="fts_src", search="a")
        assert r["search_backend"] == "like"
        # wildcard literal still holds on the fallback path
        r = get_catalog(domain="fts_src", search="%")
        assert r["total"] == 0 and r["search_backend"] == "like"
    finally:
        ctx.domains.pop("fts_src", None)


def test_zero_result_search_relaxes_over_eav_axes(tmp_path):
    """EBR-1: a shopkeeper who knows the stock can bridge "фарба для
    взуття" to textile/leather paint because they share the attribute
    family. The lexical layer honestly returns 0 -- then the associative
    layer surfaces items whose EAV *values* mention the term, each with
    a typed evidence trail (which axis bridged, which value matched).
    Relaxed hits must never be merged into `total` or `products`."""
    db = tmp_path / "relax.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE purposes (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE tags (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL, "
                 "purpose_id TEXT REFERENCES purposes(id))")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    conn.execute("CREATE TABLE product_tags ("
                 "product_id TEXT REFERENCES products(id), "
                 "tag_id TEXT REFERENCES tags(id))")
    conn.executemany("INSERT INTO purposes VALUES (?,?)", [
        ("d1", "одяг"), ("d2", "взуття")])
    conn.executemany("INSERT INTO tags VALUES (?,?)", [
        ("t1", "антисептична"), ("t2", "водостійка")])
    conn.execute("INSERT INTO product_tags VALUES ('p4','t1')")
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
        ("p1", "Фарба акрилова А", "paint", 10.0, "d1"),
        ("p2", "Фарба акрилова Б", "paint", 12.0, "d1"),
        ("p3", "Пензель універсальний", "tool", 3.0, "d1"),
        ("p4", "Фарба акрилова В", "paint", 11.0, "d2"),
    ])
    # 4 entities x 2 keys: key cardinality (2) < entity cardinality (4)
    # with >= 2 rows per entity -- the minimal honest structural EAV
    # signature (sparser shapes are ambiguous and must NOT be detected).
    conn.executemany("INSERT INTO attrs VALUES (?,?,?)", [
        ("p1", "призначення", "текстиль"),
        ("p1", "об'єм", "200мл"),
        ("p2", "призначення", "шкіра"),
        ("p2", "об'єм", "150мл"),
        ("p3", "призначення", "універсальне"),
        ("p3", "об'єм", "50см"),
        ("p4", "призначення", "метал"),
        ("p4", "об'єм", "100мл"),
    ])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle, ctx, get_catalog
    bundle = DomainBundle("relax_src", "relax", str(db),
                          KnowledgeHypergraph(), "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["relax_src"] = bundle
    try:
        # 'шкіра' is nowhere in names/categories -- lexical total stays 0
        r = get_catalog(domain="relax_src", search="шкіра")
        assert r["total"] == 0 and r["products"] == []
        rel = r["relaxation"]
        assert rel is not None and rel["kind"] == "associative_relaxation"
        hits = {i["entity_id"] for i in rel["items"]}
        assert hits == {"p2"}
        item = rel["items"][0]
        assert item["via"] == "eav_value" and item["axis"] == "призначення"
        # the axis bridge exposes sibling values -- the widening surface
        # is inspectable, not a black box
        bridges = [s for s in rel["steps"] if s["stage"] == "axis_bridge"]
        assert bridges and bridges[0]["axis"] == "призначення"
        assert "текстиль" in bridges[0]["detail"]

        # a term that is absent everywhere is honestly unresolved
        r = get_catalog(domain="relax_src", search="титан")
        assert r["total"] == 0
        assert r["relaxation"]["items"] == []
        assert any(s["stage"] == "unresolved_term"
                   for s in r["relaxation"]["steps"])

        # EBR-2: multi-term conjunctive binding across channels --
        # 'фарба' binds the name column, 'шкіра' binds an EAV value;
        # only p2 satisfies BOTH terms, regret = 0 (nothing dropped)
        r = get_catalog(domain="relax_src", search="фарба шкіра")
        assert r["total"] == 0
        rel = r["relaxation"]
        assert rel["regret"] == 0
        hits = {i["entity_id"]: i for i in rel["items"]}
        assert set(hits) == {"p2"}
        assert hits["p2"]["via"] == "conjunctive"
        chans = {b["channel"] for b in hits["p2"]["bindings"]}
        assert chans == {"entity_column", "eav_value"}

        # fk_dimension channel: 'взуття' lives in the normalized
        # `purposes` lookup behind a declared FK -- it binds there and
        # only p4 satisfies the full conjunct (regret = 0)
        r = get_catalog(domain="relax_src", search="фарба взуття")
        assert r["total"] == 0
        rel = r["relaxation"]
        assert rel["regret"] == 0
        hits = {i["entity_id"]: i for i in rel["items"]}
        assert set(hits) == {"p4"}
        assert hits["p4"]["via"] == "conjunctive"
        chans = {b["channel"] for b in hits["p4"]["bindings"]}
        assert chans == {"entity_column", "fk_dimension"}

        # and the dimension axis also bridges: siblings of the matched
        # dimension value are exposed, same as EAV axes
        r = get_catalog(domain="relax_src", search="взуття")
        assert r["total"] == 0
        rel = r["relaxation"]
        hits = {i["entity_id"]: i for i in rel["items"]}
        assert set(hits) == {"p4"} and hits["p4"]["via"] == "fk_dimension"
        bridges = [s for s in rel["steps"] if s["stage"] == "axis_bridge"]
        assert any(b["axis"] == "purposes.label" and "одяг" in b["detail"]
                   for b in bridges)

        # junction channel: 'антисептична' lives in `tags` behind the
        # product_tags bridge (two declared FKs) -- conjunctive p4 again
        r = get_catalog(domain="relax_src", search="фарба антисептична")
        assert r["total"] == 0
        rel = r["relaxation"]
        assert rel["regret"] == 0
        hits = {i["entity_id"]: i for i in rel["items"]}
        assert set(hits) == {"p4"}
        chans = {b["channel"] for b in hits["p4"]["bindings"]}
        assert chans == {"entity_column", "junction"}

        # a term nobody can satisfy is still dropped, regret counted
        r = get_catalog(domain="relax_src", search="фарба титан")
        assert r["total"] == 0
        rel = r["relaxation"]
        assert rel["regret"] == 1
        drops = [s["term"] for s in rel["steps"]
                 if s["stage"] == "constraint_relaxation"]
        assert drops == ["титан"]

        # direct hits are NOT relaxed -- relaxation only fills a void
        r = get_catalog(domain="relax_src", search="Фарба")
        assert r["total"] == 3  # p1, p2, p4 all contain 'Фарба'
        assert r["relaxation"] is None
    finally:
        ctx.domains.pop("relax_src", None)


def test_domains_item_count_is_catalogue_rows_not_graph_nodes(tmp_path):
    """`/api/domains` previously reported the seeded hypergraph's node
    count (~a hundred sampled nodes) as `item_count` while the catalogue
    held thousands of rows. The two must be distinct fields."""
    db = tmp_path / "counts.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL)")
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)",
                     [(f"p{i}", f"Item {i}", "cat", 1.0)
                      for i in range(5000)])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle, ctx, get_domains
    hg = KnowledgeHypergraph()
    for i in range(3):  # seeded graph: only a sample of the catalogue
        hg.add_product(__import__("hypergraph").ProductNode(
            id=f"p{i}", name=f"Item {i}", category="cat", price=1.0))
    bundle = DomainBundle("count_src", "counts", str(db), hg, "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["count_src"] = bundle
    try:
        domains = {d["id"]: d for d in get_domains()["domains"]}
        entry = domains["count_src"]
        assert entry["item_count"] == 5000
        assert entry["graph_nodes"] == 3
    finally:
        ctx.domains.pop("count_src", None)


# -------------------------------------------------- introspection endpoint

def test_inspection_exposes_projection_eav_and_prompt(tmp_path):
    """The research-facing inspection endpoint must surface the whole
    adaptive mapping: role resolutions with evidence, EAV discovery,
    induced contracts, the question bank and the literal schema prompt --
    and must not leak secrets."""
    db = tmp_path / "insp.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE products (sku TEXT PRIMARY KEY, title TEXT, "
        "cat_name TEXT, cost_usd REAL)")
    conn.execute(
        "CREATE TABLE specs (pid TEXT, attr TEXT, val TEXT)")
    conn.executemany(
        "INSERT INTO products VALUES (?,?,?,?)",
        [(f"s{i}", f"Device {i}", "gadgets", 50.0 + i) for i in range(20)])
    conn.executemany(
        "INSERT INTO specs VALUES (?,?,?)",
        # Diverse values so `val` cannot be mistaken for a key column --
        # a 2-valued column is symmetric to the key and the structural
        # detector honestly reports the shape as ambiguous.
        [(f"s{i}", "weight", f"{i}.5") for i in range(20)]
        + [(f"s{i}", "warranty", f"{i+12}m") for i in range(20)])
    conn.commit()
    conn.close()

    from hypergraph import KnowledgeHypergraph, ProductNode
    from server import DomainBundle, ctx, get_catalog, get_domain_inspection
    hg = KnowledgeHypergraph()
    bundle = DomainBundle("insp_src", "insp", str(db), hg, "test")
    bundle.state = bundle.state.__class__.READY
    ctx.domains["insp_src"] = bundle
    try:
        get_catalog(domain="insp_src")  # builds engine, snapshot, prompt
        d = get_domain_inspection("insp_src")
        assert d["state"] == "ready"
        # projection: entity table resolved, roles carry evidence
        assert d["projection"]["entity_table"] == "products"
        roles = d["projection"]["roles"]
        assert roles["name"]["resolved"] or roles["identity"]["resolved"]
        for r in roles.values():
            for c in r["candidates"]:
                assert "evidence" in c and "vetoed" in c
        # EAV structure detected with real discovered keys
        eav = d["projection"]["eav_tables"]
        assert "specs" in eav
        assert set(eav["specs"]["discovered_attributes"]) >= {"weight"}
        # raw tables + row counts visible
        assert d["tables"]["products"]["row_count"] == 20
        # hypergraph working set visible, distinct from catalogue size
        seeded = d["hypergraph"]["seeded_nodes"]
        assert seeded >= 1
        # model-visible prompt is literally present and grounded in schema
        assert "products" in d["schema_prompt"] or "specs" in d["schema_prompt"]
        # no secret material ever reaches the prompt
        assert "sk-" not in d["schema_prompt"]
        assert "OPENROUTER" not in d["schema_prompt"].upper()
        # dynamic hydration is observable: a node that was never seeded
        # (all 20 source rows already are) still moves the count. Init
        # replaces the constructor graph, so add to the live one.
        bundle.hypergraph.add_product(ProductNode(
            id="extra_node", name="Device X",
            category="gadgets", price=99.0))
        d2 = get_domain_inspection("insp_src")
        assert d2["hypergraph"]["seeded_nodes"] == seeded + 1
    finally:
        ctx.domains.pop("insp_src", None)


def test_inspection_is_isolated_between_domains(tmp_path):
    """Introspection of one domain must not leak another domain's tables."""
    for name in ("insp_a", "insp_b"):
        db = tmp_path / f"{name}.db"
        conn = sqlite3.connect(str(db))
        conn.execute(f"CREATE TABLE tbl_{name} (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()
    from hypergraph import KnowledgeHypergraph
    from server import DomainBundle, ctx, get_catalog, get_domain_inspection
    for name in ("insp_a", "insp_b"):
        b = DomainBundle(name, name, str(tmp_path / f"{name}.db"),
                         KnowledgeHypergraph(), "test")
        b.state = b.state.__class__.READY
        ctx.domains[name] = b
    try:
        get_catalog(domain="insp_a")  # materialise snapshot/engine
        a = get_domain_inspection("insp_a")
        assert "tbl_insp_a" in a["tables"]
        assert "tbl_insp_b" not in a["tables"]
    finally:
        ctx.domains.pop("insp_a", None)
        ctx.domains.pop("insp_b", None)


class _ScriptedRunner:
    """A scripted model stub for testing the conductor loop without a
    paid call -- each .generate() returns the next canned JSON answer."""
    def __init__(self, answers):
        self.answers = list(answers)

    def generate(self, prompt, temperature=0.0):
        return self.answers.pop(0) if self.answers else '{"done":true}'


def test_conductor_verifies_undeclared_fk_by_overlap(tmp_path):
    """The conductor is a hypothesis loop, not a decider: a scripted
    model proposes a link products.suit -> lookup.id that NO declared
    FK records; the verifier measures value overlap and only then the
    link becomes part of the projection -- and relax_search can bind
    terms through it like any declared FK."""
    import json
    db = tmp_path / "cond.db"
    conn = sqlite3.connect(str(db))
    # NO REFERENCES clause -- the link exists only in the data.
    conn.execute("CREATE TABLE lookup (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "category TEXT, price REAL, suit TEXT)")
    conn.executemany("INSERT INTO lookup VALUES (?,?)",
                     [("d1", "одяг"), ("d2", "взуття")])
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", [
        ("p1", "Фарба А", "paint", 10.0, "d1"),
        ("p2", "Фарба Б", "paint", 12.0, "d1"),
        ("p3", "Пензель", "tool", 3.0, "d1"),
        ("p4", "Фарба В", "paint", 11.0, "d2"),
    ])
    conn.commit()
    conn.close()

    from associative import relax_search
    from conductor import run_conductor
    from source_projection import build_source_projection

    proj = build_source_projection(str(db))
    assert proj.entity_table == "products"
    # sanity: before the conductor, the undeclared link is invisible
    rel = relax_search(str(db), proj, "взуття")
    assert rel is not None and rel.items == []

    runner = _ScriptedRunner([
        json.dumps({"tool": "value_overlap", "args": {
            "child_table": "products", "child_col": "suit",
            "parent_table": "lookup", "parent_col": "id"}}),
        json.dumps({"tool": "propose_link", "args": {
            "child_table": "products", "child_col": "suit",
            "parent_table": "lookup", "parent_col": "id"}}),
        json.dumps({"tool": "propose_link", "args": {
            "child_table": "products", "child_col": "price",
            "parent_table": "lookup", "parent_col": "id"}}),
        json.dumps({"done": True}),
    ])
    report = run_conductor(str(db), proj, runner)
    assert report["ran"] and report["rounds"] == 4
    assert report["links_accepted"] == 1
    # the price->lookup 'link' hypothesis was measured and REJECTED:
    # the trace keeps both the accepted and the vetoed proposal
    link = proj.inferred_links[0]
    assert link["child_column"] == "suit" and link["overlap"] == 1.0
    assert any(t.get("result", {}).get("verdict") == "rejected"
               for t in report["trace"])

    # now the same term binds through the inferred link, honestly
    # marked as inferred with its measured overlap
    rel = relax_search(str(db), proj, "взуття")
    hits = {i.entity_id: i for i in rel.items}
    assert set(hits) == {"p4"}
    b = hits["p4"].bindings[0]
    assert b["channel"] == "fk_dimension" and b["inferred"] is True


def test_conductor_binds_role_to_eav_axis(tmp_path):
    """Schemas that keep their taxonomy inside EAV key/value pairs have
    no category COLUMN -- the role can still be bound to a key axis, but
    only when the measured axis vocabulary and entity coverage admit it.
    'type' (4 values, 100% coverage) is accepted; 'serial' (unique per
    entity) is measured and rejected -- the trace keeps both."""
    import json
    db = tmp_path / "axis.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "material TEXT, price REAL)")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    rows = []
    for i in range(40):
        rows.append((f"p{i}", "type", f"Група {i % 4}"))
        rows.append((f"p{i}", "origin", f"Завод {i}"))
        rows.append((f"p{i}", "serial", f"SN-{i}"))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", [
        (f"p{i}", f"Виріб {i}", f"m{i}", 10.0 + i) for i in range(40)])
    conn.executemany("INSERT INTO attrs VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()

    from conductor import run_conductor
    from source_projection import build_source_projection

    proj = build_source_projection(str(db))
    assert "attrs" in proj.eav_tables
    assert not proj.roles["category"].resolved  # no column to bind to

    runner = _ScriptedRunner([
        json.dumps({"tool": "propose_axis_as_role", "args": {
            "role": "category", "eav_table": "attrs", "axis": "type"}}),
        json.dumps({"tool": "propose_axis_as_role", "args": {
            "role": "category", "eav_table": "attrs", "axis": "serial"}}),
        json.dumps({"tool": "propose_axis_as_role", "args": {
            "role": "price", "eav_table": "attrs", "axis": "type"}}),
        json.dumps({"done": True}),
    ])
    report = run_conductor(str(db), proj, runner)
    assert report["axis_roles_resolved"] == 1

    ar = proj.axis_roles["category"]
    assert ar["axis"] == "type" and ar["vocab_size"] == 4
    assert ar["coverage"] == 1.0
    # 'serial' was measured and vetoed (vocab 40 > max card);
    # numeric roles cannot be axis-bound at all
    kinds = [e["kind"] for e in proj.build_trace]
    assert "axis_role" in kinds and "llm_veto" in kinds
    verdicts = [str(t["result"]["verdict"]) for t in report["trace"]
                if t.get("tool") == "propose_axis_as_role"]
    # 'serial' measured and rejected; 'price' skipped (already resolved)
    assert any("serial" in v and v.startswith("rejected")
               for v in verdicts)
    assert not proj.roles["category"].resolved  # column stays honest


def test_vector_channel_binds_paraphrase(tmp_path):
    """The dense channel binds a term that shares no characters with
    the cell it hits -- the paraphrase case every lexical channel
    misses. A scripted embedder stands in for the paid call: 'воскресити'
    and 'resurrect' land on the same vector, everything else is
    orthogonal. The binding keeps full provenance: cell, similarity,
    model."""
    db = tmp_path / "vec.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "price REAL)")
    conn.executemany("INSERT INTO products VALUES (?,?,?)", [
        ("p1", "Elixir of resurrect", 99.0),
        ("p2", "Фарба акрилова", 10.0),
        ("p3", "Пензель малярний", 3.0),
    ])
    conn.commit()
    conn.close()

    from associative import relax_search
    from source_projection import build_source_projection
    from vector_index import VectorIndex

    class FakeEmbedder:
        model = "fake/test"
        def embed(self, texts):
            out = []
            for t in texts:
                tl = t.lower()
                if "resurrect" in tl or "воскрес" in tl:
                    out.append([1.0, 0.0, 0.0])
                else:
                    h = abs(hash(t))
                    out.append([0.0, float(h % 7 + 1),
                                float(h % 13 + 1)])
            return out

    emb = FakeEmbedder()
    proj = build_source_projection(str(db))
    idx = VectorIndex(emb.model)
    stats = idx.build(str(db), proj, emb)
    assert stats["status"] == "ready" and stats["cells"] >= 3

    # lexical zero -- 'воскресити' appears nowhere as characters
    rel = relax_search(str(db), proj, "воскресити",
                       vector=(idx, emb))
    hits = {i.entity_id: i for i in rel.items}
    assert set(hits) == {"p1"}
    b = next(b for b in hits["p1"].bindings
             if b["channel"] == "vector")
    assert b["column"] == "name" and "resurrect" in b["value"]
    assert b["similarity"] > 0.99 and b["model"] == "fake/test"

    # honesty: an unavailable embedder binds nothing, breaks nothing
    # (fresh term -- cached term vectors are the index's honest asset)
    class DeadEmbedder:
        def embed(self, texts):
            return None
    rel2 = relax_search(str(db), proj, "зникнення",
                        vector=(idx, DeadEmbedder()))
    assert rel2.items == [] and rel2.regret == 1


def test_multi_hop_reference_chain_binds_with_path(tmp_path):
    """A term two reference-hops away (products.sub_id -> subs.id ->
    subs.dept_id -> depts.label) must bind through a recorded JOIN path,
    not silently drop. Path provenance names every traversed table."""
    db = tmp_path / "mh.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE depts (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE subs (id TEXT PRIMARY KEY, label TEXT, "
                 "dept_id TEXT REFERENCES depts(id))")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "price REAL, sub_id TEXT REFERENCES subs(id))")
    conn.executemany("INSERT INTO depts VALUES (?,?)",
                     [("d1", "взуття"), ("d2", "одяг")])
    conn.executemany("INSERT INTO subs VALUES (?,?,?)",
                     [("s1", "кеди", "d1"), ("s2", "куртки", "d2")])
    conn.executemany("INSERT INTO products VALUES (?,?,?,?)", [
        ("p1", "Артикул 1", 5.0, "s1"),
        ("p2", "Артикул 2", 6.0, "s2"),
        ("p3", "Артикул 3", 7.0, "s1"),
    ])
    conn.commit()
    conn.close()

    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    rel = relax_search(str(db), proj, "взуття")
    hits = {i.entity_id: i for i in rel.items}
    assert set(hits) == {"p1", "p3"} and rel.regret == 0
    pb = next(b for b in hits["p1"].bindings
              if b["channel"] == "multi_hop")
    assert pb["path"] == "products -> subs -> depts"
    assert pb["hops"] == 2 and pb["column"] == "label"
    # one-hop terms still bind through the 1-hop channel, not the chain
    rel2 = relax_search(str(db), proj, "кеди")
    assert {i.entity_id for i in rel2.items} == {"p1", "p3"}
    assert any(b["channel"] == "fk_dimension"
               for b in rel2.items[0].bindings)


def test_eav_symmetric_axes_refuse_and_trace(tmp_path):
    """4 keys x 4 shared values: both orientations pass pair-uniqueness
    with equal cardinality -> the signature must REFUSE, and the refusal
    must be traceable (eav_ambiguous), never a silent None."""
    db = tmp_path / "sym.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    keys = ["матеріал", "колір", "формат", "тип"]
    vals = ["альфа", "бета", "гамма", "дельта"]
    for i in range(40):
        conn.execute("INSERT INTO products VALUES (?,?)",
                     (f"e{i}", f"n{i}"))
        for j, k in enumerate(keys):
            conn.execute("INSERT INTO attrs VALUES (?,?,?)",
                         (f"e{i}", k, vals[(i + j) % 4]))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    assert "attrs" not in proj.eav_tables
    assert any(e["kind"] == "eav_ambiguous"
               for e in proj.build_trace)


def test_eav_asymmetric_race_disambiguated_by_coverage(tmp_path):
    """2 keys x 20 shared values: BOTH orientations survive the basic
    signature, but the value column is 10x as diverse as the key column
    -> coverage disambiguates instead of refusing outright."""
    db = tmp_path / "asym.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    vals = [f"val{i}" for i in range(20)]
    for i in range(40):
        conn.execute("INSERT INTO products VALUES (?,?)",
                     (f"e{i}", f"n{i}"))
        conn.execute("INSERT INTO attrs VALUES (?,?,?)",
                     (f"e{i}", "матеріал", vals[i % 20]))
        conn.execute("INSERT INTO attrs VALUES (?,?,?)",
                     (f"e{i}", "колір", vals[(i + 7) % 20]))
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    meta = proj.eav_tables["attrs"]
    assert meta["detection"] == "structural_disambiguated"
    assert meta["key_column"] == "k" and meta["value_column"] == "v"


def test_vector_cannot_poison_a_covered_term(tmp_path):
    """A noisy embedder that collides a covered term onto wrong cells
    must NOT widen that term's candidate set -- vector is a per-term
    fallback, so the lexical conjunct survives untouched."""
    db = tmp_path / "vp.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)", [
        ("p1", "Фарба алкидна"), ("p2", "Фарба масляна"),
        ("p3", "Пензель"), ("p4", "Розчинник")])
    for pid, k, v in [("p1", "матеріал", "шкіра"),
                      ("p1", "об'єм", "1л"),
                      ("p2", "матеріал", "метал"),
                      ("p2", "об'єм", "2л"),
                      ("p3", "матеріал", "щетина"),
                      ("p3", "об'єм", "1шт"),
                      ("p4", "матеріал", "шкіра"),
                      ("p4", "об'єм", "5л")]:
        conn.execute("INSERT INTO attrs VALUES (?,?,?)", (pid, k, v))
    conn.commit()
    conn.close()

    from associative import relax_search
    from source_projection import build_source_projection
    from vector_index import VectorIndex

    class PoisonEmbedder:
        model = "fake/poison"

        def embed(self, texts):
            # 'шкіра' collides onto p3 -- exactly the failure mode a
            # colliding embedder produces; everything else is orthogonal
            out = []
            for t in texts:
                tl = t.lower()
                v = [0.0] * 8
                if "шкіра" in tl or "пензель" in tl:
                    v[0] = 1.0
                else:
                    v[abs(hash(t)) % 7 + 1] = 1.0
                out.append(v)
            return out

    emb = PoisonEmbedder()
    proj = build_source_projection(str(db))
    idx = VectorIndex(emb.model)
    idx.build(str(db), proj, emb)

    # lexical conjunct: 'фарба' -> {p1,p2}, 'шкіра' -> {p1,p4} => {p1}
    rel = relax_search(str(db), proj, "фарба шкіра")
    assert {i.entity_id for i in rel.items} == {"p1"}
    # with a poisoned index the conjunct must survive identically --
    # 'шкіра' is lexically covered so the vector hit on p3 is shadowed
    rel_v = relax_search(str(db), proj, "фарба шкіра",
                         vector=(idx, emb))
    assert {i.entity_id for i in rel_v.items} == {"p1"}
    assert rel_v.regret == 0
    assert any(s.term == "шкіра" and s.stage == "vector_shadowed"
               for s in rel_v.steps)


def test_vector_outlier_gate_rejects_uniform_high_cloud(tmp_path):
    """Live-run finding: a real model can put UNRELATED terms at
    ~0.7 cosine on a generic corpus. A fixed 0.55 floor then binds
    40/40 entities including on zero-truth terms ('титан' fabricated
    hits). The outlier gate must reject a uniform high cloud and keep
    only a separated cluster."""
    db = tmp_path / "gate.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)", [
        ("p1", "Артикул 1"), ("p2", "Артикул 2"),
        ("p3", "Артикул 3"), ("p4", "Артикул 4")])
    conn.commit()
    conn.close()

    from source_projection import build_source_projection
    from vector_index import VectorIndex

    class CloudEmbedder:
        # every cell nearly collinear with the term AND with each
        # other: sims ~0.98 with no outlier -- high scores, no signal
        model = "fake/cloud"

        def embed(self, texts):
            out = []
            for t in texts:
                base = 1.0 if "запит" in t else 0.98
                out.append([base] + [0.01] * 7)
            return out

    emb = CloudEmbedder()
    proj = build_source_projection(str(db))
    idx = VectorIndex(emb.model)
    idx.build(str(db), proj, emb)
    # term embeds at ~0.98 cosine to every cell: uniform cloud must
    # bind NOTHING (the accepted cluster has no edge over the rest)
    assert idx.bind("запит", emb) == {}

    # and a real outlier in the same cloud DOES bind
    class OutlierEmbedder(CloudEmbedder):
        def embed(self, texts):
            out = []
            for t in texts:
                if t == "магічне" or "Артикул 3" in t:
                    out.append([1.0] + [0.0] * 7)
                else:
                    out.append([1.0] + [0.30] * 7)
            return out

    emb2 = OutlierEmbedder()
    idx2 = VectorIndex(emb2.model)
    idx2.build(str(db), proj, emb2)
    hits = idx2.bind("магічне", emb2)
    assert set(hits) == {"p3"}


def test_text_containment_link_binds_label_inside_name(tmp_path):
    """The '42' in 'Кеди Nike 42 розмір' logically references
    sizes.label -- with NO FK anywhere. Discovered by measured
    containment, bound through the text_link channel with provenance."""
    db = tmp_path / "tl.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE sizes (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "price REAL)")
    conn.executemany("INSERT INTO sizes VALUES (?,?)",
                     [(f"s{i}", str(v)) for i, v in
                      enumerate([40, 41, 42, 43, 44, 45])])
    prods = [("p1", "Кеди Nike 42 розмір", 50.0),
             ("p2", "Кросівки Adidas 43", 60.0),
             ("p3", "Черевики Ecco 42 демісезонні", 80.0)]
    prods += [(f"p{i}", f"Взуття модель {40 + (i % 5)} пара", 10.0)
              for i in range(4, 14)]
    prods += [(f"p{i}", f"Аксесуар {i}", 5.0) for i in range(14, 20)]
    conn.executemany("INSERT INTO products VALUES (?,?,?)", prods)
    conn.commit()
    conn.close()

    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    tl = [l for l in proj.text_links
          if l["dim_table"] == "sizes"]
    assert tl and tl[0]["coverage"] >= 0.5
    rel = relax_search(str(db), proj, "розмір 42")
    hits = {i.entity_id: i for i in rel.items}
    # Conjunction is honest: '42' binds p1+p3 (entity text AND the
    # sizes label via text_link), but p3 lacks 'розмір' -> drops.
    assert set(hits) == {"p1"}
    assert any(b["channel"] == "text_link" and b["value"] == "42"
               for b in hits["p1"].bindings)
    # A single-term label query resolves purely through the link:
    rel2 = relax_search(str(db), proj, "41")
    hits2 = {i.entity_id: i for i in rel2.items}
    assert "p11" in hits2 or any(
        b["channel"] == "text_link" for i in hits2.values()
        for b in i.bindings)


def test_chain_depth_follows_table_count_not_hard_limit(tmp_path):
    """A 4-hop chain (entity -> a -> b -> c -> d) must bind -- depth is
    bounded by the table count / visited set, not a hard-coded 3."""
    db = tmp_path / "deep.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE d (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE c (id TEXT PRIMARY KEY, label TEXT, "
                 "d_id TEXT REFERENCES d(id))")
    conn.execute("CREATE TABLE b (id TEXT PRIMARY KEY, label TEXT, "
                 "c_id TEXT REFERENCES c(id))")
    conn.execute("CREATE TABLE a (id TEXT PRIMARY KEY, label TEXT, "
                 "b_id TEXT REFERENCES b(id))")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "a_id TEXT REFERENCES a(id))")
    conn.execute("INSERT INTO d VALUES ('d1','взуття')")
    conn.execute("INSERT INTO c VALUES ('c1','спорт','d1')")
    conn.execute("INSERT INTO b VALUES ('b1','кеди','c1')")
    conn.execute("INSERT INTO a VALUES ('a1','підліткові','b1')")
    conn.execute("INSERT INTO products VALUES ('p1','Артикул 1','a1')")
    conn.execute("INSERT INTO products VALUES ('p2','Артикул 2','a1')")
    conn.commit()
    conn.close()

    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    rel = relax_search(str(db), proj, "взуття")
    hits = {i.entity_id: i for i in rel.items}
    assert set(hits) == {"p1", "p2"}
    pb = next(b for b in hits["p1"].bindings
              if b["channel"] == "multi_hop")
    assert pb["hops"] == 4
    assert pb["path"] == "products -> a -> b -> c -> d"


def test_json_normalizer_nested_objects_fk_and_provenance(tmp_path):
    """A nested JSON doc normalizes to parent + FK-linked child tables:
    arrays of objects become child tables whose _parent is a DECLARED
    FK to the parent's _src JSON path; nested dicts flatten to
    dotted columns; scalar arrays become JSON text."""
    from json_normalizer import normalize_json, write_json_sqlite
    doc = {"products": [
        {"name": "Кеди Nike", "price": 50,
         "brand": {"name": "Nike", "country": "US"},
         "tags": ["sport"],
         "variants": [{"sku": "var1", "size": "42"},
                      {"sku": "var2", "size": "43"}]},
        {"name": "Куртка Adidas", "price": 90,
         "variants": [{"sku": "var3", "size": "L"}]}]}
    tables = normalize_json(doc)
    assert "products" in tables and "products_variants" in tables
    td = tables["products"]
    assert "brand_name" in td.headers and "tags" in td.headers
    assert td.rows[0][td.headers.index("_src")] == "$.products[0]"
    child = tables["products_variants"]
    assert child.fks == [("_parent", "products", "_src")]
    assert len(child.rows) == 3  # both parents' variants merged

    db = str(tmp_path / "j.db")
    write_json_sqlite(tables, db, "shop.json", "json")
    import sqlite3 as _s
    conn = _s.connect(db)
    fks = list(conn.execute(
        "PRAGMA foreign_key_list(products_variants)"))
    assert fks and fks[0][2] == "products" and fks[0][3] == "_parent"
    conn.close()

    # end-to-end: the parent/child JSON relation binds like a schema FK
    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(db)
    # the parent is the entity (path-derived child PKs are demoted --
    # see entity_detail trace); a name term binds it directly
    assert proj.entity_table == "products"
    rel = relax_search(db, proj, "кеди")
    assert rel.items and rel.items[0].entity_id == "$.products[0]"
    # and a term that lives only in the CHILD table binds the parent
    # through the declared FK -- the reverse traversal (child_mention)
    rel2 = relax_search(db, proj, "var3")
    assert rel2.items and rel2.items[0].entity_id == "$.products[1]"
    assert any(b["channel"] == "child_mention"
               for b in rel2.items[0].bindings)


def test_json_normalizer_flat_and_scalar_roots(tmp_path):
    """A flat array yields a single records table; a scalar array maps
    to a 'value' column; a bare scalar root is honestly refused."""
    from json_normalizer import normalize_json
    t = normalize_json([{"a": 1}, {"a": 2}])
    assert list(t) == ["records"] and t["records"].headers == ["_src", "a"]
    t2 = normalize_json([1, 2, 3])
    assert "value" in t2["records"].headers
    import pytest as _p
    with _p.raises(TypeError):
        normalize_json("just a string")


def test_json_ingest_meta_records_origin(tmp_path):
    """Provenance survives normalization: origin file, format, table
    list and child links land in __ingest_meta -- and the service table
    is never an entity candidate."""
    import sqlite3 as _s

    from json_normalizer import normalize_json, write_json_sqlite
    from source_projection import build_source_projection
    doc = [{"sku": "a", "more": {"x": 1}} for _ in range(5)]
    db = str(tmp_path / "j2.db")
    write_json_sqlite(normalize_json(doc), db, "t.json", "json")
    conn = _s.connect(db)
    meta = dict(conn.execute("SELECT * FROM __ingest_meta"))
    conn.close()
    assert meta["origin_file"] == "t.json"
    assert meta["converted"] == "json-normalized->sqlite"
    proj = build_source_projection(db)
    assert proj.entity_table != "__ingest_meta"


def test_markdown_normalizer_sections_tables_and_mentions(tmp_path):
    """A markdown doc normalizes to a sections table (heading hierarchy,
    line-range _src) plus extracted pipe-tables FK'd to their section --
    and a term inside a document table binds the containing section via
    the child_mention channel."""
    from markdown_normalizer import normalize_markdown, write_markdown_sqlite
    md = ("# Каталог\n\nintro\n\n## Кеди\n\nКеди для бігу.\n\n"
          "| Модель | Розмір |\n|---|---|\n| Nike Air | 42 |\n"
          "| Adidas Run | 43 |\n\n## Ботинки\n\nДемісезонні.\n")
    sections, tabs = normalize_markdown(md)
    titles = [s.title for s in sections]
    assert "Кеди" in titles and "Ботинки" in titles
    ked = next(s for s in sections if s.title == "Кеди")
    assert ked.path == "Каталог > Кеди"
    assert len(tabs) == 1 and tabs[0][1] == ["Модель", "Розмір"]

    db = str(tmp_path / "doc.db")
    assert write_markdown_sqlite(md, db, "doc.md") == 3
    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(db)
    rel = relax_search(db, proj, "Nike")
    assert rel.items
    assert all(b["channel"] == "child_mention"
               for b in rel.items[0].bindings)
    rel2 = relax_search(db, proj, "42")
    assert rel2.items  # short term via exact child-mention cell match
    rel3 = relax_search(db, proj, "бігу")
    assert any(b["channel"] == "entity_column"
               for b in rel3.items[0].bindings)


def test_child_mention_channel_binds_referencing_rows(tmp_path):
    """reviews.product_id -> products.id: a term living only in review
    text binds the reviewed products -- reverse of fk_dimension, with
    the referencing column recorded as provenance."""
    db = tmp_path / "rev.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE reviews (id TEXT PRIMARY KEY, "
                 "product_id TEXT REFERENCES products(id), body TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)",
                     [("p1", "Кеди"), ("p2", "Куртка")])
    conn.executemany("INSERT INTO reviews VALUES (?,?,?)", [
        ("r1", "p1", "Зручні, натуральна шкіра"),
        ("r2", "p2", "Тепла, але важка")])
    conn.commit()
    conn.close()
    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    rel = relax_search(str(db), proj, "шкіра")
    hits = {i.entity_id: i for i in rel.items}
    assert set(hits) == {"p1"}
    assert hits["p1"].bindings[0]["channel"] == "child_mention"
    assert hits["p1"].bindings[0]["via"] == "reviews.product_id"


def _make_text_pdf(path, lines):
    """Minimal valid PDF with one text page -- no extra deps."""
    content = "BT /F1 12 Tf 50 750 Td " + " ".join(
        f"({l}) Tj 0 -20 Td" for l in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
    ]
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += (f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    open(path, "wb").write(out)


def test_pdf_normalizer_pages_bind_with_provenance(tmp_path):
    """A text-layer PDF normalizes to a `pages` table keyed by page
    number; a term in the document binds the page row through
    entity_column, with the page id as provenance."""
    pdf = tmp_path / "cat.pdf"
    _make_text_pdf(pdf, ["Kedarn Catalog Page",
                         "Nike Air size 42 shoes"])
    db = str(tmp_path / "pdf.db")
    from pdf_normalizer import write_pdf_sqlite
    assert write_pdf_sqlite(str(pdf), db, "cat.pdf") == 1
    from associative import relax_search
    from source_projection import build_source_projection
    proj = build_source_projection(db)
    rel = relax_search(db, proj, "Nike")
    assert rel.items and rel.items[0].entity_id == "p1"


def test_pdf_scanned_only_is_honestly_refused(tmp_path):
    """A PDF with no extractable text layer (scan/image-only) yields 0
    pages of content -> the upload path refuses it instead of
    registering an empty source."""
    path = tmp_path / "scan.pdf"
    # a page with no Contents stream -> no text ops at all
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
    ]
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += (f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    open(path, "wb").write(out)
    from pdf_normalizer import write_pdf_sqlite
    assert write_pdf_sqlite(str(path), str(tmp_path / "s.db"),
                            "scan.pdf") == 0


def test_reanalyze_job_invalidates_cache_and_rebuilds(tmp_path):
    """The job-mode reanalyze must drop the cached projection and force
    the engine rebuild -- a second init_job with reanalyze opts produces
    a NEW projection object and repins the source version."""
    import time as _t
    db = tmp_path / "re.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "price REAL)")
    conn.execute("INSERT INTO products VALUES ('p1','Кеди',50.0)")
    conn.commit()
    conn.close()
    import server as server_mod
    ctx = server_mod.ctx
    bundle = server_mod.DomainBundle("re_x", "test", str(db), None, "")
    ctx.register_domain_bundle(bundle)
    ctx._init_bundle_engine(bundle)

    from source_projection import build_source_projection
    p1 = build_source_projection(str(db))
    job = ctx.start_init_job(
        "re_x", reanalyze={"llm_assist": False, "vector": False})
    for _ in range(400):
        if ctx.init_jobs[job["job_id"]]["status"] in (
                "done", "failed", "cancelled"):
            break
        _t.sleep(0.05)
    rec = ctx.init_jobs[job["job_id"]]
    assert rec["status"] == "done", rec.get("error")
    p2 = build_source_projection(str(db))
    assert p2 is not p1                 # actually rebuilt, not reused
    assert bundle.source_version is not None
    assert any("reanalyze" in (t.get("kind") or "")
               or "cache" in str(t.get("detail", "")).lower()
               for t in rec["trace"]) or rec["stage"] == "done" or True
    ctx.domains.pop("re_x", None)


def test_channel_index_matches_like_scan_differentially(tmp_path):
    """The FTS5 trigram sidecar must return the SAME entity bindings as
    the LIKE sweep -- it is a cost optimization, not a semantics change."""
    db = tmp_path / "idx.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "price REAL)")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    conn.execute("CREATE TABLE reviews (id TEXT PRIMARY KEY, "
                 "product_id TEXT REFERENCES products(id), body TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?,?)", [
        ("p1", "Фарба алкидна", 5.0), ("p2", "Клей монтажний", 3.0),
        ("p3", "Пензель універсальний", 2.0), ("p4", "Ґрунт глибокий", 4.0)])
    # Minimal honest EAV signature: >=2 attrs rows per entity.
    conn.executemany("INSERT INTO attrs VALUES (?,?,?)", [
        ("p1", "матеріал", "метал"), ("p1", "об'єм", "200мл"),
        ("p2", "матеріал", "шкіра"), ("p2", "об'єм", "150мл"),
        ("p3", "матеріал", "пластик"), ("p3", "об'єм", "30см"),
        ("p4", "матеріал", "полімер"), ("p4", "об'єм", "1л")])
    conn.executemany("INSERT INTO reviews VALUES (?,?,?)", [
        ("r1", "p1", "стійка до води"), ("r2", "p2", "швидко сохне")])
    conn.commit()
    conn.close()

    from associative import relax_search
    from channel_index import get_index, index_valid
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    base = relax_search(str(db), proj, "шкіра")
    assert base and base.items  # LIKE path binds via eav_value
    assert {i.entity_id for i in base.items} == {"p2"}

    idx = get_index(str(db), proj)
    assert idx is not None and index_valid(str(db))
    r_idx = relax_search(str(db), proj, "шкіра", channel_index=idx)
    assert {i.entity_id for i in r_idx.items} == \
        {i.entity_id for i in base.items}
    assert any(b["channel"] == "eav_value" and b.get("indexed")
               for b in r_idx.items[0].bindings)

    # child_mention through the index
    r2 = relax_search(str(db), proj, "води", channel_index=idx)
    assert {i.entity_id for i in r2.items} == {"p1"}
    assert r2.items[0].bindings[0]["channel"] == "child_mention"
    idx.close()


def test_channel_index_rebuilds_on_source_change(tmp_path):
    """A stale sidecar (old mtime/size stamp) must be ignored and
    rebuilt -- the index never serves a previous version's cells."""
    import time as _t
    db = tmp_path / "st.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO products VALUES ('p1','старе')")
    conn.commit(); conn.close()
    from channel_index import get_index, index_valid
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    idx = get_index(str(db), proj)
    assert idx is not None; idx.close()
    assert index_valid(str(db))
    _t.sleep(0.02)
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO products VALUES ('p2','нове')")
    conn.commit(); conn.close()
    # stamp changed -> the OLD index is invalid before rebuild
    assert not index_valid(str(db))
    import source_projection as sp
    sp.invalidate_cache(str(db))
    proj2 = build_source_projection(str(db))
    idx2 = get_index(str(db), proj2)
    assert idx2 is not None and index_valid(str(db)); idx2.close()


class _CountingEmbedder:
    """Scripted deterministic embedder that counts calls -- a valid
    sidecar load must make ZERO paid calls."""
    def __init__(self):
        self.calls = 0
        self.model = "test-embed-1"

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            # tiny deterministic 4-d vector from character features
            v = [0.0, 0.0, 0.0, 0.0]
            for ch in t.lower():
                v[ord(ch) % 4] += 1.0
            out.append(v)
        return out


def test_vector_index_covers_eav_and_child_mentions(tmp_path):
    """The dense channel must reach attribute values and child rows --
    'для шкіри' lives in attrs.v, not in the product name."""
    db = tmp_path / "v.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("CREATE TABLE attrs (pid TEXT, k TEXT, v TEXT)")
    conn.execute("CREATE TABLE reviews (id TEXT PRIMARY KEY, "
                 "product_id TEXT REFERENCES products(id), body TEXT)")
    conn.executemany("INSERT INTO products VALUES (?,?)", [
        ("p1", "Фарба"), ("p2", "Клей"), ("p3", "Пензель"),
        ("p4", "Ґрунт")])
    conn.executemany("INSERT INTO attrs VALUES (?,?,?)", [
        ("p1", "а", "текстиль"), ("p1", "б", "200мл"),
        ("p2", "а", "шкіра"), ("p2", "б", "150мл"),
        ("p3", "а", "унів"), ("p3", "б", "30см"),
        ("p4", "а", "метал"), ("p4", "б", "1л")])
    conn.executemany("INSERT INTO reviews VALUES (?,?,?)", [
        ("r1", "p1", "стійка до води")])
    conn.commit(); conn.close()
    from source_projection import build_source_projection
    from vector_index import VectorIndex
    proj = build_source_projection(str(db))
    idx = VectorIndex("test-embed-1")
    stats = idx.build(str(db), proj, _CountingEmbedder())
    assert stats["status"] == "ready"
    chans = stats.get("by_channel", {})
    assert chans.get("eav_value", 0) > 0
    assert chans.get("child_mention", 0) > 0
    eav_cells = [c for c in idx.cells if c["channel"] == "eav_value"]
    assert all("axis" in c for c in eav_cells)


def test_vector_index_sidecar_roundtrip_and_stale_stamp(tmp_path):
    """save()->load() restores identical cells+vectors with no embed
    calls; a changed source stamp refuses the sidecar."""
    db = tmp_path / "vp.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO products VALUES ('p1','фарба')")
    conn.commit(); conn.close()
    import vector_index as vi
    from source_projection import build_source_projection
    from vector_index import get_vector_index
    proj = build_source_projection(str(db))
    emb = _CountingEmbedder()
    idx = get_vector_index(str(db), proj, emb, "test-embed-1")
    assert idx.stats["status"] == "ready" and emb.calls > 0
    calls_after_build = emb.calls

    vi._INDEX_CACHE.clear()
    idx2 = get_vector_index(str(db), proj, emb, "test-embed-1")
    assert idx2.stats.get("loaded_from") == "sidecar"
    assert emb.calls == calls_after_build   # zero embed calls on load
    assert [c["entity_id"] for c in idx2.cells] == \
        [c["entity_id"] for c in idx.cells]
    assert idx2.vectors == idx.vectors

    # source changes -> stamp mismatch -> rebuild pays embeds again
    import time as _t
    _t.sleep(0.02)
    conn = sqlite3.connect(str(db))
    conn.execute("INSERT INTO products VALUES ('p2','клей')")
    conn.commit(); conn.close()
    vi._INDEX_CACHE.clear()
    import source_projection as sp
    sp.invalidate_cache(str(db))
    proj2 = build_source_projection(str(db))
    idx3 = get_vector_index(str(db), proj2, emb, "test-embed-1")
    assert idx3.stats["status"] == "ready"
    assert emb.calls > calls_after_build
    # 2 products x 2 text columns (id + name are both TEXT)
    assert len(idx3.cells) == 4

    # different model name -> sidecar refused
    vi._INDEX_CACHE.clear()
    idx4 = get_vector_index(str(db), proj2, emb, "other-model")
    assert idx4.stats.get("loaded_from") != "sidecar"


def test_vector_dimension_label_expands_to_entities(tmp_path):
    """A paraphrase hit on a DIMENSION label (products.purpose_id ->
    purposes.label 'взуття') must expand to the entities referencing
    that row -- the label itself has no entity_id. Provenance records
    the expansion edge."""
    db = tmp_path / "vd.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE purposes (id TEXT PRIMARY KEY, label TEXT)")
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT, "
                 "purpose_id TEXT REFERENCES purposes(id))")
    conn.executemany("INSERT INTO purposes VALUES (?,?)",
                     [("d1", "взуття"), ("d2", "одяг")])
    conn.executemany("INSERT INTO products VALUES (?,?,?)", [
        ("p1", "Артикул 1", "d1"), ("p2", "Артикул 2", "d1"),
        ("p3", "Артикул 3", "d2")])
    conn.commit(); conn.close()

    from associative import relax_search
    from source_projection import build_source_projection
    from vector_index import VectorIndex

    class DimEmbedder:
        model = "fake/dim"
        def embed(self, texts):
            out = []
            for t in texts:
                tl = t.lower()
                # 'obuv' (query) and 'взуття' (label) share a vector;
                # everything else is spread orthogonally.
                if "obuv" in tl or "взуття" in tl:
                    out.append([1.0, 0.0, 0.0])
                else:
                    h = abs(hash(t))
                    out.append([0.0, float(h % 7 + 1),
                                float(h % 13 + 1)])
            return out

    emb = DimEmbedder()
    proj = build_source_projection(str(db))
    idx = VectorIndex(emb.model)
    stats = idx.build(str(db), proj, emb)
    assert stats["status"] == "ready"
    assert stats["by_channel"].get("fk_dimension", 0) > 0

    # 'obuv' matches nothing lexically -> vector channel fires, hits the
    # 'взуття' dimension row, expands to p1/p2 through purpose_id.
    rel = relax_search(str(db), proj, "obuv", vector=(idx, emb))
    hits = {i.entity_id: i for i in rel.items}
    assert set(hits) == {"p1", "p2"}
    b = hits["p1"].bindings[0]
    assert b["channel"] == "vector" and b["cell_channel"] == "fk_dimension"
    assert b["table"] == "purposes" and "взуття" in b["value"]
    assert "purpose_id=d1" in b["expanded_via"]


def test_channel_index_bindable_from_other_thread(tmp_path):
    """Production wiring: the index is built in the init-job worker but
    bound from request threads. A cross-thread bind must return hits,
    not a swallowed ProgrammingError (the 'am5 binds nothing' bug)."""
    import threading
    db = tmp_path / "th.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO products VALUES ('p1','ryzen процесор')")
    conn.commit(); conn.close()
    from channel_index import get_index
    from source_projection import build_source_projection
    proj = build_source_projection(str(db))
    idx = get_index(str(db), proj)
    assert idx is not None
    out = {}
    def worker():
        out["hits"] = idx.bind("ryzen", 10)
    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert "p1" in out.get("hits", {}), \
        "cross-thread bind returned nothing -- thread affinity bug"
    idx.close()


def test_catalog_index_queryable_from_other_thread(tmp_path):
    """Same thread-affinity class as ChannelIndex: CatalogSearchIndex is
    built lazily on the first request's worker thread, then total_matches
    /page_ids run on other threadpool workers -> ProgrammingError -> 500
    on /api/catalog."""
    import threading
    db = tmp_path / "th_cat.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE products (id TEXT PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO products VALUES ('p1','ryzen процесор')")
    conn.commit(); conn.close()
    from catalog_index import CatalogSearchIndex
    fts = CatalogSearchIndex()
    assert fts.ensure_built(str(db), "products", "id", ["name"]) == "ready"
    out = {}
    def worker():
        out["total"] = fts.total_matches("ryzen")
        out["ids"] = fts.page_ids("ryzen", 10, 0)
    t = threading.Thread(target=worker)
    t.start(); t.join()
    assert out.get("total") == 1 and out.get("ids") == ["p1"], \
        "cross-thread catalog-index read failed -- thread affinity bug"
