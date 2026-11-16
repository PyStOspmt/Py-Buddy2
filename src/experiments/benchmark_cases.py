"""
Py-Buddy: deterministic benchmark cases with database-derived ground truth.

Every oracle in this module is computed by SQL and by the Value Algebra
(attribute_algebra). No LLM takes part in producing an expected answer, which
is the whole point: an expected answer written by a model is an assertion,
not a ground truth.

Case families:

  direct_price        ask the price of a specific, existing item
  cheapest_category   cheapest / most expensive item in a category
  availability        what is in stock in a category
  compat_accept       a bundle that IS compatible      -> verdict "accept"
  compat_reject       a bundle that is NOT compatible  -> verdict "reject"
  overconstrained     a budget below the minimum price -> verdict "reject"
  price_trap          ask about an item that does NOT exist -> must not invent
  conversational      greeting / meta question -> no product claims expected
"""
from __future__ import annotations

import os
import random
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from attribute_algebra import (
    CONFLICT, Concept, canonical_form, discover_concepts, fold_all,
)
from metrics import Oracle


# --------------------------------------------------------------------------
# Catalog snapshot
# --------------------------------------------------------------------------


@dataclass
class Snapshot:
    """Read-only view of the marketplace catalog, canonicalised once."""

    db_path: str
    products: Dict[str, Dict[str, object]] = field(default_factory=dict)
    #: name -> every price at which any seller lists that name.
    #: NOT name -> one price: the Rozetka catalogue has 10 000 products but
    #: only 190 distinct names, so the same name is listed by up to six
    #: sellers at six different prices. Collapsing them to a single value
    #: silently picks one seller's price and makes every other listing look
    #: like a hallucination.
    price_index: Dict[str, List[float]] = field(default_factory=dict)
    id_to_name: Dict[str, str] = field(default_factory=dict)
    concepts: List[Concept] = field(default_factory=list)
    key_to_concept: Dict[str, str] = field(default_factory=dict)
    #: product id -> {concept name -> [surface values]}
    attrs: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    _canon_cache: Dict[str, Dict[str, str]] = field(default_factory=dict, repr=False)

    def names(self) -> List[str]:
        return list(self.price_index)

    def concept_for(self, key: str) -> Optional[str]:
        return self.key_to_concept.get(key)

    def canonical_attrs(self, product_id: str) -> Dict[str, str]:
        """
        Fold every surface value of a product into one canonical value per
        concept. Memoised: the benchmark evaluates the full cpu x motherboard
        cross product, and re-folding 10 000 products' EAV rows each time is
        what turns a 1.5M-pair scan into minutes instead of milliseconds.
        """
        cached = self._canon_cache.get(product_id)
        if cached is not None:
            return cached
        # F50: the strip context is the schema's own key vocabulary --
        # source-derived, not a baked-in lexicon.
        all_keys_ctx = " ".join(self.key_to_concept)
        out: Dict[str, str] = {}
        for key, values in self.attrs.get(product_id, {}).items():
            cname = self.key_to_concept.get(key)
            if cname is None:
                continue
            folded = fold_all(values, key=all_keys_ctx or key)
            if folded is CONFLICT or folded is None:
                continue
            out[cname] = str(folded)
        self._canon_cache[product_id] = out
        return out


def load_snapshot(db_path: str) -> Snapshot:
    """Load and canonicalise the catalog. Concept discovery runs once."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"catalog not found: {db_path}. Run "
            f"`python data/generate_marketplace_scale.py` first."
        )
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    conn.row_factory = sqlite3.Row

    snap = Snapshot(db_path=db_path)
    for r in conn.execute(
        "SELECT id, name, category, brand, price, merchant_id, stock_quantity, "
        "performance_score FROM products"
    ):
        snap.products[r["id"]] = dict(r)
        # F10: an absent price is NULL, not 0.0 -- it must not enter the
        # price index, where it would masquerade as "free".
        if r["price"] is not None:
            snap.price_index.setdefault(r["name"], []).append(float(r["price"]))
        snap.id_to_name[r["id"]] = r["name"]

    key_values: Dict[str, Set[str]] = {}
    key_scopes: Dict[str, Set[str]] = {}
    for r in conn.execute(
        "SELECT m.product_id, m.meta_key, m.meta_value, p.category "
        "FROM product_eav_meta m JOIN products p ON p.id = m.product_id"
    ):
        key, val = r["meta_key"], r["meta_value"]
        if key == "vendor":       # a seller id, not a product attribute
            continue
        key_values.setdefault(key, set()).add(val)
        key_scopes.setdefault(key, set()).add(r["category"])
        snap.attrs.setdefault(r["product_id"], {}).setdefault(key, []).append(val)
    conn.close()

    # Two passes. The first merges redundant keys into shared concepts; the
    # second gives every still-unassigned key its own concept. A key with no
    # redundant twin is not "no attribute" — it is an attribute that happens
    # to be spelled one way. Without the second pass, a catalogue with no
    # duplicated key names yields zero concepts and every accessor silently
    # returns None.
    sorted_kv = {k: sorted(v) for k, v in key_values.items()}
    grouped = discover_concepts(sorted_kv, key_scopes, min_keys=2)
    assigned = {k for c in grouped for k in c.keys}
    leftover = {k: v for k, v in sorted_kv.items() if k not in assigned}
    singletons = discover_concepts(leftover, None, min_keys=1, allow_merge=False)

    concepts = grouped + singletons
    snap.concepts = concepts
    for c in concepts:
        for k in c.keys:
            snap.key_to_concept[k] = c.name
    return snap


def concept_index(snap: Snapshot) -> Dict[str, Concept]:
    return {c.name: c for c in snap.concepts}


# --------------------------------------------------------------------------
# Compatibility oracle — deterministic, Value-Algebra based
# --------------------------------------------------------------------------


def _concept(snap: Snapshot, name: str) -> Optional[Concept]:
    return next((c for c in snap.concepts if c.name == name), None)


def _local_value(
    snap: Snapshot,
    product_id: str,
    *,
    shape: Optional[str] = None,
    dimension: Optional[str] = None,
    marker_values: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """
    The canonical value this product holds *for the matching concept*.

    Resolution is product-local on purpose. Picking "the first concept of
    dimension P" is wrong the moment a catalogue has two unmerged P-typed
    keys (which is exactly what happens when key names are not redundant):
    the product's own attribute would be looked up under a stranger's
    concept and silently come back None.
    """
    markers = {str(canonical_form(m)) for m in (marker_values or ())}
    for cname, val in snap.canonical_attrs(product_id).items():
        c = _concept(snap, cname)
        if c is None:
            continue
        if shape is not None and c.shape.value != shape:
            continue
        if dimension is not None and c.dimension != dimension:
            continue
        if markers:
            key_ctx = " ".join(snap.key_to_concept) or " ".join(c.keys)
            vocab = {str(canonical_form(v, key=key_ctx)) for v in c.vocabulary}
            if not (vocab & markers):
                continue
        return val
    return None


_SOCKET_MARKERS = ("am4", "am5", "lga1700", "lga1851", "lga1200", "sAM5")


def sockets_of(snap: Snapshot, product_id: str) -> Optional[str]:
    """Canonical socket, or None when the product declares none."""
    return _local_value(snap, product_id, shape="token", marker_values=_SOCKET_MARKERS)


def pcd_of(snap: Snapshot, product_id: str) -> Optional[str]:
    return _local_value(snap, product_id, shape="duplex")


def diameter_mm(snap: Snapshot, product_id: str) -> Optional[float]:
    val = _local_value(snap, product_id, shape="measure", dimension="L")
    if val is None:
        return None
    try:
        return float(str(val).removesuffix("mm"))
    except ValueError:
        return None


def power_watts(snap: Snapshot, product_id: str) -> Optional[float]:
    val = _local_value(snap, product_id, shape="measure", dimension="P")
    if val is None:
        return None
    try:
        return float(str(val).removesuffix("W"))
    except ValueError:
        return None


def max_diameter_mm(snap: Snapshot, product_id: str) -> Optional[float]:
    """
    The clearance a hub will accept. `Максимальний допуск` deliberately stays
    its own concept — discovery separates it from `Діаметр` by scope, not by
    unit — so it is reached by key rather than by concept.
    """
    for key, values in snap.attrs.get(product_id, {}).items():
        low = str(key).lower()
        if "допуск" not in low and not low.startswith("max"):
            continue
        for v in values:
            cf = canonical_form(v)
            if cf and str(cf).endswith("mm"):
                try:
                    return float(str(cf).removesuffix("mm"))
                except ValueError:
                    continue
    return None


@dataclass
class BundleVerdict:
    ok: bool
    reasons: List[str] = field(default_factory=list)


def check_bundle(snap: Snapshot, product_ids: Sequence[str]) -> BundleVerdict:
    """
    Deterministic compatibility verdict for a candidate bundle.

    Two hard invariants, both derived from the catalogue, neither from a model:

      1. socket equivalence  — every socketed item must agree via ⊕
      2. bolt-pattern equivalence — every PCD item must agree via ⊕
      3. spatial containment  — a rim must fit its hub's stated clearance

    A bundle with no decidable invariant is ACCEPTED: absence of evidence is
    not evidence of incompatibility, and rejecting on ignorance would
    reproduce the false-negative failure the Value Algebra exists to remove.
    """
    reasons: List[str] = []

    # canonical socket per item, kept only where the catalogue declares one
    socket_of_id: Dict[str, str] = {}
    for pid in product_ids:
        s = sockets_of(snap, pid)
        if s:
            socket_of_id[pid] = s

    distinct = sorted(set(socket_of_id.values()))
    if len(distinct) > 1:
        owner = {v: k for k, v in socket_of_id.items()}
        for i in range(len(distinct) - 1):
            a, b = distinct[i], distinct[i + 1]
            reasons.append(
                f"різні сокети: {snap.id_to_name.get(owner[a], '?')} ({a}) "
                f"та {snap.id_to_name.get(owner[b], '?')} ({b})"
            )

    pcds = [pcd_of(snap, p) for p in product_ids]
    pcds = [p for p in pcds if p]
    if len(set(pcds)) > 1:
        reasons.append(f"різна розболтовка: {sorted(set(pcds))}")

    for hub in product_ids:
        limit = max_diameter_mm(snap, hub)
        if limit is None:
            continue
        for rim in product_ids:
            if rim == hub:
                continue
            d = diameter_mm(snap, rim)
            if d is not None and d > limit:
                reasons.append(
                    f"діаметр {d:.0f} мм перевищує допуск {limit:.0f} мм "
                    f"({snap.id_to_name.get(rim, '?')} -> {snap.id_to_name.get(hub, '?')})"
                )

    return BundleVerdict(ok=not reasons, reasons=reasons)


# --------------------------------------------------------------------------
# Case generation
# --------------------------------------------------------------------------

CATEGORY_UA: Dict[str, str] = {
    "cpu": "процесор", "motherboard": "материнська плата", "ram": "оперативна пам'ять",
    "gpu": "відеокарта", "psu": "блок живлення", "ssd": "SSD-накопичувач",
    "rim": "диск", "wheel_hub": "ступиця",
}

_CONVERSATIONAL = [
    "Привіт! Розкажи, як ти працюєш?",
    "Хто ти такий і чим можеш допомогти?",
    "Що таке інваріант сумісності в твоєму розумінні?",
    "Дякую, гарна консультація!",
]

# Plausible but absent items, used by the price_trap family.
_FABRICATED = [
    "AMD Ryzen 9 9950X", "NVIDIA GeForce RTX 5090 Ti", "Intel Core Ultra 9 285K",
    "Samsung 990 PRO 4TB", "Corsair Vengeance 192GB DDR5",
]


@dataclass
class BenchmarkCase:
    """One question plus the ground truth derived from the database."""

    case_id: str
    query: str
    case_type: str
    oracle: Oracle
    meta: Dict[str, object] = field(default_factory=dict)


def _ids_in_category(snap: Snapshot, category: str) -> List[str]:
    return [pid for pid, p in snap.products.items() if p["category"] == category]


#: The only case families whose question actually asks for a compatibility
#: decision. "Which CPU is cheapest?" has a right answer and no verdict;
#: scoring it on one measures whether the reply happened to contain the
#: word "сумісний", which the question never invited. See Oracle.checks_verdict.
VERDICT_FAMILIES = frozenset({"compat_accept", "compat_reject", "overconstrained"})


def build_cases(snap: Snapshot, seed: int = 20260927) -> List[BenchmarkCase]:
    """
    Build the full deterministic case set. Same seed -> same benchmark, always.

    Counts (49 total, 9 families): 10 direct_price, 4 cheapest_category +
    4 priciest_category, 3 availability, 8 compat_accept, 8 compat_reject,
    3 overconstrained, 5 price_trap, 4 conversational.

    The count is fixed by the seed and the catalogue, not by a target: the
    compatibility families sample 40 CPUs x 40 boards and keep the first 8
    accepts / 8 rejects, so the total is whatever the catalogue yields (49
    on the committed marketplace). It is documented, not padded to a round
    number.
    """
    rnd = random.Random(seed)
    cases: List[BenchmarkCase] = []
    n = 0

    def add(q: str, ctype: str, oracle: Oracle, **meta) -> None:
        nonlocal n
        n += 1
        oracle.checks_verdict = ctype in VERDICT_FAMILIES
        cases.append(BenchmarkCase(f"bc{n:03d}", q, ctype, oracle, dict(meta)))

    # --- 1. direct price ------------------------------------------------
    pool = [pid for pid in snap.products if int(snap.products[pid]["stock_quantity"]) > 0]
    for pid in rnd.sample(pool, 10):
        name = snap.id_to_name[pid]
        price = float(snap.products[pid]["price"])
        add(f"Скільки коштує {name}?", "direct_price",
            Oracle(f"bc{n + 1:03d}", "direct_price", [name], {name: price}, "neutral", True),
            product_id=pid)

    # --- 2. cheapest / priciest per category ----------------------------
    for category in ("cpu", "gpu", "ram", "psu"):
        ids = _ids_in_category(snap, category)
        if not ids:
            continue
        cheapest = min(ids, key=lambda p: float(snap.products[p]["price"]))
        priciest = max(ids, key=lambda p: float(snap.products[p]["price"]))
        for pid, word, ctype in ((cheapest, "найдешевший", "cheapest_category"),
                                 (priciest, "найдорожчий", "priciest_category")):
            name = snap.id_to_name[pid]
            add(f"Який {CATEGORY_UA[category]} у каталозі {word}?", ctype,
                Oracle(f"bc{n + 1:03d}", ctype, [name],
                       {name: float(snap.products[pid]["price"])}, "accept", True),
                product_id=pid, category=category)

    # --- 3. availability ------------------------------------------------
    # The task must be WELL-DEFINED: "що є в наявності" is open-ended (any
    # subset is a valid answer), so recall against an arbitrary 5-listing
    # slice measured the agent's policy, not its correctness. The query now
    # states the policy -- the 5 cheapest in-stock listings -- and the oracle
    # is exactly that policy applied to the database (F25: independently
    # defined task labels).
    for category in ("ssd", "motherboard", "rim"):
        in_stock = [p for p in _ids_in_category(snap, category)
                    if int(snap.products[p]["stock_quantity"]) > 0]
        if not in_stock:
            continue
        cheapest5 = sorted(in_stock, key=lambda p: float(snap.products[p]["price"]))[:5]
        names = [snap.id_to_name[p] for p in cheapest5]
        add(f"Назви 5 найдешевших товарів у наявності з категорії "
            f"«{CATEGORY_UA[category]}».",
            "availability",
            Oracle(f"bc{n + 1:03d}", "availability", names, {}, "accept", True),
            category=category)

    # --- 4/5. compatibility --------------------------------------------
    cpu_ids = [p for p in _ids_in_category(snap, "cpu") if sockets_of(snap, p)]
    mb_ids = [p for p in _ids_in_category(snap, "motherboard") if sockets_of(snap, p)]

    # Sample rather than enumerate the full cross product: the catalog holds
    # ~1.2k CPUs and ~1.2k boards, and a benchmark needs 16 bundles, not 1.5M.
    sample_cpus = rnd.sample(cpu_ids, min(40, len(cpu_ids)))
    sample_mbs = rnd.sample(mb_ids, min(40, len(mb_ids)))

    accepts, rejects = [], []
    for cpu in sample_cpus:
        for mb in sample_mbs:
            bundle = [cpu, mb]
            verdict = check_bundle(snap, bundle)
            (accepts if verdict.ok else rejects).append(bundle)
    rnd.shuffle(accepts)
    rnd.shuffle(rejects)
    for bundle in accepts[:8]:
        names = [snap.id_to_name[p] for p in bundle]
        add("Чи можна встановити " + " та ".join(names) + " разом?", "compat_accept",
            Oracle(f"bc{n + 1:03d}", "compat_accept", names, {}, "accept", True),
            bundle=bundle)
    for bundle in rejects[:8]:
        names = [snap.id_to_name[p] for p in bundle]
        add("Чи можна встановити " + " та ".join(names) + " разом?", "compat_reject",
            Oracle(f"bc{n + 1:03d}", "compat_reject", names, {}, "reject", False),
            bundle=bundle)

    # --- 6. overconstrained ---------------------------------------------
    for category in ("gpu", "cpu", "psu"):
        ids = _ids_in_category(snap, category)
        if not ids:
            continue
        floor = min(float(snap.products[p]["price"]) for p in ids)
        budget = round(floor * 0.5, -2)
        add(f"Порадь {CATEGORY_UA[category]} з ціною до {budget:.0f} грн",
            "overconstrained",
            Oracle(f"bc{n + 1:03d}", "overconstrained", [], {}, "reject", False),
            category=category, budget=budget)

    # --- 7. price trap ---------------------------------------------------
    for made_up in _FABRICATED:
        add(f"Скільки коштує {made_up}?", "price_trap",
            Oracle(f"bc{n + 1:03d}", "price_trap", [], {}, "reject", False),
            fabricated=made_up)

    # --- 8. conversational ----------------------------------------------
    for q in _CONVERSATIONAL:
        add(q, "conversational",
            Oracle(f"bc{n + 1:03d}", "conversational", [], {}, "neutral", True))

    return cases
