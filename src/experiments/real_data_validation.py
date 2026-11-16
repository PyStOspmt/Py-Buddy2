"""Py-Buddy: validation on real, public catalogues.

The paper's numbers come from `data/marketplace_scale.db`, a catalogue this
repository generated for itself. This module re-runs the *foundational* layers
against catalogues that nobody in this repository wrote:

    real_off.db     Open Food Facts product database (ODbL 1.0)
    real_prices.db  Open Prices observations  (ODbL 1.0)
    real_wdc.db     WDC Products 2017         (CC-BY 4.0, Web Data Commons)

produced by `data/ingest_real_data.py`.

What is measured, and what deliberately is *not*
------------------------------------------------
Measured here (all domain-independent -- they hold for food as well as for
PC parts):

  section 1  catalogue shape and key discovery on real heterogeneous EAV
  section 2  the Value Algebra (⊕) against real surface spellings and units:
             how many distinct strings it unifies, how many it keeps apart,
             and where a plain string-equality normaliser would have failed
  section 3  cross-offer agreement inside real groups (WDC `cluster_id`;
             GTIN groups in Open Prices) -- does the algebra DETECT the
             disagreements two real shops state about one product?
  section 4  grounded answering over the real catalogue with SQL-derived
             oracles, against the BM25 retrieval baseline

NOT measured here: the compatibility families (socket / bolt pattern /
clearance) of `benchmark_cases.py`. Those invariants are PC-domain axioms; a
food catalogue has no sockets, and inventing a "compatibility" question about
camembert to make a table look full is exactly the kind of self-flattery this
module exists to prevent. Section 4 therefore scores only the families whose
ground truth SQL can compute: prices, availability, and absent products.

Answer-side language is the project's own (Ukrainian cue words in
`offline_agent.py`); the catalogue is real. That keeps cue-word parsing in the
distribution the agent was built for while removing every invented product,
price and attribute.

Usage
-----
    python src/real_data_validation.py --db data/real_off.db    --label off
    python src/real_data_validation.py --db data/real_prices.db --label prices
    python src/real_data_validation.py --db data/real_wdc.db    --label wdc
    python src/real_data_validation.py --db data/real_off.db --out docs/real_off.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
CORE = os.path.abspath(os.path.join(HERE, "..", "core"))
if CORE not in sys.path:
    sys.path.insert(0, CORE)

from attribute_algebra import CONFLICT, canonical_form, fold_all  # noqa: E402
from benchmark_cases import Snapshot, load_snapshot               # noqa: E402
from key_unit_profiles import profile_key_units, summarise as summarise_units  # noqa: E402
from metrics import Oracle, aggregate, score_case                 # noqa: E402

#: Pairs compared per key in section 2. Real catalogues carry keys with tens of
#: thousands of values; the comparison is quadratic in the sample, so the sample
#: is capped and the cap is reported instead of quietly deciding significance.
MAX_VALUES_PER_KEY = 200
SAMPLE_VALUES = 6


def _is_zero_canonical(canon: Optional[str]) -> bool:
    """True when a canonical form carries a magnitude of exactly zero.

    Needed to keep the headline honest. On Open Food Facts most of the pairs
    the algebra unifies are `0.0 g` against `0.0 mg`: correct, and worth
    nothing -- zero is zero in any unit, and it is the easiest possible merge.
    Reporting them as evidence of unit understanding would overstate the
    result, so they are counted separately instead of being quietly included.
    """
    if not canon:
        return False
    m = re.match(r"^[-+]?\d*\.?\d+", canon)
    return bool(m) and float(m.group(0)) == 0.0


def _naive(text: object) -> str:
    """The normaliser this work argues against: fold unicode, lower, collapse.

    It is the strongest *string* normaliser that needs no domain model, which
    is what makes it the honest competitor for the algebra in section 2.
    """
    return " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())


def _pct(num: int, den: int) -> float:
    return round(100.0 * num / den, 2) if den else 0.0


# --------------------------------------------------------------------------
# Section 1: the real catalogue as the algebra sees it
# --------------------------------------------------------------------------


def report_catalog(snap: Snapshot) -> Dict[str, object]:
    """Shape of the real data, plus what key discovery made of it."""
    multi_key_concepts = [c for c in snap.concepts if len(c.keys) > 1]
    keyed = sum(1 for c in snap.concepts if c.keys)
    price_counts: Dict[str, int] = {}
    priced = 0
    for pid, row in snap.products.items():
        try:
            p = float(row["price"])
        except (KeyError, TypeError, ValueError):
            continue
        if p > 0:
            priced += 1
        price_counts[snap.id_to_name.get(pid, pid)] = price_counts.get(
            snap.id_to_name.get(pid, pid), 0) + 1
    multi_offer_names = sum(1 for n, c in price_counts.items() if c > 1)
    grouped_rows = sum(len(c.keys) for c in multi_key_concepts)
    return {
        "products": len(snap.products),
        "distinct_names": len(snap.price_index),
        "distinct_prices_quoted": len({round(float(p), 2) for v in snap.price_index.values()
                                       for p in v}),
        "priced_products": priced,
        "price_coverage_pct": _pct(priced, len(snap.products)),
        "names_listed_more_than_once": multi_offer_names,
        "eav_rows": sum(len(v) for v in snap.attrs.values()),
        "eav_keys": len({k for per in snap.attrs.values() for k in per}),
        "concepts": len(snap.concepts),
        "keys_with_a_concept": keyed,
        "concepts_over_several_keys": len(multi_key_concepts),
        "keys_grouped_by_discovery": grouped_rows,
        "discovered_groups": [
            {"name": c.name, "shape": c.shape.value, "dimension": c.dimension,
             "unit": c.unit, "keys": c.keys[:8], "vocabulary": c.vocabulary[:6]}
            for c in multi_key_concepts[:5]
        ],
    }


# --------------------------------------------------------------------------
# Section 2: the Value Algebra on real surface spellings
# --------------------------------------------------------------------------


def report_algebra(snap: Snapshot, sample: int = SAMPLE_VALUES) -> Dict[str, object]:
    """What ⊕ does to the values a real catalogue actually stores.

    Three questions, in this order:

      1. can the value be canonicalised at all  (coverage)
      2. does ⊕ unify two spellings the string normaliser keeps apart (gain)
      3. does ⊕ keep apart two spellings the string normaliser would wrongly
         merge (that is the safety half of the same claim)

    The comparison is measured over distinct values *within one key*, which is
    the harshest setting available: values under one key are supposed to be
    comparable, so any failure there is a failure of the algebra, not of the
    pairing.
    """
    per_key: Dict[str, List[str]] = {}
    for per in snap.attrs.values():
        for key, values in per.items():
            bucket = per_key.setdefault(key, [])
            room = MAX_VALUES_PER_KEY - len(bucket)
            if room > 0:
                bucket.extend(values[:room])

    totals = {"values": 0, "canonicalisable": 0, "pairs": 0, "naive_unified": 0,
              "algebra_unified": 0, "algebra_gain": 0, "gain_zero": 0, "kept_apart": 0,
              "gain_examples": [], "kept_apart_examples": []}
    rows: List[Dict[str, object]] = []

    for key, values in sorted(per_key.items()):
        uniq = sorted(set(values))
        if not uniq:
            continue
        canon: Dict[str, Optional[str]] = {}
        for v in uniq:
            c = canonical_form(v, key=key)
            canon[v] = str(c) if isinstance(c, str) else None
        canonicalisable = sum(1 for v in uniq if canon[v] is not None)

        pairs = naive_pairs = algebra_pairs = gain = kept_apart = 0
        gain_zero = 0
        for i in range(len(uniq)):
            a = uniq[i]
            for j in range(i + 1, len(uniq)):
                b = uniq[j]
                is_naive = _naive(a) == _naive(b)
                is_algebra = canon[a] is not None and canon[a] == canon[b]
                pairs += 1
                naive_pairs += is_naive
                algebra_pairs += is_algebra
                if is_algebra and not is_naive:
                    gain += 1
                    gain_zero += _is_zero_canonical(canon[a])
                    if len(totals["gain_examples"]) < sample:
                        totals["gain_examples"].append(
                            {"key": key, "left": a, "right": b, "canonical": canon[a],
                             "zero_magnitude": _is_zero_canonical(canon[a])})
                elif is_naive and not is_algebra:
                    kept_apart += 1
                    if len(totals["kept_apart_examples"]) < sample:
                        totals["kept_apart_examples"].append(
                            {"key": key, "left": a, "right": b})

        totals["values"] += len(uniq)
        totals["canonicalisable"] += canonicalisable
        totals["pairs"] += pairs
        totals["naive_unified"] += naive_pairs
        totals["algebra_unified"] += algebra_pairs
        totals["algebra_gain"] += gain
        totals["gain_zero"] += gain_zero
        totals["kept_apart"] += kept_apart
        rows.append({
            "key": key, "distinct_values": len(uniq),
            "canonicalisable": canonicalisable, "pairs": pairs,
            "naive_unified": naive_pairs, "algebra_unified": algebra_pairs,
            "algebra_gain": gain, "gain_zero_magnitude": gain_zero,
            "kept_apart": kept_apart,
        })

    rows.sort(key=lambda r: (-int(r["algebra_gain"]), -int(r["distinct_values"])))
    return {
        "values_examined": totals["values"],
        "canonicalisable_pct": _pct(totals["canonicalisable"], totals["values"]),
        "value_pairs_compared": totals["pairs"],
        "unified_by_string_normaliser": totals["naive_unified"],
        "unified_by_algebra": totals["algebra_unified"],
        "algebra_gain_pairs": totals["algebra_gain"],
        "algebra_gain_pct_of_pairs": _pct(totals["algebra_gain"], totals["pairs"]),
        "algebra_gain_zero_magnitude_pairs": totals["gain_zero"],
        "algebra_gain_substantive_pairs": totals["algebra_gain"] - totals["gain_zero"],
        "kept_apart_pairs": totals["kept_apart"],
        "gain_examples": totals["gain_examples"],
        "kept_apart_examples": totals["kept_apart_examples"],
        "top_keys": rows[:15],
        "max_values_per_key": MAX_VALUES_PER_KEY,
    }


# --------------------------------------------------------------------------
# Section 3: does the algebra see the disagreements real offers state?
# --------------------------------------------------------------------------

GROUP_KEYS = ("cluster_id", "gtin")


def _group_key(snap: Snapshot) -> Optional[str]:
    present = {k for per in snap.attrs.values() for k in per}
    for candidate in GROUP_KEYS:
        if candidate in present:
            return candidate
    return None


def _worst_minority(units: Dict[str, object]) -> str:
    """One-line console example: the worst key and the unit that is odd in it."""
    offenders = units.get("top_minority_unit_keys") or []
    if not offenders:
        return "none"
    worst = offenders[0]
    rare = sorted((worst.get("minority_units") or {}).items(),
                  key=lambda kv: (-kv[1], kv[0]))[:1]
    token = f"{rare[0][0]} x{rare[0][1]}" if rare else "?"
    return f"{worst.get('key')} in {worst.get('dominant')} but {token}"


def report_unit_profiles(snap: Snapshot) -> Dict[str, object]:
    """Per-key unit usage of a real catalogue: dominant unit, minority units.

    Descriptive only, and deliberately so. The measurement behind
    `key_unit_profiles` shows that inferring `g == mg` from co-occurrence would
    have "corroborated" 915 Open Food Facts cells that are themselves
    unit-entry errors; this report labels keys and never merges values.
    """
    values_by_key: Dict[str, List[str]] = {}
    for attrs in snap.attrs.values():
        for key, values in attrs.items():
            bucket = values_by_key.setdefault(key, [])
            room = MAX_VALUES_PER_KEY - len(bucket)
            if room > 0:
                bucket.extend(str(v) for v in values[:room])
    return summarise_units(profile_key_units(values_by_key))


def report_groups(snap: Snapshot, sample: int = SAMPLE_VALUES) -> Dict[str, object]:
    """Fold every attribute set of one real group and count real disagreements.

    A group is a set of offers that a public dataset says are the SAME product
    (WDC `cluster_id`, or a shared GTIN in Open Prices). Any attribute the
    members state differently is therefore a genuine disagreement between two
    real shops, not a synthetic contradiction -- and a system that silently
    averages such values produces a number no shop ever quoted.
    """
    gkey = _group_key(snap)
    if gkey is None:
        return {"group_key": None,
                "note": "no cluster_id / gtin attribute in this catalogue"}

    groups: Dict[str, List[str]] = {}
    for pid, per in snap.attrs.items():
        for value in per.get(gkey, []):
            groups.setdefault(str(value), []).append(pid)

    conflicts: Dict[str, int] = {}
    agreements: Dict[str, int] = {}
    examples: List[Dict[str, object]] = []
    multi = 0
    for gvalue, members in groups.items():
        if len(members) < 2:
            continue
        multi += 1
        keys = {k for pid in members for k in snap.attrs.get(pid, {})} - {gkey}
        for key in keys:
            values = [v for pid in members for v in snap.attrs.get(pid, {}).get(key, [])]
            if len(values) < 2:
                continue
            folded = fold_all(list(values), key=key)
            if folded is CONFLICT:
                conflicts[key] = conflicts.get(key, 0) + 1
                if len(examples) < sample:
                    examples.append({"group": gvalue, "key": key,
                                     "values": sorted(set(values))[:6],
                                     "members": len(members)})
            else:
                agreements[key] = agreements.get(key, 0) + 1

    compared = sum(conflicts.values()) + sum(agreements.values())
    return {
        "group_key": gkey,
        "groups_total": len(groups),
        "groups_with_several_offers": multi,
        "attribute_sets_compared": compared,
        "conflicting_attribute_sets": sum(conflicts.values()),
        "conflict_rate_pct": _pct(sum(conflicts.values()), compared),
        "conflicts_by_key": sorted(conflicts.items(), key=lambda kv: -kv[1])[:15],
        "agreements_by_key_top": sorted(agreements.items(), key=lambda kv: -kv[1])[:10],
        "conflict_examples": examples,
    }


def report_price_dispersion(snap: Snapshot) -> Dict[str, object]:
    """How far apart the real offers of one product are priced.

    Reported because it is the number a "just average the listings" system
    gets wrong: the catalogue does not contain one price per product, and the
    spread between real shops is the size of the error it would make.
    """
    gkey = _group_key(snap)
    groups: Dict[str, List[float]] = {}
    if gkey is not None:
        buckets: Dict[str, List[str]] = {}
        for pid, per in snap.attrs.items():
            for value in per.get(gkey, []):
                buckets.setdefault(str(value), []).append(pid)
        for gvalue, members in buckets.items():
            prices = []
            for pid in members:
                try:
                    p = float(snap.products[pid]["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                if p > 0:
                    prices.append(p)
            if len(prices) > 1:
                groups[gvalue] = prices
    else:
        for name, prices in snap.price_index.items():
            positive = [float(p) for p in prices if float(p) > 0]
            if len(positive) > 1:
                groups[name] = positive

    if not groups:
        return {"groups_with_several_prices": 0,
                "note": "no group has more than one real price"}

    spreads = sorted(max(v) / min(v) for v in groups.values() if min(v) > 0)
    median = spreads[len(spreads) // 2]
    over_50 = sum(1 for s in spreads if s > 1.5)
    return {
        "groups_with_several_prices": len(groups),
        "median_max_over_min": round(median, 3),
        "p90_max_over_min": round(spreads[int(len(spreads) * 0.9)], 3),
        "groups_where_spread_exceeds_50pct": over_50,
        "share_over_50pct": _pct(over_50, len(spreads)),
        "worst_examples": [
            {"group": g, "prices": sorted(round(p, 2) for p in v)[:6]}
            for g, v in sorted(groups.items(),
                               key=lambda kv: -(max(kv[1]) / min(kv[1])))[:5]
            if min(v) > 0
        ],
    }


# --------------------------------------------------------------------------
# Section 4: grounded answering over the real catalogue
# --------------------------------------------------------------------------

#: Query templates. Their cue words are the project's own (`offline_agent.py`
#: parses Ukrainian), and the product names inside them are the real stored
#: names, so name matching is exercised against real spelling.
Q_PRICE = "Скільки коштує {name}?"
Q_CHEAPEST = "Який товар найдешевший у категорії {category}?"
Q_AVAILABLE = "Що є в наявності в категорії {category}?"

CONVERSATIONAL = (
    "Привіт! Розкажи, як ти працюєш?",
    "Хто ти такий і чим можеш допомогти?",
    "Дякую, гарна консультація!",
)


class RealCase:
    """One real-data question plus the answer SQL says is right."""

    def __init__(self, case_id: str, case_type: str, query: str, oracle: Oracle):
        self.case_id = case_id
        self.case_type = case_type
        self.query = query
        self.oracle = oracle


def _ro_query(db_path: str, sql: str, args: Sequence[object] = ()) -> List[tuple]:
    import sqlite3
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    try:
        return list(conn.execute(sql, args))
    finally:
        conn.close()


def build_real_cases(snap: Snapshot, n: int = 60, seed: int = 7,
                     absent_names: Optional[Sequence[str]] = None) -> List[RealCase]:
    """Price / availability / absent-product cases with SQL ground truth."""
    rng = random.Random(seed)
    db = snap.db_path
    cases: List[RealCase] = []

    # --- direct price -----------------------------------------------------
    priced = _ro_query(db, "SELECT name, MIN(price) FROM products WHERE price > 0 "
                           "GROUP BY name HAVING COUNT(*) >= 1 ORDER BY name")
    if not priced:
        print("    [note] no priced products in this catalogue: price families are "
              "skipped, not faked (an item nobody offers has no price to check)",
              flush=True)
    picked = (rng.sample(priced, min(max(1, n // 3), len(priced))) if priced else [])
    for i, (name, _) in enumerate(picked):
        prices = [round(float(p), 2) for p in snap.price_index.get(name, [])]
        cases.append(RealCase(
            f"real_price_{i}", "direct_price", Q_PRICE.format(name=name),
            Oracle(case_id=f"real_price_{i}", case_type="direct_price",
                   expected_products=[name],
                   expected_prices={name: prices or [float(priced[i][1])]},
                   checks_verdict=False)))

    # --- cheapest in a real category --------------------------------------
    cats = _ro_query(db, "SELECT category, COUNT(DISTINCT name) FROM products "
                         "WHERE price > 0 GROUP BY category "
                         "HAVING COUNT(DISTINCT name) >= 3 ORDER BY 2 DESC LIMIT 12")
    for i, (category, _) in enumerate(cats[:max(1, n // 6)]):
        row = _ro_query(db, "SELECT name, MIN(price) FROM products WHERE category = ? "
                            "AND price > 0 GROUP BY name ORDER BY 2 LIMIT 1", (category,))
        if not row:
            continue
        name = row[0][0]
        cases.append(RealCase(
            f"real_cheapest_{i}", "cheapest_category",
            Q_CHEAPEST.format(category=category),
            Oracle(case_id=f"real_cheapest_{i}", case_type="cheapest_category",
                   expected_products=[name],
                   expected_prices={name: [round(float(p), 2)
                                           for p in snap.price_index.get(name, [])]},
                   checks_verdict=False)))

    # --- availability -----------------------------------------------------
    for i, (category, _) in enumerate(cats[:max(1, n // 6)]):
        rows = _ro_query(db, "SELECT name FROM products WHERE category = ? "
                             "GROUP BY name ORDER BY name LIMIT 3", (category,))
        names = [r[0] for r in rows]
        if not names:
            continue
        cases.append(RealCase(
            f"real_avail_{i}", "availability",
            Q_AVAILABLE.format(category=category),
            Oracle(case_id=f"real_avail_{i}", case_type="availability",
                   expected_products=names[:1], checks_verdict=False)))

    # --- products that really are absent -----------------------------------
    # Taken from a different real catalogue and verified absent by SELECT, so
    # the trap is "a real product this catalogue does not carry", never an
    # invented name.
    if absent_names:
        known = {n.lower() for n in snap.price_index}
        absent = [x for x in absent_names if x and x.lower() not in known]
        for i, name in enumerate(absent[:max(1, n // 6)]):
            cases.append(RealCase(
                f"real_absent_{i}", "absent_product", Q_PRICE.format(name=name),
                Oracle(case_id=f"real_absent_{i}", case_type="absent_product",
                       expected_products=[], checks_verdict=False)))

    # --- meta questions (nothing to invent) --------------------------------
    for i, q in enumerate(CONVERSATIONAL):
        cases.append(RealCase(f"real_meta_{i}", "conversational", q,
                              Oracle(case_id=f"real_meta_{i}", case_type="conversational",
                                     expected_products=[], checks_verdict=False)))
    return cases


def run_systems(snap: Snapshot, cases: Sequence[RealCase],
                verbose: bool = True) -> Dict[str, object]:
    """Score the proposed system and the retrieval baseline on the same cases.

    Both run offline (no language model): this compares the deterministic tool
    layer against BM25 retrieval, so any difference is attributable to the
    architecture and not to a model's luck.
    """
    from baseline_rag import RetrievalBaseline
    from offline_agent import OfflineDeterministicAgent

    pybuddy = OfflineDeterministicAgent(snap)
    baseline = RetrievalBaseline(snap.db_path, runner=None)
    out: Dict[str, object] = {}
    for system in ("pybuddy_offline", "bm25_rag"):
        scores = []
        for case in cases:
            t0 = time.time()
            verdict = None
            if system == "pybuddy_offline":
                res = pybuddy.answer(case.query)
                text = str(res.get("text", ""))
                steps = int(res.get("steps", 0))
                cost = float(res.get("cost_usd", 0.0))
                verdict = res.get("verdict")
            else:
                text, steps, cost = baseline.answer(case.query)
                text, steps, cost = str(text), int(steps), float(cost)
            s = score_case(text, case.oracle, snap.names(), snap.price_index,
                           steps=steps, latency_s=time.time() - t0, cost_usd=cost,
                           structured_verdict=verdict)
            scores.append(s)
            if verbose and not s.is_correct:
                print(f"    [miss] {system} {case.case_id} [{case.case_type}] "
                      f"{s.violations}")
        agg = aggregate(scores)
        out[system] = {"aggregate": agg.__dict__,
                       "cases": [s.to_dict() for s in scores]}
    return out


def absent_names_from(db_path: str, exclude: str, limit: int = 40) -> List[str]:
    """Real product names from *other* catalogues in the same directory.

    Used by the absent-product family: a name that is real somewhere else is
    still verified absent here by SELECT, so the trap stays honest.
    """
    import sqlite3
    folder = os.path.dirname(os.path.abspath(db_path)) or "."
    names: List[str] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.startswith("real_") or not fname.endswith(".db"):
            continue
        other = os.path.join(folder, fname)
        if os.path.abspath(other) == os.path.abspath(exclude):
            continue
        try:
            conn = sqlite3.connect("file:%s?mode=ro" % other.replace("\\", "/"), uri=True)
            for (name,) in conn.execute("SELECT name FROM products "
                                        "WHERE price > 0 GROUP BY name LIMIT 20"):
                names.append(str(name))
            conn.close()
        except Exception:                                # noqa: BLE001
            continue
        if len(names) >= limit:
            break
    return names[:limit]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True, help="real_*.db produced by the ingestor")
    ap.add_argument("--label", default="", help="name used in the report")
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None, help="write the full report as JSON here")
    args = ap.parse_args(argv)

    label = args.label or os.path.splitext(os.path.basename(args.db))[0]
    snap = load_snapshot(args.db)
    print(f"[real] {label}: {len(snap.products)} products, "
          f"{len(snap.concepts)} discovered concepts", flush=True)

    catalog = report_catalog(snap)
    print(f"[real] catalogue: {catalog['eav_rows']} EAV rows over "
          f"{catalog['eav_keys']} keys; price coverage "
          f"{catalog['price_coverage_pct']}%; "
          f"{catalog['names_listed_more_than_once']} names listed more than once; "
          f"{catalog['keys_grouped_by_discovery']} keys grouped into "
          f"{catalog['concepts_over_several_keys']} multi-key concepts")

    algebra = report_algebra(snap)
    print(f"[real] algebra: {algebra['canonicalisable_pct']}% of "
          f"{algebra['values_examined']} distinct values canonicalisable; unified by "
          f"string normaliser {algebra['unified_by_string_normaliser']} vs by algebra "
          f"{algebra['unified_by_algebra']} (gain {algebra['algebra_gain_pairs']}, "
          f"of which {algebra['algebra_gain_zero_magnitude_pairs']} are zero-magnitude "
          f"pairs such as `0.0 g` == `0.0 mg`)")

    groups = report_groups(snap)
    if groups.get("group_key") is None:
        print(f"[real] groups: not applicable -- {groups.get('note')}")
    else:
        print(f"[real] groups: {groups.get('groups_with_several_offers', 0)} groups of "
              f"several real offers; {groups.get('conflicting_attribute_sets', 0)} "
              f"conflicting attribute sets ({groups.get('conflict_rate_pct', 0)}%)")

    units = report_unit_profiles(snap)
    if units["keys_profiled"]:
        print(f"[real] units: {units['keys_profiled']} keys profiled; "
              f"{units['keys_with_minority_unit']} carry a minority unit "
              f"({units['minority_values_flagged']} values likely mistyped, e.g. "
              f"{_worst_minority(units)})")

    dispersion = report_price_dispersion(snap)
    if dispersion.get("groups_with_several_prices"):
        print(f"[real] prices: {dispersion['groups_with_several_prices']} groups with "
              f"several real prices, median max/min {dispersion['median_max_over_min']}")

    cases = build_real_cases(snap, n=args.cases, seed=args.seed,
                             absent_names=absent_names_from(args.db, args.db))
    print(f"[real] cases: {len(cases)} with SQL-derived ground truth", flush=True)
    systems = run_systems(snap, cases)
    for name, data in systems.items():
        agg = data["aggregate"]  # type: ignore[index]
        print(f"[real] {name}: execution accuracy {agg['execution_accuracy_pct']}%, "
              f"hallucination rate {agg['hallucination_rate_pct']}%, price error rate "
              f"{agg['price_error_rate_pct']}%, NVFG {agg['nvfg_pct']}%")

    report = {
        "label": label,
        "db": os.path.abspath(args.db),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "catalog": catalog,
        "algebra": algebra,
        "unit_profiles": units,
        "groups": groups,
        "price_dispersion": dispersion,
        "cases": len(cases),
        "systems": systems,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[real] report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())






