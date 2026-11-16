"""
Py-Buddy: Open Dynamic Contract Induction (ODCI)
Automates zero-config induction of Virtual Hypergraph constraints across arbitrary e-commerce domains.
Maps raw schemas and attributes to the 4 Universal Meta-Ontology Classes:
1. EquivalenceConstraint (Matching Keys)
2. CumulativeCapacityConstraint (Resource Demand vs Provider Capacity)
3. MutualExclusionConstraint (Conflict Sets)
4. SpatialBoundingConstraint (Dimensional Clearances)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Sequence, Set, Tuple
import json
import re

from attribute_algebra import (
    UNIT_TO_BASE,
    Shape,
    base_of,
    discover_concepts,
    spec_of,
)
from hypergraph import (
    KnowledgeHypergraph,
    ProductNode,
    BaseConstraint,
    EquivalenceConstraint,
    CumulativeCapacityConstraint,
    MutualExclusionConstraint,
    SpatialBoundingConstraint,
)
from openrouter_client import OpenRouterAgentRunner

# F40/F47: the term lists below are LABELLED SEEDS, not universal rules.
# They were observed to separate matching attributes from commercial /
# logistics noise in the pilot catalogues; the load-bearing checks are
# structural (a single-valued attribute cannot discriminate; a capacity
# pairing must be supported by the sample's own numbers). A caller may
# replace or empty any seed via the keyword arguments -- the constants
# are data, not hidden logic.
NON_MATCHING_SEED_KEYS = frozenset({
    "brand", "manufacturer", "color", "status", "warranty", "name", "id",
    "description", "vendor", "merchant", "seller", "shop", "store",
    "supplier", "rating", "vendor_rating", "delivery_days", "price",
    "image", "url", "sku", "колір", "статус", "гарантія", "магазин",
    "продавець", "рейтинг", "доставка",
})
DEMAND_SEED_TERMS = (
    "tdp", "watt", "draw", "weight", "mass", "dose", "demand",
    "consumption", "power_consumption",
    "потужність", "споживання", "тепловиділення", "вага", "маса", "доза",
    "витрата",
)
CAPACITY_SEED_TERMS = (
    "psu_wattage", "wattage", "capacity", "payload", "max_safe_dose",
    "power_supply", "limit", "supply", "psu",
    "бж", "живлення", "ємність", "вантажопідйомність", "макс_доза",
    "ліміт", "запас",
)
# A demand name that also looks like a capacity name is a capacity.
_CAPACITY_OVERLAP_TERMS = (
    "supply", "psu", "capacity", "payload", "max", "limit",
    "бж", "макс", "ліміт",
)
DIMENSION_SEED_TERMS = (
    "length", "width", "height", "diameter", "radius",
    "діаметр", "довжина", "ширина", "висота", "радіус",
)
MAX_DIMENSION_SEED_TERMS = (
    "max_length", "clearance", "max_width", "max_height", "max_diameter",
    "arch_clearance", "макс_діаметр", "макс_довжина", "кліренс", "допуск",
)
# A dimension name that also looks like a limit is a limit.
_MAX_DIM_OVERLAP_TERMS = ("max", "макс", "clearance", "допуск")

EQ = "eq"
LE = "le"
_SYMBOLIC_SHAPES = (Shape.TOKEN, Shape.DUPLEX, Shape.RATIO)


@dataclass(frozen=True, order=True)
class StructuralRule:
    """Domain-free compatibility invariant induced via Value Algebra & dimensions."""
    kind: str
    subject: Tuple[str, ...]
    obj: Tuple[str, ...]
    headroom: float = field(default=0.0, compare=False)

    def __str__(self) -> str:
        if self.kind == EQ:
            return f"eq({', '.join(self.subject)})"
        return f"le({', '.join(self.subject)} <= {', '.join(self.obj)})"

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "subject": list(self.subject),
            "obj": list(self.obj),
            "headroom": round(self.headroom, 4),
        }


def _structural_measure(value: object) -> Optional[Tuple[float, str]]:
    sp = spec_of(str(value))
    if sp.shape is not Shape.MEASURE or not sp.dimension or not sp.numbers:
        return None
    base = base_of(sp.dimension, sp.unit)
    factor = UNIT_TO_BASE.get(sp.unit or base, 1.0)
    return sp.numbers[0] * factor, sp.dimension


def _structural_covers(provider: Sequence[float], consumer: Sequence[float]) -> bool:
    if not provider or not consumer:
        return False
    if sorted(provider) == sorted(consumer):
        return False
    return min(provider) >= max(consumer)


def _key_values(products: Sequence[ProductNode]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for p in products:
        for k, v in p.attributes.items():
            out.setdefault(k, []).append(str(v))
    return out


def _key_scopes(products: Sequence[ProductNode]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for p in products:
        for k in p.attributes:
            out.setdefault(k, set()).add(p.category)
    return out


def induce_structural_rules(products: Sequence[ProductNode]) -> List[StructuralRule]:
    """
    Induce compatibility rules from a catalogue without domain vocabulary.

    1) Symbolic attributes (`TOKEN`, `DUPLEX`, `RATIO`) spanning >=2 categories
       become multi-alias equivalence rules (e.g. `socket` <-> `Сокет`).
    2) Dimensional measurements (`Shape.MEASURE` with shared dimension) where
       all values in an object group cover all values in a distinct-category
       subject group become universal containment (`<=`) rules.
    """
    if not products:
        return []

    key_values = _key_values(products)
    key_scopes = _key_scopes(products)

    concepts = discover_concepts(key_values, key_scopes, min_keys=1)
    groups = [tuple(sorted(c.keys)) for c in concepts]
    rules: List[StructuralRule] = []

    def group_scope(keys: Sequence[str]) -> Set[str]:
        out: Set[str] = set()
        for k in keys:
            out |= key_scopes.get(k, set())
        return out

    def group_measure(keys: Sequence[str]) -> Tuple[List[float], Optional[str]]:
        vals: List[float] = []
        dim: Optional[str] = None
        for k in keys:
            for v in key_values.get(k, ()):
                m = _structural_measure(v)
                if m is None:
                    continue
                if dim is not None and m[1] != dim:
                    return ([], None)
                dim = m[1]
                vals.append(m[0])
        return (vals, dim)

    for concept in concepts:
        if concept.shape not in _SYMBOLIC_SHAPES:
            continue
        if len(group_scope(concept.keys)) < 2:
            continue
        keys_sorted = tuple(sorted(concept.keys))
        rules.append(StructuralRule(EQ, keys_sorted, keys_sorted))

    measurable: List[Tuple[Tuple[str, ...], str, List[float]]] = []
    for keys in groups:
        vals, dim = group_measure(keys)
        if dim and vals:
            measurable.append((keys, dim, vals))

    for i, (sub_keys, sub_dim, sub_vals) in enumerate(measurable):
        for obj_keys, obj_dim, obj_vals in measurable[i + 1:]:
            if sub_dim != obj_dim:
                continue
            if group_scope(sub_keys) & group_scope(obj_keys):
                continue
            forward = _structural_covers(obj_vals, sub_vals)
            backward = _structural_covers(sub_vals, obj_vals)
            if forward == backward:
                continue
            subj, obj = (sub_keys, obj_keys) if forward else (obj_keys, sub_keys)
            subj_vals, obj_vals_sel = (sub_vals, obj_vals) if forward else (obj_vals, sub_vals)
            ceiling = max(obj_vals_sel)
            headroom = (max(subj_vals) / ceiling) if ceiling else 1.0
            rules.append(StructuralRule(LE, tuple(sorted(subj)), tuple(sorted(obj)), headroom))

    return sorted(set(rules))


class ContractInductor:
    """
    Induces mathematical compatibility invariants from raw product catalogs or schema profiles.

    The structural core is domain-free; the name-term SEEDS that nominate
    demand/capacity/dimension candidates are labelled, overridable data
    (F40/F47), and every induced rule stays a `candidate` that may flag
    conflicts but never vouches for compatibility (F12).
    """
    def __init__(self, openrouter_runner: Optional[OpenRouterAgentRunner] = None,
                 max_induced: Optional[int] = None):
        self.runner = openrouter_runner
        self.max_induced = max_induced
        #: Populated by induce_contracts_from_products: how many rules were
        #: induced, kept and dropped under the cap. Truncation is always
        #: visible here and must be surfaced by callers (T6).
        self.last_induction_report: Optional[Dict[str, int]] = None

    @staticmethod
    def _truncate_induced(constraints: List[BaseConstraint],
                          cap: int) -> List[BaseConstraint]:
        """Deterministically keep `cap` induced rules.

        Resource/dimension rules (capacity, spatial, exclusion) carry
        higher per-rule signal than equivalence keys, so they are kept
        first; equivalence rules are ordered by category coverage so a
        truncated set still spans the widest observed scopes.
        """
        def _key(c: BaseConstraint) -> Tuple[int, int, str]:
            coverage = -len(getattr(c, "categories", None) or [])
            priority = 1 if isinstance(c, EquivalenceConstraint) else 0
            return (priority, coverage, c.description or "")

        return sorted(constraints, key=_key)[:cap]

    def _induce_structural_constraints(
        self,
        products: List[ProductNode],
        headroom_ratio: Optional[float] = None,
        non_matching_keys: Optional[Set[str]] = None,
    ) -> List[BaseConstraint]:
        """Materialize structural rules as live candidate constraints (F12)."""
        nm_keys = NON_MATCHING_SEED_KEYS if non_matching_keys is None else non_matching_keys
        filtered = [
            ProductNode(
                id=p.id,
                name=p.name,
                category=p.category,
                price=p.price,
                attributes={k: v for k, v in p.attributes.items() if k not in nm_keys},
            )
            for p in products
        ]
        key_items: Dict[str, Set[str]] = {}
        for p in filtered:
            for k in p.attributes:
                key_items.setdefault(k, set()).add(p.id)

        def _split_cooccurring_aliases(keys: Tuple[str, ...]) -> List[Tuple[str, ...]]:
            """Keep aliases merged only when they appear on mostly disjoint products."""
            groups: List[List[str]] = []
            for k in keys:
                k_set = key_items.get(k, set())
                placed = False
                for g in groups:
                    conflict = False
                    for existing in g:
                        e_set = key_items.get(existing, set())
                        denom = min(len(k_set), len(e_set))
                        if denom and len(k_set & e_set) > 0.25 * denom:
                            conflict = True
                            break
                    if not conflict:
                        g.append(k)
                        placed = True
                        break
                if not placed:
                    groups.append([k])
            return [tuple(g) for g in groups]

        structural = induce_structural_rules(filtered)
        constraints: List[BaseConstraint] = []
        covered_eq_keys: Set[str] = set()
        covered_le_pairs: Set[Tuple[str, str]] = set()
        hr = float(headroom_ratio) if headroom_ratio is not None else 1.0

        for rule in structural:
            if rule.kind == EQ and rule.subject:
                for alias_group in _split_cooccurring_aliases(rule.subject):
                    primary = alias_group[0]
                    aliases = alias_group[1:]
                    label = " / ".join(alias_group)
                    c = EquivalenceConstraint(
                        attr_name=primary,
                        categories=None,
                        description=f"Структурне узгодження [{label}]",
                        alias_attrs=aliases,
                    )
                    c.status = "candidate"
                    constraints.append(c)
                    covered_eq_keys.update(alias_group)
            elif rule.kind == LE and rule.subject and rule.obj:
                d_label = " / ".join(rule.subject)
                c_label = " / ".join(rule.obj)
                c = CumulativeCapacityConstraint(
                    demand_attr=rule.subject[0],
                    capacity_attr=rule.obj[0],
                    headroom_ratio=hr,
                    description=f"Структурне обмеження ({d_label}) <= ({c_label})",
                    demand_attrs=rule.subject,
                    capacity_attrs=rule.obj,
                )
                c.status = "candidate"
                constraints.append(c)
                for sk in rule.subject:
                    for ok in rule.obj:
                        covered_le_pairs.add((sk, ok))

        # Additive fallback for bare unitless float attributes (no dimension tokens)
        # verified by the caller's sample evidence (`test_f11` / raw float datasets).
        for legacy_c in self._induce_deterministic(
            products, non_matching_keys=nm_keys, headroom_ratio=headroom_ratio
        ):
            if isinstance(legacy_c, EquivalenceConstraint):
                if legacy_c.attr_name in NON_MATCHING_SEED_KEYS:
                    continue
                if legacy_c.attr_name in covered_eq_keys:
                    continue
                covered_eq_keys.add(legacy_c.attr_name)
                constraints.append(legacy_c)
            elif isinstance(legacy_c, CumulativeCapacityConstraint):
                if (legacy_c.demand_attr, legacy_c.capacity_attr) in covered_le_pairs:
                    continue
                covered_le_pairs.add((legacy_c.demand_attr, legacy_c.capacity_attr))
                constraints.append(legacy_c)
            else:
                constraints.append(legacy_c)
        return constraints

    def induce_contracts_from_products(
        self,
        products: List[ProductNode],
        use_llm: bool = False,
        headroom_ratio: Optional[float] = None,
        max_induced: Optional[int] = None,
    ) -> List[BaseConstraint]:
        """
        Extracts constraints from a collection of ProductNode instances.
        Uses unified domain-free structural induction by default (F12: candidate),
        or LLM-guided meta-typing if use_llm=True.
        `headroom_ratio` is the source/user-stated safety margin for
        capacity rules; None means the induced rule asserts the exact
        bound (F47 -- no fabricated margin).
        """
        induced: Optional[List[BaseConstraint]] = None
        if use_llm and self.runner and self.runner.api_key:
            try:
                llm_constraints = self._induce_via_llm(products)
                if llm_constraints:
                    induced = llm_constraints
            except Exception:
                # Graceful fallback to deterministic induction if LLM
                # network/format fails; the fallback path is taken below.
                pass

        if induced is None:
            induced = self._induce_structural_constraints(products, headroom_ratio=headroom_ratio)

        cap = max_induced if max_induced is not None else self.max_induced
        if cap is not None and len(induced) > cap:
            self.last_induction_report = {
                "induced": len(induced), "kept": cap,
                "dropped": len(induced) - cap, "cap": cap,
            }
            return self._truncate_induced(induced, cap)
        self.last_induction_report = {
            "induced": len(induced), "kept": len(induced),
            "dropped": 0, "cap": cap if cap is not None else 0,
        }
        return induced

    def _induce_deterministic(
        self,
        products: List[ProductNode],
        *,
        non_matching_keys: Optional[Set[str]] = None,
        demand_terms: Tuple[str, ...] = DEMAND_SEED_TERMS,
        capacity_terms: Tuple[str, ...] = CAPACITY_SEED_TERMS,
        dimension_terms: Tuple[str, ...] = DIMENSION_SEED_TERMS,
        max_dimension_terms: Tuple[str, ...] = MAX_DIMENSION_SEED_TERMS,
        headroom_ratio: Optional[float] = None,
    ) -> List[BaseConstraint]:
        """
        Deterministic, domain-agnostic constraint discovery based on attribute co-occurrence and naming semantics.

        F12: everything this method produces is a correlation observed in the
        passed-in sample, promoted by naming heuristics. Every induced rule is
        therefore marked `status="candidate"`: it may signal a conflict, but
        its pass is never counted as evidence of compatibility. Only rules
        authored or independently validated elsewhere may vouch.

        F40/F47: the name-term lists are labelled seeds (module constants),
        caller-overridable; the structural filters -- single-valued
        attributes cannot discriminate, pairings must be supported by the
        sample's own numbers -- are the domain-free part.
        """
        constraints: List[BaseConstraint] = []
        if non_matching_keys is None:
            non_matching_keys = set(NON_MATCHING_SEED_KEYS)

        # 1. Map attributes to categories that possess them
        attr_to_categories: Dict[str, Set[str]] = {}
        attr_types: Dict[str, Set[str]] = {}
        attr_values: Dict[str, List[float]] = {}
        attr_distinct: Dict[str, Set[str]] = {}

        for p in products:
            for k, v in p.attributes.items():
                if k not in attr_to_categories:
                    attr_to_categories[k] = set()
                    attr_types[k] = set()
                attr_to_categories[k].add(p.category)
                attr_types[k].add(type(v).__name__)
                attr_distinct.setdefault(k, set()).add(repr(v))
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    attr_values.setdefault(k, []).append(float(v))

        # 2. Induce Equivalence Constraints (Matching Keys)
        # Any categorical/string attribute shared across >= 2 different categories is an equivalence candidate
        for attr, cats in attr_to_categories.items():
            # F40: the STRUCTURAL filter first -- an attribute that takes a
            # single value across the whole sample cannot discriminate
            # anything, whatever its name, so no fixed list is needed to
            # exclude it. The labelled seed set then excludes commercial /
            # logistics keys a structural criterion cannot see.
            if len(attr_distinct.get(attr, ())) <= 1:
                continue
            if attr.lower() in non_matching_keys:
                continue
            
            types = attr_types[attr]
            is_string_like = "str" in types or "bool" in types
            
            if len(cats) >= 2 and is_string_like:
                constraints.append(EquivalenceConstraint(
                    attr_name=attr,
                    categories=list(cats),
                    description=f"Авто-індукована еквівалентність за ключем '{attr}' між категоріями {list(cats)}"
                ))

        # 3. Induce Cumulative Capacity Constraints
        # Look for numeric demand indicators and capacity providers (Multilingual + Dimensional)
        demand_candidates = []
        capacity_candidates = []

        for attr, types in attr_types.items():
            if not any(t in ("int", "float") for t in types):
                continue
            attr_l = attr.lower()
            if any(term in attr_l for term in demand_terms) and not any(
                    cap_term in attr_l for cap_term in _CAPACITY_OVERLAP_TERMS):
                demand_candidates.append(attr)
            if any(term in attr_l for term in capacity_terms):
                capacity_candidates.append(attr)

        for dem in demand_candidates:
            for cap in capacity_candidates:
                # Classify on the folded name -- a "TDP" key was previously
                # missed entirely because the pairing terms are lowercase.
                dem_l, cap_l = dem.lower(), cap.lower()
                is_power = any(w in dem_l for w in ["tdp", "watt", "draw", "consumption", "потужн", "спожив"]) and any(w in cap_l for w in ["watt", "psu", "power", "бж", "живлен"])
                is_weight = any(w in dem_l for w in ["weight", "mass", "вага", "маса"]) and any(w in cap_l for w in ["payload", "capacity", "mass", "вантаж", "ємніст"])
                is_dose = ("dose" in dem_l or "доза" in dem_l) and ("dose" in cap_l or "доза" in cap_l)

                if is_power or is_weight or is_dose:
                    # F11: the pairing must be supported by the sample itself.
                    # A "capacity" whose observed values are smaller than the
                    # "demand" it is paired with would flag every bundle as a
                    # conflict -- the numbers refute the naming heuristic, so
                    # no rule is induced from that pairing.
                    dem_vals = attr_values.get(dem, [])
                    cap_vals = attr_values.get(cap, [])
                    if not dem_vals or not cap_vals:
                        continue
                    if sum(cap_vals) / len(cap_vals) <= sum(dem_vals) / len(dem_vals):
                        continue
                    # F47: the margin is not baked in. The source/user may
                    # state one explicitly; without a stated margin the
                    # induced rule asserts the exact bound (ratio 1.0) --
                    # claiming a 10% safety allowance the data never
                    # mentioned would fabricate a constraint.
                    margin = 1.0 if headroom_ratio is None else float(headroom_ratio)
                    margin_note = ("точна межа" if margin == 1.0
                                   else f"запас {margin:g}")
                    constraints.append(CumulativeCapacityConstraint(
                        demand_attr=dem,
                        capacity_attr=cap,
                        headroom_ratio=margin,
                        description=f"Авто-індукований кумулятивний баланс: {dem} <= {margin:g} * {cap} ({margin_note})"
                    ))

        # 4. Induce Spatial Bounding Constraints (Multilingual + Dimension-based)
        dim_candidates = []
        max_dim_candidates = []

        for attr, types in attr_types.items():
            if not any(t in ("int", "float") for t in types):
                continue
            attr_l = attr.lower()
            if any(term in attr_l for term in dimension_terms) and not any(m in attr_l for m in _MAX_DIM_OVERLAP_TERMS):
                dim_candidates.append(attr)
            if any(term in attr_l for term in max_dimension_terms):
                max_dim_candidates.append(attr)

        for dim in dim_candidates:
            for max_dim in max_dim_candidates:
                dim_l, max_l = dim.lower(), max_dim.lower()
                matches_diam = ("diameter" in dim_l or "діаметр" in dim_l) and ("diameter" in max_l or "діаметр" in max_l)
                matches_len = ("length" in dim_l or "довжина" in dim_l) and ("length" in max_l or "довжина" in max_l)
                if matches_diam or matches_len:
                    constraints.append(SpatialBoundingConstraint(
                        dimension_attr=dim,
                        max_dimension_attr=max_dim,
                        description=f"Авто-індукований габаритний допуск: {dim} <= {max_dim}"
                    ))

        for c in constraints:
            c.status = "candidate"
        return constraints

    def _induce_via_llm(self, products: List[ProductNode]) -> List[BaseConstraint]:
        """
        Uses OpenRouter DeepSeek to analyze product attributes and infer Meta-Constraints.
        """
        # Sample unique attribute schemas by category
        schema_by_cat: Dict[str, Dict[str, Any]] = {}
        for p in products:
            if p.category not in schema_by_cat:
                schema_by_cat[p.category] = {}
            for k, v in p.attributes.items():
                if k not in schema_by_cat[p.category]:
                    schema_by_cat[p.category][k] = v

        prompt = f"""Ти — науковий аналітик онтологій систем штучного інтелекту.
Проаналізуй атрибути категорій товарів каталогу:
{json.dumps(schema_by_cat, ensure_ascii=False, indent=2)}

Вияви та згрупуй відношення сумісності у 4 універсальні мета-класи:
1. "equivalence": {{ "attr_name": "...", "categories": ["..."] }} (ключі сумісності, що мають збігатися)
2. "capacity": {{ "demand_attr": "...", "capacity_attr": "...", "headroom_ratio": 0.9 }} (сума споживання <= місткість)
3. "exclusion": {{ "condition_a": {{...}}, "condition_b": {{...}} }} (взаємне виключення)
4. "spatial": {{ "dimension_attr": "...", "max_dimension_attr": "..." }} (габаритні допуски)

Поверни виключно валідний JSON-список об'єктів такого формату:
[
  {{"type": "equivalence", "attr_name": "...", "categories": [...]}},
  {{"type": "capacity", "demand_attr": "...", "capacity_attr": "..."}}
]
Ніяких пояснень чи тексту поза JSON блоком.
"""
        response = self.runner.generate(prompt, temperature=0.0)
        match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
        if match:
            raw_json = match.group(0)
        else:
            raw_json = re.sub(r"```(?:json)?\s*", "", response)
            raw_json = re.sub(r"```\s*", "", raw_json).strip()
        
        parsed = json.loads(raw_json)
        constraints: List[BaseConstraint] = []
        for item in parsed:
            t = item.get("type")
            if t == "equivalence":
                constraints.append(EquivalenceConstraint(
                    attr_name=item["attr_name"],
                    categories=item.get("categories"),
                    description=f"LLM-індукована еквівалентність за '{item['attr_name']}'"
                ))
            elif t == "capacity":
                constraints.append(CumulativeCapacityConstraint(
                    demand_attr=item["demand_attr"],
                    capacity_attr=item["capacity_attr"],
                    headroom_ratio=item.get("headroom_ratio", 0.9),
                    description=f"LLM-індукований баланс {item['demand_attr']} <= {item['capacity_attr']}"
                ))
            elif t == "exclusion":
                constraints.append(MutualExclusionConstraint(
                    condition_a=item["condition_a"],
                    condition_b=item["condition_b"],
                    description=item.get("description", "LLM-індуковане взаємне виключення")
                ))
            elif t == "spatial":
                constraints.append(SpatialBoundingConstraint(
                    dimension_attr=item["dimension_attr"],
                    max_dimension_attr=item["max_dimension_attr"],
                    description=f"LLM-індукований допуск {item['dimension_attr']} <= {item['max_dimension_attr']}"
                ))
        return constraints

    def build_virtual_hypergraph(self, products: List[ProductNode], use_llm: bool = False) -> KnowledgeHypergraph:
        """
        Creates and returns a fully configured KnowledgeHypergraph with zero manual rules.
        """
        kg = KnowledgeHypergraph()
        for p in products:
            kg.add_product(p)
            
        induced_constraints = self.induce_contracts_from_products(products, use_llm=use_llm)
        for c in induced_constraints:
            kg.add_constraint(c)
            
        return kg
