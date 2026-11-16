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
    subject_categories: Tuple[str, ...] = ()
    object_categories: Tuple[str, ...] = ()

    def __str__(self) -> str:
        if self.kind == EQ:
            return f"eq({', '.join(self.subject)})"
        scope = ""
        if self.subject_categories or self.object_categories:
            sc = "/".join(self.subject_categories) or "*"
            oc = "/".join(self.object_categories) or "*"
            scope = f" [{sc} -> {oc}]"
        return f"le({', '.join(self.subject)} <= {', '.join(self.obj)}{scope})"

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "kind": self.kind,
            "subject": list(self.subject),
            "obj": list(self.obj),
            "headroom": round(self.headroom, 4),
        }
        if self.subject_categories:
            out["subject_categories"] = list(self.subject_categories)
        if self.object_categories:
            out["object_categories"] = list(self.object_categories)
        return out


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

    from attribute_algebra import canonical_form

    all_keys_ctx = " ".join(sorted(key_values))
    for concept in concepts:
        if concept.shape not in _SYMBOLIC_SHAPES:
            continue
        if len(group_scope(concept.keys)) < 2:
            continue
        canon_vals = {
            canonical_form(v, key=all_keys_ctx)
            for v in concept.vocabulary
        } - {None}
        if len(canon_vals) <= 1:
            continue
        keys_sorted = tuple(sorted(concept.keys))
        rules.append(StructuralRule(EQ, keys_sorted, keys_sorted))

    def group_measure_by_category(keys: Sequence[str]) -> Tuple[Dict[str, List[float]], Optional[str]]:
        by_cat: Dict[str, List[float]] = {}
        dim: Optional[str] = None
        key_set = set(keys)
        for p in products:
            for k, v in p.attributes.items():
                if k not in key_set:
                    continue
                m = _structural_measure(v)
                if m is None:
                    continue
                if dim is not None and m[1] != dim:
                    return ({}, None)
                dim = m[1]
                by_cat.setdefault(p.category, []).append(m[0])
        return (by_cat, dim)

    measurable: List[Tuple[Tuple[str, ...], str, List[float]]] = []
    for keys in groups:
        vals, dim = group_measure(keys)
        if dim and vals:
            measurable.append((keys, dim, vals))

    paired_groups: Set[Tuple[str, ...]] = set()
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
            paired_groups.add(tuple(sorted(subj)))
            paired_groups.add(tuple(sorted(obj)))
            rules.append(StructuralRule(LE, tuple(sorted(subj)), tuple(sorted(obj)), headroom))

    # Shared-key cross-category capacity dominance: when a single physical
    # concept (e.g. wattage keys reused by suppliers across CPUs, GPUs, PSUs)
    # partitions into cleanly separated consumer and provider category ranges.
    for keys, _dim, _vals in measurable:
        keys_sorted = tuple(sorted(keys))
        if keys_sorted in paired_groups:
            continue
        by_cat, _ = group_measure_by_category(keys)
        if len(by_cat) < 2:
            continue
        cats = sorted(by_cat)
        providers: List[str] = []
        consumers_by_provider: Dict[str, List[str]] = {}
        for obj_cat in cats:
            dominated = [
                sub_cat for sub_cat in cats
                if sub_cat != obj_cat
                and _structural_covers(by_cat[obj_cat], by_cat[sub_cat])
                and not _structural_covers(by_cat[sub_cat], by_cat[obj_cat])
            ]
            if dominated:
                # Only treat obj_cat as top provider if it is not itself dominated.
                is_sub = any(
                    other != obj_cat
                    and _structural_covers(by_cat[other], by_cat[obj_cat])
                    and not _structural_covers(by_cat[obj_cat], by_cat[other])
                    for other in cats
                )
                if not is_sub:
                    providers.append(obj_cat)
                    consumers_by_provider[obj_cat] = dominated
        for obj_cat in providers:
            sub_cats = tuple(sorted(consumers_by_provider[obj_cat]))
            sub_vals = [v for c in sub_cats for v in by_cat[c]]
            ceiling = max(by_cat[obj_cat])
            headroom = (max(sub_vals) / ceiling) if ceiling else 1.0
            rules.append(StructuralRule(
                LE, keys_sorted, keys_sorted, headroom,
                subject_categories=sub_cats,
                object_categories=(obj_cat,),
            ))

    return sorted(set(rules))


class ContractInductor:
    """
    Domain-free Open Dynamic Contract Inductor.

    Discovers compatibility invariants strictly from Value Algebra shapes,
    canonical value overlaps, and dimensional measurement dominance -- zero
    domain keyword/attribute/category vocabularies. Every induced rule stays
    `status="candidate"` (F12: vetoes conservatively, never vouches).
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
        nm_keys = set(non_matching_keys) if non_matching_keys else set()
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
                    subject_categories=rule.subject_categories or None,
                    object_categories=rule.object_categories or None,
                )
                c.status = "candidate"
                constraints.append(c)
        return constraints

    def induce_contracts_from_products(
        self,
        products: List[ProductNode],
        use_llm: bool = False,
        headroom_ratio: Optional[float] = None,
        max_induced: Optional[int] = None,
        non_matching_keys: Optional[Set[str]] = None,
    ) -> List[BaseConstraint]:
        """
        Extracts constraints from a collection of ProductNode instances.
        Uses unified domain-free structural induction by default (F12: candidate),
        or LLM-guided meta-typing if use_llm=True.
        `headroom_ratio` is the source/user-stated safety margin for
        capacity rules; None means the induced rule asserts the exact
        bound (F47 -- no fabricated margin).
        `non_matching_keys` may optionally pass data-derived offer keys to skip.
        """
        induced: Optional[List[BaseConstraint]] = None
        if use_llm and self.runner and self.runner.api_key:
            try:
                llm_constraints = self._induce_via_llm(products)
                if llm_constraints:
                    induced = llm_constraints
            except Exception:
                pass

        if induced is None:
            induced = self._induce_structural_constraints(
                products,
                headroom_ratio=headroom_ratio,
                non_matching_keys=non_matching_keys,
            )

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
