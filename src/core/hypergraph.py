"""
Py-Buddy: Knowledge Hypergraph Engine
Handles n-ary compositional compatibility verification for e-commerce catalogs.
"""

import hashlib
from types import MappingProxyType
from typing import Set, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from attribute_algebra import Shape, UNIT_TO_BASE, base_of, canonical_form, spec_of


def _canonical_eq_value(raw_val: Any, key_ctx: Optional[str] = None) -> str:
    """Project an attribute value canonically via Value Algebra; fall back verbatim for free text."""
    text = str(raw_val)
    canon = canonical_form(text, key=key_ctx)
    return canon if canon is not None else text.strip()


def _extract_measure_value(raw_val: Any) -> Optional[Tuple[float, Optional[str]]]:
    """Parse a bare numeric or dimensional quantity (e.g. '120W', '19.0\"', 'R18') to canonical units."""
    if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
        return float(raw_val), None
    s = str(raw_val).strip()
    if not s:
        return None
    try:
        return float(s), None
    except ValueError:
        pass
    try:
        sp = spec_of(s)
        if sp.shape is Shape.MEASURE and sp.dimension and sp.numbers:
            base = base_of(sp.dimension, sp.unit)
            factor = UNIT_TO_BASE.get(sp.unit or base, 1.0)
            return float(sp.numbers[0]) * factor, sp.dimension
        if sp.numbers:
            return float(sp.numbers[0]), sp.dimension
    except Exception:
        pass
    return None

@dataclass(frozen=True)
class ProductNode:
    """
    One catalogue row.

    `attributes` is wrapped read-only (`MappingProxyType`) on construction so no
    downstream session can mutate shared node attributes in place; updates require
    constructing a new node.

    `hash=False` on `attributes` keeps catalogue-scale sets keyed cleanly by node identity.
    """
    id: str
    name: str
    category: str
    # Absent prices stay None so unpriced rows never appear free (price=0.0).
    price: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict, hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.attributes, MappingProxyType):
            object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

@dataclass(frozen=True)
class HyperEdge:
    id: str
    name: str
    node_ids: Set[str]
    weight: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)

class BaseConstraint(ABC):
    """
    Abstract Universal Invariant for Virtual Hypergraph evaluation.
    Evaluates n-ary compatibility over a set of product nodes dynamically.

    F12: a constraint carries an epistemic status. "verified" means the rule
    was authored or independently validated and may vouch for compatibility;
    "candidate" means the rule was induced from observed correlations and may
    signal a conflict, but its pass is not evidence -- an unverified rule
    that happens to hold says nothing about compatibility.
    """
    status: str = "verified"

    def rule_id(self) -> str:
        """Stable deterministic identifier for admin confirmation & UI inspection."""
        parts = [type(self).__name__]
        for k in sorted(vars(self)):
            if k.startswith("_") or k in ("status", "description"):
                continue
            v = getattr(self, k)
            if isinstance(v, (list, tuple, set)):
                norm_v = ",".join(sorted(str(x) for x in v)) if isinstance(v, set) else ",".join(str(x) for x in v)
            elif isinstance(v, dict):
                norm_v = ",".join(f"{dk}={v[dk]}" for dk in sorted(v))
            else:
                norm_v = str(v)
            parts.append(f"{k}={norm_v}")
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
        return f"{type(self).__name__.lower()[:6]}_{digest}"

    @abstractmethod
    def validate(self, nodes: List[ProductNode]) -> Tuple[bool, Optional[str]]:
        """Returns (is_valid, optional_conflict_explanation)."""

    def applies_to(self, nodes: List[ProductNode]) -> bool:
        """
        F15: does this rule actually constrain the given set?

        A rule that cannot apply to a set (e.g. a socket rule over a set with
        no motherboard) passes vacuously; counting that pass as evidence
        reports unverified sets as "compatible". The built-in rule classes
        override this with honest applicability checks. The default stays
        conservative (True) so custom rules keep their legacy voice — but
        every rule shipped in this module must answer honestly.
        """
        return True

    def coverage_gaps(self, nodes: List[ProductNode]) -> List[str]:
        """
        R6: which in-scope members of the set this rule could not see.

        A rule may apply to a set yet stay blind to some of its members
        (a scoped product that lacks the key attribute). Such a rule must
        not vouch for compatibility of the whole set -- its pass covers
        only what it saw. The default reports no gaps so legacy/custom
        rules keep their behaviour.
        """
        return []

@dataclass
class EquivalenceConstraint(BaseConstraint):
    """
    Meta-Class 1: Point Equivalence / Matching Key (e.g. Socket, PCD, Size, Thread).
    Requires matching attribute values among products that possess this attribute.
    Supports synonymous `alias_attrs` induced by Value Algebra (e.g. `socket` & `Сокет`).
    """
    attr_name: str
    categories: Optional[List[str]] = None
    description: str = ""
    alias_attrs: Tuple[str, ...] = ()

    def _all_keys(self) -> Tuple[str, ...]:
        seen = []
        for k in (self.attr_name, *self.alias_attrs):
            if k and k not in seen:
                seen.append(k)
        return tuple(seen)

    def _key_ctx(self) -> str:
        return " ".join(self._all_keys())

    def _canonical_value(self, raw_val: Any) -> str:
        return _canonical_eq_value(raw_val, self._key_ctx())

    def _node_key_value(self, node: ProductNode) -> Optional[Tuple[str, Any]]:
        for k in self._all_keys():
            if k in node.attributes:
                return k, node.attributes[k]
        return None

    def _matching_nodes(self, nodes: List[ProductNode]) -> List[Tuple[ProductNode, str, Any]]:
        out = []
        for n in nodes:
            if self.categories is not None and n.category not in self.categories:
                continue
            kv = self._node_key_value(n)
            if kv is not None:
                out.append((n, kv[0], kv[1]))
        return out

    def applies_to(self, nodes: List[ProductNode]) -> bool:
        # The rule only speaks when at least two products carry the key.
        return len(self._matching_nodes(nodes)) >= 2

    def validate(self, nodes: List[ProductNode]) -> Tuple[bool, Optional[str]]:
        matching_nodes = self._matching_nodes(nodes)
        if len(matching_nodes) < 2:
            return True, None

        _, _, base_val = matching_nodes[0]
        base_node = matching_nodes[0][0]
        base_canon = self._canonical_value(base_val)
        for node, _, val in matching_nodes[1:]:
            if self._canonical_value(val) != base_canon:
                desc = self.description or f"Невідповідність ключа '{self.attr_name}'"
                return False, f"{desc}: '{base_node.name}' ({base_val}) несумісний з '{node.name}' ({val})"
        return True, None

    def coverage_gaps(self, nodes: List[ProductNode]) -> List[str]:
        # A member inside the rule's scope that carries no key is UNSEEN:
        # "socket rule passes over two AM5 boards" says nothing about the
        # third board whose socket was never read (R6 review: absence was
        # silently counted as compatible).
        return [
            f"'{n.name}' ({n.category}): атрибут '{self.attr_name}' відсутній — сумісність не перевірена"
            for n in nodes
            if (self.categories is None or n.category in self.categories)
            and self._node_key_value(n) is None
        ]

@dataclass
class CumulativeCapacityConstraint(BaseConstraint):
    """
    Meta-Class 2: Cumulative Capacity / Inequality Constraint (e.g. TDP vs PSU Wattage, Weight vs Payload).
    Ensures that the sum of demands does not exceed the provider's capacity.

    A build is carried by ONE provider per category, so capacity is the
    MAXIMUM provider's value, not the sum: two 500 W power supplies do
    not give the machine 1000 W -- the machine still has one socket for
    power. The previous version summed them, which is the same wrong
    physics in a new shape (R6 review). Demand still sums across
    categories, because that is what a single provider actually feeds.

    `provider_mode="additive"` opts back into summing for resource kinds
    whose units genuinely combine (e.g. two 8 GB RAM sticks are 16 GB of
    installed memory). The default is "single" because that is the
    fail-closed reading: crediting spare parts as installed capacity is
    the dangerous direction of error, not the safe one.
    """
    demand_attr: str
    capacity_attr: str
    headroom_ratio: float = 1.0  # 1.0 means demand <= capacity; 0.8 means 20% safety headroom
    provider_mode: str = "single"  # "single": max provider; "additive": providers sum
    description: str = ""
    demand_attrs: Tuple[str, ...] = ()
    capacity_attrs: Tuple[str, ...] = ()
    subject_categories: Optional[Tuple[str, ...]] = None
    object_categories: Optional[Tuple[str, ...]] = None

    def _demand_keys(self) -> Tuple[str, ...]:
        seen = []
        for k in (self.demand_attr, *self.demand_attrs):
            if k and k not in seen:
                seen.append(k)
        return tuple(seen)

    def _capacity_keys(self) -> Tuple[str, ...]:
        seen = []
        for k in (self.capacity_attr, *self.capacity_attrs):
            if k and k not in seen:
                seen.append(k)
        return tuple(seen)

    @staticmethod
    def _extract_measure(raw_val: Any) -> Optional[Tuple[float, Optional[str]]]:
        return _extract_measure_value(raw_val)

    def _is_demand_node(self, node: ProductNode) -> bool:
        if self.subject_categories is not None and node.category not in self.subject_categories:
            return False
        return any(k in node.attributes for k in self._demand_keys())

    def _is_capacity_node(self, node: ProductNode) -> bool:
        if self.object_categories is not None and node.category not in self.object_categories:
            return False
        return any(k in node.attributes for k in self._capacity_keys())

    def applies_to(self, nodes: List[ProductNode]) -> bool:
        # The inequality only speaks when there is both something to carry
        # and something that carries it.
        return (any(self._is_capacity_node(n) for n in nodes)
                and any(self._is_demand_node(n) for n in nodes))

    def coverage_gaps(self, nodes: List[ProductNode]) -> List[str]:
        # Members carrying neither role were never weighed: the inequality
        # is silent about them, and silence must not read as a pass.
        return [
            f"'{n.name}' ({n.category}): ні '{self.demand_attr}', ні '{self.capacity_attr}' — поза перевіркою місткості"
            for n in nodes
            if not self._is_demand_node(n) and not self._is_capacity_node(n)
        ]

    def validate(self, nodes: List[ProductNode]) -> Tuple[bool, Optional[str]]:
        total_demand = 0.0
        has_demand = False
        capacity_by_category: Dict[str, float] = {}
        d_keys = self._demand_keys()
        c_keys = self._capacity_keys()
        for n in nodes:
            if self.subject_categories is None or n.category in self.subject_categories:
                for dk in d_keys:
                    if dk in n.attributes:
                        m = self._extract_measure(n.attributes[dk])
                        if m is not None:
                            total_demand += m[0]
                            has_demand = True
                        break
            if self.object_categories is None or n.category in self.object_categories:
                for ck in c_keys:
                    if ck in n.attributes:
                        m = self._extract_measure(n.attributes[ck])
                        if m is not None:
                            cap = m[0]
                            if self.provider_mode == "additive":
                                capacity_by_category[n.category] = (
                                    capacity_by_category.get(n.category, 0.0) + cap)
                            else:
                                capacity_by_category[n.category] = max(
                                    capacity_by_category.get(n.category, 0.0), cap)
                        break

        if not capacity_by_category or not has_demand or total_demand == 0.0:
            return True, None

        desc = self.description or f"Дефіцит місткості '{self.capacity_attr}'"
        for category, cap_val in sorted(capacity_by_category.items()):
            if total_demand > cap_val * self.headroom_ratio:
                return False, (f"{desc}: сумарна потреба ({total_demand:g}) "
                               f"перевищує місткість '{category}' ({cap_val:g})")
        return True, None

@dataclass
class MutualExclusionConstraint(BaseConstraint):
    """
    Meta-Class 3: Mutual Exclusion / Conflict Set.
    Prohibits co-existence of conflicting product attribute combinations.
    """
    condition_a: Dict[str, Any]
    condition_b: Dict[str, Any]
    description: str = ""

    def _matches_any(self, nodes: List[ProductNode], cond: Dict[str, Any]) -> bool:
        return any(
            all(n.attributes.get(k) == v for k, v in cond.items()) for n in nodes
        )

    def applies_to(self, nodes: List[ProductNode]) -> bool:
        # The prohibition only speaks when at least one side of the conflict
        # pair is present in the set.
        return (self._matches_any(nodes, self.condition_a)
                or self._matches_any(nodes, self.condition_b))

    def validate(self, nodes: List[ProductNode]) -> Tuple[bool, Optional[str]]:
        has_a = self._matches_any(nodes, self.condition_a)
        has_b = self._matches_any(nodes, self.condition_b)
        
        if has_a and has_b:
            desc = self.description or "Виявлено взаємовиключні позиції в одному комплекті."
            return False, f"Конфлікт взаємного виключення: {desc}"
        return True, None

@dataclass
class SpatialBoundingConstraint(BaseConstraint):
    """
    Meta-Class 4: Spatial / Tolerance Bounding (e.g. GPU Length vs Case Max Length, Rim vs Arch).
    Ensures dimension <= max_dimension using canonical dimensional units when present.
    """
    dimension_attr: str
    max_dimension_attr: str
    description: str = ""
    dimension_attrs: Tuple[str, ...] = ()
    max_dimension_attrs: Tuple[str, ...] = ()
    subject_categories: Optional[Tuple[str, ...]] = None
    object_categories: Optional[Tuple[str, ...]] = None

    def _dim_keys(self) -> Tuple[str, ...]:
        seen = []
        for k in (self.dimension_attr, *self.dimension_attrs):
            if k and k not in seen:
                seen.append(k)
        return tuple(seen)

    def _max_keys(self) -> Tuple[str, ...]:
        seen = []
        for k in (self.max_dimension_attr, *self.max_dimension_attrs):
            if k and k not in seen:
                seen.append(k)
        return tuple(seen)

    @staticmethod
    def _first_measure(node: ProductNode, keys: Tuple[str, ...]) -> Optional[float]:
        for k in keys:
            if k in node.attributes:
                m = _extract_measure_value(node.attributes[k])
                if m is not None:
                    return m[0]
        return None

    def applies_to(self, nodes: List[ProductNode]) -> bool:
        d_keys, m_keys = self._dim_keys(), self._max_keys()
        has_item = any(
            (self.subject_categories is None or n.category in self.subject_categories)
            and any(k in n.attributes for k in d_keys)
            for n in nodes
        )
        has_container = any(
            (self.object_categories is None or n.category in self.object_categories)
            and any(k in n.attributes for k in m_keys)
            for n in nodes
        )
        return has_item and has_container

    def validate(self, nodes: List[ProductNode]) -> Tuple[bool, Optional[str]]:
        d_keys, m_keys = self._dim_keys(), self._max_keys()
        items = [
            n for n in nodes
            if (self.subject_categories is None or n.category in self.subject_categories)
            and any(k in n.attributes for k in d_keys)
        ]
        containers = [
            n for n in nodes
            if (self.object_categories is None or n.category in self.object_categories)
            and any(k in n.attributes for k in m_keys)
        ]

        for item in items:
            dim_val = self._first_measure(item, d_keys)
            if dim_val is None:
                continue
            for container in containers:
                max_val = self._first_measure(container, m_keys)
                if max_val is None:
                    continue
                if dim_val > max_val:
                    desc = self.description or f"Габаритний допуск '{self.dimension_attr}'"
                    return False, f"{desc}: габарит '{item.name}' ({dim_val:g}) перевищує допуск '{container.name}' ({max_val:g})"
        return True, None

class KnowledgeHypergraph:
    """
    Hybrid Neuro-Symbolic Hypergraph Engine.
    Supports both:
    1. Virtual Implicit Hypergraph: Dynamic evaluation of n-ary Universal Constraints (Zero-Config).
    2. Static Explicit Hyperedges: Legacy enumerated sub-graphs.
    """
    def __init__(self):
        self.nodes: Dict[str, ProductNode] = {}
        self.hyperedges: Dict[str, HyperEdge] = {}
        self._node_to_edges: Dict[str, Set[str]] = {}
        self.constraints: List[BaseConstraint] = []
        # Two indices, both maintained in `add_product` and both earning
        # their keep in `find_compatible_additions`. Without them that method
        # validated every one of 10 000 rows against every constraint, and
        # the obvious fix -- "add a category filter" -- was already a
        # full-table scan wearing a filter.
        self._by_category: Dict[str, Set[str]] = {}
        self._by_attribute_value: Dict[Tuple[str, str], Set[str]] = {}

    def add_product(self, product: ProductNode) -> None:
        # F16: replacing a node must not leave its stale memberships behind.
        # The previous version only added, so a re-registered product stayed
        # in its old category index and old attribute-value buckets forever.
        old = self.nodes.get(product.id)
        if old is not None:
            old_bucket = self._by_category.get(old.category)
            if old_bucket is not None:
                old_bucket.discard(product.id)
            for key, value in old.attributes.items():
                value_bucket = self._by_attribute_value.get((key, str(value)))
                if value_bucket is not None:
                    value_bucket.discard(product.id)
        self.nodes[product.id] = product
        self._node_to_edges.setdefault(product.id, set())
        self._by_category.setdefault(product.category, set()).add(product.id)
        for key, value in product.attributes.items():
            self._by_attribute_value.setdefault((key, str(value)), set()).add(product.id)

    @staticmethod
    def _conflicting() -> str:
        """Sentinel: a scope-committed key already disagrees within the basket."""
        return "__conflict__"

    def _equivalence_commitments(
        self, current_selection: Set[str]
    ) -> List[Tuple["EquivalenceConstraint", Dict[str, str]]]:
        """
        F16: the only provably necessary prefilter is equality on the keys
        an EquivalenceConstraint actually compares, taken from selection
        members inside that constraint's category scope.

        The previous filter also treated spatial dimension attributes as
        equality commitments and ignored constraint scope entirely, so a
        40-unit item was excluded from a basket holding a 30-unit item even
        though both fit the same 100-unit bound (P22), and an out-of-scope
        node's attribute value vetoed in-scope candidates.
        """
        commitments: List[Tuple[EquivalenceConstraint, Dict[str, str]]] = []
        for c in self.constraints:
            if not isinstance(c, EquivalenceConstraint):
                continue
            if c.status != "verified":
                # F12: an induced correlation must not silently veto
                # candidates either.
                continue
            scope = set(c.categories) if c.categories else None
            committed: Dict[str, str] = {}
            for pid in current_selection:
                node = self.nodes.get(pid)
                if node is None:
                    continue
                if scope is not None and node.category not in scope:
                    continue
                kv = c._node_key_value(node)
                if kv is not None:
                    _, val = kv
                    value = c._canonical_value(val)
                    for ak in c._all_keys():
                        previous = committed.setdefault(ak, value)
                        if previous != value:
                            committed[ak] = self._conflicting()
            commitments.append((c, committed))
        return commitments

    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Registers a dynamic mathematical invariant (Virtual Hypergraph)."""
        self.constraints.append(constraint)

    def set_constraint_status(self, rule_id: str, new_status: str) -> Optional[BaseConstraint]:
        """Promote or demote a constraint by its canonical rule_id ('verified' | 'candidate')."""
        if new_status not in ("verified", "candidate"):
            raise ValueError(f"Unsupported constraint status: {new_status!r}")
        for c in self.constraints:
            if c.rule_id() == rule_id:
                c.status = new_status
                return c
        return None

    def add_hyperedge(self, edge: HyperEdge) -> None:
        """Registers an explicit static hyperedge (Legacy)."""
        for nid in edge.node_ids:
            if nid not in self.nodes:
                raise ValueError(f"Node {nid} referenced in hyperedge {edge.id} does not exist in graph.")
        self.hyperedges[edge.id] = edge
        for nid in edge.node_ids:
            self._node_to_edges[nid].add(edge.id)

    def is_compatible(self, product_ids: Set[str]) -> bool:
        """
        Predicate Compat(S):
        - If verified dynamic constraints are registered, evaluates them all.
        - Otherwise, checks if product_ids is a subset of at least one static hyperedge.

        F12: candidate (induced, unverified) constraints do not flip this
        predicate -- a correlation that happens to hold is not evidence of
        compatibility, and this predicate's contract is the permissive retail
        basket default, not a verification claim.
        """
        if not product_ids:
            return True
        for pid in product_ids:
            if pid not in self.nodes:
                return False

        # Virtual Hypergraph dynamic evaluation
        active_nodes = [self.nodes[pid] for pid in product_ids]
        verified = [c for c in self.constraints if c.status == "verified"]
        if verified:
            for c in verified:
                valid, _ = c.validate(active_nodes)
                if not valid:
                    return False
            return True
        # F12: candidate (induced, unverified) rules may conservatively veto
        # -- a false "incompatible" is annoying, a false "compatible" is
        # harmful -- but they can never vouch.
        for c in self.constraints:
            if c.status == "verified":
                continue
            if c.applies_to(active_nodes):
                valid, _ = c.validate(active_nodes)
                if not valid:
                    return False

        # Fallback to static explicit hyperedge evaluation if bundles exist
        if self.hyperedges:
            first_pid = next(iter(product_ids))
            candidate_edges = self._node_to_edges.get(first_pid, set())
            for edge_id in candidate_edges:
                edge = self.hyperedges[edge_id]
                if product_ids.issubset(edge.node_ids):
                    return True
            return False

        # Domain-Agnostic Basket Default (ZCCS = 0):
        # In general consumer catalogs (food, groceries, marketplace retail, books),
        # where no physical assembly constraints or static hyperedges exist,
        # products in a cart are compatible by default for joint checkout.
        return True

    def compatibility_verdict(self, product_ids: Set[str]) -> Dict[str, Any]:
        """
        F15: compatibility as a verdict over evaluated evidence, with
        "unknown" distinct from "compatible".

        - "compatible": every id resolved and every rule that actually
          applies to the set (or a static hyperedge) validated it.
        - "incompatible": an applicable rule or a hyperedge rejected the set.
        - "unknown": nothing was evaluated — no id resolved, no rule applied
          to the set, or this graph carries neither constraints nor
          hyperedges. A rule that passes vacuously (its subject attributes
          are absent from the set) is not evidence: the live run of
          2026-10-04 showed a CPU+PSU pair reported "compatible" with
          evaluated_rules=2 although neither registered rule could speak
          about that pair. The retail basket default (ZCCS) is a
          joint-checkout assumption, not evidence of physical or semantic
          compatibility, so it is never reported here as "compatible".
          `is_compatible` keeps the legacy basket semantics for callers that
          explicitly want them.
        """
        unresolved = sorted(pid for pid in product_ids if pid not in self.nodes)
        resolved = [pid for pid in product_ids if pid in self.nodes]
        base = {"unresolved_ids": unresolved}
        if not resolved:
            return dict(base, verdict="unknown", evaluated_rules=0, conflicts=[],
                        candidate_conflicts=[])
        if self.constraints:
            active_nodes = [self.nodes[pid] for pid in resolved]
            applied = [c for c in self.constraints if c.applies_to(active_nodes)]
            verified = [c for c in applied if c.status == "verified"]
            candidates = [c for c in applied if c.status != "verified"]
            # F12: candidate (induced) rules may signal, never vouch. Their
            # conflicts are reported separately and never become the verdict.
            candidate_conflicts: List[str] = []
            for c in candidates:
                valid, reason = c.validate(active_nodes)
                if not valid:
                    candidate_conflicts.append(reason or repr(c))
            if verified:
                conflicts: List[str] = []
                # R6: a rule that was blind to in-scope members cannot
                # vouch for the whole set. Its observed conflicts still
                # count -- a violation on the seen part is real evidence --
                # but its silence on unseen members is not a pass.
                gaps: List[str] = []
                vouching: List[BaseConstraint] = []
                for c in verified:
                    g = c.coverage_gaps(active_nodes)
                    if g:
                        gaps.extend(g)
                    else:
                        vouching.append(c)
                    valid, reason = c.validate(active_nodes)
                    if not valid:
                        conflicts.append(reason or repr(c))
                if conflicts:
                    return dict(base, verdict="incompatible",
                                evaluated_rules=len(verified),
                                conflicts=conflicts,
                                coverage_gaps=gaps,
                                candidate_conflicts=candidate_conflicts)
                if vouching:
                    return dict(base, verdict="compatible",
                                evaluated_rules=len(verified), conflicts=[],
                                coverage_gaps=gaps,
                                candidate_conflicts=candidate_conflicts)
                if gaps:
                    # Every applicable verified rule was blind somewhere:
                    # nothing vouched, so the honest verdict is unknown.
                    return dict(base, verdict="unknown",
                                evaluated_rules=len(verified), conflicts=[],
                                coverage_gaps=gaps,
                                candidate_conflicts=candidate_conflicts)
            if candidate_conflicts:
                # F12/F15: a distinct, honest state -- an unverified induced
                # rule signals a conflict. Conservative enough to act on, but
                # never reported as verified physics.
                return dict(base, verdict="candidate_conflict",
                            evaluated_rules=0, conflicts=[],
                            candidate_conflicts=candidate_conflicts)
            # No verified rule actually applied: registered-but-inapplicable
            # or unverified rules are not evidence, so fall through instead of
            # vouching.
        if self.hyperedges:
            resolved_set = set(resolved)
            candidate_edges = self._node_to_edges.get(resolved[0], set())
            for edge_id in candidate_edges:
                if resolved_set.issubset(self.hyperedges[edge_id].node_ids):
                    return dict(base, verdict="compatible",
                                evaluated_rules=len(candidate_edges), conflicts=[],
                                candidate_conflicts=[])
            return dict(base, verdict="incompatible",
                        evaluated_rules=len(candidate_edges),
                        conflicts=["Немає статичного гіперребра, що покриває цей набір товарів."],
                        candidate_conflicts=[])
        return dict(base, verdict="unknown", evaluated_rules=0, conflicts=[],
                    candidate_conflicts=candidate_conflicts
                    if self.constraints else [])

    def _candidate_ids(self, target_category: Optional[str]) -> List[str]:
        """Rows worth testing, taken from the category index when there is one."""
        if target_category:
            return sorted(self._by_category.get(target_category, ()))
        return sorted(self.nodes)

    def find_compatible_additions(self, current_selection: Set[str], target_category: Optional[str] = None) -> List[ProductNode]:
        """
        Finds all products in the catalog that can be added to current_selection while maintaining full compatibility.

        F16: the only prefilter applied here is equality on the keys an
        EquivalenceConstraint compares, within that constraint's own category
        scope. Candidates that lack the key are kept (absence is not a
        violation); spatial, capacity and mutual-exclusion predicates are
        left to `is_compatible`, because no cheap necessary predicate exists
        for them that cannot silently remove a reachable result.
        """
        if not current_selection:
            return [self.nodes[pid] for pid in self._candidate_ids(target_category)]

        # If using Virtual Hypergraph: test candidates dynamically
        if self.constraints:
            commitments = self._equivalence_commitments(current_selection)
            # An existing conflict inside a rule's scope cannot be repaired
            # by adding a node: every superset stays incompatible.
            for _c, committed in commitments:
                if any(value == self._conflicting() for value in committed.values()):
                    return []
            valid_additions = []
            for pid in self._candidate_ids(target_category):
                if pid in current_selection:
                    continue
                node = self.nodes[pid]
                excluded = False
                for c, committed in commitments:
                    if c.categories is not None and node.category not in c.categories:
                        continue      # outside the rule's scope: it says nothing
                    for key, value in committed.items():
                        if key in node.attributes and c._canonical_value(node.attributes[key]) != value:
                            excluded = True
                            break
                    if excluded:
                        break
                if excluded:
                    continue          # a matching key already rules it out
                if self.is_compatible(current_selection | {pid}):
                    valid_additions.append(node)
            return valid_additions

        # Fallback to static hyperedge intersection
        first_pid = next(iter(current_selection))
        candidate_edge_ids = self._node_to_edges.get(first_pid, set())
        valid_addition_ids = set()
        for edge_id in candidate_edge_ids:
            edge = self.hyperedges[edge_id]
            if current_selection.issubset(edge.node_ids):
                valid_addition_ids.update(edge.node_ids - current_selection)

        additions = [self.nodes[nid] for nid in valid_addition_ids]
        if target_category:
            additions = [p for p in additions if p.category == target_category]
        return additions

    def find_conflicts(self, product_ids: Set[str]) -> List[str]:
        """
        Identifies and explains pairwise or group incompatibilities in product_ids.
        Uses exact mathematical explanations when dynamic constraints are active.
        """
        conflicts = []
        if self.constraints:
            active_nodes = [self.nodes[pid] for pid in product_ids if pid in self.nodes]
            for c in self.constraints:
                valid, explanation = c.validate(active_nodes)
                if not valid and explanation:
                    conflicts.append(explanation)
            return conflicts

        # Static fallback only if explicit bundle hyperedges exist
        if self.hyperedges:
            pids = list(product_ids)
            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    pair = {pids[i], pids[j]}
                    if not self.is_compatible(pair):
                        n1 = self.nodes.get(pids[i])
                        n2 = self.nodes.get(pids[j])
                        t1 = n1.name if n1 else pids[i]
                        t2 = n2.name if n2 else pids[j]
                        conflicts.append(f"Несумісність: '{t1}' та '{t2}' не мають спільного гіперребра сумісності.")
            if not conflicts and len(product_ids) > 2:
                conflicts.append(f"Комплект {product_ids} у такій комбінації не підтримується жодним гіперребром.")
            return conflicts

        # Open basket domain with no assembly constraints has zero conflicts
        return []

