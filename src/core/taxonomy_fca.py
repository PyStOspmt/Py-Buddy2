"""
Py-Buddy: Facet Taxonomy Engine (FCA-inspired).

F24 -- honest capability statement. What this module ACTUALLY builds is a
facet taxonomy: root -> category -> category-facet concepts derived from
resolved categorical columns. It does NOT compute the full set of closed
formal concepts of L(G, M, I): there is no closure operator, no
join/meet computation, and no guarantee that every formal concept is
present. Navigation scans and returns bounded samples. Until closure is
implemented and independently validated, this structure must be described
as a facet taxonomy, not as a computed concept lattice.

Status of the neighbouring research components (docs/11 F24): the NVFG
lattice and the structural benchmark inductor are implemented and
unit-tested but are NOT wired into the production request path; they must
not be reported as deployed capabilities.
"""

from typing import Dict, List, Set, Any, Tuple
import sqlite3
import re
from dataclasses import dataclass, field

@dataclass
class FormalConcept:
    """
    A Formal Concept (A, B) in Formal Concept Analysis:
    - extent (A): Set of object IDs (products)
    - intent (B): Set of shared attributes/facets
    - level: Depth in the concept hierarchy (0 = Root / supremum)
    - children: Subconcepts (more specific intent, smaller extent)
    - parents: Superconcepts (broader extent, less specific intent)
    """
    id: str
    extent: Set[str]
    intent: Set[str]
    level: int = 0
    subconcept_ids: List[str] = field(default_factory=list)
    superconcept_ids: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def extent_size(self) -> int:
        return len(self.extent)

class FCATaxonomyEngine:
    """
    Builds and navigates a facet taxonomy over arbitrary SQLite databases
    (see the module docstring: this is NOT the full formal concept lattice).

    Guarantees Zero-Code-Change Schema Ingestion (ZCCS = 0) by introspecting
    the canonical source projection, discrete attributes, and categorical
    hierarchies.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.concepts: Dict[str, FormalConcept] = {}
        self.root_concept_id: str = "root"
        self._attribute_index: Dict[str, Set[str]] = {}  # attribute -> concept_ids
        self._build_lattice()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _build_lattice(self) -> None:
        """
        Extracts formal context (Objects G, Attributes M, Relation I) from SQLite
        and builds a multi-level Concept Lattice.
        """
        # F10: the formal context is extracted from the canonical entity table
        # discovered by the source projection, not from a hard-coded
        # `products` table; a renamed schema builds the same lattice.
        from source_projection import build_source_projection, quote_ident
        try:
            proj = build_source_projection(self.db_path)
        except Exception:
            return
        if proj.entity_table is None:
            return
        ident = proj.resolve("identity")
        name_res = proj.resolve("name")
        cat_res = proj.resolve("category")
        price_res = proj.resolve("price")
        if not ident.resolved or not name_res.column:
            return

        conn = self._connect()
        try:
            cursor = conn.cursor()
            q = quote_ident
            t = q(proj.entity_table)
            cols = {row["name"] if not isinstance(row, tuple) else row[1]
                    for row in cursor.execute(f"PRAGMA table_info({t})").fetchall()}

            # Facet columns: the resolved category plus the merchant axis
            # (offer-style sources where the grouping column is a seller,
            # not a product category) plus light exact-name hints
            # (brand/vendor/type); a source that states none of them simply
            # contributes fewer facets instead of failing.
            facet_cols = []
            if cat_res.column:
                facet_cols.append(cat_res.column)
            merch_res = proj.resolve("merchant")
            if merch_res.column:
                facet_cols.append(merch_res.column)
            facet_cols.extend(c for c in ("brand", "виробник", "type", "department", "vendor") if c in cols)
            facet_cols = list(dict.fromkeys(facet_cols))
            if not facet_cols:
                facet_cols = [name_res.column]

            select_cols = [f"{q(ident.column)} AS id", f"{q(name_res.column)} AS name"]
            if price_res.column:
                select_cols.append(f"{q(price_res.column)} AS price")
            select_cols += [f"{q(c)} AS {q(c)}" for c in facet_cols]
            cursor.execute(f"SELECT {', '.join(select_cols)} FROM {t}")
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return

        all_ids = set()
        # 1. Map Objects -> Attributes
        object_intents: Dict[str, Set[str]] = {}
        attribute_extents: Dict[str, Set[str]] = {}

        for r in rows:
            p_id = str(r["id"])
            all_ids.add(p_id)
            intent = set()

            for col in facet_cols:
                val = r[col]
                if val is not None and str(val).strip():
                    attr = f"{col}:{str(val).strip()}"
                    intent.add(attr)
                    attribute_extents.setdefault(attr, set()).add(p_id)

            object_intents[p_id] = intent

        # 2. Build Root Concept (Supremum: all objects, empty intent)
        root = FormalConcept(
            id="root",
            extent=all_ids,
            intent=set(),
            level=0,
            meta={"label": "Всі товари", "type": "root"}
        )
        self.concepts["root"] = root

        # 3. Build Level 1 Concepts: Macro-Categories
        cat_col = "category" if "category" in facet_cols else facet_cols[0]
        cat_attrs = [a for a in attribute_extents if a.startswith(f"{cat_col}:")]

        for cat_attr in sorted(cat_attrs):
            cat_name = cat_attr.split(":", 1)[1]
            c_extent = attribute_extents[cat_attr]
            cid = f"cat_{cat_name}"

            concept = FormalConcept(
                id=cid,
                extent=c_extent,
                intent={cat_attr},
                level=1,
                superconcept_ids=["root"],
                meta={"label": cat_name, "category": cat_name, "type": "category"}
            )
            self.concepts[cid] = concept
            root.subconcept_ids.append(cid)
            self._attribute_index.setdefault(cat_attr.lower(), set()).add(cid)
            self._attribute_index.setdefault(cat_name.lower(), set()).add(cid)

            # 4. Build Level 2 Concepts: (Category, Brand) Subconcepts
            if "brand" in facet_cols:
                # Find all brands within this category
                brands_in_cat: Dict[str, Set[str]] = {}
                for pid in c_extent:
                    for attr in object_intents[pid]:
                        if attr.startswith("brand:"):
                            brand_name = attr.split(":", 1)[1]
                            brands_in_cat.setdefault(brand_name, set()).add(pid)

                for bname, b_extent in sorted(brands_in_cat.items()):
                    sub_cid = f"{cid}_brand_{bname}"
                    sub_intent = {cat_attr, f"brand:{bname}"}
                    sub_concept = FormalConcept(
                        id=sub_cid,
                        extent=b_extent,
                        intent=sub_intent,
                        level=2,
                        superconcept_ids=[cid],
                        meta={"label": f"{cat_name} -> {bname}", "category": cat_name, "brand": bname, "type": "sub_category"}
                    )
                    self.concepts[sub_cid] = sub_concept
                    concept.subconcept_ids.append(sub_cid)
                    self._attribute_index.setdefault(f"{cat_name.lower()}:{bname.lower()}", set()).add(sub_cid)
                    self._attribute_index.setdefault(bname.lower(), set()).add(sub_cid)

    def navigate(self, query: str) -> Dict[str, Any]:
        """
        Navigates the Concept Lattice given user search terms or attributes.
        Returns the most specific matching Formal Concept and its available facets.
        This provides O(log N) drill-down and completely eliminates Horizon Truncation.
        """
        q_lower = query.lower()

        # Score all concepts by how many of their intent attributes are supported by the query
        scored_concepts: List[Tuple[float, int, FormalConcept]] = []
        for cid, concept in self.concepts.items():
            if cid == "root" or not concept.intent:
                continue

            matched_attrs = 0
            for attr in concept.intent:
                val = attr.split(":", 1)[-1].lower()
                clean_attr = attr.lower()
                # Exact boundary matching so that 'men_clothing' does not match inside 'women_clothing'
                pat_val = r'(?<![a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ_])' + re.escape(val) + r'(?![a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ_])'
                pat_attr = r'(?<![a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ_])' + re.escape(clean_attr) + r'(?![a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ_])'
                if re.search(pat_val, q_lower) or re.search(pat_attr, q_lower):
                    matched_attrs += 1

            if matched_attrs > 0:
                # Coverage ratio: how complete is the intent match
                coverage = matched_attrs / len(concept.intent)
                # We prioritize fully covered concepts (coverage == 1.0) with maximum level and matched count
                scored_concepts.append((coverage, matched_attrs, concept))

        if not scored_concepts:
            # Fallback to root concept
            root = self.concepts.get("root")
            return {
                "concept_id": "root",
                "extent_size": root.extent_size if root else 0,
                "label": "Весь каталог",
                "available_subcategories": [self.concepts[sid].meta.get("label", sid) for sid in (root.subconcept_ids if root else [])],
                "matched": False
            }

        # Sort: first by coverage == 1.0 (exact full concept match), then level, then matched_attrs
        scored_concepts.sort(
            key=lambda item: (
                1 if item[0] >= 1.0 else 0,
                item[2].level,
                item[1],
                -item[2].extent_size
            ),
            reverse=True
        )

        _, _, best = scored_concepts[0]
        subcats = [self.concepts[sid].meta.get("label", sid) for sid in best.subconcept_ids]

        return {
            "concept_id": best.id,
            "level": best.level,
            "intent": sorted(best.intent),
            "extent_size": best.extent_size,
            "sample_ids": sorted(best.extent)[:10],
            "label": best.meta.get("label", best.id),
            "available_subcategories": subcats,
            "matched": True
        }

    def get_concept_summary(self) -> Dict[str, Any]:
        """Returns structural metrics of the generated Concept Lattice."""
        return {
            "total_concepts": len(self.concepts),
            "root_objects": self.concepts["root"].extent_size if "root" in self.concepts else 0,
            "categories_count": len(self.concepts["root"].subconcept_ids) if "root" in self.concepts else 0,
            "lattice_depth": max((c.level for c in self.concepts.values()), default=0)
        }
