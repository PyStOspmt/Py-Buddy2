"""
Py-Buddy: Domain Catalog Registry
Modular repository for multi-domain product nodes and universal constraints.
Decouples catalog initialization from the server transport layer.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import os

from hypergraph import (
    KnowledgeHypergraph,
    ProductNode
)
from source_projection import quote_ident

# Single repository-root anchor (docs/13 C2): this module lives in src/core/,
# so the repository root is three levels up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# T6: bound induced-rule cardinality so attribute-dense sources cannot
# explode the constraint set unboundedly. Truncation is recorded on
# `kg.induction_report` and surfaced via /api/domains -- never silent.
MAX_INDUCED_CONSTRAINTS = 100

class DomainCatalogRegistry:
    """
    Manages multi-domain catalog configurations and registers their respective
    Universal Meta-Constraints for Virtual Hypergraph dynamic evaluation.
    """
    @classmethod
    def create_pc_catalog(cls, hypergraph_json_path: Optional[str] = None) -> KnowledgeHypergraph:
        """Initializes the PC Hardware catalog and registers its contracts dynamically."""
        db_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'catalog.db'))
        if os.path.exists(db_path):
            return cls.create_database_catalog(db_path, seed_limit=120)

        kg = KnowledgeHypergraph()
        if hypergraph_json_path and os.path.exists(hypergraph_json_path):
            with open(hypergraph_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for n in data.get("nodes", []):
                attrs = dict(n.get("attributes", {}))
                kg.add_product(ProductNode(n["id"], n["name"], n["category"], n["price"], attrs))

        from contract_inductor import ContractInductor
        inductor = ContractInductor(max_induced=MAX_INDUCED_CONSTRAINTS)
        for c in inductor.induce_contracts_from_products(list(kg.nodes.values())):
            kg.add_constraint(c)
        kg.induction_report = inductor.last_induction_report
        return kg

    @classmethod
    def create_auto_catalog(cls, db_path: Optional[str] = None) -> KnowledgeHypergraph:
        """Initializes the Auto Parts catalog dynamically from SQLite with zero hardcoded domain constraints."""
        default_db = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'auto_parts.db'))
        target_db = db_path or default_db

        if os.path.exists(target_db):
            return cls.create_database_catalog(target_db, seed_limit=100)
        return KnowledgeHypergraph()

    @classmethod
    def create_marketplace_catalog(cls, db_path: str) -> KnowledgeHypergraph:
        """
        Initializes the Rozetka Marketplace Hypergraph dynamically from SQLite.
        Preloads a diverse canonical seed set for interactive exploration,
        and provides on-demand dynamic Lazy Projection from the 10,000+ item SQLite DB
        with automated mathematical constraint induction.
        """
        if os.path.exists(db_path):
            return cls.create_database_catalog(db_path, seed_limit=120)
        return KnowledgeHypergraph()

    @staticmethod
    def _fk_category_join(proj, cat_res) -> Optional[Tuple[str, str, str, str]]:
        """When entity has no direct category column, inspect unique FK dimension label."""
        if cat_res.resolved or not proj.entity_table:
            return None
        tbl = proj.snapshot.tables.get(proj.entity_table)
        if tbl is None:
            return None
        candidates: List[Tuple[str, str, str, str]] = []
        for col in tbl.columns.values():
            if not getattr(col, "is_foreign_key", False):
                continue
            tgt_tbl_name = getattr(col, "fk_target_table", None)
            tgt_col_name = getattr(col, "fk_target_column", None)
            if not tgt_tbl_name or not tgt_col_name or tgt_tbl_name == tbl.name:
                continue
            tgt_tbl = proj.snapshot.tables.get(tgt_tbl_name)
            if tgt_tbl is None or tgt_tbl.row_count < 2 or tgt_tbl.row_count > 100:
                continue
            pk_set = set(tgt_tbl.primary_keys)
            lbl_cols = [
                tc.name
                for tc in tgt_tbl.columns.values()
                if tc.name not in pk_set
                and not any(tok in (tc.data_type or "").upper() for tok in ("INT", "FLOAT", "REAL", "NUM", "DOUBLE", "BLOB"))
            ]
            if len(lbl_cols) == 1:
                candidates.append((col.name, tgt_tbl_name, tgt_col_name, lbl_cols[0]))
        return candidates[0] if len(candidates) == 1 else None

    @staticmethod
    def _projection_query_parts(proj, name_res, cat_res, price_res, ident) -> Tuple[str, str, bool]:
        """Build aliased SELECT list, optional FK category JOIN clause, and category presence."""
        q = quote_ident
        name_sel = f"e.{q(name_res.column)} AS name" if name_res.column else f"e.{q(ident.column)} AS name"
        price_sel = f"e.{q(price_res.column)} AS price" if price_res.column else "NULL AS price"
        join_sql = ""
        if cat_res.column:
            cat_sel = f"e.{q(cat_res.column)} AS category"
            has_cat = True
        else:
            fk_cat = DomainCatalogRegistry._fk_category_join(proj, cat_res)
            if fk_cat is not None:
                fk_col, dim_tbl, dim_pk, dim_lbl = fk_cat
                join_sql = f" LEFT JOIN {q(dim_tbl)} dim ON e.{q(fk_col)} = dim.{q(dim_pk)}"
                cat_sel = f"dim.{q(dim_lbl)} AS category"
                has_cat = True
            else:
                cat_sel = "NULL AS category"
                has_cat = False
        return f"e.{q(ident.column)} AS id, {name_sel}, {cat_sel}, {price_sel}", join_sql, has_cat

    @staticmethod
    def _projection_select(proj, name_res, cat_res, price_res, ident) -> str:
        """Build the projection SELECT list from resolved roles (F10)."""
        q = quote_ident
        name_sel = f"{q(name_res.column)} AS name" if name_res.column else f"{q(ident.column)} AS name"
        cat_sel = f"{q(cat_res.column)} AS category" if cat_res.column else "NULL AS category"
        price_sel = f"{q(price_res.column)} AS price" if price_res.column else "NULL AS price"
        return f"{q(ident.column)} AS id, {name_sel}, {cat_sel}, {price_sel}"

    @staticmethod
    def _collect_eav_meta(cursor, proj, entity_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Pull EAV metadata for the given entity ids from every detected EAV table."""
        meta_by_prod: Dict[str, Dict[str, Any]] = {}
        if not entity_ids:
            return meta_by_prod
        q = quote_ident
        placeholders = ",".join("?" for _ in entity_ids)
        for eav_name, meta in proj.eav_tables.items():
            try:
                cursor.execute(
                    f"SELECT {q(meta['entity_fk'])}, {q(meta['key_column'])}, {q(meta['value_column'])} "
                    f"FROM {q(eav_name)} WHERE {q(meta['entity_fk'])} IN ({placeholders})",
                    entity_ids,
                )
                for eid, k, v in cursor.fetchall():
                    if k is None:
                        continue
                    meta_by_prod.setdefault(str(eid), {})[str(k)] = v
            except Exception:
                # An EAV hypothesis that does not join simply contributes nothing.
                continue
        return meta_by_prod

    @staticmethod
    def _non_intrinsic_eav_keys(cursor, proj) -> set:
        """
        Structural offer/seller-axis detection (zero domain vocabulary):
        an EAV key whose canonical Value-Algebra meanings disagree across
        listings for the SAME named product entity cannot be an intrinsic
        compatibility attribute of that product (e.g. per-offer seller/promo tags),
        whereas spelling variants (`"AM4"` vs `"Socket AM4"`, `"65W"` vs `"65 Вт"`)
        fold to the same canonical value and are preserved cleanly.
        """
        from attribute_algebra import canonical_form

        non_intrinsic: set = set()
        table = proj.entity_table
        ident = proj.resolve("identity")
        name_res = proj.resolve("name")
        if not table or not ident.resolved or not name_res.column:
            return non_intrinsic
        q = quote_ident
        t = q(table)
        id_c = q(ident.column)
        nm_c = q(name_res.column)
        for eav_name, meta in proj.eav_tables.items():
            et = q(eav_name)
            fk = q(meta["entity_fk"])
            kc = q(meta["key_column"])
            vc = q(meta["value_column"])
            try:
                # Single-pass O(N) scan; canonical_form runs only on (name, key)
                # pairs whose raw spellings actually differ across offers.
                cursor.execute(
                    f"SELECT ent.{nm_c}, eav.{kc}, eav.{vc} "
                    f"FROM {t} ent "
                    f"JOIN {et} eav ON eav.{fk} = ent.{id_c} "
                    f"WHERE ent.{nm_c} IS NOT NULL AND eav.{kc} IS NOT NULL AND eav.{vc} IS NOT NULL"
                )
                first_raw: Dict[Tuple[str, str], str] = {}
                diff_pairs: set = set()
                by_nk_raw: Dict[Tuple[str, str], set] = {}
                for nm, k, v in cursor.fetchall():
                    nk = (str(nm), str(k))
                    sv = str(v)
                    prev = first_raw.get(nk)
                    if prev is None:
                        first_raw[nk] = sv
                    elif prev != sv:
                        if nk not in diff_pairs:
                            diff_pairs.add(nk)
                            by_nk_raw[nk] = {prev, sv}
                        else:
                            by_nk_raw[nk].add(sv)
                if not diff_pairs:
                    continue
                all_keys_ctx = " ".join(sorted({k for (_nm, k) in diff_pairs}))
                for (_nm, k), raw_vals in by_nk_raw.items():
                    if k in non_intrinsic:
                        continue
                    canon_vals = set()
                    for rv in raw_vals:
                        cf = canonical_form(rv, key=all_keys_ctx)
                        canon_vals.add(cf if cf is not None else rv.strip())
                    if len(canon_vals) > 1:
                        non_intrinsic.add(k)
            except Exception:
                continue
        return non_intrinsic

    @staticmethod
    def _build_raw_node(pid: str, name: str, category: str, price: Optional[float],
                        raw_meta: Dict[str, Any], name_from_identity: bool = False) -> ProductNode:
        attrs = {str(k): str(v) for k, v in raw_meta.items() if v is not None and str(v).strip() != ""}
        if name_from_identity:
            attrs["name_source"] = "identity_column"
        return ProductNode(id=pid, name=name, category=category, price=price, attributes=attrs)

    @staticmethod
    def load_product_dynamically(db_path: str, product_id: str) -> Optional[ProductNode]:
        """
        Lazy on-demand projection of a single product from any supported SQLite DB.

        F10: the projection is driven by the canonical source projection, not
        by a hard-coded `products` table shape. A source without a resolvable
        price column still projects; the price axis is simply absent (None,
        never an invented 0.0).
        """
        return DomainCatalogRegistry.load_products_dynamically(
            db_path, [product_id]).get(product_id)

    @staticmethod
    def load_products_dynamically(db_path: str, product_ids: List[str]) -> Dict[str, "ProductNode"]:
        """
        Bulk variant of `load_product_dynamically` (F34): one connection, one
        `IN (...)` select, and one batched metadata pass -- replacing the
        per-row lazy hydration that made a catalog page cost N+1 queries.
        """
        import sqlite3
        from source_projection import build_source_projection

        ids = [str(i) for i in product_ids]
        if not ids or not os.path.exists(db_path):
            return {}
        try:
            proj = build_source_projection(db_path)
        except Exception:
            return {}
        table = proj.entity_table
        ident = proj.resolve("identity")
        if table is None or not ident.resolved:
            return {}
        name_res = proj.resolve("name")
        cat_res = proj.resolve("category")
        price_res = proj.resolve("price")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            select_list, join_sql, has_cat = DomainCatalogRegistry._projection_query_parts(
                proj, name_res, cat_res, price_res, ident
            )
            placeholders = ", ".join("?" for _ in ids)
            cursor.execute(
                f"SELECT {select_list} FROM {quote_ident(table)} e{join_sql} "
                f"WHERE e.{quote_ident(ident.column)} IN ({placeholders})",
                ids,
            )
            rows = cursor.fetchall()
            pids = [str(r[0]) for r in rows]
            meta_by_prod = DomainCatalogRegistry._collect_eav_meta(cursor, proj, pids)
        finally:
            conn.close()

        out: Dict[str, ProductNode] = {}
        for r in rows:
            pid = str(r[0])
            raw_meta = dict(meta_by_prod.get(pid, {}))
            # F10: an absent price stays absent; a NULL name falls back to the
            # identity value with an honest marker instead of a fabricated label.
            price = None if r[3] is None else float(r[3])
            name = r[1] if (name_res.column and r[1] is not None) else pid
            category = (r[2] or "general") if has_cat else "general"
            node = DomainCatalogRegistry._build_raw_node(
                pid, name, category, price, raw_meta,
                name_from_identity=not bool(name_res.column),
            )
            out[pid] = node
        return out

    @classmethod
    def create_database_catalog(cls, db_path: str, seed_limit: int = 120) -> KnowledgeHypergraph:
        """
        Universal domain-agnostic hypergraph projection from any SQLite database.

        F10: the projection is driven by the canonical source projection
        (source_projection.py), not by a hard-coded `products` table shape.
        A source without a resolvable price column still projects -- its items
        simply have no price axis -- and a renamed schema needs no product-code
        changes. Zero domain vocabularies or seed dictionaries are applied.
        """
        import sqlite3
        from source_projection import build_source_projection

        kg = KnowledgeHypergraph()
        if not os.path.exists(db_path):
            return kg
        try:
            proj = build_source_projection(db_path)
        except Exception:
            return kg
        table = proj.entity_table
        ident = proj.resolve("identity")
        if table is None or not ident.resolved:
            return kg
        name_res = proj.resolve("name")
        cat_res = proj.resolve("category")
        price_res = proj.resolve("price")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            select_list, join_sql, has_cat = DomainCatalogRegistry._projection_query_parts(
                proj, name_res, cat_res, price_res, ident
            )
            group_col = name_res.column or ident.column
            base_sql = (
                f"SELECT {select_list} FROM {quote_ident(table)} e{join_sql} "
                f"GROUP BY e.{quote_ident(group_col)}"
            )
            try:
                cursor.execute(f"{base_sql} ORDER BY MIN(e.rowid) LIMIT {int(seed_limit)}")
            except sqlite3.OperationalError:
                cursor.execute(f"{base_sql} LIMIT {int(seed_limit)}")
            rows = cursor.fetchall()

            pids = [str(r[0]) for r in rows]
            meta_by_prod = DomainCatalogRegistry._collect_eav_meta(cursor, proj, pids)
            non_intrinsic_keys = DomainCatalogRegistry._non_intrinsic_eav_keys(cursor, proj)
        finally:
            conn.close()

        for r in rows:
            pid = str(r[0])
            raw_meta = dict(meta_by_prod.get(pid, {}))
            # F10: an absent price stays absent (None, never 0.0); a NULL name
            # falls back to the identity value with an honest marker.
            price = None if r[3] is None else float(r[3])
            name = r[1] if (name_res.column and r[1] is not None) else pid
            category = (r[2] or "general") if has_cat else "general"
            node = DomainCatalogRegistry._build_raw_node(
                pid, name, category, price, raw_meta,
                name_from_identity=not bool(name_res.column),
            )
            kg.add_product(node)

        # Dynamic Contract Induction for any ingested SQLite domain (zero seeds)
        from contract_inductor import ContractInductor
        inductor = ContractInductor(max_induced=MAX_INDUCED_CONSTRAINTS)
        for c in inductor.induce_contracts_from_products(
            list(kg.nodes.values()),
            non_matching_keys=non_intrinsic_keys or None,
        ):
            kg.add_constraint(c)
        kg.induction_report = inductor.last_induction_report

        return kg


