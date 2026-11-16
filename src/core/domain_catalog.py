"""
Py-Buddy: Domain Catalog Registry
Modular repository for multi-domain product nodes and universal constraints.
Decouples catalog initialization from the server transport layer.
"""

from typing import Dict, List, Any, Optional
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
    def _legacy_merchant_meta(
        cursor,
        entity_ids: List[str],
        entity_table: str = "products",
        ident_col: str = "id",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Merchant enrichment is a marketplace-schema feature. For any other
        table shape it is simply absent (F18: absent stays absent).
        """
        vendor_by_prod: Dict[str, Dict[str, Any]] = {}
        if not entity_ids:
            return vendor_by_prod
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='merchants'")
        has_merchants_tbl = bool(cursor.fetchone())
        q = quote_ident
        cursor.execute(f"PRAGMA table_info({q(entity_table)})")
        prod_cols = {col[1] for col in cursor.fetchall()}
        if not (has_merchants_tbl and "merchant_id" in prod_cols):
            return vendor_by_prod
        placeholders = ",".join("?" for _ in entity_ids)
        cursor.execute(
            f"SELECT e.{q(ident_col)}, m.name, m.rating, m.delivery_days "
            f"FROM {q(entity_table)} e LEFT JOIN merchants m ON e.merchant_id = m.id "
            f"WHERE e.{q(ident_col)} IN ({placeholders})",
            entity_ids,
        )
        for pid, vname, vrating, vdeliv in cursor.fetchall():
            vendor_by_prod[str(pid)] = (vname, vrating, vdeliv)
        return vendor_by_prod

    @staticmethod
    def _merge_vendor_meta(raw_meta: Dict[str, Any], vend: Optional[tuple]) -> None:
        # F18: only stated values are passed through; missing ones stay missing.
        if not vend:
            return
        vname, vrating, vdeliv = vend
        if vname:
            raw_meta["vendor"] = str(vname)
        if vrating is not None:
            raw_meta["vendor_rating"] = str(vrating)
        if vdeliv is not None:
            raw_meta["delivery_days"] = str(vdeliv)

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
        from marketplace_normalizer import MarketplaceEAVNormalizer
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
            select_list = DomainCatalogRegistry._projection_select(proj, name_res, cat_res, price_res, ident)
            placeholders = ", ".join("?" for _ in ids)
            cursor.execute(
                f"SELECT {select_list} FROM {quote_ident(table)} "
                f"WHERE {quote_ident(ident.column)} IN ({placeholders})",
                ids,
            )
            rows = cursor.fetchall()
            pids = [str(r[0]) for r in rows]
            meta_by_prod = DomainCatalogRegistry._collect_eav_meta(cursor, proj, pids)
            vendor_by_prod = DomainCatalogRegistry._legacy_merchant_meta(
                cursor, pids, entity_table=table, ident_col=ident.column
            )
        finally:
            conn.close()

        out: Dict[str, ProductNode] = {}
        for r in rows:
            pid = str(r[0])
            raw_meta = dict(meta_by_prod.get(pid, {}))
            DomainCatalogRegistry._merge_vendor_meta(raw_meta, vendor_by_prod.get(pid))
            # F10: an absent price stays absent; a NULL name falls back to the
            # identity value with an honest marker instead of a fabricated label.
            price = None if r[3] is None else float(r[3])
            name = r[1] if (name_res.column and r[1] is not None) else pid
            category = (r[2] or "general") if cat_res.column else "general"
            node = MarketplaceEAVNormalizer.build_product_node(pid, name, category, price, raw_meta)
            if not name_res.column:
                node = ProductNode(
                    id=node.id, name=node.name, category=node.category, price=node.price,
                    attributes={**dict(node.attributes), "name_source": "identity_column"},
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
        changes.
        """
        import sqlite3
        from marketplace_normalizer import MarketplaceEAVNormalizer
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
            select_list = DomainCatalogRegistry._projection_select(proj, name_res, cat_res, price_res, ident)
            group_col = name_res.column or ident.column
            cursor.execute(
                f"SELECT {select_list} FROM {quote_ident(table)} "
                f"GROUP BY {quote_ident(group_col)} LIMIT {int(seed_limit)}"
            )
            rows = cursor.fetchall()

            pids = [str(r[0]) for r in rows]
            meta_by_prod = DomainCatalogRegistry._collect_eav_meta(cursor, proj, pids)
            vendor_by_prod = DomainCatalogRegistry._legacy_merchant_meta(
                cursor, pids, entity_table=table, ident_col=ident.column
            )
        finally:
            conn.close()

        for r in rows:
            pid = str(r[0])
            raw_meta = dict(meta_by_prod.get(pid, {}))
            DomainCatalogRegistry._merge_vendor_meta(raw_meta, vendor_by_prod.get(pid))
            # F10: an absent price stays absent (None, never 0.0); a NULL name
            # falls back to the identity value with an honest marker.
            price = None if r[3] is None else float(r[3])
            name = r[1] if (name_res.column and r[1] is not None) else pid
            category = (r[2] or "general") if cat_res.column else "general"
            node = MarketplaceEAVNormalizer.build_product_node(pid, name, category, price, raw_meta)
            if not name_res.column:
                node = ProductNode(
                    id=node.id, name=node.name, category=node.category, price=node.price,
                    attributes={**dict(node.attributes), "name_source": "identity_column"},
                )
            kg.add_product(node)

        # Dynamic Contract Induction for any ingested SQLite domain
        from contract_inductor import ContractInductor
        inductor = ContractInductor(max_induced=MAX_INDUCED_CONSTRAINTS)
        for c in inductor.induce_contracts_from_products(list(kg.nodes.values())):
            kg.add_constraint(c)
        kg.induction_report = inductor.last_induction_report

        return kg


