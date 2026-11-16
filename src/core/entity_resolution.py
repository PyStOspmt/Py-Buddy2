"""
Py-Buddy: Multi-Vendor Entity Resolution & Attribute Harmonization (Slot 4 / C6 Resolution).

In multi-vendor e-commerce marketplaces (Rozetka, Amazon, Open Prices), a physical product
is listed by multiple merchants under slight variations of title, rev suffixes, and chaotic EAV values.
This module clusters raw database records into canonical product entities, harmonizing their
attribute specs via the Value Algebra (⊕) while preserving distinct merchant provenance and offers.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from attribute_algebra import canonical_form, fold_all, CONFLICT
from metrics import _norm_key
from source_projection import build_source_projection, quote_ident


@dataclass
class MerchantOffer:
    """
    One merchant's listing. F19/F18: a field the source does not state stays
    None -- the old COALESCE defaults invented a 1-day delivery, a 5.0
    rating and a "Store Direct" merchant for every listing that had none.
    """
    listing_id: str
    merchant_id: Optional[str]
    merchant_name: Optional[str]
    price: Optional[float]
    stock_quantity: Optional[int]
    delivery_days: Optional[int]
    rating: Optional[float]
    raw_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class CanonicalProductEntity:
    canonical_id: str
    canonical_name: str
    category: str
    brand: Optional[str]
    harmonized_attributes: Dict[str, str] = field(default_factory=dict)
    offers: List[MerchantOffer] = field(default_factory=list)
    # F19: attributes on which sellers disagree are recorded with full
    # provenance instead of collapsing to a convenient first value.
    attribute_conflicts: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    @property
    def min_price(self) -> Optional[float]:
        priced = [o.price for o in self.offers if o.price is not None]
        return min(priced) if priced else None

    @property
    def max_price(self) -> Optional[float]:
        priced = [o.price for o in self.offers if o.price is not None]
        return max(priced) if priced else None

    @property
    def seller_count(self) -> int:
        return len({o.merchant_id for o in self.offers if o.merchant_id is not None})

    def get_cheapest_offer(self) -> Optional[MerchantOffer]:
        priced = [o for o in self.offers if o.price is not None]
        if not priced:
            return None
        # Tie-break only on values the merchant actually states; an absent
        # delivery time or rating is treated as the worst case, not as a
        # fabricated best case.
        return min(priced, key=lambda o: (
            o.price,
            o.delivery_days if o.delivery_days is not None else 10 ** 9,
            -(o.rating if o.rating is not None else 0.0),
        ))


class MarketplaceEntityResolver:
    """
    Deterministically resolves and clusters multi-vendor product listings into
    Canonical Entities based on core normalized title signatures and EAV attribute profiles.
    """

    # Strips structural listing noise (e.g. '(Rev 1)', '[Bulk]', '(v2)', etc.)
    _CLEAN_RE = re.compile(r"\s*\(rev\s*\d+\)|\s*\[.*?\]|\s*\(v\d+\)|\s*\(batch\s*\d+\)", re.IGNORECASE)

    @classmethod
    def normalize_entity_key(cls, name: str) -> str:
        """
        Extracts invariant product identity key (e.g., 'AMD Ryzen 7 7800X3D (Rev 2)' -> 'amd ryzen 7 7800x3d').
        """
        clean = cls._CLEAN_RE.sub("", name or "").strip()
        norm = _norm_key(clean)
        # Normalize double whitespaces
        return " ".join(norm.split())

    @classmethod
    def resolve_from_database(
        cls,
        db_path: str,
        category: Optional[str] = None,
        limit_listings: int = 10000
    ) -> Dict[str, CanonicalProductEntity]:
        """
        Introspects the database and resolves all product listings into canonical entities.
        """
        if not os.path.exists(db_path):
            return {}
        try:
            proj = build_source_projection(db_path)
        except Exception:
            return {}
        table = proj.entity_table
        ident = proj.resolve("identity")
        if table is None or not ident.resolved or not ident.column:
            return {}
        name_res = proj.resolve("name")
        cat_res = proj.resolve("category")
        price_res = proj.resolve("price")
        stock_res = proj.resolve("stock")

        conn = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(db_path).replace("\\", "/"), uri=True)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        q = quote_ident

        cur.execute(f"PRAGMA table_info({q(table)})")
        entity_cols = {col[1] for col in cur.fetchall()}

        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='merchants'")
        has_merchants = bool(cur.fetchone()) and "merchant_id" in entity_cols
        merchant_cols = set()
        if has_merchants:
            cur.execute("PRAGMA table_info(merchants)")
            merchant_cols = {row[1] for row in cur.fetchall()}

        aliases = {"id": "merchant_id", "name": "merchant_name",
                   "delivery_days": "delivery_days", "rating": "rating"}
        merchant_select = ", ".join(
            f"m.{col} as {alias}" if col in merchant_cols else f"NULL as {alias}"
            for col, alias in aliases.items()
        ) if has_merchants else ", ".join(
            f"NULL as {alias}" for alias in aliases.values()
        )
        name_col = name_res.column or ident.column
        cat_sel = f"p.{q(cat_res.column)} AS category" if cat_res.column else "'general' AS category"
        brand_sel = "p.brand AS brand" if "brand" in entity_cols else "NULL AS brand"
        price_sel = f"p.{q(price_res.column)} AS price" if price_res.column else "NULL AS price"
        stock_sel = f"p.{q(stock_res.column)} AS stock_quantity" if stock_res.column else "NULL AS stock_quantity"

        query = f"""
        SELECT p.{q(ident.column)} AS id, p.{q(name_col)} AS name,
               {cat_sel}, {brand_sel}, {price_sel}, {stock_sel},
               {merchant_select}
        FROM {q(table)} p
        """
        if has_merchants:
            query += " LEFT JOIN merchants m ON p.merchant_id = m.id "

        where = []
        params: List[Any] = []
        if category and cat_res.column:
            where.append(f"p.{q(cat_res.column)} = ?")
            params.append(category)

        if where:
            query += " WHERE " + " AND ".join(where)

        query += f" LIMIT {int(limit_listings)}"
        cur.execute(query, params)
        rows = cur.fetchall()

        # Fetch EAV metadata via canonical source projection (fallback to legacy table names only if needed)
        eav_by_prod: Dict[str, Dict[str, str]] = {}
        eav_sources = proj.eav_tables or {}
        if eav_sources:
            for tbl_name, meta in eav_sources.items():
                try:
                    cur.execute(
                        f"SELECT {q(meta['entity_fk'])}, {q(meta['key_column'])}, {q(meta['value_column'])} "
                        f"FROM {q(tbl_name)}"
                    )
                    for pid, k, v in cur.fetchall():
                        if k is not None:
                            eav_by_prod.setdefault(str(pid), {})[str(k)] = str(v) if v is not None else ""
                except Exception:
                    continue
        else:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('product_eav_meta', 'product_meta')")
            meta_tbl = cur.fetchone()
            if meta_tbl:
                tbl_name = meta_tbl[0]
                cur.execute(f"PRAGMA table_info({q(tbl_name)})")
                mcols = {row[1] for row in cur.fetchall()}
                if {"product_id", "meta_key", "meta_value"} <= mcols:
                    kcol, vcol = "meta_key", "meta_value"
                elif {"product_id", "attr_key", "attr_value"} <= mcols:
                    kcol, vcol = "attr_key", "attr_value"
                else:
                    kcol = vcol = None
                if kcol and vcol:
                    cur.execute(
                        f"SELECT product_id, {q(kcol)}, {q(vcol)} FROM {q(tbl_name)}"
                    )
                    for pid, k, v in cur.fetchall():
                        if k is not None:
                            eav_by_prod.setdefault(str(pid), {})[str(k)] = (
                                str(v) if v is not None else ""
                            )

        conn.close()

        # Group by normalized entity key
        clusters: Dict[str, List[Tuple[sqlite3.Row, Dict[str, str]]]] = {}
        for r in rows:
            pname = str(r["name"] if r["name"] is not None else r["id"])
            ekey = cls.normalize_entity_key(pname)
            clusters.setdefault(ekey, []).append((r, eav_by_prod.get(str(r["id"]), {})))

        entities: Dict[str, CanonicalProductEntity] = {}
        for ekey, group in clusters.items():
            first_row, _ = group[0]
            # Canonical name is the shortest representative title without revision noise
            raw_name = str(first_row["name"] if first_row["name"] is not None else first_row["id"])
            canon_name = cls._CLEAN_RE.sub("", raw_name).strip() or raw_name
            # F19: the identity is derived from the entity key itself, so it
            # is stable under input permutation. The old index-based id
            # (`canon_{category}_{idx}`) changed whenever the iteration order
            # -- or the set of clusters -- changed.
            canon_id = "canon_" + ekey.replace(" ", "_")

            offers: List[MerchantOffer] = []
            attr_values: Dict[str, List[Tuple[str, str]]] = {}

            for row, raw_meta in group:
                offer = MerchantOffer(
                    listing_id=row["id"],
                    merchant_id=row["merchant_id"],
                    merchant_name=row["merchant_name"],
                    # F18/F19: absent stays absent -- no invented 0.0 price.
                    price=float(row["price"]) if row["price"] is not None else None,
                    stock_quantity=int(row["stock_quantity"]) if row["stock_quantity"] is not None else None,
                    delivery_days=int(row["delivery_days"]) if row["delivery_days"] is not None else None,
                    rating=float(row["rating"]) if row["rating"] is not None else None,
                    raw_attributes=raw_meta
                )
                offers.append(offer)

                for k, v in raw_meta.items():
                    attr_values.setdefault(k, []).append((str(row["id"]), str(v)))

            # Harmonize attributes across all sellers via the Value Algebra (⊕).
            # F19: a disagreement is recorded with full provenance, never
            # resolved by picking a convenient first value.
            harmonized: Dict[str, str] = {}
            conflicts: Dict[str, List[Tuple[str, str]]] = {}
            all_keys_ctx = " ".join(attr_values)
            for k, vals in attr_values.items():
                folded = fold_all([v for _, v in vals], key=all_keys_ctx)
                # F20: fold preserves unparseable values as raw tokens; a
                # canonical attribute may only carry a canonical value.
                if (folded is not None and folded is not CONFLICT
                        and canonical_form(str(folded), key=all_keys_ctx) == folded):
                    harmonized[k] = str(folded)
                elif folded is CONFLICT:
                    conflicts[k] = list(vals)
                # An all-equal unparseable value stays visible in each
                # offer's raw_attributes; it is not promoted to a fact.

            entity = CanonicalProductEntity(
                canonical_id=canon_id,
                attribute_conflicts=conflicts,
                canonical_name=canon_name,
                category=first_row["category"],
                brand=first_row["brand"],
                harmonized_attributes=harmonized,
                offers=offers
            )
            entities[canon_id] = entity

        return entities
