"""
Py-Buddy: Domain-Agnostic Deep Profiler (Trump 1)
Automated database schema introspection, EAV flattening, and knowledge_snapshot generation.
Zero domain-specific hardcoding.
"""

from typing import Dict, List, Any, Optional
import sqlite3
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

@dataclass
class ColumnProfile:
    name: str
    data_type: str
    is_primary_key: bool = False
    is_nullable: bool = True
    is_foreign_key: bool = False
    fk_target_table: Optional[str] = None
    fk_target_column: Optional[str] = None
    semantic_type: str = "unknown"  # "identifier", "price", "category", "spec_numeric", "text_search", "generic"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    distinct_sample_values: List[Any] = field(default_factory=list)
    distinct_count: int = 0

@dataclass
class TableProfile:
    name: str
    row_count: int = 0
    is_eav: bool = False
    eav_entity_fk: Optional[str] = None
    eav_key_col: Optional[str] = None
    eav_val_col: Optional[str] = None
    columns: Dict[str, ColumnProfile] = field(default_factory=dict)
    primary_keys: List[str] = field(default_factory=list)

@dataclass
class DatabaseKnowledgeSnapshot:
    engine: str
    database_name: str
    created_at: str
    tables: Dict[str, TableProfile] = field(default_factory=dict)
    eav_projections: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DomainAgnosticProfiler:
    """
    Introspects arbitrary SQL databases to build a semantic Knowledge Snapshot.
    Detects EAV structures (like WordPress/WooCommerce wp_postmeta or Magento catalog_product_entity_*).
    Infers semantic types from column names, data types, and value distributions.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database file not found: {db_path}")

    def profile_database(self) -> DatabaseKnowledgeSnapshot:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        db_name = os.path.basename(self.db_path)
        snapshot = DatabaseKnowledgeSnapshot(
            engine="sqlite",
            database_name=db_name,
            created_at=datetime.now(timezone.utc).isoformat()
        )

        # 1. Discover all user tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row['name'] for row in cursor.fetchall()]

        for table in tables:
            table_profile = self._profile_table(cursor, table)
            snapshot.tables[table] = table_profile

        # 2. Detect EAV structures and build virtual projections
        self._detect_eav_structures(snapshot, cursor)

        conn.close()
        return snapshot

    def _profile_table(self, cursor: sqlite3.Cursor, table_name: str) -> TableProfile:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM \"{table_name}\"")
        row_count = cursor.fetchone()['cnt']

        table_profile = TableProfile(name=table_name, row_count=row_count)

        # Introspect columns via PRAGMA table_info
        cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
        col_rows = cursor.fetchall()

        # Introspect Foreign Keys
        cursor.execute(f"PRAGMA foreign_key_list(\"{table_name}\")")
        fk_rows = cursor.fetchall()
        fk_map = {}
        for fk in fk_rows:
            # fk: id, seq, table, from, to, on_update, on_delete, match
            fk_map[fk['from']] = (fk['table'], fk['to'])

        for col in col_rows:
            c_name = col['name']
            c_type = col['type'].upper()
            is_pk = bool(col['pk'])
            is_nullable = not bool(col['notnull'])

            if is_pk:
                table_profile.primary_keys.append(c_name)

            fk_info = fk_map.get(c_name)
            is_fk = fk_info is not None
            fk_target_tbl = fk_info[0] if is_fk else None
            fk_target_col = fk_info[1] if is_fk else None

            # Determine sample values and distinct count
            distinct_vals = []
            distinct_count = 0
            min_val = None
            max_val = None

            if row_count > 0:
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT \"{c_name}\") as d_cnt FROM \"{table_name}\"")
                    distinct_count = cursor.fetchone()['d_cnt']

                    # Sample distinct values (up to 15)
                    cursor.execute(f"SELECT DISTINCT \"{c_name}\" FROM \"{table_name}\" WHERE \"{c_name}\" IS NOT NULL LIMIT 15")
                    distinct_vals = [r[0] for r in cursor.fetchall()]

                    # Check for numeric min/max
                    if any(t in c_type for t in ["INT", "FLOAT", "REAL", "NUMERIC", "DOUBLE"]):
                        cursor.execute(f"SELECT MIN(\"{c_name}\") as min_v, MAX(\"{c_name}\") as max_v FROM \"{table_name}\"")
                        m_row = cursor.fetchone()
                        if m_row and m_row['min_v'] is not None:
                            min_val = float(m_row['min_v'])
                            max_val = float(m_row['max_v'])
                except Exception:
                    # Range stats are enrichment; the column keeps its
                    # structural profile without them.
                    pass

            # Infer semantic type
            semantic_type = self._infer_semantic_type(c_name, c_type, is_pk, is_fk, distinct_count, row_count, distinct_vals)

            col_profile = ColumnProfile(
                name=c_name,
                data_type=c_type,
                is_primary_key=is_pk,
                is_nullable=is_nullable,
                is_foreign_key=is_fk,
                fk_target_table=fk_target_tbl,
                fk_target_column=fk_target_col,
                semantic_type=semantic_type,
                min_value=min_val,
                max_value=max_val,
                distinct_sample_values=distinct_vals,
                distinct_count=distinct_count
            )
            table_profile.columns[c_name] = col_profile

        return table_profile

    def _infer_semantic_type(
        self,
        name: str,
        dtype: str,
        is_pk: bool,
        is_fk: bool,
        distinct_count: int,
        total_rows: int,
        samples: List[Any]
    ) -> str:
        """
        F09: this is a NAME HINT, not ground truth. A column called
        `amount_due` can hold ratings while a column called `cena` holds
        prices. Consumers must treat `semantic_type` as one evidence tier
        inside the canonical source projection (source_projection.py), where
        hints can be outvoted by structural or value evidence and ambiguity
        stays visible.
        """
        name_lower = name.lower()
        if is_pk:
            return "primary_key"
        if is_fk or name_lower.endswith("_id") or name_lower == "id":
            return "foreign_key"

        # Price / Currency detection
        price_keywords = ["price", "cost", "amount", "fee", "rate", "ціна", "вартість", "msrp"]
        if any(kw in name_lower for kw in price_keywords):
            return "price"

        # Inventory / Stock detection
        stock_keywords = ["stock", "quantity", "inventory", "count", "залишок", "кількість", "qty"]
        if any(kw in name_lower for kw in stock_keywords):
            return "stock"

        # Category / Brand / Discrete Classification
        cat_keywords = ["category", "brand", "type", "kind", "genre", "status", "категорія", "бренд", "виробник"]
        if any(kw in name_lower for kw in cat_keywords) or (0 < distinct_count <= 50 and total_rows > 50):
            return "category"

        # Text search (names, titles, descriptions)
        text_keywords = ["name", "title", "description", "desc", "label", "назва", "опис", "модель", "model"]
        if any(kw in name_lower for kw in text_keywords):
            return "text_search"

        # Numeric specs
        if any(t in dtype for t in ["INT", "FLOAT", "REAL", "NUMERIC", "DOUBLE"]):
            return "spec_numeric"

        return "generic"

    def _detect_eav_structures(self, snapshot: DatabaseKnowledgeSnapshot, cursor: sqlite3.Cursor) -> None:
        """
        Detects tables following Entity-Attribute-Value pattern:
        Typically: (entity_id/post_id, meta_key/attribute_code, meta_value/value)
        """
        for table_name, table in snapshot.tables.items():
            key_col = None
            val_col = None
            entity_fk = None

            for col in table.columns.values():
                c_low = col.name.lower()
                if c_low in ["meta_key", "attribute_code", "attribute_name", "key", "param_name", "property"]:
                    key_col = col.name
                elif c_low in ["meta_value", "attribute_value", "value", "val", "param_value"]:
                    val_col = col.name
                elif col.is_foreign_key or c_low in ["post_id", "entity_id", "product_id", "item_id"]:
                    entity_fk = col.name

            if key_col and val_col and entity_fk:
                table.is_eav = True
                table.eav_entity_fk = entity_fk
                table.eav_key_col = key_col
                table.eav_val_col = val_col

                # Inspect distinct EAV keys to form virtual schema projection
                try:
                    cursor.execute(f"SELECT DISTINCT \"{key_col}\" FROM \"{table_name}\" LIMIT 50")
                    distinct_keys = [r[0] for r in cursor.fetchall() if r[0] is not None]
                    snapshot.eav_projections[table_name] = {
                        "entity_fk": entity_fk,
                        "key_column": key_col,
                        "value_column": val_col,
                        "discovered_attributes": distinct_keys
                    }
                except Exception:
                    # Key sampling failure leaves the table detected but
                    # without its attribute vocabulary.
                    pass

    def export_snapshot_json(self, output_path: str) -> str:
        snapshot = self.profile_database()
        data = snapshot.to_dict()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

    @staticmethod
    def generate_compact_schema_context(snapshot_dict: Dict[str, Any]) -> str:
        """
        Reads snapshot dict and converts it into a high-density,
        anti-hallucination prompt context for SQL generation in ReAct agents.
        """
        data = snapshot_dict
        lines = [f"### Database Schema Snapshot ({data['engine'].upper()} - {data['database_name']})"]
        for tbl_name, tbl in data["tables"].items():
            pk_str = f" [PK: {', '.join(tbl['primary_keys'])}]" if tbl['primary_keys'] else ""
            eav_str = " (EAV Table)" if tbl.get('is_eav') else ""
            lines.append(f"- **Table `{tbl_name}`** ({tbl['row_count']} rows){pk_str}{eav_str}:")
            
            for col_name, col in tbl["columns"].items():
                meta_bits = [col['data_type']]
                if col['semantic_type'] != 'generic':
                    meta_bits.append(f"sem={col['semantic_type']}")
                if col.get('min_value') is not None and col.get('max_value') is not None:
                    meta_bits.append(f"range=[{col['min_value']:.1f}, {col['max_value']:.1f}]")
                if col.get('distinct_sample_values') and col['semantic_type'] == 'category':
                    samples = col['distinct_sample_values'][:10]
                    meta_bits.append(f"samples={samples}")
                
                lines.append(f"  - `{col_name}`: {', '.join(meta_bits)}")

        # Collect top category samples across tables to explicitly ground LLM in store domain
        discovered_categories = []
        for tbl_name, tbl in data["tables"].items():
            for col_name, col in tbl["columns"].items():
                if col['semantic_type'] == 'category' and col.get('distinct_sample_values'):
                    for s in col['distinct_sample_values']:
                        if s and s not in discovered_categories:
                            discovered_categories.append(str(s))

        if discovered_categories:
            lines.append(f"\n### Наявні категорії каталогу: {discovered_categories[:15]}")

        if data.get("eav_projections"):
            lines.append("\n### EAV Virtual Projections:")
            for eav_tbl, meta in data["eav_projections"].items():
                attrs = ", ".join(meta.get("discovered_attributes", [])[:8])
                lines.append(f"- `{eav_tbl}` links to `{meta['entity_fk']}` with attributes: [{attrs}...]")

        return "\n".join(lines)
