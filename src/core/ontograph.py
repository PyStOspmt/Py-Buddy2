"""
Py-Buddy: Dynamic Domain Taxonomy Tree (OntoGraph 2.0)
Constructs an adaptive, multi-level hierarchical knowledge tree and mindmap
from arbitrary SQLite product databases. Eliminates Horizon Truncation,
structures chaotic/dirty catalogs, discovers EAV metadata parameters,
and powers visual taxonomy navigation for both AI Agents and the Web Console.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import sqlite3
import re
import os
import json
import hashlib
from dataclasses import dataclass, field

@dataclass
class OntoNode:
    id: str
    name: str
    level: int  # 0: Root, 1: Macro-Department, 2: Category, 3: Brand/Facet, 4: Model Cluster
    item_count: int = 0
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    parameters: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)
    raw_categories: List[str] = field(default_factory=list)
    children: List['OntoNode'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level,
            "item_count": self.item_count,
            "price_min": self.price_min,
            "price_max": self.price_max,
            "parameters": self.parameters,
            "sample_ids": self.sample_ids,
            "raw_categories": self.raw_categories,
            "children": [c.to_dict() for c in self.children]
        }

@dataclass
class DomainOntologyProfile:
    domain_concept: str
    macro_departments: Dict[str, List[str]] = field(default_factory=dict)
    decision_facets: Dict[str, List[str]] = field(default_factory=dict)
    routing_synonyms: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain_concept": self.domain_concept,
            "macro_departments": self.macro_departments,
            "decision_facets": self.decision_facets,
            "routing_synonyms": self.routing_synonyms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DomainOntologyProfile":
        return cls(
            domain_concept=data.get("domain_concept", "Каталог товарів"),
            macro_departments=data.get("macro_departments", {}),
            decision_facets=data.get("decision_facets", {}),
            routing_synonyms=data.get("routing_synonyms", {}),
        )

class SemanticOntologyInductor:
    """
    Neuro-Symbolic Domain Ontology Inductor.
    Discovers semantic macro-departments, domain decision facets, and customer query synonyms
    via LLM analysis of product evidence samples from SQLite,
    grounded with deterministic symbolic fallback clustering and strict DB invariants (ZCCS = 0).
    """

    @classmethod
    def induce(
        cls,
        categories: List[str],
        sample_rows_by_cat: Dict[str, List[sqlite3.Row]],
        discovered_params: Dict[str, List[str]],
        runner: Optional[Any] = None,
        cache_path: Optional[str] = None,
        source_version: Optional[Tuple[int, int]] = None
    ) -> DomainOntologyProfile:
        # F33: the cache is published inside a versioned envelope. A payload
        # written for a different source file (or a pre-envelope cache with
        # no version at all) is stale evidence and is recomputed, not read.
        from source_lifecycle import read_versioned_cache, write_versioned_cache
        if cache_path:
            cached = read_versioned_cache(cache_path, source_version)
            if isinstance(cached, dict):
                if "domain_concept" in cached:
                    return DomainOntologyProfile.from_dict(cached)
                return DomainOntologyProfile(
                    domain_concept="Каталог товарів",
                    macro_departments=cached
                )

        # Try LLM induction when runner is active
        if runner and getattr(runner, "api_key", None):
            profile = cls._induce_with_llm(categories, sample_rows_by_cat, discovered_params, runner)
            if profile and profile.macro_departments:
                if cache_path:
                    write_versioned_cache(cache_path, source_version, profile.to_dict())
                return profile

        # Deterministic symbolic fallback
        profile = cls._induce_fallback(categories, discovered_params)
        if cache_path:
            write_versioned_cache(cache_path, source_version, profile.to_dict())
        return profile

    @classmethod
    def _induce_with_llm(
        cls,
        categories: List[str],
        sample_rows_by_cat: Dict[str, List[sqlite3.Row]],
        discovered_params: Dict[str, List[str]],
        runner: Any
    ) -> Optional[DomainOntologyProfile]:
        clean_to_raw: Dict[str, List[str]] = {}
        lookup_keys: Dict[str, List[str]] = {}
        for c in categories:
            cl = OntoGraph._clean_category_name(c)
            clean_to_raw.setdefault(cl, []).append(c)
            lookup_keys.setdefault(cl.strip().lower(), []).append(c)
            lookup_keys.setdefault(str(c).strip().lower(), []).append(c)

        prompt_samples = []
        for cl, raw_list in list(clean_to_raw.items())[:60]:
            if cl == "Інше / Різне":
                continue
            examples = []
            for r in raw_list:
                for row in sample_rows_by_cat.get(r, [])[:2]:
                    if "name" in row.keys() and row["name"]:
                        examples.append(str(row["name"]))
            prompt_samples.append({
                "category": cl,
                "examples": examples[:2],
                "attributes": discovered_params.get(raw_list[0], [])[:3] if raw_list else []
            })

        prompt = f"""Ти — провідний експерт із семантичного моделювання та таксономії знань в електронній комерції.
Перед тобою вибірка категорій та товарів із каталогу:
{json.dumps(prompt_samples, ensure_ascii=False, indent=2)}

Виконай нейро-символьний аналіз предметної області:
1. Визнач концептуальний домен ("domain_concept") українською мовою (наприклад: "Комп'ютерні комплектуючі", "Лікарські засоби та аптека", "Автозапчастини", "Продукти харчування").
2. Згрупуй категорії у 5-10 логічних макро-розділів ("macro_departments"). Кожен макро-розділ повинен містити список категорій із наданого переліку (поле "category").
3. Визнач ключові фасети вибору ("decision_facets") — 3-5 найважливіших технічних/споживчих характеристик для кожної групи (наприклад, для ПК: сокет, пам'ять; для ліків: діюча речовина, форма, дозування; для авто: сумісність, сторона, двигун).
4. Запропонуй типові синоніми пошуку ("routing_synonyms") — словник, де ключ це розмовний термін покупця, а значення — точна назва категорії.

Поверни ВИКЛЮЧНО валідний JSON-об'єкт строго за схемою:
{{
  "domain_concept": "Назва домену",
  "macro_departments": {{
    "Назва макро-розділу 1": ["категорія_1", "категорія_2"],
    "Назва макро-розділу 2": ["категорія_3"]
  }},
  "decision_facets": {{
    "Назва макро-розділу 1": ["параметр_1", "параметр_2"]
  }},
  "routing_synonyms": {{
    "народна назва чи синонім": "Назва категорії"
  }}
}}
Жодного вступного чи пояснювального тексту поза JSON.
"""
        try:
            raw = runner.generate(prompt, temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                return None
            data = json.loads(m.group(0))
            if not isinstance(data, dict):
                return None

            macro_deps = data.get("macro_departments", {})
            if not isinstance(macro_deps, dict) or len(macro_deps) < 1:
                return None

            result_deps: Dict[str, List[str]] = {}
            assigned_raw: Set[str] = set()

            for dept_name, cl_cats in macro_deps.items():
                raw_list = []
                if isinstance(cl_cats, list):
                    for cl in cl_cats:
                        k = str(cl).strip().lower()
                        if k in lookup_keys:
                            for raw_c in lookup_keys[k]:
                                if raw_c not in assigned_raw:
                                    raw_list.append(raw_c)
                                    assigned_raw.add(raw_c)
                        elif cl in clean_to_raw:
                            for raw_c in clean_to_raw[cl]:
                                if raw_c not in assigned_raw:
                                    raw_list.append(raw_c)
                                    assigned_raw.add(raw_c)
                if raw_list:
                    result_deps[str(dept_name)] = raw_list

            unassigned = [c for c in categories if c not in assigned_raw]
            if unassigned:
                result_deps["Інше / Різне"] = unassigned

            return DomainOntologyProfile(
                domain_concept=str(data.get("domain_concept") or "Каталог товарів"),
                macro_departments=result_deps,
                decision_facets=data.get("decision_facets", {}) if isinstance(data.get("decision_facets"), dict) else {},
                routing_synonyms=data.get("routing_synonyms", {}) if isinstance(data.get("routing_synonyms"), dict) else {}
            )
        except Exception:
            return None

    @classmethod
    def _induce_fallback(cls, categories: List[str], discovered_params: Dict[str, List[str]]) -> DomainOntologyProfile:
        """
        Deterministic symbolic fallback:
        - If categories <= 12, preserves 1:1 clean category structure without fake departments.
        - If categories > 12, clusters by structural delimiters (/, :, >, -) or shared domain prefix phrases.
        - Strictly forbids single-token splitting ('tokens[0]') to eliminate nonsensical clusters.
        """
        if len(categories) <= 12:
            return DomainOntologyProfile(
                domain_concept="Каталог товарів",
                macro_departments={},
                decision_facets=discovered_params
            )

        groups: Dict[str, List[str]] = {}
        for c in categories:
            cl = OntoGraph._clean_category_name(c)
            if cl == "Інше / Різне":
                groups.setdefault("Інше / Різне", []).append(c)
                continue

            delim_match = re.split(r"\s*[/\\:>|]\s*", c)
            if len(delim_match) > 1 and len(delim_match[0].strip()) >= 3:
                macro = OntoGraph._clean_category_name(delim_match[0].strip())
            else:
                words = re.findall(r"[\w]+", cl)
                if len(words) >= 2 and len(words[0]) >= 4:
                    macro = f"{words[0].capitalize()} {words[1]}"
                elif len(words) == 1 and len(words[0]) >= 4:
                    macro = words[0].capitalize()
                else:
                    macro = "Основні категорії"
            groups.setdefault(macro, []).append(c)

        if len(groups) > 15:
            consolidated: Dict[str, List[str]] = {}
            misc: List[str] = []
            for gname, gcats in groups.items():
                if len(gcats) >= 2:
                    consolidated[gname] = gcats
                else:
                    misc.extend(gcats)
            if misc:
                consolidated["Супутні та інші товари"] = misc
            groups = consolidated

        return DomainOntologyProfile(
            domain_concept="Каталог товарів",
            macro_departments=groups,
            decision_facets=discovered_params
        )

class OntoGraph:
    """
    Dynamic Domain Taxonomy Tree (OntoGraph Engine 2.0).
    Builds and navigates an interactive, hierarchical category mindmap over any SQLite catalog.
    Zero-code-change schema ingestion (ZCCS = 0) with EAV metadata discovery and adaptive clustering.
    """
    def __init__(self, db_path: str, openrouter_runner: Optional[Any] = None, use_ai_clustering: bool = True):
        self.db_path = db_path
        self.runner = openrouter_runner
        self.use_ai_clustering = use_ai_clustering
        self.root: Optional[OntoNode] = None
        self.profile: Optional[DomainOntologyProfile] = None
        self._lookup: Dict[str, OntoNode] = {}
        self._token_index: Dict[str, List[OntoNode]] = {}
        self._direct_token_index: Dict[str, List[OntoNode]] = {}
        if os.path.exists(db_path):
            self.build()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{os.path.abspath(self.db_path).replace(chr(92), '/')}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _clean_category_name(raw: str) -> str:
        """Strips language tags (en:, fr:), replaces punctuation/dashes, capitalizes cleanly."""
        if not raw:
            return "Інше / Різне"
        cleaned = re.sub(r"^[a-z]{2}:", "", str(raw).strip(), flags=re.IGNORECASE)
        cleaned = cleaned.replace("_", " ").replace("-", " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        low = cleaned.lower()
        if low in {"null", "undefined", "unknown", "none", "", "nan"}:
            return "Інше / Різне"
        return cleaned.capitalize()

    def _get_cache_path(self) -> str:
        h = hashlib.md5(os.path.abspath(self.db_path).encode("utf-8")).hexdigest()[:10]
        # This module lives in src/core/, so the repository data/ dir is three
        # levels up (docs/13 C2).
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, f".ontology_cache_{h}.json")

    def build(self) -> OntoNode:
        """
        Dynamically inspects the database, EAV tables, and constructs the adaptive OntoGraph.

        F10: the table and column roles come from the canonical source
        projection, not from an allow-list of table names -- a renamed schema
        builds the same ontology without product-code changes.
        """
        from source_projection import build_source_projection, quote_ident
        try:
            proj = build_source_projection(self.db_path)
        except Exception:
            proj = None
        ident = proj.resolve("identity") if proj else None
        if proj is None or proj.entity_table is None or not ident.resolved:
            self.root = OntoNode(id="root", name="Порожній каталог", level=0)
            return self.root

        conn = self._connect()
        try:
            cursor = conn.cursor()
            main_table = proj.entity_table
            q = quote_ident

            cursor.execute(f"PRAGMA table_info({q(main_table)})")
            cols = {row["name"]: row["name"] for row in cursor.fetchall()}

            # Column roles from the projection; brand stays a light exact-name
            # hint and is simply absent when the source does not state it.
            pk_col = ident.column
            name_res = proj.resolve("name")
            cat_res = proj.resolve("category")
            price_res = proj.resolve("price")
            name_col = name_res.column or pk_col
            cat_col = cat_res.column
            merch_col = proj.resolve("merchant").column
            price_col = price_res.column
            brand_col = next((c for c in ("brand", "виробник", "manufacturer") if c in cols), None)
            # Grouping axis: the product category when the source has one,
            # otherwise the merchant axis of an offer-style source. The
            # axis is reported under its real name in `parameters`, not
            # silently relabelled "category" (a7).
            group_col = cat_col or merch_col
            axis_label = ("category" if cat_col
                          else "merchant" if merch_col else None)

            # Aggregated reading instead of materialising every row: the
            # tree needs only per-group count/min/max plus bounded ID/name
            # samples, so a million-row catalogue is summarised by GROUP BY
            # and window functions rather than fetched whole into RAM.
            if group_col:
                grp_expr = (
                    f"CASE WHEN {q(group_col)} IS NULL "
                    f"OR TRIM(CAST({q(group_col)} AS TEXT)) IN ('','0','0.0') "
                    f"THEN 'Загальні товари' "
                    f"ELSE TRIM(CAST({q(group_col)} AS TEXT)) END")
            else:
                grp_expr = "'Загальні товари'"
            brand_expr = (
                f"CASE WHEN {q(brand_col)} IS NULL "
                f"OR TRIM(CAST({q(brand_col)} AS TEXT))='' "
                f"THEN 'Інші виробники' "
                f"ELSE TRIM(CAST({q(brand_col)} AS TEXT)) END"
            ) if brand_col else "'Інші виробники'"
            price_expr = q(price_col) if price_col else "NULL"

            cursor.execute(
                f"SELECT COUNT(*), MIN({price_expr}), MAX({price_expr}) "
                f"FROM {q(main_table)}")
            total_rows, root_pmin, root_pmax = cursor.fetchone()

            cursor.execute(
                f"SELECT {grp_expr} AS g, COUNT(*), "
                f"MIN({price_expr}), MAX({price_expr}) "
                f"FROM {q(main_table)} GROUP BY g")
            # cat -> (count, min, max)
            cat_stats = {str(c): (cnt, mn, mx)
                         for c, cnt, mn, mx in cursor.fetchall()}

            cursor.execute(
                f"SELECT {grp_expr} AS g, {brand_expr} AS b, COUNT(*), "
                f"MIN({price_expr}), MAX({price_expr}) "
                f"FROM {q(main_table)} GROUP BY g, b")
            # cat -> {brand -> (count, min, max)}
            brand_stats: Dict[str, Dict[str, tuple]] = {}
            for c, b, cnt, mn, mx in cursor.fetchall():
                brand_stats.setdefault(str(c), {})[str(b)] = (cnt, mn, mx)

            # Bounded samples: <=10 ids per category for node sample_ids,
            # <=3 names per category for the inductor prompt, <=5 ids per
            # (category, brand) pair for brand nodes.
            cursor.execute(
                f"SELECT g, b, pid, pname, rn_cat, rn_brand FROM ("
                f"  SELECT {grp_expr} AS g, {brand_expr} AS b, "
                f"  {q(pk_col)} AS pid, {q(name_col)} AS pname, "
                f"  ROW_NUMBER() OVER (PARTITION BY {grp_expr} "
                f"      ORDER BY {q(pk_col)}) AS rn_cat, "
                f"  ROW_NUMBER() OVER (PARTITION BY {grp_expr}, {brand_expr} "
                f"      ORDER BY {q(pk_col)}) AS rn_brand "
                f"  FROM {q(main_table)}) AS base "
                f"WHERE rn_cat <= 10 OR rn_brand <= 5")
            cat_id_samples: Dict[str, List[str]] = {}
            cat_name_samples: Dict[str, List[str]] = {}
            brand_id_samples: Dict[str, Dict[str, List[str]]] = {}
            for g, b, pid, pname, rn_cat, rn_brand in cursor.fetchall():
                g, b = str(g), str(b)
                if rn_cat <= 10:
                    cat_id_samples.setdefault(g, []).append(str(pid))
                if rn_cat <= 3 and pname is not None:
                    cat_name_samples.setdefault(g, []).append(str(pname))
                if rn_brand <= 5:
                    brand_id_samples.setdefault(g, {}).setdefault(
                        b, []).append(str(pid))

            # Discover top EAV attributes per category from every detected
            # EAV table (name-based or structural), joined through its own
            # entity reference column.
            cat_eav_params: Dict[str, List[str]] = {}
            if group_col:
                for meta_table, meta in proj.eav_tables.items():
                    try:
                        cursor.execute(f"""
                        SELECT p.{q(group_col)}, m.{q(meta['key_column'])}, COUNT(*) as cnt
                        FROM {q(meta_table)} m
                        JOIN {q(main_table)} p ON p.{q(pk_col)} = m.{q(meta['entity_fk'])}
                        GROUP BY p.{q(cat_col)}, m.{q(meta['key_column'])}
                        ORDER BY cnt DESC
                        """)
                        for c_raw, mkey, _ in cursor.fetchall():
                            if c_raw:
                                cat_eav_params.setdefault(str(c_raw), []).append(str(mkey))
                    except Exception:
                        continue

            # Also discover parameters from flat table columns
            standard_cols = {pk_col, name_col, cat_col, merch_col, brand_col, price_col,
                             "stock_quantity", "merchant_id", "created_at", "updated_at",
                             "performance_score", "image_url", "url"}
            standard_cols.discard(None)
            flat_attr_cols = [c for c in cols if c not in standard_cols]
            if flat_attr_cols and group_col:
                for col_name in flat_attr_cols:
                    try:
                        cursor.execute(f"SELECT DISTINCT {q(group_col)} FROM {q(main_table)} WHERE {q(col_name)} IS NOT NULL")
                        for (c_raw,) in cursor.fetchall():
                            if c_raw:
                                existing = cat_eav_params.setdefault(str(c_raw), [])
                                if col_name not in existing:
                                    existing.append(col_name)
                    except Exception:
                        # A column that cannot be probed contributes no
                        # parameter candidates; other columns still do.
                        pass
        finally:
            conn.close()

        if not total_rows:
            self.root = OntoNode(id="root", name="Каталог без товарів", level=0)
            return self.root

        def _fnum(x: Any) -> Optional[float]:
            """Stated numeric min/max or honest None -- a non-numeric
            price value is reported absent, never coerced or crashy."""
            if x is None:
                return None
            try:
                return round(float(x), 2)
            except (TypeError, ValueError):
                return None

        # Root Node Aggregates
        self.root = OntoNode(
            id="root",
            name="Каталог товарів (Головна)",
            level=0,
            item_count=total_rows,
            price_min=_fnum(root_pmin),
            price_max=_fnum(root_pmax),
            parameters=[x for x in (axis_label,
                                    "brand" if brand_col else None,
                                    "price" if price_col else None) if x]
        )
        self._lookup["root"] = self.root

        # Group rows by raw category
        distinct_cats = list(cat_stats.keys())
        sample_rows_by_cat = {
            c: [{"name": n} for n in cat_name_samples.get(c, [])]
            for c in distinct_cats
        }

        cache_file = self._get_cache_path()
        from source_lifecycle import source_version
        self.profile = SemanticOntologyInductor.induce(
            categories=distinct_cats,
            sample_rows_by_cat=sample_rows_by_cat,
            discovered_params=cat_eav_params,
            runner=self.runner if self.use_ai_clustering else None,
            cache_path=cache_file,
            source_version=source_version(self.db_path)
        )

        use_macro = bool(self.profile.macro_departments) and len(distinct_cats) > 12
        macro_mapping = self.profile.macro_departments if use_macro else {}

        def _agg(member_cats: List[str]) -> Tuple[int, Any, Any]:
            """count, min, max over a set of grouped categories."""
            cnt, lo, hi = 0, None, None
            for c in member_cats:
                st = cat_stats.get(c)
                if not st:
                    continue
                cnt += st[0]
                mn, mx = _fnum(st[1]), _fnum(st[2])
                if mn is not None:
                    lo = mn if lo is None else min(lo, mn)
                if mx is not None:
                    hi = mx if hi is None else max(hi, mx)
            return cnt, lo, hi

        def _samples(member_cats: List[str], cap: int) -> List[str]:
            out: List[str] = []
            for c in member_cats:
                for pid in cat_id_samples.get(c, []):
                    out.append(pid)
                    if len(out) >= cap:
                        return out
            return out

        def _cat_params(raw_cat: str) -> List[str]:
            params = list(cat_eav_params.get(raw_cat, []))
            for f in self.profile.decision_facets.get(raw_cat, []):
                if f not in params:
                    params.append(f)
            return params

        # Build tree structure
        if use_macro and macro_mapping:
            # 3-level structure: Root -> Macro-Department -> Categories -> Brands
            for dept_name, member_cats in macro_mapping.items():
                present = [mc for mc in member_cats if mc in cat_stats]
                if not present:
                    continue
                d_count, d_p_min, d_p_max = _agg(present)

                dept_id = f"dept_{re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ]', '_', dept_name)}"
                dept_params = list(self.profile.decision_facets.get(dept_name, []))
                dept_node = OntoNode(
                    id=dept_id,
                    name=dept_name,
                    level=1,
                    item_count=d_count,
                    price_min=d_p_min,
                    price_max=d_p_max,
                    parameters=dept_params,
                    raw_categories=member_cats,
                    sample_ids=_samples(present, 10)
                )
                self.root.children.append(dept_node)
                self._lookup[dept_id] = dept_node
                self._index_tokens(dept_name, dept_node)

                # Subcategories inside this department
                for mc in present:
                    c_count, c_min, c_max = _agg([mc])
                    clean_name = self._clean_category_name(mc)
                    params = _cat_params(mc)

                    cat_id = f"{dept_id}_cat_{re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ]', '_', mc)}"
                    cat_node = OntoNode(
                        id=cat_id,
                        name=clean_name,
                        level=2,
                        item_count=c_count,
                        price_min=c_min,
                        price_max=c_max,
                        parameters=params[:6],
                        raw_categories=[mc],
                        sample_ids=_samples([mc], 8)
                    )
                    dept_node.children.append(cat_node)
                    self._lookup[cat_id] = cat_node
                    self._index_tokens(clean_name, cat_node)
                    self._index_tokens(mc, cat_node)

                    # Level 3: Brands / Vendors
                    self._attach_brand_nodes(
                        cat_node, brand_stats.get(mc, {}),
                        brand_id_samples.get(mc, {}), params, cat_id)

        else:
            # Standard Clean Structure (<= 12 categories): Root -> Category -> Brands
            for raw_cat, st in sorted(cat_stats.items(), key=lambda kv: -kv[1][0]):
                c_count, c_min, c_max = _agg([raw_cat])
                clean_name = self._clean_category_name(raw_cat)
                params = _cat_params(raw_cat)

                cat_id = f"cat_{re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ]', '_', raw_cat)}"
                cat_node = OntoNode(
                    id=cat_id,
                    name=clean_name,
                    level=1,
                    item_count=c_count,
                    price_min=c_min,
                    price_max=c_max,
                    parameters=params[:6],
                    raw_categories=[raw_cat],
                    sample_ids=_samples([raw_cat], 10)
                )
                self.root.children.append(cat_node)
                self._lookup[cat_id] = cat_node
                self._index_tokens(clean_name, cat_node)
                self._index_tokens(raw_cat, cat_node)

                # Level 2: Brands / Vendors
                self._attach_brand_nodes(
                    cat_node, brand_stats.get(raw_cat, {}),
                    brand_id_samples.get(raw_cat, {}), params, cat_id)

        # Index semantic routing synonyms from the inducted ontology profile
        if self.profile and self.profile.routing_synonyms:
            for synonym, target_cat in self.profile.routing_synonyms.items():
                for node in self._lookup.values():
                    if node.name.lower() == str(target_cat).lower():
                        self._index_tokens(synonym, node)
                        break

        return self.root

    def _attach_brand_nodes(
        self,
        parent_node: OntoNode,
        brands: Dict[str, tuple],
        brand_samples: Dict[str, List[str]],
        params: List[str],
        base_id: str,
    ) -> None:
        """Groups products under a category node by brand/vendor."""
        def _fnum(x: Any) -> Optional[float]:
            try:
                return round(float(x), 2)
            except (TypeError, ValueError):
                return None

        for bname, (cnt, mn, mx) in sorted(brands.items(), key=lambda kv: -kv[1][0]):
            b_id = f"{base_id}_b_{re.sub(r'[^a-zA-Z0-9а-яА-ЯёЁіІїЇєЄґҐ]', '_', bname)}"
            b_node = OntoNode(
                id=b_id,
                name=bname,
                level=parent_node.level + 1,
                item_count=cnt,
                price_min=_fnum(mn),
                price_max=_fnum(mx),
                parameters=params,
                raw_categories=parent_node.raw_categories,
                sample_ids=brand_samples.get(bname, [])
            )
            parent_node.children.append(b_node)
            self._lookup[b_id] = b_node
            self._index_tokens(bname, b_node, direct=True)
            self._index_tokens(f"{parent_node.name} {bname}", b_node, direct=False)

    def _index_tokens(self, text: str, node: OntoNode, *, direct: bool = True) -> None:
        words = re.findall(r"[A-Za-z0-9а-яА-ЯёЁіІїЇєЄґҐ_]+", text.lower())
        for w in words:
            if len(w) >= 2:
                self._token_index.setdefault(w, []).append(node)
                if direct:
                    self._direct_token_index.setdefault(w, []).append(node)

    def navigate(self, query: str) -> Dict[str, Any]:
        """
        Drills down into the taxonomy tree to find the most specific relevant node.
        Returns the category breadcrumb, depth level, item count, price range,
        discovered EAV parameters, and the raw SQL category strings for database querying.
        """
        if not self.root:
            return {"found": False, "message": "Дерево таксономії порожнє"}

        q_lower = query.lower()
        words = re.findall(r"[A-Za-z0-9а-яА-ЯёЁіІїЇєЄґҐ_]+", q_lower)

        hits_by_id: Dict[int, Tuple[Set[str], Set[str], OntoNode]] = {}
        for w in words:
            for node in self._token_index.get(w, ()):
                entry = hits_by_id.setdefault(id(node), (set(), set(), node))
                entry[0].add(w)
            for node in self._direct_token_index.get(w, ()):
                entry = hits_by_id.setdefault(id(node), (set(), set(), node))
                entry[0].add(w)
                entry[1].add(w)

        if not hits_by_id:
            return {
                "found": False,
                "node_id": self.root.id,
                "name": self.root.name,
                "level": 0,
                "item_count": self.root.item_count,
                "price_range": {"min": self.root.price_min, "max": self.root.price_max},
                "available_departments": [c.name for c in self.root.children[:10]],
                "message": "Вкажіть точнішу категорію або товар.",
                "hint": "Категорію чи бренд не знайдено у дереві; для пошуку за назвою чи моделлю використайте catalog_search або sql_query із LIKE.",
            }

        ranked = sorted(
            hits_by_id.values(),
            key=lambda item: (len(item[0]), len(item[1]), item[2].level, item[2].item_count),
            reverse=True,
        )
        best = ranked[0][2]
        other_matches = [
            {
                "node_id": n.id,
                "name": n.name,
                "level": n.level,
                "item_count": n.item_count,
                "price_range": {"min": n.price_min, "max": n.price_max},
                "raw_categories": n.raw_categories,
            }
            for _m, _d, n in ranked[1:6]
            if n.id != best.id and (n.name.lower() == best.name.lower() or len(_m) == len(ranked[0][0]))
        ]

        res = {
            "found": True,
            "node_id": best.id,
            "name": best.name,
            "level": best.level,
            "item_count": best.item_count,
            "price_range": {"min": best.price_min, "max": best.price_max},
            "parameters": best.parameters,
            "sample_ids": best.sample_ids,
            "raw_db_categories": best.raw_categories,
            "sub_branches": [c.name for c in best.children[:12]]
        }
        if other_matches:
            res["other_matches"] = other_matches
        return res

    def to_tree_dict(self) -> Dict[str, Any]:
        """Returns the full hierarchical mindmap tree for Web UI rendering."""
        return self.root.to_dict() if self.root else {}
