"""
Py-Buddy: Deterministic Catalogue Access for the Constraint Layer (ADR-012).

The LLM supplies optimization intent only (category, budget, minimum performance);
all candidate entities are materialized by verified SQL SELECT queries attached to
execution_history so the grounding gate can audit every final claim.
"""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Dict, List, Optional, Sequence, Set, Tuple

from pareto_relaxer import CandidateSolution
from source_projection import build_source_projection, quote_ident

#: How much above the stated budget we are willing to look. A hard budget
#: with an empty result is the "empty-set deadlock" the project exists to
#: fix, but the relaxation must be visible and bounded, never unbounded.
BUDGET_RELAX_FACTOR = 1.5

#: Ceiling on how many entities one tool call may bring into the core.
MAX_CANDIDATES = 40


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect("file:%s?mode=ro" % os.path.abspath(db_path).replace("\\", "/"),
                           uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def _get_columns(conn: sqlite3.Connection, table_name: str) -> Set[str]:
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cur.fetchall()}


def _projection_for(db_path: str):
    """Canonical projection of the source, or None when it cannot be built."""
    try:
        return build_source_projection(db_path)
    except Exception:
        return None


def fetch_candidates(
    db_path: str,
    category: Optional[str],
    budget_max: Optional[float] = None,
    min_performance: Optional[float] = None,
    limit: int = MAX_CANDIDATES,
    *,
    hard_limit: int = MAX_CANDIDATES,
) -> List[CandidateSolution]:
    """
    Materialise real catalogue rows as Pareto candidates.

    F10: the query is built from the canonical source projection, not from a
    hard-coded `products` shape. A source without a resolvable price column
    still yields candidates -- their price is None and the budget filter is
    honestly skipped, because there is no price axis to filter on.

    `budget_max`/`min_performance` bound a SCORING POOL with a declared
    margin (BUDGET_RELAX_FACTOR, -15) -- the fuzzy relaxer needs near-miss
    items to grade. They are NOT a feasibility test: strict or relaxed
    satisfaction of the stated bounds is decided by the NVFG solver on a
    pool fetched WITHOUT these bounds (pass None, as pareto_relax now does).
    `hard_limit` lifts the display cap for that solver pool; the caller is
    responsible for reporting pool truncation in `search_space`.
    """
    limit = max(1, min(int(limit), int(hard_limit)))
    proj = _projection_for(db_path)
    if proj is None or proj.entity_table is None:
        return []
    ident = proj.resolve("identity")
    if not ident.resolved:
        return []
    name_res = proj.resolve("name")
    cat_res = proj.resolve("category")
    price_res = proj.resolve("price")
    perf_res = proj.resolve("performance")
    stock_res = proj.resolve("stock")

    conn = _connect(db_path)
    try:
        q = quote_ident
        select_parts = [
            f"p.{q(ident.column)} AS id",
            f"p.{q(name_res.column)} AS name" if name_res.column else f"p.{q(ident.column)} AS name",
            f"p.{q(price_res.column)} AS price" if price_res.column else "NULL AS price",
            f"p.{q(perf_res.column)} AS performance_score" if perf_res.column else "NULL AS performance_score",
        ]

        # Merchant enrichment is a marketplace-schema feature; for any other
        # shape it is absent (F18: absent stays absent -- no invented delivery
        # days, no invented ratings).
        cols = _get_columns(conn, proj.entity_table)
        if proj.entity_table == "products" and _has_table(conn, "merchants") and "merchant_id" in cols:
            select_parts.append("m.delivery_days AS delivery_days")
            select_parts.append("m.rating AS merchant_rating")
            from_clause = f"FROM {q(proj.entity_table)} AS p LEFT JOIN merchants m ON m.id = p.merchant_id"
        else:
            select_parts.append("NULL AS delivery_days")
            select_parts.append("NULL AS merchant_rating")
            from_clause = f"FROM {q(proj.entity_table)} AS p"

        where_parts = []
        if stock_res.column:
            where_parts.append(f"p.{q(stock_res.column)} > 0")

        params: List[object] = []
        if category and cat_res.column:
            where_parts.append(f"p.{q(cat_res.column)} = ?")
            params.append(category)
        if budget_max is not None and budget_max > 0 and price_res.column:
            where_parts.append(f"p.{q(price_res.column)} <= ?")
            params.append(round(float(budget_max) * BUDGET_RELAX_FACTOR, 2))
        if min_performance is not None and perf_res.column:
            where_parts.append(f"p.{q(perf_res.column)} >= ?")
            params.append(float(min_performance) - 15.0)   # bounded relaxation

        order_by = f"p.{q(price_res.column)} ASC" if price_res.column else f"p.{q(ident.column)} ASC"
        where_sql = " AND ".join(where_parts) if where_parts else "1=1"
        sql = f"SELECT {', '.join(select_parts)} {from_clause} WHERE {where_sql} ORDER BY {order_by} LIMIT ?"
        params.append(limit)

        rows = conn.execute(sql, params).fetchall()
        return [
            CandidateSolution(
                item_id=str(r["id"]),
                name=str(r["name"]),
                price=float(r["price"]) if r["price"] is not None else None,
                performance_score=float(r["performance_score"]) if r["performance_score"] is not None else None,
                delivery_days=int(r["delivery_days"]) if r["delivery_days"] is not None else None,
            )
            for r in rows
        ]
    finally:
        conn.close()


#: F34: the mention resolver normalises names in Python, so the candidate
#: set cannot be produced by a single LIKE. The row scan is therefore
#: explicitly bounded: within the scanned prefix the match is exact; beyond
#: it the resolver may miss (a miss degrades to the general agent path --
#: it can never fabricate a resolution).
_MENTION_SCAN_CAP = 50_000


def resolve_mentions(
    db_path: str,
    text: str,
    *,
    category: Optional[str] = None,
    limit: int = 12,
    scan_cap: int = _MENTION_SCAN_CAP,
) -> List[CandidateSolution]:
    """
    Find catalogue products the user literally named.
    Matching is longest-first so that a longer name is preferred over a
    shorter one that happens to be a prefix of it.
    """
    if not text:
        return []
    proj = _projection_for(db_path)
    if proj is None or proj.entity_table is None:
        return []
    ident = proj.resolve("identity")
    if not ident.resolved:
        return []
    name_res = proj.resolve("name")
    cat_res = proj.resolve("category")
    price_res = proj.resolve("price")
    perf_res = proj.resolve("performance")

    q = quote_ident
    name_sel = f"p.{q(name_res.column)} AS name" if name_res.column else f"p.{q(ident.column)} AS name"
    price_sel = f"p.{q(price_res.column)} AS price" if price_res.column else "NULL AS price"
    perf_sel = f"p.{q(perf_res.column)} AS performance_score" if perf_res.column else "NULL AS performance_score"
    sql = f"SELECT p.{q(ident.column)} AS id, {name_sel}, {price_sel}, {perf_sel} FROM {q(proj.entity_table)} AS p"
    params: List[object] = []
    if category and cat_res.column:
        sql += f" WHERE p.{q(cat_res.column)}=?"
        params.append(category)
    sql += f" LIMIT {max(1, int(scan_cap))}"

    conn = _connect(db_path)
    try:
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    haystack = _key(text)
    hits: List[CandidateSolution] = []
    for r in sorted(rows, key=lambda x: -len(str(x["name"]))):
        if _key(r["name"]) and _key(r["name"]) in haystack:
            hits.append(CandidateSolution(
                item_id=str(r["id"]), name=str(r["name"]),
                price=float(r["price"]) if r["price"] is not None else None,
                performance_score=float(r["performance_score"]) if r["performance_score"] is not None else None,
                delivery_days=None,
            ))
            if len(hits) >= limit:
                break
    return hits


def _key(text: str) -> str:
    """Normalisation shared by the resolver and the metric's name matcher."""
    import unicodedata as _ud
    s = _ud.normalize("NFKD", (text or "").lower())
    s = "".join(ch for ch in s if not _ud.combining(ch))
    return re.sub(r"[\s_\-.,()\[\]\"'«»]+", "", s)


def fetch_by_ids(db_path: str, item_ids: Sequence[str]) -> List[CandidateSolution]:
    """Materialise an explicit set of catalogue rows by id."""
    if not item_ids:
        return []
    proj = _projection_for(db_path)
    if proj is None or proj.entity_table is None:
        return []
    ident = proj.resolve("identity")
    if not ident.resolved:
        return []
    name_res = proj.resolve("name")
    price_res = proj.resolve("price")
    perf_res = proj.resolve("performance")

    conn = _connect(db_path)
    try:
        q = quote_ident
        select_parts = [
            f"p.{q(ident.column)} AS id",
            f"p.{q(name_res.column)} AS name" if name_res.column else f"p.{q(ident.column)} AS name",
            f"p.{q(price_res.column)} AS price" if price_res.column else "NULL AS price",
            f"p.{q(perf_res.column)} AS performance_score" if perf_res.column else "NULL AS performance_score",
        ]
        # F18: absent merchant data stays absent. A COALESCE default here used
        # to invent a 3-day delivery for every merchant that had none.
        cols = _get_columns(conn, proj.entity_table)
        if proj.entity_table == "products" and _has_table(conn, "merchants") and "merchant_id" in cols:
            select_parts.append("m.delivery_days AS delivery_days")
            from_clause = f"FROM {q(proj.entity_table)} AS p LEFT JOIN merchants m ON m.id = p.merchant_id"
        else:
            select_parts.append("NULL AS delivery_days")
            from_clause = f"FROM {q(proj.entity_table)} AS p"

        marks = ",".join("?" * len(item_ids))
        sql = f"SELECT {', '.join(select_parts)} {from_clause} WHERE p.{q(ident.column)} IN ({marks})"
        rows = conn.execute(sql, list(item_ids)).fetchall()
        return [
            CandidateSolution(
                item_id=str(r["id"]), name=str(r["name"]),
                price=float(r["price"]) if r["price"] is not None else None,
                performance_score=float(r["performance_score"]) if r["performance_score"] is not None else None,
                delivery_days=int(r["delivery_days"]) if r["delivery_days"] is not None else None,
            )
            for r in rows
        ]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# Read-only SQL construction for the fast path
# --------------------------------------------------------------------------

# Superlatives are a CLOSED linguistic class (like prepositions), not domain
# vocabulary: every language has them and they are not catalogue content.
# This is the one place a short word list is legitimate, and it is kept here
# rather than scattered through the router. Everything else -- brand names,
# categories, product names -- must come from the database.
_EXTREMUM_PATTERNS = (
    ("ASC", re.compile(r"\b(?:найдешевш\w*|щонайдешев\w*|cheapest|least\s+expensive)\b", re.IGNORECASE)),
    ("DESC", re.compile(r"\b(?:найдорожч\w*|преміальн\w*|most\s+expensive)\b", re.IGNORECASE)),
)

# Structural fast-path gate: every remaining content token after removing the
# superlative and catalogue-matched values must be a closed-class function word
# or generic item designator; any unexplained token falls back to Tier-2 ReAct.
_FUNCTION_TOKENS = frozenset({
    "який", "яка", "яке", "які", "скільки", "в", "у", "на", "і", "та",
    "з", "зі", "для", "чи", "будь", "ласка", "є", "по", "точна",
    "точний", "точне", "ціна", "ціну", "ціни", "вартість", "коштує",
    "наявності", "наявність", "магазин", "магазині", "магазину",
    "the", "a", "an", "what", "which", "is", "are", "how", "much",
    "does", "do", "cost", "of", "in", "for", "and", "or", "please",
    "show", "me", "tell", "stock", "shop", "store",
})
_ENTITY_TOKENS = frozenset({
    "товар", "товари", "товару", "продукт", "продукти", "позиція",
    "позиції", "деталь", "деталі", "модель", "моделі", "штука",
    "item", "items", "product", "products", "thing", "model",
    "models", "one",
})
_TOKEN_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)?", re.UNICODE)


def detect_extremum(text: str) -> Optional[str]:
    """Return 'ASC', 'DESC' or None for a price superlative."""
    t = text or ""
    for direction, pattern in _EXTREMUM_PATTERNS:
        if pattern.search(t):
            return direction
    return None


def _unexplained_tokens(text: str, matched_terms: Sequence[str]) -> List[str]:
    """
    Content tokens the fast path cannot express: every token must be a
    function word, a generic entity noun, or part of a catalogue value
    the SQL actually filters on. A budget ("за 500"), a spec ("16 гб"),
    or an unresolved selector ("материнська плата") stays unexplained.
    """
    residual = (text or "").lower()
    for _direction, pattern in _EXTREMUM_PATTERNS:
        residual = pattern.sub(" ", residual)
    hit_tokens = set()
    for term in matched_terms:
        hit_tokens.update(str(term).lower().split())
    return [tok for tok in _TOKEN_RE.findall(residual)
            if tok not in _FUNCTION_TOKENS and tok not in _ENTITY_TOKENS
            and tok not in hit_tokens and not tok.isdigit()]


def build_extremum_sql(
    db_path: str,
    text: str,
    *,
    limit: int = 1,
) -> Optional[Tuple[str, List[object]]]:
    """
    Build a read-only superlative query from the catalogue's own vocabulary.

    The previous implementation hard-coded two brand names:

        if "найдешевш" in q and "amd" in q:  ... LIKE '%ryzen%' ...
        if "найдешевш" in q and "intel" in q: ... LIKE '%intel%' ...

    A store that sold anything else fell through to the expensive tier, and
    a store that sold something else with "amd" in the name got a
    confidently wrong answer. The filter here is built from the distinct
    brand and category values that actually exist in this catalogue, and the
    terms are passed as bound parameters.

    Returns (sql, params), or None when the text carries no superlative.
    """
    direction = detect_extremum(text)
    if direction is None:
        return None

    # F10: the query is built from the canonical source projection. A source
    # without a resolvable price column has no "cheapest/most expensive"
    # answer at all, so the superlative path honestly declines (None) instead
    # of guessing a column.
    proj = _projection_for(db_path)
    if proj is None or proj.entity_table is None:
        return None
    ident = proj.resolve("identity")
    name_res = proj.resolve("name")
    price_res = proj.resolve("price")
    cat_res = proj.resolve("category")
    stock_res = proj.resolve("stock")
    if not ident.resolved or not name_res.column or not price_res.column:
        return None

    q = quote_ident
    t = q(proj.entity_table)
    conn = _connect(db_path)
    try:
        brand_col = next((c for c in ("brand", "виробник", "manufacturer")
                          if c in _get_columns(conn, proj.entity_table)), None)
        brands = ([r[0] for r in conn.execute(f"SELECT DISTINCT {q(brand_col)} FROM {t} WHERE {q(brand_col)} IS NOT NULL")]
                  if brand_col else [])
        categories = ([r[0] for r in conn.execute(f"SELECT DISTINCT {q(cat_res.column)} FROM {t} WHERE {q(cat_res.column)} IS NOT NULL")]
                      if cat_res.column else [])
    finally:
        conn.close()

    low = (text or "").lower()

    def mentioned(values: Sequence[str]) -> List[str]:
        hits = [v for v in values if v and v.lower() in low]
        return sorted(hits, key=len, reverse=True)   # longest wins

    brand_hits = mentioned(brands)
    cat_hits = mentioned(categories)

    # F49: every content token of the query must be explained -- the
    # superlative phrase, a catalogue value the SQL filters on, or a
    # closed-class word. Anything else means the query asks for more
    # than a bare price extremum (a spec, a budget, compatibility, an
    # unresolved category name), and the fast path honestly declines.
    if _unexplained_tokens(text, brand_hits + cat_hits):
        return None

    where = [f"{q(price_res.column)} IS NOT NULL"]
    if stock_res.column:
        where.append(f"{q(stock_res.column)} > 0")
    params: List[object] = []
    if brand_hits and brand_col:
        where.append(f"LOWER({q(brand_col)}) IN (%s)" % ",".join("?" * len(brand_hits)))
        params.extend(b.lower() for b in brand_hits)
    elif cat_hits and cat_res.column:
        where.append(f"{q(cat_res.column)} IN (%s)" % ",".join("?" * len(cat_hits)))
        params.extend(cat_hits)

    cat_sel = q(cat_res.column) if cat_res.column else "'general' as category"
    sql = (f"SELECT {q(ident.column)}, {q(name_res.column)}, {cat_sel}, {q(price_res.column)} "
           f"FROM {t} WHERE {' AND '.join(where)} ORDER BY {q(price_res.column)} {direction} LIMIT ?")
    params.append(max(1, int(limit)))
    return sql, params


# --------------------------------------------------------------------------
# Grounding set for the Zero-Hallucination Gate
# --------------------------------------------------------------------------


class GroundingSet:
    """
    The only products and prices a final answer is allowed to mention.

    Built from tool results, never from model output. ADR-012: the gate
    verifies against this, and `recommended_items` is projected out of it
    rather than accepted from the LLM.
    """

    #: Bound on retained source rows so an over-broad query cannot grow
    #: the per-run evidence set without limit.
    _MAX_ROWS = 500

    def __init__(self) -> None:
        self.names: Set[str] = set()
        # F19: every observed price is kept. The old first-value-wins dict
        # merged distinct offers of one name into whichever price arrived
        # first, silently discarding the others.
        self.prices: Dict[str, List[float]] = {}
        self.ids: Set[str] = set()
        self._name_to_id: Dict[str, str] = {}
        # F22: the raw admitted rows, kept for the post-generation claim
        # verifier -- sentence-level claims need attribute values, not
        # just the name/price projection.
        self.rows: List[Dict[str, object]] = []

    def add_solution(self, c: CandidateSolution) -> None:
        self.names.add(c.name)
        if c.price is not None:
            self.prices.setdefault(c.name, []).append(c.price)
        self.ids.add(c.item_id)
        self._name_to_id.setdefault(c.name, c.item_id)

    def extend(self, solutions: Sequence[CandidateSolution]) -> None:
        for c in solutions:
            self.add_solution(c)

    def extend_rows(self, rows: Sequence[Dict[str, object]],
                    provenance: Optional[Dict[str, str]] = None) -> None:
        """
        Add rows from a sanitised SELECT result.

        Accepts the dict shape produced by `SQLSanitizer.execute_safe_query`.
        Rows whose price cannot be read are still admitted as names: an
        unpriced name is verifiable, an unpriced price is not.

        F04: `provenance` maps result column names to "source" or "derived".
        A derived `name` column (a literal, an expression, an alias of one)
        means the row's identity was manufactured by the query, not read
        from a table — the row grounds nothing. A derived `price` column
        still admits the name, but never the number.
        """
        for row in rows:
            if not isinstance(row, dict):
                continue
            if provenance is not None and provenance.get("name", "source") != "source":
                continue
            name = row.get("name")
            if not name:
                continue
            self.names.add(str(name))
            if len(self.rows) < self._MAX_ROWS:
                self.rows.append(dict(row))
            if row.get("id"):
                self.ids.add(str(row["id"]))
                self._name_to_id.setdefault(str(name), str(row["id"]))
            for key in ("price", "ЦІНА", "цена"):
                if key in row:
                    if provenance is not None and provenance.get(key, "source") != "source":
                        break
                    try:
                        self.prices.setdefault(str(name), []).append(float(row[key]))
                    except (TypeError, ValueError):
                        # Unparseable price cell: the product contributes
                        # no price observation.
                        pass
                    break

    def projected_recommendations(self) -> List[Dict[str, object]]:
        """
        The recommended items a response is allowed to contain.

        Derived entirely from grounded rows. If the model wanted to
        recommend something, that thing had to come back from a query.

        F19: when the grounded rows disagree on a name's price, the
        projection reports the conflict instead of quietly picking the
        first observed value.
        """
        out: List[Dict[str, object]] = []
        for name in sorted(self.names):
            observed = self.prices.get(name, [])
            distinct = sorted(set(observed))
            entry: Dict[str, object] = {
                "id": self._name_to_id.get(name, ""), "name": name,
                "price": distinct[0] if len(distinct) == 1 else None,
            }
            if len(distinct) > 1:
                entry["price_conflicts"] = distinct
            out.append(entry)
        return out

    def verification_rows(self) -> List[Dict[str, object]]:
        """
        Raw evidence rows for the post-generation claim verifier.

        The admitted SELECT rows first (they carry whatever attribute
        columns the run actually retrieved); the projected rows as a
        fallback so a name+price-only run still yields verifiable
        evidence -- and a spec claim about an attribute nobody queried
        still has nothing to lean on.
        """
        out = list(self.rows)
        covered = {str(r.get("name")) for r in out}
        for entry in self.projected_recommendations():
            if entry["name"] not in covered:
                out.append(dict(entry))
        return out
