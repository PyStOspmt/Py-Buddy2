"""
Py-Buddy: Deterministic ReAct Sanitizer & Zero-Hallucination Gate (Trump 2)
Strict Read-Only SQL AST enforcement, Query injection prevention, Forced LIMIT injection,
and empirical grounding verification.
"""

from typing import ClassVar, Dict, List, Any, Optional, Sequence, Tuple, Set
import os
import sqlite3
import re

class SecurityViolationError(Exception):
    """Raised when an unsafe or mutating query is attempted."""

def escape_like(term: str) -> str:
    """Escape LIKE metacharacters so a user term matches literally.

    `%` and `_` are LIKE wildcards: unescaped, `search=%` pages the
    whole table instead of looking for a literal percent (stress
    finding T4). Queries using the escaped value must append
    `ESCAPE '\\'` to the LIKE clause.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

LIKE_ESCAPE_CLAUSE = " ESCAPE '\\'"

class HallucinationError(Exception):
    """Raised when synthesized claims lack empirical grounding in execution history."""

class SQLSanitizer:
    """
    Guarantees deterministic, read-only SQL execution.
    - Blocks statement chaining (semicolon attacks)
    - Rejects DDL/DML mutating verbs
    - Injects or clamps LIMIT clauses
    - Enforces read-only SQLite connections
    """
    FORBIDDEN_VERBS: ClassVar[Set[str]] = {
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", 
        "REPLACE", "GRANT", "REVOKE", "PRAGMA", "ATTACH", "DETACH", "VACUUM",
        "EXEC", "EXECUTE", "MERGE"
    }

    def __init__(self, default_limit: int = 20, max_limit: int = 50,
                 max_work_steps: int = 5_000_000):
        self.default_limit = default_limit
        self.max_limit = max_limit
        # Per-query CPU budget in SQLite VM steps. Bounded so a query
        # cannot run forever, but large enough for legitimate dense-EAV
        # joins and ORDER BY over mid-size catalogues (the previous
        # hard abort at 50k steps killed a 31-row EAV join on a 37k-row
        # attribute table).
        self.max_work_steps = max_work_steps

    def sanitize_and_prepare_query(self, raw_query: str) -> str:
        """
        Validates the SQL query and injects safe limits.
        Returns the sanitized query ready for execution.
        """
        cleaned = raw_query.strip()
        # Remove trailing semicolon if present
        if cleaned.endswith(";"):
            cleaned = cleaned[:-1].strip()

        # Auto-escape unescaped Ukrainian/Cyrillic internal apostrophes (e.g. 'пам'ять' -> 'пам''ять')
        # to prevent SQLite syntax errors near Cyrillic letter sequences.
        cleaned = re.sub(r"(?<=[а-яА-ЯёЁіІїЇєЄґҐ])['’ʼ](?=[а-яА-ЯёЁіІїЇєЄґҐ])", "''", cleaned)

        # Check for multiple statements via semicolon
        # Mask out semicolons inside single or double quotes
        query_without_strings = re.sub(r"'(?:''|[^'])*'", "''", cleaned)
        query_without_strings = re.sub(r'"(?:""|[^"])*"', '""', query_without_strings)

        if ";" in query_without_strings:
            raise SecurityViolationError("Multi-statement query chaining (semicolon) is strictly prohibited.")

        # Extract tokens (words) from stripped query
        tokens = [t.upper() for t in re.findall(r"\b[A-Za-z_]+\b", query_without_strings)]
        if not tokens:
            raise SecurityViolationError("Empty SQL query.")

        # Ensure query starts with SELECT or WITH
        first_word = tokens[0]
        if first_word not in ("SELECT", "WITH"):
            raise SecurityViolationError(f"Query must begin with SELECT or WITH; got: '{first_word}'")

        # Verify no forbidden mutating verbs exist anywhere in the statement
        for token in tokens:
            if token in self.FORBIDDEN_VERBS:
                raise SecurityViolationError(f"Mutating or unsafe SQL token '{token}' is prohibited.")

        # Limit enforcement
        sanitized = self._enforce_limit(cleaned)
        return sanitized

    def _enforce_limit(self, query: str) -> str:
        """
        Ensures a LIMIT clause is present and does not exceed self.max_limit.

        A parameterised `LIMIT ?` counts as present. It previously did not,
        so the clause below was appended to it and every such query died
        with `near "LIMIT": syntax error` — a failure that the fast path
        swallowed and turned into a silent fall-through to the expensive
        tier.
        """
        from query_awareness import mask_sql_literals

        masked = mask_sql_literals(query)
        limit_match = re.search(r"\bLIMIT\s+(\d+)", masked, flags=re.IGNORECASE)
        if limit_match:
            val = int(limit_match.group(1))
            if val > self.max_limit:
                # Clamp to max_limit
                return query[:limit_match.start(1)] + str(self.max_limit) + query[limit_match.end(1):]
            return query
        if re.search(r"\bLIMIT\s+\?", masked, flags=re.IGNORECASE):
            return query
        # Append default limit
        return f"{query} LIMIT {self.default_limit}"

    def _fuzzy_fallback_search(self, conn: sqlite3.Connection, query: str,
                               limit: int = 5, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Adaptive fallback search when an exact SQL query returns 0 rows.
        Extracts search terms from LIKE or = clauses and matches them with relevance scoring.

        F10: the fallback searches the canonical entity table discovered by
        the source projection, not a hard-coded `products` table.
        """
        try:
            from source_projection import build_source_projection, quote_ident
            proj = None
            if db_path:
                try:
                    proj = build_source_projection(db_path)
                except Exception:
                    proj = None
            if proj is None:
                # Recover the file path from the connection itself.
                try:
                    row = conn.execute("PRAGMA database_list").fetchone()
                    if row and row[2]:
                        proj = build_source_projection(row[2])
                except Exception:
                    proj = None
            if proj is None or proj.entity_table is None:
                return []
            ident = proj.resolve("identity")
            name_res = proj.resolve("name")
            if not ident.resolved or not name_res.column:
                return []
            cat_res = proj.resolve("category")
            price_res = proj.resolve("price")
            q = quote_ident
            t = q(proj.entity_table)

            cursor = conn.cursor()
            cols = {row[1] for row in cursor.execute(f"PRAGMA table_info({t})").fetchall()}
            brand_col = next((c for c in ("brand", "виробник", "manufacturer") if c in cols), None)
            has_price = price_res.column is not None

            # Extract search terms from LIKE clauses or quoted strings
            like_matches = re.findall(r"(?:LIKE|=)\s*['\"]%?([^%'\"\n]+)%?['\"]", query, flags=re.IGNORECASE)
            tokens: List[str] = []
            for m in like_matches:
                words = re.findall(r"[A-Za-z0-9а-яА-ЯёЁіІїЇєЄґҐ]+", m)
                tokens.extend([w.lower() for w in words if len(w) >= 2])
            tokens = list(dict.fromkeys(tokens))
            if not tokens:
                return []

            name_expr = f"LOWER({q(name_res.column)})"
            score_parts = []
            where_parts = []
            for tok in tokens:
                clean_t = tok.replace("'", "''")
                if brand_col:
                    score_parts.append(f"(CASE WHEN {name_expr} LIKE '%{clean_t}%' OR LOWER({q(brand_col)}) LIKE '%{clean_t}%' THEN 2 ELSE 0 END)")
                    where_parts.append(f"{name_expr} LIKE '%{clean_t}%' OR LOWER({q(brand_col)}) LIKE '%{clean_t}%'")
                else:
                    score_parts.append(f"(CASE WHEN {name_expr} LIKE '%{clean_t}%' THEN 2 ELSE 0 END)")
                    where_parts.append(f"{name_expr} LIKE '%{clean_t}%'")

            score_expr = " + ".join(score_parts)
            where_expr = " OR ".join(where_parts)

            # Preserve category hierarchy if present in original query
            cat_match = re.search(r"category\s*=\s*['\"]([^'\"]+)['\"]", query, flags=re.IGNORECASE)
            if cat_match and cat_res.column:
                clean_cat = cat_match.group(1).replace("'", "''")
                where_expr = f"({where_expr}) AND {q(cat_res.column)} = '{clean_cat}'"

            order_clause = f"({score_expr}) DESC" + (f", {q(price_res.column)} ASC" if has_price else "")

            cols_to_select = [f"{q(ident.column)}", f"{q(name_res.column)}"]
            if cat_res.column:
                cols_to_select.append(f"{q(cat_res.column)}")
            if brand_col:
                cols_to_select.append(f"{q(brand_col)}")
            if has_price:
                cols_to_select.append(f"{q(price_res.column)}")

            fallback_sql = f"SELECT {', '.join(cols_to_select)} FROM {t} WHERE {where_expr} ORDER BY {order_clause} LIMIT {int(limit)}"
            cursor.execute(fallback_sql)
            rows = cursor.fetchall()
            col_names = [d[0] for d in cursor.description]
            return [dict(zip(col_names, r)) for r in rows]
        except Exception:
            return []

    def execute_safe_query(
        self,
        db_path: str,
        query: str,
        params: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the sanitized query against a read-only SQLite database.

        `params` are bound through the driver, never interpolated into the
        SQL text. Queries the system builds itself (see
        `catalog_query.build_extremum_sql`) use `?` placeholders, and binding
        is the reason this is safer than string formatting: a catalogue
        value containing a quote can never become executable SQL.
        """
        sanitized_query = self.sanitize_and_prepare_query(query)

        # F31: fail-closed. If the read-only URI connection cannot be opened
        # there is no fallback to a writable connection — refusing to serve
        # the query is the only safe outcome.
        uri_path = "file:%s?mode=ro" % os.path.abspath(db_path).replace("\\", "/")
        try:
            conn = sqlite3.connect(uri_path, uri=True)
        except Exception as exc:
            raise SecurityViolationError(
                f"Read-only connection to the database failed ({exc}); "
                "refusing to fall back to a writable connection.")

        conn.row_factory = sqlite3.Row
        # F31: engine-level policy, not token matching. The authorizer denies
        # everything except plain reads, and query_only makes the connection
        # unable to write even if a check above is ever bypassed.
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(self._read_only_authorizer)
        # F31: a CPU/work budget. The handler fires every ~50k VM steps;
        # returning non-zero aborts the query once the budget is spent.
        _remaining = [self.max_work_steps]
        def _work_budget() -> int:
            _remaining[0] -= 50_000
            return 1 if _remaining[0] <= 0 else 0
        conn.set_progress_handler(_work_budget, 50_000)
        cursor = conn.cursor()

        try:
            bound = list(params) if params else None
            if bound is not None:
                # F31: a bound LIMIT value is clamped like a literal one.
                sanitized_query, bound = self._clamp_bound_limit(sanitized_query, bound)
                cursor.execute(sanitized_query, bound)
            else:
                cursor.execute(sanitized_query)
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description] if cursor.description else []
            result_rows = [dict(row) for row in rows]

            res: Dict[str, Any] = {
                "success": True,
                "sanitized_query": sanitized_query,
                "columns": col_names,
                "rows": result_rows,
                "row_count": len(result_rows)
            }
            # F04: provenance of every result column. Only "source" columns
            # (bare columns of real tables) may ground catalogue facts.
            try:
                res["column_provenance"] = self._projection_provenance(
                    sanitized_query, conn, col_names)
            except Exception:
                # Fail-closed: if provenance cannot be established, nothing
                # from this result may ground facts.
                res["column_provenance"] = {c: "derived" for c in col_names}

            try:
                from query_awareness import annotate_sql_window
                res.update(annotate_sql_window(
                    conn, sanitized_query, bound, result_rows, self.max_work_steps))
            except Exception:
                pass

            # Adaptive Zero-Row Recovery: If no rows matched, attempt fuzzy fallback
            if len(result_rows) == 0:
                fuzzy_hits = self._fuzzy_fallback_search(conn, sanitized_query, limit=5, db_path=db_path)
                if fuzzy_hits:
                    res["fuzzy_suggestions"] = fuzzy_hits
                    res["hint"] = (
                        f"Точного збігу за критеріями не знайдено (0 рядків), але в базі знайдено {len(fuzzy_hits)} схожих товарів. "
                        f"Рекомендовано використати ці товари або уточнити запит: "
                        + ", ".join(h.get("name", "") for h in fuzzy_hits)
                    )

            return res
        except Exception as e:
            msg = str(e)
            if "interrupted" in msg.lower():
                msg = (f"query interrupted: work budget exceeded "
                       f"({self.max_work_steps} VM steps)")
            return {
                "success": False,
                "sanitized_query": sanitized_query,
                "error": msg,
                "rows": [],
                "row_count": 0
            }
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------
    # F31: engine-level read-only policy and work budget.
    # ------------------------------------------------------------------
    # SQLite authorizer action codes. The interpreter's own constants are
    # used when available; the integers are the documented fallback.
    _A_OK = getattr(sqlite3, "SQLITE_OK", 0)
    _A_DENY = getattr(sqlite3, "SQLITE_DENY", 1)
    _A_PRAGMA = getattr(sqlite3, "SQLITE_PRAGMA", 19)
    _A_READ = getattr(sqlite3, "SQLITE_READ", 20)
    _A_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
    _A_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
    _A_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)

    @classmethod
    def _read_only_authorizer(cls, action, arg1, arg2, db_name, trigger):
        if action in (cls._A_SELECT, cls._A_READ, cls._A_FUNCTION, cls._A_RECURSIVE):
            return cls._A_OK
        if action == cls._A_PRAGMA and arg1 in ("table_info", "query_only"):
            return cls._A_OK
        return cls._A_DENY

    def _clamp_bound_limit(self, query: str,
                           params: List[Any]) -> Tuple[str, List[Any]]:
        """Clamp a bound `LIMIT ?` value exactly like a literal LIMIT."""
        from query_awareness import mask_sql_literals

        masked = mask_sql_literals(query)
        m = re.search(r"\bLIMIT\s+\?", masked, flags=re.IGNORECASE)
        if not m:
            return query, params
        idx = masked[:m.start()].count("?")
        if idx < len(params):
            try:
                val = int(params[idx])
            except (TypeError, ValueError):
                return query, params
            if val > self.max_limit:
                params[idx] = self.max_limit
        return query, params

    # ------------------------------------------------------------------
    # F04: projection provenance. Read-only is not provenance: a SELECT that
    # manufactures its own name and price literals must not ground facts.
    # ------------------------------------------------------------------
    @staticmethod
    def _table_columns(conn: sqlite3.Connection) -> Dict[str, Set[str]]:
        """table -> set of column names, for every table in the database."""
        out: Dict[str, Set[str]] = {}
        try:
            for (tname,) in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"):
                try:
                    safe = str(tname).replace('"', '""')
                    out[str(tname)] = {
                        str(row[1]) for row in conn.execute(f'PRAGMA table_info("{safe}")')
                    }
                except Exception:
                    continue
        except Exception:
            # A table that cannot be introspected is absent from the
            # provenance map; its columns default to "derived".
            pass
        return out

    @staticmethod
    def _split_top_level(segment: str) -> List[str]:
        """Split a SELECT list on commas that are not inside parens/strings."""
        parts: List[str] = []
        depth = 0
        cur: List[str] = []
        in_str = False
        i = 0
        while i < len(segment):
            ch = segment[i]
            if in_str:
                cur.append(ch)
                if ch == "'":
                    if i + 1 < len(segment) and segment[i + 1] == "'":
                        cur.append("'")
                        i += 1
                    else:
                        in_str = False
            elif ch == "'":
                in_str = True
                cur.append(ch)
            elif ch == "(":
                depth += 1
                cur.append(ch)
            elif ch == ")":
                depth -= 1
                cur.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(cur))
                cur = []
            else:
                cur.append(ch)
            i += 1
        if cur:
            parts.append("".join(cur))
        return parts

    @staticmethod
    def _outermost_select_segment(
            query: str) -> Optional[Tuple[str, List[Tuple[Optional[str], str]]]]:
        """
        Returns (select_list, tables) for the outermost SELECT: the first
        SELECT token at parenthesis depth 0, the text up to the next depth-0
        FROM, and the (alias, table) references of that FROM clause.
        Returns None when the shape cannot be parsed (fail-closed).
        """
        depth = 0
        select_start: Optional[int] = None
        select_end: Optional[int] = None
        tables: List[Tuple[Optional[str], str]] = []
        state = "before"
        alias_expect = False
        tokens = re.finditer(
            r"\b(SELECT|FROM|JOIN|LEFT|RIGHT|INNER|OUTER|CROSS|AS|ON|WHERE|"
            r"GROUP|ORDER|LIMIT|HAVING|WINDOW|UNION)\b|,|\(|\)",
            query, flags=re.IGNORECASE)
        for m in tokens:
            tok = m.group(0)
            if tok == "(":
                depth += 1
                continue
            if tok == ")":
                depth -= 1
                continue
            if depth != 0:
                continue
            up = tok.upper()
            if state == "before" and up == "SELECT":
                state = "select_list"
                select_start = m.end()
            elif state == "select_list" and up == "FROM":
                select_end = m.start()
                state = "from_clause"
                alias_expect = False
            elif state == "from_clause":
                if up in ("LEFT", "RIGHT", "INNER", "OUTER", "CROSS"):
                    continue
                if up == "JOIN":
                    alias_expect = False
                elif up == "AS":
                    continue
                elif up in ("ON", "WHERE", "GROUP", "ORDER", "LIMIT",
                            "HAVING", "WINDOW"):
                    state = "after_from"
                elif up == "UNION":
                    # A second top-level arm could project anything; the
                    # provenance of the combined columns is unknowable here.
                    return None
                elif tok == ",":
                    alias_expect = False
                elif not alias_expect:
                    tables.append((None, tok))
                    alias_expect = True
                else:
                    tables[-1] = (tok, tables[-1][1])
                    alias_expect = False
        if select_start is None:
            return None
        select_list = query[select_start: select_end if select_end is not None else len(query)]
        return select_list, tables

    def _projection_provenance(self, query: str, conn: sqlite3.Connection,
                               col_names: Sequence[str]) -> Dict[str, str]:
        """
        F04: classify each result column of the outermost SELECT as
        "source" (a bare column of a real table) or "derived" (a literal,
        an expression, an aggregate, or an alias of one). Fail-closed:
        anything the classifier cannot pin to a table column is "derived".
        """
        table_columns = self._table_columns(conn)
        all_cols: Set[str] = set()
        for cols in table_columns.values():
            all_cols.update(cols)
        parsed = self._outermost_select_segment(query)
        if parsed is None:
            return {c: "derived" for c in col_names}
        select_list, tables = parsed
        from_tables = {t for _a, t in tables if t}
        prov: Dict[str, str] = {}
        for item in self._split_top_level(select_list):
            stripped = item.strip()
            if not stripped:
                continue
            if stripped == "*":
                for tname in from_tables:
                    all_cols.update(table_columns.get(tname, set()))
                    for col in table_columns.get(tname, set()):
                        prov.setdefault(col, "source")
                continue
            m_qstar = re.match(r'^([\w"\[\]]+)\.\*$', stripped)
            if m_qstar:
                alias = m_qstar.group(1).strip('"[]')
                for a, tname in tables:
                    if a == alias or (a is None and tname == alias):
                        for col in table_columns.get(tname, set()):
                            prov.setdefault(col, "source")
                continue
            alias_name: Optional[str] = None
            m_as = re.match(
                r'^(.*?)\s+AS\s+("?[\w]+"?|\[[\w]+\])$',
                stripped, flags=re.IGNORECASE)
            if m_as:
                expr, alias_name = m_as.group(1).strip(), m_as.group(2).strip('"[]')
            else:
                expr = stripped
            m_col = re.match(r'^([\w"\[\]]+)(?:\.([\w"\[\]]+))?$', expr)
            if m_col:
                colpart = (m_col.group(2) or m_col.group(1)).strip('"[]')
                if colpart in all_cols:
                    prov[alias_name or colpart] = "source"
                    continue
            # Literal / expression / aggregate / unknown -> derived.
            prov.setdefault(alias_name or stripped, "derived")
        # Any result column the parser never saw is derived (fail-closed).
        return {c: prov.get(c, "derived") for c in col_names}

class ZeroHallucinationGate:
    """
    Verifies that all entities, product names, IDs, and prices stated in the final
    synthesis are strictly grounded in empirical query outputs stored in execution_history.
    """
    @staticmethod
    def extract_grounded_catalog(execution_history: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Builds a lookup index of all verified products returned by database queries
        or hypergraph lookups in the current session.
        Keyed by string product ID and lowercased title/name.
        """
        grounded_items: Dict[str, Dict[str, Any]] = {}
        
        for record in execution_history:
            rows = record.get("result", {}).get("rows", [])
            for row in rows:
                p_id = str(row.get("id") or row.get("product_id") or row.get("item_id") or "")
                title = str(row.get("name") or row.get("title") or row.get("model") or "")
                
                if p_id:
                    grounded_items[p_id.lower()] = row
                if title:
                    grounded_items[title.lower()] = row
                    
        return grounded_items

    @classmethod
    def verify_recommendation_grounding(
        cls, 
        claimed_items: List[Dict[str, Any]], 
        execution_history: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Checks claimed products against empirical history.
        Each claimed item should have: {"id": ..., "price": ..., "name": ...}
        Returns (is_valid, list_of_violations).
        """
        grounded = cls.extract_grounded_catalog(execution_history)
        violations = []

        if not claimed_items:
            return True, []

        for item in claimed_items:
            c_id = str(item.get("id", "")).lower()
            c_name = str(item.get("name", "")).lower()
            c_price = item.get("price")

            # Must match either by ID or name in grounded catalog
            match = grounded.get(c_id) or grounded.get(c_name)
            if not match:
                violations.append(f"Product '{item.get('name') or c_id}' was NEVER retrieved from the database (Hallucination).")
                continue

            # Verify price consistency if price is claimed
            if c_price is not None:
                grounded_price = match.get("price") or match.get("cost") or match.get("amount")
                if grounded_price is not None:
                    # 1% relative tolerance (the documented contract).
                    # The previous absolute 1.0 band was +/-20% on a 5 UAH
                    # item and +/-0.002% on a 50000 UAH one -- a tolerance
                    # that changes meaning with the currency scale is not a
                    # tolerance, it is a rounding accident.
                    actual = float(grounded_price)
                    claimed = float(c_price)
                    tolerance = max(0.01 * abs(actual), 1e-9)
                    if abs(claimed - actual) > tolerance:
                        violations.append(
                            f"Price discrepancy for '{item.get('name')}': claimed {c_price}, actual in DB is {grounded_price}."
                        )

        return (len(violations) == 0), violations
