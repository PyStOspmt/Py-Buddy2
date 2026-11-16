"""Domain-agnostic SQL window telemetry, catalog search, and ReAct cycle state helpers."""
from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

_AGG_CALL_RE = re.compile(
    r"\b(?:COUNT|MIN|MAX|SUM|AVG|TOTAL|GROUP_CONCAT)\s*\(", re.IGNORECASE
)
_DEPTH_CLAUSE_RE = re.compile(
    r"\b(SELECT|FROM|GROUP\s+BY|HAVING|WINDOW|UNION|INTERSECT|EXCEPT|ORDER\s+BY|LIMIT|OFFSET)\b",
    re.IGNORECASE,
)


def mask_sql_literals(sql: str) -> str:
    """Replace single-quoted literals and comments with equal-length spaces."""
    n = len(sql)
    out = list(sql)
    i = 0
    while i < n:
        ch = sql[i]
        if ch == "'":
            out[i] = " "
            i += 1
            while i < n:
                if sql[i] == "'":
                    out[i] = " "
                    i += 1
                    if i < n and sql[i] == "'":
                        out[i] = " "
                        i += 1
                        continue
                    break
                out[i] = " "
                i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] != "\n":
                out[i] = " "
                i += 1
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            out[i] = " "
            out[i + 1] = " "
            i += 2
            while i < n:
                if sql[i] == "*" and i + 1 < n and sql[i + 1] == "/":
                    out[i] = " "
                    out[i + 1] = " "
                    i += 2
                    break
                out[i] = " " if sql[i] != "\n" else "\n"
                i += 1
            continue
        i += 1
    return "".join(out)


def _depth_zero_positions(masked: str) -> Tuple[List[Tuple[int, int, str]], List[int]]:
    """Return outermost depth-0 clause spans and depth map for every character."""
    clauses: List[Tuple[int, int, str]] = []
    depths = [0] * len(masked)
    depth = 0
    in_dq = False
    for i, ch in enumerate(masked):
        if ch == '"':
            in_dq = not in_dq
        elif not in_dq:
            if ch == "(":
                depths[i] = depth
                depth += 1
                continue
            if ch == ")":
                depth = max(0, depth - 1)
        depths[i] = depth

    for m in _DEPTH_CLAUSE_RE.finditer(masked):
        if depths[m.start()] == 0:
            kw = re.sub(r"\s+", " ", m.group(1).upper())
            clauses.append((m.start(), m.end(), kw))
    return clauses, depths


def _parse_int_or_param(
    tok: str, tok_offset: int, masked: str, params: Optional[Sequence[Any]]
) -> Optional[int]:
    if tok == "?":
        if params is None:
            return None
        q_idx = masked[:tok_offset].count("?")
        if 0 <= q_idx < len(params):
            try:
                return int(params[q_idx])
            except (TypeError, ValueError):
                return None
        return None
    try:
        return int(tok)
    except ValueError:
        return None


def inspect_sql_shape(
    sql: str, params: Optional[Sequence[Any]] = None
) -> Dict[str, Any]:
    """Inspect outermost SQL clauses safely with literal/comment masking."""
    masked = mask_sql_literals(sql or "")
    clauses, depths = _depth_zero_positions(masked)

    select_start = next((end for _s, end, kw in clauses if kw == "SELECT"), None)
    from_start = next((s for s, _e, kw in clauses if kw == "FROM"), None)
    has_group_by = any(kw == "GROUP BY" for _s, _e, kw in clauses)
    has_union = any(kw in ("UNION", "INTERSECT", "EXCEPT") for _s, _e, kw in clauses)
    order_spans = [(s, e) for s, e, kw in clauses if kw == "ORDER BY"]
    limit_spans = [(s, e) for s, e, kw in clauses if kw == "LIMIT"]
    has_order_by = bool(order_spans)

    is_scalar_aggregate = False
    if select_start is not None and not has_group_by and not has_union:
        seg_end = from_start if from_start is not None and from_start > select_start else len(masked)
        depth0_select = "".join(
            masked[idx] if (depths[idx] == 0 or (masked[idx] == "(" and idx > 0 and depths[idx - 1] == 0)) else " "
            for idx in range(select_start, seg_end)
        )
        if _AGG_CALL_RE.search(depth0_select):
            is_scalar_aggregate = True

    limit_applied: Optional[int] = None
    offset_applied = 0
    last_limit_span = limit_spans[-1] if limit_spans else None
    if last_limit_span is not None:
        tail = masked[last_limit_span[1]:]
        m_lim = re.match(
            r"^\s*(\d+|\?)(?:\s*,\s*(\d+|\?))?(?:\s+OFFSET\s+(\d+|\?))?",
            tail,
            flags=re.IGNORECASE,
        )
        if m_lim:
            g1, g2, g3 = m_lim.group(1), m_lim.group(2), m_lim.group(3)
            base_off = last_limit_span[1]
            if g2 is not None:
                off_v = _parse_int_or_param(g1, base_off + m_lim.start(1), masked, params)
                lim_v = _parse_int_or_param(g2, base_off + m_lim.start(2), masked, params)
                offset_applied = max(0, off_v or 0)
                limit_applied = lim_v
            else:
                limit_applied = _parse_int_or_param(g1, base_off + m_lim.start(1), masked, params)
                if g3 is not None:
                    off_v = _parse_int_or_param(g3, base_off + m_lim.start(3), masked, params)
                    offset_applied = max(0, off_v or 0)

    return {
        "masked": masked,
        "has_order_by": has_order_by,
        "has_group_by": has_group_by,
        "has_union": has_union,
        "is_scalar_aggregate": is_scalar_aggregate,
        "order_span": order_spans[-1] if order_spans else None,
        "limit_span": last_limit_span,
        "limit_applied": limit_applied,
        "offset_applied": offset_applied,
    }


def build_count_probe_query(
    sql: str, params: Optional[Sequence[Any]] = None
) -> Optional[Tuple[str, List[Any]]]:
    """Strip trailing outermost LIMIT/OFFSET (and redundant ORDER BY) to count total matches."""
    shape = inspect_sql_shape(sql, params)
    limit_span = shape["limit_span"]
    if limit_span is None:
        return None
    cut_pos = limit_span[0]
    order_span = shape["order_span"]
    if (
        order_span is not None
        and not shape["has_group_by"]
        and not shape["has_union"]
        and order_span[0] < cut_pos
    ):
        cut_pos = order_span[0]
    inner = (sql[:cut_pos] or "").strip()
    if not inner:
        return None
    kept_qmarks = shape["masked"][:cut_pos].count("?")
    probe_params = list(params[:kept_qmarks]) if params is not None else []
    return f"SELECT COUNT(*) FROM ({inner})", probe_params


def annotate_sql_window(
    conn: sqlite3.Connection,
    sanitized_query: str,
    bound_params: Optional[Sequence[Any]],
    result_rows: List[Dict[str, Any]],
    max_work_steps: int,
) -> Dict[str, Any]:
    """Compute exact or bounded window completeness metadata for a sanitized SELECT."""
    shape = inspect_sql_shape(sanitized_query, bound_params)
    returned = len(result_rows)
    limit_applied = shape["limit_applied"]
    offset = shape["offset_applied"]
    ordered = bool(shape["has_order_by"])
    is_agg = bool(shape["is_scalar_aggregate"])

    total: Optional[int]
    has_more = False
    truncated = False

    if is_agg or limit_applied is None or limit_applied <= 0:
        total = returned if is_agg else offset + returned
    elif returned < limit_applied:
        total = offset + returned
    else:
        total = None
        probe = build_count_probe_query(sanitized_query, bound_params)
        if probe is not None:
            probe_sql, probe_params = probe
            probe_rem = [min(500_000, max(50_000, int(max_work_steps) // 4))]

            def _probe_budget() -> int:
                probe_rem[0] -= 50_000
                return 1 if probe_rem[0] <= 0 else 0

            try:
                conn.set_progress_handler(_probe_budget, 50_000)
                cnt_row = conn.execute(probe_sql, probe_params).fetchone()
                if cnt_row is not None and cnt_row[0] is not None:
                    total = int(cnt_row[0])
            except Exception:
                total = None
            finally:
                conn.set_progress_handler(None, 0)
        if total is not None:
            has_more = total > (offset + returned)
            truncated = has_more
        else:
            has_more = True
            truncated = True

    if returned == 0:
        showing = "0 of 0" if total == 0 else "0"
    elif total is not None:
        showing = f"{offset + 1}-{offset + returned} of {total}"
    else:
        showing = f"{offset + 1}-{offset + returned} (truncated)"

    meta: Dict[str, Any] = {
        "truncated": truncated,
        "ordered": ordered,
        "offset": offset,
        "limit_applied": limit_applied,
        "total": total,
        "has_more": has_more,
        "showing": showing,
    }
    if has_more:
        meta["next_offset"] = offset + returned
    if truncated and not ordered:
        scope_txt = f"{returned} із {total}" if total is not None else f"перші {returned}"
        meta["warning_code"] = "TRUNCATED_UNORDERED"
        meta["notice"] = (
            f"УВАГА: показано лише {scope_txt} рядків без ORDER BY (порядок довільний). "
            "Мінімум, максимум чи повнота вибірки НЕ гарантовані — повтори запит з "
            "ORDER BY <колонка> ASC/DESC або використай catalog_search."
        )
    elif truncated and ordered:
        meta["warning_code"] = "TRUNCATED_ORDERED"
        meta["notice"] = (
            f"Показано впорядковану вибірку ({showing}); "
            f"наступну сторінку можна отримати через OFFSET {offset + returned}."
        )
    return meta


def _escape_like_literal(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_catalog_window(
    db_path: str,
    *,
    query: str = "",
    category: Optional[str] = None,
    sort: str = "price_asc",
    limit: int = 10,
    offset: int = 0,
    search_index: Any = None,
    max_work_steps: int = 5_000_000,
) -> Dict[str, Any]:
    """Domain-agnostic FTS5/LIKE paginated search with total counts and category breakdown."""
    from catalog_index import CatalogSearchIndex, MIN_TERM_LENGTH
    from source_projection import build_source_projection, quote_ident

    if not db_path or not os.path.exists(db_path):
        return {"success": False, "error": "Source database not found.", "rows": []}
    proj = build_source_projection(db_path)
    if proj is None or proj.entity_table is None:
        return {"success": False, "error": "Entity table could not be resolved.", "rows": []}

    ident = proj.resolve("identity")
    name_res = proj.resolve("name")
    cat_res = proj.resolve("category")
    price_res = proj.resolve("price")
    stock_res = proj.resolve("stock")
    if not ident.resolved or not name_res.column:
        return {"success": False, "error": "Identity or name column unresolved.", "rows": []}

    limit = max(1, min(25, int(limit or 10)))
    offset = max(0, int(offset or 0))
    raw_query = str(query or "").strip()[:300]
    terms = [t for t in raw_query.split() if t]

    q = quote_ident
    table_q = q(proj.entity_table)
    id_col, name_col = q(ident.column), q(name_res.column)
    cat_col = q(cat_res.column) if cat_res.column else None
    price_col = q(price_res.column) if price_res.column else None
    stock_col = q(stock_res.column) if stock_res.column else None

    backend = "sql_scan"
    fts_ids: Optional[List[str]] = None
    fts_terms = [t for t in terms if len(t) >= MIN_TERM_LENGTH and "%" not in t and "_" not in t]
    if fts_terms and len(fts_terms) == len(terms):
        fts = search_index if isinstance(search_index, CatalogSearchIndex) else CatalogSearchIndex()
        text_cols = [c for c in (name_res.column, cat_res.column) if c]
        if fts.ensure_built(db_path, proj.entity_table, ident.column, text_cols) == "ready":
            expr = " ".join('"' + t.replace('"', '""') + '"' for t in fts_terms)
            try:
                fts_cnt = fts.total_matches(expr)
                if fts_cnt == 0:
                    fts_ids = []
                    backend = "fts5_trigram"
                elif fts_cnt <= 500:
                    fts_ids = fts.page_ids(expr, 500, 0)
                    backend = "fts5_trigram"
            except sqlite3.Error:
                fts_ids = None

    where_clauses: List[str] = []
    params: List[Any] = []
    if fts_ids is not None:
        if not fts_ids:
            where_clauses.append("1 = 0")
        else:
            where_clauses.append(f"{id_col} IN ({','.join('?' * len(fts_ids))})")
            params.extend(fts_ids)
    else:
        for term in terms:
            pats = sorted({f"%{_escape_like_literal(v)}%" for v in (term, term.lower(), term.upper(), term.capitalize())})
            cols_to_match = [name_col] + ([cat_col] if cat_col else [])
            parts = ["(" + " OR ".join(f"{c} LIKE ? ESCAPE '\\'" for _ in pats) + ")" for c in cols_to_match]
            where_clauses.append("(" + " OR ".join(parts) + ")")
            for _ in cols_to_match:
                params.extend(pats)
        if terms:
            backend = "sql_like"

    if category is not None and str(category).strip() and cat_col:
        where_clauses.append(f"LOWER({cat_col}) = LOWER(?)")
        params.append(str(category).strip())

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    sort_mode = str(sort or "price_asc").strip().lower()
    if sort_mode == "price_asc" and price_col:
        order_sql = f"ORDER BY ({price_col} IS NULL) ASC, {price_col} ASC, {id_col} ASC"
    elif sort_mode == "price_desc" and price_col:
        order_sql = f"ORDER BY ({price_col} IS NULL) ASC, {price_col} DESC, {id_col} ASC"
    elif sort_mode == "name_asc":
        order_sql = f"ORDER BY {name_col} ASC, {id_col} ASC"
    else:
        sort_mode = "default"
        order_sql = f"ORDER BY {id_col} ASC"

    uri_path = "file:%s?mode=ro" % os.path.abspath(db_path).replace("\\", "/")
    conn = sqlite3.connect(uri_path, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON")
    rem = [int(max_work_steps)]
    conn.set_progress_handler(lambda: (rem.__setitem__(0, rem[0] - 50_000), 1 if rem[0] <= 0 else 0)[1], 50_000)

    try:
        total = int(conn.execute(f"SELECT COUNT(*) FROM {table_q} {where_sql}", params).fetchone()[0])

        breakdown: List[Dict[str, Any]] = []
        if cat_col and total > 0:
            bd_rows = conn.execute(
                f"SELECT COALESCE(CAST({cat_col} AS TEXT), 'general') AS grp, COUNT(*) AS cnt "
                f"FROM {table_q} {where_sql} GROUP BY grp ORDER BY cnt DESC LIMIT 8",
                params,
            ).fetchall()
            breakdown = [{"category": str(r[0]), "count": int(r[1])} for r in bd_rows]

        sel_cols = [f"{id_col} AS id", f"{name_col} AS name"]
        prov = {"id": "source", "name": "source"}
        if cat_col:
            sel_cols.append(f"{cat_col} AS category")
            prov["category"] = "source"
        if price_col:
            sel_cols.append(f"{price_col} AS price")
            prov["price"] = "source"
        if stock_col:
            sel_cols.append(f"{stock_col} AS stock")
            prov["stock"] = "source"

        rows_raw = conn.execute(
            f"SELECT {', '.join(sel_cols)} FROM {table_q} {where_sql} {order_sql} LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        rows = [dict(r) for r in rows_raw]
    finally:
        conn.close()

    returned = len(rows)
    has_more = total > (offset + returned)
    showing = f"{offset + 1}-{offset + returned} of {total}" if returned else (f"0 of {total}")
    out: Dict[str, Any] = {
        "success": True,
        "backend": backend,
        "sort": sort_mode,
        "total": total,
        "returned": returned,
        "row_count": returned,
        "offset": offset,
        "limit": limit,
        "truncated": has_more,
        "ordered": sort_mode != "default",
        "has_more": has_more,
        "showing": showing,
        "category_breakdown": breakdown,
        "rows": rows,
        "column_provenance": prov,
    }
    if has_more:
        out["next_offset"] = offset + returned
    return out


def build_cycle_state_summary(
    *,
    step: int,
    max_steps: int,
    grounded_count: int,
    tool_name: str,
    result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build purely numeric/boolean ReAct cycle telemetry (no DB text; F32-safe)."""
    res = result if isinstance(result, dict) else {}
    summary: Dict[str, Any] = {
        "step": int(step),
        "max_steps": int(max_steps),
        "remaining_steps": max(0, int(max_steps) - int(step)),
        "grounded_items": int(grounded_count),
        "last_tool": str(tool_name),
    }
    for key in ("row_count", "total", "offset", "next_offset", "truncated", "ordered", "has_more", "warning_code"):
        if key in res and res[key] is not None:
            summary[f"last_{key}"] = res[key]
    return summary
