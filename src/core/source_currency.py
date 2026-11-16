"""Domain-agnostic source currency resolution and formatting helpers."""

import json
import os
import re
import sqlite3
from typing import Any, Optional

from source_projection import build_source_projection, quote_ident

UNSET_CURRENCY: Any = object()
_ISO_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
_INVALID_CURRENCY_TOKENS = frozenset({"NONE", "NULL", "MIXED", "UNKNOWN", "N/A", "NA"})


def normalize_currency_code(raw: Optional[str]) -> Optional[str]:
    """Normalize explicit ISO-4217 currency codes; return None when unknown or absent."""
    if raw is None:
        return None
    txt = str(raw).strip().upper()
    if not txt or txt in _INVALID_CURRENCY_TOKENS:
        return None
    if txt in {"₴", "ГРН", "ГРН.", "UAH"}:
        return "UAH"
    if txt in {"$", "USD"}:
        return "USD"
    if txt in {"€", "EUR"}:
        return "EUR"
    if _ISO_CURRENCY_RE.match(txt):
        return txt
    return None


def read_db_currency(db_path: Optional[str]) -> Optional[str]:
    """Inspect source DB metadata or uniform currency column without domain hardcoding."""
    if not db_path or not os.path.exists(db_path):
        return None
    ro_uri = "file:%s?mode=ro" % os.path.abspath(db_path).replace("\\", "/")
    try:
        conn = sqlite3.connect(ro_uri, uri=True)
    except sqlite3.Error:
        return None
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        for meta_table in ("ingest_meta", "__ingest_meta"):
            if meta_table not in tables:
                continue
            try:
                rows = dict(
                    conn.execute(f"SELECT key, value FROM {meta_table}").fetchall()
                )
            except sqlite3.Error:
                continue
            price_cov = str(rows.get("price_coverage") or "").strip().lower()
            if price_cov.startswith("none"):
                return None
            for key in ("currency_kept", "currency"):
                code = normalize_currency_code(rows.get(key))
                if code is not None:
                    return code

        # Fallback: uniform explicit `currency` column on the entity table.
        try:
            proj = build_source_projection(db_path)
            ent = proj.entity_table
        except Exception:
            ent = None
        if ent and ent in tables:
            cols = {
                str(r[1]).lower(): str(r[1])
                for r in conn.execute(f"PRAGMA table_info({quote_ident(ent)})").fetchall()
            }
            cur_col = cols.get("currency")
            if cur_col:
                vals = [
                    r[0]
                    for r in conn.execute(
                        f"SELECT DISTINCT {quote_ident(cur_col)} "
                        f"FROM {quote_ident(ent)} "
                        f"WHERE {quote_ident(cur_col)} IS NOT NULL "
                        f"AND TRIM({quote_ident(cur_col)}) != '' LIMIT 3"
                    ).fetchall()
                ]
                normalized = {
                    c for c in (normalize_currency_code(v) for v in vals) if c is not None
                }
                if len(normalized) == 1:
                    return next(iter(normalized))
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return None


def _read_registry_currency(db_path: Optional[str]) -> Optional[str]:
    """Read explicit declarative currency from sources.demo.json/runtime matching db_path."""
    if not db_path:
        return None
    target_real = os.path.realpath(os.path.abspath(db_path))
    data_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
    )
    project_root = os.path.abspath(os.path.join(data_dir, ".."))
    from server_config import env_setting
    runtime_path = env_setting("PYBUDDY_RUNTIME_REGISTRY", "AINSA_RUNTIME_REGISTRY") or os.path.join(
        data_dir, "sources.runtime.json"
    )
    for reg_path in (runtime_path, os.path.join(data_dir, "sources.demo.json")):
        try:
            with open(reg_path, "r", encoding="utf-8") as fh:
                entries = json.load(fh).get("sources", [])
        except (OSError, ValueError):
            continue
        for entry in entries:
            cands = entry.get("db_candidates") or (
                [entry["db_path"]] if entry.get("db_path") else []
            )
            for cand in cands:
                cand_abs = (
                    cand
                    if os.path.isabs(cand)
                    else os.path.abspath(os.path.join(project_root, cand))
                )
                if os.path.realpath(cand_abs) == target_real:
                    code = normalize_currency_code(entry.get("currency"))
                    if code is not None:
                        return code
    return None


def resolve_source_currency(
    db_path: Optional[str],
    configured_currency: Any = UNSET_CURRENCY,
) -> Optional[str]:
    """Resolve source currency from explicit arg, DB provenance, or declarative registry."""
    if configured_currency is not UNSET_CURRENCY:
        if configured_currency is None:
            return None
        explicit = normalize_currency_code(configured_currency)
        if explicit is not None:
            return explicit
    from_db = read_db_currency(db_path)
    if from_db is not None:
        return from_db
    return _read_registry_currency(db_path)


def format_currency_label(currency: Optional[str], style: str = "prose") -> str:
    """Return a clean unit suffix for grounded prices, or empty string when unknown."""
    code = normalize_currency_code(currency)
    if code is None:
        return ""
    if code == "UAH":
        return "₴" if style == "symbol" else "грн"
    return code


def format_price_with_currency(
    price: float,
    currency: Optional[str] = None,
    decimals: int = 2,
) -> str:
    """Format a numeric price paired with its verified unit (never inventing UAH)."""
    num = f"{float(price):.{decimals}f}"
    label = format_currency_label(currency, style="prose")
    return f"{num} {label}" if label else num
