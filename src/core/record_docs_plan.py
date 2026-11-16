"""
Retrieval Fabric: plan-time helpers for `record_documents` (private).

All the bounded, measured SQL that decides foldability and key
uniqueness lives here so `record_documents.py` stays within the
project's 350-line bound; the public entry point is
`record_documents.plan_documents` (a thin wrapper over `build_plan`).

Key rules (structural, measured -- never name-based):

* an id column must be UNIQUE+NOT NULL over the full table, else the
  merge would silently collapse two entities into one id;
* a dimension's parent key must be unique over the parent table,
  else the LEFT JOIN would duplicate entity rows;
* identifier-like := nearly-unique AND whitespace-free AND short AND
  digit-bearing (digit_share >= 0.5): single-word unique NAMES and
  lookup labels are kept, codes/UUIDs/ISBNs are not;
* dimension/bridge label columns skip the identifier-like test
  (labels are unique by design) and get a stricter code-table rule
  instead (digit_share >= 0.95 AND distinct_ratio >= 0.9).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

from record_documents import (
    MAX_LABEL_COLS,
    MIN_NONNULL_RATIO,
    BridgeSpec,
    DimSpec,
    DocPlan,
    EavSpec,
    _ro_connect,
)
from source_projection import (
    SourceProjection,
    _is_numeric,
    _is_text,
    _looks_temporal,
    quote_ident,
)

logger = logging.getLogger(__name__)

_STAT_CHARS = "0123456789 .,+-:/%"


@dataclass
class _ColStats:
    nonnull: int = 0
    distinct: int = 0
    avg_len: float = 0.0
    ws_share: float = 0.0
    numeric_share: float = 0.0
    digit_share: float = 0.0


def _col_stats(conn: sqlite3.Connection, table: str,
               column: str) -> _ColStats:
    """One bounded aggregate query over the whole column."""
    c, t = quote_ident(column), quote_ident(table)
    row = conn.execute(
        f"SELECT COUNT({c}), COUNT(DISTINCT {c}), "
        f"AVG(LENGTH(CAST({c} AS TEXT))), "
        f"SUM(CASE WHEN instr(CAST({c} AS TEXT),' ')>0 OR "
        f"instr(CAST({c} AS TEXT),char(9))>0 OR "
        f"instr(CAST({c} AS TEXT),char(10))>0 THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN TRIM(CAST({c} AS TEXT),'{_STAT_CHARS}')='' "
        f"THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN CAST({c} AS TEXT) GLOB '*[0-9]*' "
        f"THEN 1 ELSE 0 END) FROM {t}").fetchone()
    nn = int(row[0] or 0)
    st = _ColStats(nonnull=nn, distinct=int(row[1] or 0),
                   avg_len=float(row[2] or 0.0))
    if nn:
        st.ws_share = float(row[3] or 0) / nn
        st.numeric_share = float(row[4] or 0) / nn
        st.digit_share = float(row[5] or 0) / nn
    return st


def _exclusion_reason(col, st: _ColStats, row_count: int,
                      label: bool) -> str | None:
    """None = foldable text; otherwise the structural reason to skip.

    `label` (dimension/bridge label column): unique-by-design lookup
    labels skip the identifier-like test; only a near-unique digit
    code table is still an identifier, not a label."""
    if _is_numeric(col):
        return "numeric"
    if _looks_temporal(col):
        return "temporal"
    if row_count <= 0 or st.nonnull / row_count < MIN_NONNULL_RATIO:
        return "sparse"
    if st.numeric_share >= 0.95:
        return "numeric"
    if st.nonnull and st.distinct == 1:
        return "constant"          # no discriminative information
    ratio = st.distinct / st.nonnull if st.nonnull else 0.0
    if label:
        if st.digit_share >= 0.95 and ratio >= 0.9:
            return "identifier_like"
        return None
    if (ratio >= 0.9 and st.ws_share <= 0.05 and st.avg_len <= 40
            and st.digit_share >= 0.5):
        return "identifier_like"
    return None


def _foldable_cols(conn, table_prof, key_cols: set,
                   excluded: list, *, label: bool = False,
                   skip: set = frozenset()) -> list:
    """[(name, distinct_ratio, avg_len)] of foldable columns; every
    rejected candidate is recorded in `excluded` with its reason.
    `skip` columns were already decided elsewhere (id_not_unique)."""
    rows = table_prof.row_count or 0
    out = []
    for col in table_prof.columns.values():
        if col.name in skip:
            continue
        if col.name in key_cols:
            excluded.append(
                (f"{table_prof.name}.{col.name}", "id_column"))
            continue
        st = _col_stats(conn, table_prof.name, col.name)
        reason = _exclusion_reason(col, st, rows, label)
        if reason is None:
            ratio = st.distinct / st.nonnull if st.nonnull else 0.0
            out.append((col.name, ratio, st.avg_len))
        else:
            excluded.append((f"{table_prof.name}.{col.name}", reason))
    # rd/2: distinctiveness first -- the most identifying column leads
    # the document, low-cardinality attributes trail it.
    out.sort(key=lambda p: (-p[1], p[2], p[0]))
    return out


def _is_unique_key(conn, table: str, col: str) -> bool:
    """Non-null AND distinct over the full table (one bounded
    aggregate query) -- the only safe join/merge key."""
    c, t = quote_ident(col), quote_ident(table)
    n, d, nn = conn.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT {c}), COUNT({c}) "
        f"FROM {t}").fetchone()
    return bool(n) and n == d and nn == n


def _fk_links(proj: SourceProjection, table: str) -> list:
    """FK-like columns of `table`: (col, target_table, target_col)."""
    tprof = proj.snapshot.tables.get(table)
    links = []
    if tprof is not None:
        for c in tprof.columns.values():
            if c.is_foreign_key and c.fk_target_table:
                links.append((c.name, c.fk_target_table,
                              c.fk_target_column))
    for l in getattr(proj, "inferred_links", []) or []:
        if l.get("child_table") == table:
            links.append((l["child_column"], l["parent_table"],
                          l["parent_column"]))
    return links


def _pick_id_column(proj: SourceProjection, tprof, conn,
                    excluded: list) -> tuple:
    """(id_col, rejected): first UNIQUE+NOT NULL candidate wins --
    resolved identity, then the single PK, then rowid (always
    unique). Rejected candidates are recorded as id_not_unique -- a
    non-unique id would merge distinct entities silently."""
    cands, rejected = [], set()
    ident = proj.resolve("identity")
    if ident.resolved and ident.column:
        cands.append(ident.column)
    if len(tprof.primary_keys) == 1:
        cands.append(tprof.primary_keys[0])
    for col in dict.fromkeys(cands):
        if col in tprof.columns \
                and _is_unique_key(conn, tprof.name, col):
            return col, rejected
        rejected.add(col)
        excluded.append((f"{tprof.name}.{col}", "id_not_unique"))
    return "rowid", rejected


def _label_cols(proj, conn, ptab, pkey, excluded):
    """(key_col, label_cols) of a parent/other table, or None."""
    pprof = proj.snapshot.tables.get(ptab)
    if pprof is None:
        return None
    if pkey is None and len(pprof.primary_keys) == 1:
        pkey = pprof.primary_keys[0]      # `REFERENCES t` -> its PK
    if pkey not in pprof.columns:
        return None
    if not _is_unique_key(conn, ptab, pkey):
        excluded.append((f"{ptab}.{pkey}", "parent_key_not_unique"))
        return None
    labels = _foldable_cols(conn, pprof, {pkey}, excluded,
                            label=True)
    if not labels:
        return None
    return pkey, tuple(n for n, _, _ in labels[:MAX_LABEL_COLS])


def _plan_dims(proj, conn, id_col, excluded) -> list:
    ent, dims, seen = proj.entity_table, [], set()
    for col, ptab, pcol in _fk_links(proj, ent):
        if col == id_col or ptab == ent or col in seen:
            continue
        found = _label_cols(proj, conn, ptab, pcol, excluded)
        if found is None:
            continue
        seen.add(col)
        dims.append(DimSpec(col, ptab, found[0], found[1]))
    return dims


def _plan_bridges(proj, conn, excluded) -> list:
    ent, bridges = proj.entity_table, []
    for tname in proj.snapshot.tables:
        if tname == ent or tname in proj.eav_tables \
                or tname.startswith("__"):
            continue
        links = _fk_links(proj, tname)
        ent_side = [l for l in links if l[1] == ent]
        other = [l for l in links if l[1] != ent]
        if len(links) != 2 or len(ent_side) != 1 or len(other) != 1:
            continue
        (ecol, _, _), (ocol, otab, okey) = ent_side[0], other[0]
        found = _label_cols(proj, conn, otab, okey, excluded)
        if found is None:
            continue
        bridges.append(BridgeSpec(tname, ecol, ocol, otab,
                                  found[0], found[1]))
    return bridges


def _plan_eav(proj, excluded) -> list:
    specs = []
    for tname, meta in (proj.eav_tables or {}).items():
        ecol = meta.get("entity_fk") or meta.get("entity_column")
        kcol, vcol = meta.get("key_column"), meta.get("value_column")
        tprof = proj.snapshot.tables.get(tname)
        cols = tprof.columns if tprof else {}
        if not (ecol and kcol and vcol
                and all(c in cols for c in (ecol, kcol, vcol))):
            excluded.append((f"{tname}.*", "eav_columns_missing"))
            continue
        if not _is_text(cols[vcol]):
            excluded.append((f"{tname}.{vcol}", "eav_value_not_text"))
            continue
        specs.append(EavSpec(tname, ecol, kcol, vcol))
    return specs


def build_plan(proj: SourceProjection, db_path: str
               ) -> DocPlan | None:
    """Derive a render plan from projection structure; None if no
    entity table was resolved (caller reports it)."""
    if proj.entity_table is None:
        return None
    tprof = proj.snapshot.tables[proj.entity_table]
    conn = _ro_connect(db_path)
    excluded: list = []
    try:
        id_col, rejected = _pick_id_column(proj, tprof, conn, excluded)
        own = _foldable_cols(conn, tprof, {id_col}, excluded,
                             skip=rejected)
        dims = _plan_dims(proj, conn, id_col, excluded)
        bridges = _plan_bridges(proj, conn, excluded)
        eav = _plan_eav(proj, excluded)
    finally:
        conn.close()
    return DocPlan(
        entity_table=proj.entity_table, id_column=id_col,
        own_columns=tuple(n for n, _, _ in own),
        eav=tuple(eav), dimensions=tuple(dims), bridges=tuple(bridges),
        excluded=tuple(sorted(set(excluded))),
        display_column=(own[0][0] if own and own[0][1] >= 0.5
                        else None))
