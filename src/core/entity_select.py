"""C1: entity-table selection -- pure structural helpers for
`SourceProjection._choose_entity_table`.

Kept in a private module so the already-oversized projection file
does not grow further. Everything is derived from declared/inferred
FK edges and bounded aggregate SQL -- never from identifier
spellings. The function returns a plain dict; the caller owns the
trace events and the public attributes.

Selection pipeline (per the C1 contract):
  Step 0  detail demotion: a child PK set that EXTENDS a parent
          candidate's PKs at a non-alphanumeric boundary
          ('$.docs[0].variants[1]' extends '$.docs[0]'; '1234' does
          NOT extend '1') -- the detail leaves candidacy and counts
          as a facet of its parent.
  Graph   relation edges = declared FKs + proj.inferred_links.
          EAV attachment: the EAV entity column maps to the candidate
          whose PK contains >= 98% of its DISTINCT values (measured).
          Junctions: non-candidate, non-EAV tables with >= 2 FK edges
          to distinct tables; each referenced pair is an undirected
          bridge facet for both endpoints.
          Facets of X = bridge edges touching X + EAV tables
          attached to X + non-candidate FK child tables of X.
  Step A  fold lookups: X leaves if another remaining candidate Y
          has a MANY-TO-ONE edge Y->X (the child column repeats
          values: distinct < row count -- a shared lookup, not a
          1:1 extension row) and X has no edge back to Y. Evaluated
          simultaneously, so lookup chains fold in one pass. Folded
          candidates with a measured free-text column (avg >= 80
          chars over non-null) become alternatives and force
          ambiguous=True.
  Step B  rank by (facets desc, rows desc, name asc); ambiguous
          also when the top two tie on facets and the larger row
          count is at most 3x the smaller.
"""
from __future__ import annotations

import itertools
import logging
import sqlite3

logger = logging.getLogger(__name__)

_EXT_CHILD_SAMPLE = 200     # child pk rows probed for extensions
_EXT_PARENT_SAMPLE = 5000   # parent pk rows compared against
_EXT_COVERAGE = 0.9         # share of extending child pks
_DISTINCT_PROBE = 10000     # distinct eav entity values probed
_EAV_COVERAGE = 0.98        # containment share to attach
_AVG_LEN_PROBE = 5000       # non-null values averaged (bounded)
_FREE_TEXT_AVG = 80.0       # avg chars that make a column free text


def _fk_edges(tables: dict, inferred_links: list) -> list:
    """Column-level child -> parent edges:
    [(child_table, child_column, parent_table)]."""
    edges = set()
    for tname, tprof in tables.items():
        for col in tprof.columns.values():
            if col.is_foreign_key and col.fk_target_table:
                edges.add((tname, col.name, col.fk_target_table))
    for link in inferred_links or []:
        child, parent = link.get("child_table"), link.get("parent_table")
        col = link.get("child_column")
        if child and parent and col:
            edges.add((child, col, parent))
    return sorted(edges)


def _detail_map(conn: sqlite3.Connection, tables: dict,
                cand_names: set) -> dict:
    """{child: info} -- child pk values extend a parent's pk at a
    NON-ALPHANUMERIC boundary via a declared FK. '1234' vs '1' is a
    numeric coincidence, not an extension; '$.docs[0].x' vs
    '$.docs[0]' is."""
    from source_projection import quote_ident  # deferred (circular)
    out = {}
    for name in sorted(cand_names):
        tprof = tables[name]
        for col in tprof.columns.values():
            parent = col.fk_target_table if col.is_foreign_key else None
            if not parent or parent not in cand_names or parent == name:
                continue
            p_pk = tables[parent].primary_keys[0]
            try:
                child_pks = [str(r[0]) for r in conn.execute(
                    f"SELECT {quote_ident(tprof.primary_keys[0])} FROM "
                    f"{quote_ident(name)} LIMIT {_EXT_CHILD_SAMPLE}")]
                parent_pks = {str(r[0]) for r in conn.execute(
                    f"SELECT {quote_ident(p_pk)} FROM "
                    f"{quote_ident(parent)} LIMIT {_EXT_PARENT_SAMPLE}")}
            except sqlite3.Error:
                continue
            n = sum(1 for cpk in child_pks if any(
                len(cpk) > len(ppk) and cpk.startswith(ppk)
                and not cpk[len(ppk)].isalnum()
                for ppk in parent_pks))
            if child_pks and n / len(child_pks) >= _EXT_COVERAGE:
                out[name] = {"parent": parent,
                             "coverage": n / len(child_pks),
                             "via": col.name, "parent_pk": p_pk}
                break
    return out


def _eav_attachments(conn: sqlite3.Connection, tables: dict,
                     eav_tables: dict, cand_names: set) -> dict:
    """{eav_table: candidate} -- the candidate whose PK contains
    >= _EAV_COVERAGE of the eav entity column's DISTINCT values
    (bounded SQL; no attachment when none reaches the bar)."""
    from source_projection import quote_ident
    out = {}
    for eav, meta in sorted(eav_tables.items()):
        ecol = meta.get("entity_fk") or meta.get("entity_column")
        if not ecol or eav not in tables:
            continue
        probe = (f"(SELECT DISTINCT {quote_ident(ecol)} AS v FROM "
                 f"{quote_ident(eav)} LIMIT {_DISTINCT_PROBE})")
        try:
            total = conn.execute(
                f"SELECT COUNT(*) FROM {probe}").fetchone()[0]
            if not total:
                continue
            best, best_cov = None, 0.0
            for cand in sorted(cand_names):
                pk = tables[cand].primary_keys[0]
                hit = conn.execute(
                    f"SELECT COUNT(*) FROM {probe} WHERE v IN "
                    f"(SELECT {quote_ident(pk)} FROM "
                    f"{quote_ident(cand)})").fetchone()[0]
                if hit / total > best_cov:
                    best, best_cov = cand, hit / total
        except sqlite3.Error:
            continue
        if best is not None and best_cov >= _EAV_COVERAGE:
            out[eav] = {"candidate": best, "coverage": best_cov}
    return out


def _junctions(tables: dict, cand_names: set, eav_tables: dict,
               edges: list) -> list:
    """Non-candidate, non-EAV tables with >= 2 FK edges to distinct
    tables: each referenced PAIR is an undirected bridge edge for
    both endpoints. Returns [(a, b, junction_table)]."""
    refs: dict = {}
    for child, _col, parent in edges:
        refs.setdefault(child, set()).add(parent)
    out = []
    for tname in sorted(tables):
        if (tname in cand_names or tname in eav_tables
                or tname.startswith("__")):
            continue
        targets = sorted(refs.get(tname, ()))
        if len(targets) < 2:
            continue
        out.extend((a, b, tname)
                   for a, b in itertools.combinations(targets, 2))
    return out


def _facets(cand_names: set, edges: list, detail_of: dict,
            eav_attach: dict, bridge_edges: list) -> dict:
    """facets[X] = bridge edges touching X + EAV tables attached to
    X + non-candidate FK child tables of X (details left candidacy
    so they count here; eav/junctions excluded from the child count
    to keep each evidence kind distinct)."""
    eav_set = {e for e in eav_attach}
    juncs = {j for _a, _b, j in bridge_edges}
    counts = {c: 0 for c in cand_names}
    for a, b, _j in bridge_edges:
        if a in counts:
            counts[a] += 1
        if b in counts:
            counts[b] += 1
    for meta in eav_attach.values():
        counts[meta["candidate"]] += 1
    seen_children = set()
    for child, _col, parent in edges:
        if parent in counts and child not in cand_names \
                and child not in eav_set and child not in juncs \
                and (child, parent) not in seen_children:
            seen_children.add((child, parent))
            counts[parent] += 1
    return counts


def _is_lookup_edge(tables: dict, child: str, col: str) -> bool:
    """A lookup edge is MANY-TO-ONE: the child's reference column
    repeats values (distinct < row count). A 1:1 reference is an
    extension row (products <-> reviews), not a shared dictionary."""
    tprof, cprof = tables.get(child), None
    if tprof is not None:
        cprof = tprof.columns.get(col)
    if tprof is None or cprof is None or cprof.distinct_count is None:
        return False
    return cprof.distinct_count < (tprof.row_count or 0)


def _fold_lookups(edges: list, remaining: set, tables: dict) -> tuple:
    """X folds when another REMAINING candidate Y has a many-to-one
    edge Y->X (X is Y's shared lookup) and X has no edge back to Y.
    Evaluated simultaneously on the current set so a lookup chain
    folds in one pass."""
    parents_of, inbound_of = {}, {}
    for child, col, parent in edges:
        parents_of.setdefault(child, set()).add(parent)
        if _is_lookup_edge(tables, child, col):
            inbound_of.setdefault(parent, set()).add(child)
    folded = {}
    for x in sorted(remaining):
        parents = parents_of.get(x, set()) & remaining
        one_way = [y for y in inbound_of.get(x, set()) & remaining
                   if y not in parents]
        if one_way:
            folded[x] = sorted(one_way)
    return folded, remaining - set(folded)


def _free_text_column(conn: sqlite3.Connection, tprof) -> str | None:
    """Name of a non-PK text column whose measured avg length over
    non-null values is >= _FREE_TEXT_AVG (bounded probe), else None."""
    from source_projection import _is_text, quote_ident
    for col in tprof.columns.values():
        if col.name in set(tprof.primary_keys) or not _is_text(col):
            continue
        try:
            avg = conn.execute(
                f"SELECT AVG(LENGTH(v)) FROM (SELECT "
                f"CAST({quote_ident(col.name)} AS TEXT) AS v FROM "
                f"{quote_ident(tprof.name)} WHERE "
                f"{quote_ident(col.name)} IS NOT NULL "
                f"LIMIT {_AVG_LEN_PROBE})").fetchone()[0]
        except sqlite3.Error:
            continue
        if avg is not None and avg >= _FREE_TEXT_AVG:
            return col.name
    return None


def rank_entity_candidates(conn: sqlite3.Connection, tables: dict,
                           eav_tables: dict, inferred_links: list,
                           cand_names: set) -> dict:
    """Run Steps 0/A/B and return everything the caller needs for
    trace events and the public attributes."""
    edges = _fk_edges(tables, inferred_links)
    detail_of = _detail_map(conn, tables, cand_names)
    eav_attach = _eav_attachments(conn, tables, eav_tables, cand_names)
    bridge_edges = _junctions(tables, cand_names, eav_tables, edges)
    facets = _facets(cand_names, edges, detail_of, eav_attach,
                     bridge_edges)
    remaining = cand_names - set(detail_of)
    folded, kept = _fold_lookups(edges, remaining, tables)
    alternatives, free_cols = [], {}
    for x in sorted(folded):
        col = _free_text_column(conn, tables[x])
        if col is not None:
            alternatives.append(x)
            free_cols[x] = col
    ranked = sorted(kept, key=lambda n: (
        -facets.get(n, 0), -(tables[n].row_count or 0), n))
    ambiguous = bool(alternatives)
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        ra, rb = tables[a].row_count or 0, tables[b].row_count or 0
        if facets.get(a, 0) == facets.get(b, 0) \
                and max(ra, rb) <= 3 * min(ra, rb):
            ambiguous = True
    return {
        "entity": ranked[0] if ranked else None,
        "ambiguous": ambiguous,
        "alternatives": alternatives,
        "detail_of": detail_of, "folded": folded,
        "free_cols": free_cols, "facets": facets,
        "ranked": ranked, "eav_attach": eav_attach,
        "bridge_edges": bridge_edges,
    }
