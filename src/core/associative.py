"""
EBR: evidence-bound associative recall over a source projection.

Two stages, each leaving a typed trail:

  * Stage A -- multi-channel conjunctive binding. Each query term is
    bound independently to either an entity column (name/category) or
    to an EAV *value* axis. An item that satisfies EVERY term through
    any mix of channels is a full-satisfaction candidate: the lexical
    path missed it only because part of the constraint lived inside
    attributes, not because the item is absent ("фарба для шкіри" --
    'фарба' binds the name, 'шкіра' binds axis `призначення`).
  * Stage B -- ordered relaxation. Terms that bind to nothing are
    dropped one step at a time; each drop is a recorded
    `constraint_relaxation` step with regret measured as the number of
    dropped constraints. This is the trace a minimal-regret lattice
    (docs/04, slot 5) needs: the widening path is data, not guesswork.

Nothing relaxed is merged into direct hits: callers receive a separate
`relaxation` payload and must not present it as an exact match.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from sanitizer import LIKE_ESCAPE_CLAUSE, escape_like
from source_projection import SourceProjection, quote_ident

MAX_RELAXED_ITEMS = 40
MAX_AXIS_SIBLINGS = 20
MAX_PER_TERM_SCAN = 500
MIN_TERM = 3  # below this a term is too ambiguous to widen on

# Binding channels -- the "why" attached to every candidate. Channels
# are structural patterns discovered by the profiler, not per-database
# configuration: any schema shape matching a pattern becomes a channel.
CH_ENTITY = "entity_column"   # term hit a text column on the entity table
CH_EAV = "eav_value"          # term hit a value inside an EAV axis
CH_FKDIM = "fk_dimension"     # term hit a value in a table the entity
                              # references through a declared FK
CH_JUNC = "junction"          # term hit a dimension behind a
                              # many-to-many bridge (two-FK table)
CH_PATH = "multi_hop"         # term reached through a >=2-edge chain of
                              # reference links (entity->subs->depts);
                              # binding carries the full path
CH_TLINK = "text_link"        # term hit a dimension label that lives
                              # inside entity text (no FK at all)
CH_MENTION = "child_mention"  # term hit text in a table that FKs INTO
                              # the entity table (reviews, doc tables)
CH_VEC = "vector"             # cosine-similar cell in the dense index
                              # (paraphrase/semantic -- opt-in channel)

MAX_PATH_HOPS = None          # None = naturally bounded by the table
                              # count (visited set prevents cycles)
MAX_PATHS = 32                # structural path enumeration budget
MAX_TLINK_LABELS = 20         # matched dimension labels per term


@dataclass
class RelaxStep:
    stage: str               # "constraint_relaxation" | "axis_bridge"
                           # | "unresolved_term"
    term: str
    axis: str | None = None        # the EAV key that bridged, if any
    matched_value: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "term": self.term,
                "axis": self.axis, "matched_value": self.matched_value,
                "detail": self.detail}


@dataclass
class RelaxedItem:
    entity_id: str
    via: str                       # "conjunctive" or the single channel
    bindings: list = field(default_factory=list)  # per-term evidence
    axis: str | None = None        # first EAV axis bound (display aid)
    matched_value: str | None = None
    name: str | None = None        # hydrated from the entity table

    def to_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "via": self.via,
                "axis": self.axis, "matched_value": self.matched_value,
                "name": self.name, "bindings": self.bindings}


@dataclass
class RelaxationResult:
    steps: list = field(default_factory=list)
    items: list = field(default_factory=list)
    terms_used: list = field(default_factory=list)
    regret: int = 0                # number of dropped query terms

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "associative_relaxation",
            "terms_used": self.terms_used,
            "regret": self.regret,
            "steps": [s.to_dict() for s in self.steps],
            "items": [i.to_dict() for i in self.items],
        }


def _case_variants(term: str) -> list[str]:
    """LIKE folds ASCII only; Cyrillic needs explicit variants (T4)."""
    return sorted({term, term.lower(), term.upper(), term.capitalize()})


def _like_pred(col: str, pats: list) -> str:
    return "(" + " OR ".join(
        f"{quote_ident(col)} LIKE ?{LIKE_ESCAPE_CLAUSE}"
        for _ in pats) + ")"


def _scan_term(cur, proj: SourceProjection, term: str,
               eav: dict, per_term_limit: int,
               idx=None) -> dict:
    """Bind one term across every discovered lexical evidence channel.

    Returns entity_id -> list of binding dicts. Each binding records the
    channel that carried it, so a candidate's provenance is inspectable.
    The dense vector channel is deliberately NOT scanned here: a noisy
    embedding can fabricate same-term bindings and poison the
    conjunctive intersection. The vector channel is applied by the
    caller as a per-term fallback -- only for terms with zero lexical
    coverage (its actual job: paraphrase). See T16 residual.
    """
    from source_projection import _is_text  # reuse the type predicate

    q = quote_ident
    hits: dict[str, list] = {}
    pats = _case_variants(term)
    params = [f"%{escape_like(p)}%" for p in pats]

    ident = proj.resolve("identity")
    profile = proj.snapshot.tables.get(proj.entity_table)
    if not ident.column or profile is None:
        return hits

    if idx is not None and len(term) >= 3:
        # Indexed path: one trigram MATCH replaces the per-column and
        # per-EAV-table LIKE sweeps (channels 1, 2, 7). Semantics are
        # identical -- trigram substring match, same provenance shape.
        for rid, bs in idx.bind(term, per_term_limit).items():
            hits.setdefault(rid, []).extend(bs)
    else:
        # Channel 1 -- every text column on the entity table, not only
        # the ones that happened to resolve to a role.
        for col in profile.columns.values():
            if not _is_text(col):
                continue
            sql = (f"SELECT {q(ident.column)}, {q(col.name)} "
                   f"FROM {q(proj.entity_table)} "
                   f"WHERE {_like_pred(col.name, pats)} "
                   f"LIMIT {int(per_term_limit)}")
            for rid, v in cur.execute(sql, params):
                hits.setdefault(str(rid), []).append(
                    {"term": term, "channel": CH_ENTITY,
                     "column": col.name, "value": str(v)[:120]})

        # Channel 2 -- values inside detected EAV axes.
        for tbl, meta in eav.items():
            ent = meta.get("entity_fk") or meta.get("entity_column")
            key = meta.get("key_column")
            val = meta.get("value_column")
            if not (ent and key and val):
                continue
            sql = (f"SELECT {q(ent)}, {q(key)}, {q(val)} "
                   f"FROM {q(tbl)} WHERE {_like_pred(val, pats)} "
                   f"LIMIT {int(per_term_limit)}")
            for rid, k, v in cur.execute(sql, params):
                hits.setdefault(str(rid), []).append(
                    {"term": term, "channel": CH_EAV, "table": tbl,
                     "axis": str(k), "value": str(v)[:120]})

    # Unified relation registry: declared FKs (overlap=None) plus
    # conductor-verified inferred links (measured overlap). Every
    # downstream channel consumes this registry, so a link found by ANY
    # detector works like a written-down foreign key.
    links: dict[str, list] = {}  # child_table -> [(child_col, parent, parent_col, overlap)]
    for tname, tprof in proj.snapshot.tables.items():
        for col in tprof.columns.values():
            if col.is_foreign_key and col.fk_target_table:
                tgt0 = proj.snapshot.tables.get(col.fk_target_table)
                jc = col.fk_target_column or (
                    tgt0.primary_keys[0]
                    if tgt0 and tgt0.primary_keys else None)
                if jc:
                    links.setdefault(tname, []).append(
                        (col.name, col.fk_target_table, jc, None))
    for l in getattr(proj, "inferred_links", []):
        links.setdefault(l["child_table"], []).append(
            (l["child_column"], l["parent_table"],
             l["parent_column"], l.get("overlap")))

    # Channel 3 -- normalized dimensions the entity references
    # (products.purpose_id -> purposes.name), declared or inferred.
    for col_name, tgt, join_col, overlap in links.get(proj.entity_table, []):
        tprof = proj.snapshot.tables.get(tgt)
        if tprof is None:
            continue
        for tcol in tprof.columns.values():
            if not _is_text(tcol):
                continue
            ref = "r." + q(tcol.name)
            pred = ("(" + " OR ".join(
                f"{ref} LIKE ?{LIKE_ESCAPE_CLAUSE}"
                for _ in pats) + ")")
            sql = (f"SELECT e.{q(ident.column)}, r.{q(tcol.name)} "
                   f"FROM {q(proj.entity_table)} e "
                   f"JOIN {q(tgt)} r ON e.{q(col_name)} = r.{q(join_col)} "
                   f"WHERE {pred} LIMIT {int(per_term_limit)}")
            try:
                rows = cur.execute(sql, params).fetchall()
            except sqlite3.Error:
                continue
            for rid, v in rows:
                b = {"term": term, "channel": CH_FKDIM, "table": tgt,
                     "column": tcol.name, "value": str(v)[:120]}
                if overlap is not None:
                    b["inferred"] = True
                    b["overlap"] = overlap
                hits.setdefault(str(rid), []).append(b)

    # Channel 4 -- junction bridges: a table with >=2 verified links,
    # one to the entity, is a many-to-many bridge (product_tags ->
    # products + tags). Links may be declared OR conductor-inferred.
    for jname in proj.snapshot.tables:
        if jname == proj.entity_table or jname in eav:
            continue
        jlinks = links.get(jname) or []
        ent_links = [l for l in jlinks if l[1] == proj.entity_table]
        dim_links = [l for l in jlinks if l[1] != proj.entity_table]
        if not ent_links or not dim_links:
            continue
        for ef_col, _, _, ef_ov in ent_links:
            for df_col, dim, join_col, df_ov in dim_links:
                dprof = proj.snapshot.tables.get(dim)
                if dprof is None or not join_col:
                    continue
                inferred_ov = (df_ov if df_ov is not None else ef_ov)
                for tcol in dprof.columns.values():
                    if not _is_text(tcol):
                        continue
                    ref = "r." + q(tcol.name)
                    pred = ("(" + " OR ".join(
                        f"{ref} LIKE ?{LIKE_ESCAPE_CLAUSE}"
                        for _ in pats) + ")")
                    sql = (
                        f"SELECT e.{q(ident.column)}, r.{q(tcol.name)} "
                        f"FROM {q(proj.entity_table)} e "
                        f"JOIN {q(jname)} j "
                        f"ON e.{q(ident.column)} = j.{q(ef_col)} "
                        f"JOIN {q(dim)} r "
                        f"ON j.{q(df_col)} = r.{q(join_col)} "
                        f"WHERE {pred} LIMIT {int(per_term_limit)}")
                    try:
                        rows = cur.execute(sql, params).fetchall()
                    except sqlite3.Error:
                        continue
                    for rid, v in rows:
                        b = {"term": term, "channel": CH_JUNC,
                             "table": dim, "via_table": jname,
                             "column": tcol.name,
                             "value": str(v)[:120]}
                        if inferred_ov is not None:
                            b["inferred"] = True
                            b["overlap"] = inferred_ov
                        hits.setdefault(str(rid), []).append(b)

    # Channel 5 -- reference-chain paths: entity -> A -> B -> ...
    # following child->parent links (declared or inferred) for up to
    # MAX_PATH_HOPS edges. The 1-hop case is channel 3; this covers
    # products.sub_id -> subs -> depts and deeper normalisation chains.
    # Path provenance is recorded in every binding.
    paths: list[list[tuple]] = []  # each: [(child_col, parent, parent_col, overlap)]
    frontier = [[(c, p, pc, ov)]
                for c, p, pc, ov in links.get(proj.entity_table, [])]
    max_hops = MAX_PATH_HOPS or len(proj.snapshot.tables)
    while frontier and len(paths) < MAX_PATHS:
        path = frontier.pop()
        last_parent = path[-1][1]
        if len(path) >= 2:
            paths.append(path)
        if len(path) >= max_hops:
            continue
        visited = {proj.entity_table} | {p[1] for p in path}
        for c, p, pc, ov in links.get(last_parent, []):
            if p not in visited:
                frontier.append(path + [(c, p, pc, ov)])
    for path in paths:
        dprof = proj.snapshot.tables.get(path[-1][1])
        if dprof is None:
            continue
        # JOIN e.a1 = t1.pk, t1.a2 = t2.pk, ... -- every hop is a
        # verified link, so the join path is evidence, not a guess.
        joins, alias = [], "e"
        for i, (c, p, pc, _ov) in enumerate(path):
            a = f"t{i}"
            joins.append(
                f"JOIN {q(p)} {a} ON {alias}.{q(c)} = {a}.{q(pc)}")
            alias = a
        inferred_ov = next((ov for *_, ov in path if ov is not None), None)
        path_desc = " -> ".join(
            [proj.entity_table] + [p for _, p, _, _ in path])
        for tcol in dprof.columns.values():
            if not _is_text(tcol):
                continue
            ref = f"{alias}.{q(tcol.name)}"
            pred = ("(" + " OR ".join(
                f"{ref} LIKE ?{LIKE_ESCAPE_CLAUSE}"
                for _ in pats) + ")")
            sql = (f"SELECT e.{q(ident.column)}, {ref} "
                   f"FROM {q(proj.entity_table)} e {' '.join(joins)} "
                   f"WHERE {pred} LIMIT {int(per_term_limit)}")
            try:
                rows = cur.execute(sql, params).fetchall()
            except sqlite3.Error:
                continue
            for rid, v in rows:
                b = {"term": term, "channel": CH_PATH,
                     "table": path[-1][1], "column": tcol.name,
                     "value": str(v)[:120],
                     "path": path_desc, "hops": len(path)}
                if inferred_ov is not None:
                    b["inferred"] = True
                    b["overlap"] = inferred_ov
                hits.setdefault(str(rid), []).append(b)

    # Channel 6 -- text-containment links: a dimension label living
    # inside entity text without any FK. The term hits the label; the
    # entities containing that label bind through it ('42' -> sizes
    # label '42' -> products whose name carries the '42' token).
    text_links = getattr(proj, "text_links", []) or []
    etext = [c.name for c in profile.columns.values() if _is_text(c)]
    for link in text_links:
        dtab, lcol = link["dim_table"], link["label_column"]
        try:
            labs = [str(r[0]) for r in cur.execute(
                f"SELECT DISTINCT {q(lcol)} FROM {q(dtab)} "
                f"WHERE {_like_pred(lcol, pats)} "
                f"LIMIT {MAX_TLINK_LABELS}", params)]
        except sqlite3.Error:
            continue
        if not etext:
            break
        expr = "LOWER(' '||" + "||' '||".join(
            f"COALESCE({q(c)},'')" for c in etext) + "||' ')"
        for lab in labs:
            try:
                rows = cur.execute(
                    f"SELECT {q(ident.column)} "
                    f"FROM {q(proj.entity_table)} "
                    f"WHERE {expr} LIKE ?{LIKE_ESCAPE_CLAUSE} "
                    f"LIMIT {int(per_term_limit)}",
                    [f"% {escape_like(lab.lower())} %"]).fetchall()
            except sqlite3.Error:
                continue
            for (rid,) in rows:
                hits.setdefault(str(rid), []).append({
                    "term": term, "channel": CH_TLINK,
                    "table": dtab, "column": lcol,
                    "value": str(lab)[:120],
                    "coverage": link.get("coverage")})

    # Channel 7 -- child mentions: tables that hold a FK INTO the
    # entity table carry content ABOUT specific entities (reviews of a
    # product, a markdown table inside a document section). The term
    # hits their text columns; the referenced entities bind through
    # the recorded FK -- the reverse of fk_dimension. Skipped when the
    # channel index already answered it.
    for tname, tprof in (proj.snapshot.tables.items()
                         if idx is None or len(term) < 3 else ()):
        fks_to_ent = [c.name for c in tprof.columns.values()
                      if c.is_foreign_key
                      and c.fk_target_table == proj.entity_table]
        if not fks_to_ent:
            continue
        for fkcol in fks_to_ent:
            for tcol in tprof.columns.values():
                if not _is_text(tcol) or tcol.name == fkcol:
                    continue
                try:
                    rows = cur.execute(
                        f"SELECT {q(fkcol)}, {q(tcol.name)} "
                        f"FROM {q(tname)} "
                        f"WHERE {_like_pred(tcol.name, pats)} "
                        f"LIMIT {int(per_term_limit)}", params)
                except sqlite3.Error:
                    continue
                for rid, v in rows:
                    if rid is None:
                        continue
                    hits.setdefault(str(rid), []).append({
                        "term": term, "channel": CH_MENTION,
                        "table": tname, "column": tcol.name,
                        "value": str(v)[:120],
                        "via": f"{tname}.{fkcol}"})
    return hits


def _scan_term_exact(cur, proj: SourceProjection, term: str,
                     per_term_limit: int) -> dict:
    """Anchored binding for terms below MIN_TERM ('42', '41').

    A short term earns the right to bind only by being an EXACT value
    in a discovered vocabulary -- a text_link dimension label or a
    1-hop dimension label -- never by substring LIKE, which would let
    '4' match everything. Entities are then bound through the same
    evidence channels: FK equality for dimensions, token-anchored
    containment for text links. A short term absent from every
    vocabulary returns {} and is honestly dropped by the caller.
    """
    from source_projection import _is_text

    q = quote_ident
    hits: dict[str, list] = {}

    ident = proj.resolve("identity")
    profile = proj.snapshot.tables.get(proj.entity_table)
    if not ident.column or profile is None:
        return hits
    variants = _case_variants(term)

    # Path 0 -- the term IS a complete cell value on the entity table
    # itself ('42' == a size cell), not a substring.
    for col in profile.columns.values():
        if not _is_text(col):
            continue
        try:
            rows = cur.execute(
                f"SELECT {q(ident.column)}, {q(col.name)} "
                f"FROM {q(proj.entity_table)} "
                f"WHERE LOWER({q(col.name)}) IN "
                f"({','.join('?' * len(variants))}) "
                f"LIMIT {int(per_term_limit)}",
                [v.lower() for v in variants]).fetchall()
        except sqlite3.Error:
            continue
        for rid, v in rows:
            hits.setdefault(str(rid), []).append(
                {"term": term, "channel": CH_ENTITY,
                 "column": col.name, "value": str(v)[:120]})

    # Path A -- 1-hop dimension labels: term IS the label value.
    links = [(c.name, c.fk_target_table, c.fk_target_column)
             for c in profile.columns.values()
             if c.is_foreign_key and c.fk_target_table]
    for child_col, parent, parent_col in links:
        pprof = proj.snapshot.tables.get(parent)
        if pprof is None:
            continue
        for tcol in pprof.columns.values():
            if not _is_text(tcol):
                continue
            try:
                pks = [r[0] for r in cur.execute(
                    f"SELECT {q(parent_col)} FROM {q(parent)} "
                    f"WHERE LOWER({q(tcol.name)}) IN "
                    f"({','.join('?' * len(variants))}) "
                    f"LIMIT {int(per_term_limit)}",
                    [v.lower() for v in variants])]
            except sqlite3.Error:
                continue
            if not pks:
                continue
            ph = ",".join("?" * len(pks))
            try:
                rows = cur.execute(
                    f"SELECT {q(ident.column)} "
                    f"FROM {q(proj.entity_table)} "
                    f"WHERE {q(child_col)} IN ({ph}) "
                    f"LIMIT {int(per_term_limit)}", pks).fetchall()
            except sqlite3.Error:
                continue
            for (rid,) in rows:
                hits.setdefault(str(rid), []).append({
                    "term": term, "channel": CH_FKDIM,
                    "table": parent, "column": tcol.name,
                    "value": str(term)[:120]})

    # Path B -- text-containment links: term IS a dimension label that
    # lives inside entity text.
    etext = [c.name for c in profile.columns.values() if _is_text(c)]
    expr = ("LOWER(' '||" + "||' '||".join(
        f"COALESCE({q(c)},'')" for c in etext) + "||' ')") if etext else None
    for link in getattr(proj, "text_links", []) or []:
        try:
            labs = [str(r[0]) for r in cur.execute(
                f"SELECT DISTINCT {q(link['label_column'])} "
                f"FROM {q(link['dim_table'])} "
                f"WHERE LOWER({q(link['label_column'])}) IN "
                f"({','.join('?' * len(variants))}) "
                f"LIMIT {MAX_TLINK_LABELS}",
                [v.lower() for v in variants])]
        except sqlite3.Error:
            continue
        if expr is None:
            break
        for lab in labs:
            try:
                rows = cur.execute(
                    f"SELECT {q(ident.column)} "
                    f"FROM {q(proj.entity_table)} "
                    f"WHERE {expr} LIKE ?{LIKE_ESCAPE_CLAUSE} "
                    f"LIMIT {int(per_term_limit)}",
                    [f"% {escape_like(lab.lower())} %"]).fetchall()
            except sqlite3.Error:
                continue
            for (rid,) in rows:
                hits.setdefault(str(rid), []).append({
                    "term": term, "channel": CH_TLINK,
                    "table": link["dim_table"],
                    "column": link["label_column"],
                    "value": str(lab)[:120],
                    "coverage": link.get("coverage")})

    # Path C -- exact cell match in a table that FKs into the entity
    # (a child row whose value IS the short term, e.g. size '42').
    for tname, tprof in proj.snapshot.tables.items():
        fks_to_ent = [c.name for c in tprof.columns.values()
                      if c.is_foreign_key
                      and c.fk_target_table == proj.entity_table]
        if not fks_to_ent:
            continue
        for fkcol in fks_to_ent:
            for tcol in tprof.columns.values():
                if not _is_text(tcol) or tcol.name == fkcol:
                    continue
                try:
                    rows = cur.execute(
                        f"SELECT {q(fkcol)}, {q(tcol.name)} "
                        f"FROM {q(tname)} "
                        f"WHERE LOWER({q(tcol.name)}) IN "
                        f"({','.join('?' * len(variants))}) "
                        f"LIMIT {int(per_term_limit)}",
                        [v.lower() for v in variants])
                except sqlite3.Error:
                    continue
                for rid, v in rows:
                    if rid is None:
                        continue
                    hits.setdefault(str(rid), []).append({
                        "term": term, "channel": CH_MENTION,
                        "table": tname, "column": tcol.name,
                        "value": str(v)[:120],
                        "via": f"{tname}.{fkcol}"})
    return hits


def relax_search(
    db_path: str,
    proj: SourceProjection,
    query: str,
    limit: int = MAX_RELAXED_ITEMS,
    vector: tuple | None = None,
    channel_index=None,
) -> RelaxationResult | None:
    """Widen a zero-result lexical query across the evidence channels.

    Returns None when the source has nothing to widen over (no entity
    table resolved). Returns a result with empty `items` when widening
    ran but honestly found nothing.
    """
    eav = getattr(proj, "eav_tables", {}) or {}
    if not proj.entity_table:
        return None

    terms = [t for t in (query or "").split() if t]
    if not terms:
        return RelaxationResult()

    conn = sqlite3.connect(
        "file:{}?mode=ro".format(
            os.path.abspath(db_path).replace("\\", "/")),
        uri=True)
    try:
        cur = conn.cursor()
        result = RelaxationResult()

        # Stage A: per-term coverage across all channels. The vector
        # channel is a per-term FALLBACK: it binds only when the lexical
        # channels found nothing for that term. Otherwise a noisy dense
        # hit would widen a covered term's candidate set and poison the
        # conjunctive intersection (bench-caught failure mode: a
        # colliding embedder turned ebr+v recall to 0 on conj_mix).
        coverage = {}
        vector_shadowed: list[str] = []
        for term in terms:
            if len(term) < MIN_TERM:
                # A sub-MIN term ('42') may still be a valid label --
                # it binds only by exact membership in a discovered
                # vocabulary, never by substring.
                coverage[term] = _scan_term_exact(
                    cur, proj, term, MAX_PER_TERM_SCAN)
            else:
                coverage[term] = _scan_term(
                    cur, proj, term, eav, MAX_PER_TERM_SCAN,
                    idx=channel_index)
            if coverage[term]:
                if vector is not None:
                    vector_shadowed.append(term)
                continue
            if vector is None:
                continue
            idx, embedder = vector
            for rid, bs in idx.bind(term, embedder).items():
                coverage[term].setdefault(rid, []).extend(bs)
        for t in vector_shadowed:
            result.steps.append(RelaxStep(
                stage="vector_shadowed", term=t,
                detail=("vector channel skipped -- term already bound "
                        "by lexical channels; dense hits cannot widen "
                        "a covered term")))

        # Stage B: constraint relaxation -- terms with zero coverage
        # cannot be satisfied on this source at all; dropping them is the
        # minimal honest regret (each dropped term is recorded).
        satisfiable = [t for t in terms if coverage[t]]
        dropped = [t for t in terms if not coverage[t]]
        result.regret = len(dropped)
        for t in dropped:
            result.steps.append(RelaxStep(
                stage="unresolved_term", term=t,
                detail="term binds to no column and no EAV value"))
            result.steps.append(RelaxStep(
                stage="constraint_relaxation", term=t,
                detail="dropped to keep the conjunct satisfiable"))

        if satisfiable:
            common = None
            for t in satisfiable:
                ids = set(coverage[t])
                common = ids if common is None else (common & ids)
            for rid in sorted(common or ()):
                if len(result.items) >= limit:
                    break
                bindings = []
                for t in satisfiable:
                    bindings.extend(coverage[t][rid])
                attr_b = next((b for b in bindings
                               if b["channel"] != CH_ENTITY), None)
                if len(satisfiable) > 1:
                    via = "conjunctive"
                elif attr_b is not None:
                    via = attr_b["channel"]
                else:
                    via = CH_ENTITY
                result.items.append(RelaxedItem(
                    entity_id=rid, via=via, bindings=bindings,
                    axis=attr_b["axis"] if attr_b and "axis" in attr_b
                        else (f"{attr_b['table']}.{attr_b['column']}"
                              if attr_b and "table" in attr_b
                              else (attr_b.get("column") if attr_b
                                    else None)),
                    matched_value=attr_b["value"] if attr_b else None))
                result.terms_used.extend(satisfiable)

        # Axis bridge: for every attribute axis that carried a term
        # (EAV key or dimension column), enumerate sibling values -- the
        # widening surface stays inspectable.
        bridged = set()
        for term in satisfiable:
            for bindings in coverage[term].values():
                for b in bindings:
                    if b["channel"] == CH_EAV:
                        sig = (term, "eav", b["table"], b["axis"])
                        if sig in bridged:
                            continue
                        bridged.add(sig)
                        meta = eav[b["table"]]
                        sib = cur.execute(
                            f"SELECT DISTINCT "
                            f"{quote_ident(meta['value_column'])} "
                            f"FROM {quote_ident(b['table'])} "
                            f"WHERE "
                            f"{quote_ident(meta['key_column'])} = ? "
                            f"LIMIT ?",
                            (b["axis"], MAX_AXIS_SIBLINGS)).fetchall()
                        axis = b["axis"]
                    elif b["channel"] in (CH_FKDIM, CH_JUNC):
                        sig = (term, b["channel"], b["table"], b["column"])
                        if sig in bridged:
                            continue
                        bridged.add(sig)
                        sib = cur.execute(
                            f"SELECT DISTINCT {quote_ident(b['column'])} "
                            f"FROM {quote_ident(b['table'])} LIMIT ?",
                            (MAX_AXIS_SIBLINGS,)).fetchall()
                        axis = f"{b['table']}.{b['column']}"
                    else:
                        continue
                    siblings = [str(s[0])[:80] for s in sib]
                    result.steps.append(RelaxStep(
                        stage="axis_bridge", term=term, axis=axis,
                        detail=("sibling values on the same axis: "
                                + ", ".join(siblings))))

        # Hydrate display names through the projection's own role
        # bindings -- no assumed column names.
        if result.items:
            ident = proj.resolve("identity")
            nm = proj.resolve("name")
            if ident.column:
                ids = [i.entity_id for i in result.items]
                ph = ",".join("?" * len(ids))
                sel = (quote_ident(nm.column) if nm.column
                       else quote_ident(ident.column))
                rows = cur.execute(
                    f"SELECT {quote_ident(ident.column)}, {sel} "
                    f"FROM {quote_ident(proj.entity_table)} "
                    f"WHERE {quote_ident(ident.column)} IN ({ph})",
                    ids).fetchall()
                names = {str(r[0]): r[1] for r in rows}
                for it in result.items:
                    it.name = names.get(it.entity_id)

        return result
    finally:
        conn.close()
