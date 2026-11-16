"""
Py-Buddy: registration-time fiduciary probe (external-review P1.2).

Why this exists
---------------
The FIC audit machinery used to live in evaluation code only: a static
lint over source text that could say "no commercial token is named" and
nothing about what the serving path actually does. This probe is the
behavioural complement: at source initialisation it recomputes the
production ordering over a fixed evaluation set under the TRUE
commercial state of the catalogue and under a PERMUTED one (every
detected commercial channel column re-assigned deterministically), then
issues a `FiduciaryCertificate` over the rank movement.

Scope, honestly
---------------
* The scorer under test is the production ordering primitive
  (`ORDER BY price` over in-stock items -- the same key the deterministic
  core and `fetch_candidates` use). An LLM layer could still be leaked a
  commercial field through a prompt; the probe does not cover that.
* Detection of "commercial" columns uses the declared
  `COMMERCIAL_CONCEPTS` table from `fiduciary_audit` -- a labelled
  heuristic list of industry-standard field names, not hidden domain
  knowledge. A channel spelled differently is simply not detected, and
  the report says so (`NO_COMMERCIAL_CHANNEL`).
* The permutation is ONE intervention per probe run, not the universal
  quantifier of the formal claim. Statuses carry that limit.
* The probe never mutates the source: the permuted state lives in a
  temporary copy that is deleted after the audit.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from typing import Any, Callable, Dict, List, Optional, Sequence

from fiduciary_audit import (
    COMMERCIAL_CONCEPTS, audit_ranking, certify,
)
from source_lifecycle import source_version
from source_projection import build_source_projection

#: How many cheapest in-stock items per category the probe fixes as the
#: evaluation set, and how many categories are covered. The comparison is
#: bounded on purpose: a probe that scans the whole catalogue would make
#: every source registration pay for a full audit.
PROBE_TOP_K = 5
PROBE_MAX_CATEGORIES = 3
PROBE_MAX_ITEMS = 50


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _column_tokens(name: str) -> set:
    return {t for t in name.lower().replace("-", "_").split("_") if t}


def detect_commercial_channels(
    conn: sqlite3.Connection,
    entity_table: str,
    id_col: str,
) -> List[Dict[str, str]]:
    """
    Find commercial-looking columns in tables linkable to the entity set.

    A channel = (table, column, concept, link_column). Detection is by
    the declared COMMERCIAL_CONCEPTS spelling table; linkage is
    empirical: the channel table must own a column whose values actually
    match entity ids (checked on a sample, not assumed from the name).
    Undetected spellings are an honest blind spot, reported by the caller
    as `NO_COMMERCIAL_CHANNEL` -- never guessed.
    """
    spellings = {
        s.lower(): concept
        for concept, ss in COMMERCIAL_CONCEPTS.items() for s in ss
    }
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    entity_ids = set()
    try:
        for r in conn.execute(
                f"SELECT {id_col} FROM {entity_table} LIMIT 200").fetchall():
            entity_ids.add(str(r[0]))
    except Exception:
        return []

    channels: List[Dict[str, str]] = []
    for table in tables:
        if table == entity_table:
            continue
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        col_names = [c[1] for c in cols]
        # Candidate link columns: *_id columns whose values match entity ids.
        link_col = None
        for c in col_names:
            if c == id_col or c.endswith("_id") or c == "id":
                try:
                    rows = conn.execute(
                        f"SELECT {c} FROM {table} LIMIT 50").fetchall()
                except Exception:
                    continue
                if any(str(r[0]) in entity_ids for r in rows):
                    link_col = c
                    break
        if link_col is None:
            continue
        for c in col_names:
            if c == link_col:
                continue
            tokens = _column_tokens(c)
            joined = c.lower()
            concept = None
            for spelling, cand in spellings.items():
                if spelling in tokens or spelling in joined:
                    concept = cand
                    break
            if concept:
                channels.append({
                    "table": table, "column": c,
                    "concept": concept, "link": link_col,
                })
    return channels


def _evaluation_set(
    conn: sqlite3.Connection,
    entity_table: str,
    id_col: str,
    price_col: str,
    stock_col: Optional[str],
    cat_col: Optional[str],
) -> List[str]:
    """
    The fixed set the audit ranks: top-K cheapest in-stock items per
    category -- identical in spirit to the experiment's `product_set`,
    but built through the canonical projection so it works on any source
    shape the projection resolves.
    """
    stock_sql = f" AND {stock_col} > 0" if stock_col else ""
    ids: List[str] = []
    if cat_col:
        cats = [r[0] for r in conn.execute(
            f"SELECT DISTINCT {cat_col} FROM {entity_table} "
            f"WHERE {cat_col} IS NOT NULL LIMIT ?",
            (PROBE_MAX_CATEGORIES,)).fetchall()]
        for cat in cats:
            rows = conn.execute(
                f"SELECT {id_col} FROM {entity_table} "
                f"WHERE {cat_col} = ? AND {price_col} IS NOT NULL{stock_sql} "
                f"ORDER BY {price_col} ASC, {id_col} ASC LIMIT ?",
                (cat, PROBE_TOP_K)).fetchall()
            ids.extend(str(r[0]) for r in rows)
    else:
        rows = conn.execute(
            f"SELECT {id_col} FROM {entity_table} "
            f"WHERE {price_col} IS NOT NULL{stock_sql} "
            f"ORDER BY {price_col} ASC, {id_col} ASC LIMIT ?",
            (PROBE_MAX_ITEMS,)).fetchall()
        ids.extend(str(r[0]) for r in rows)
    return ids[:PROBE_MAX_ITEMS]


def _production_scores(
    conn: sqlite3.Connection,
    entity_table: str,
    id_col: str,
    price_col: str,
    pids: Sequence[str],
) -> Dict[str, float]:
    """
    The ordering the deterministic core serves: cheapest first, i.e.
    score = -price. Commercial columns are never read here -- that is
    what the probe is certifying.
    """
    marks = ",".join("?" * len(pids))
    rows = conn.execute(
        f"SELECT {id_col}, {price_col} FROM {entity_table} "
        f"WHERE {id_col} IN ({marks})", tuple(pids)).fetchall()
    return {str(r[0]): -float(r[1]) for r in rows if r[1] is not None}


def _permute_channels(
    conn: sqlite3.Connection,
    channels: Sequence[Dict[str, str]],
    seed: int = 0,
) -> None:
    """
    Re-assign every detected channel's values deterministically.

    Boolean-ish channels get the alternating assignment from
    `deterministic_permutation` (uncorrelated with the true state by
    construction). Wider numeric columns get a rotation of their own
    observed values: the commercial STATE changes, but no magnitude is
    invented that the source did not contain.
    """
    for ch in channels:
        table, col, link = ch["table"], ch["column"], ch["link"]
        rows = conn.execute(
            f"SELECT {link}, {col} FROM {table} ORDER BY {link}"
        ).fetchall()
        if not rows:
            continue
        distinct = {r[1] for r in rows}
        if len(distinct) <= 2:
            # Alternating assignment -- the adversarial flip.
            values = sorted(distinct)
            hi = values[-1]
            lo = values[0]
            for i, r in enumerate(rows):
                conn.execute(
                    f"UPDATE {table} SET {col} = ? WHERE {link} = ?",
                    (hi if (i + seed) % 2 == 0 else lo, r[0]))
        else:
            ordered = [r[1] for r in rows]
            rotated = ordered[1:] + ordered[:1]
            for r, v in zip(rows, rotated):
                conn.execute(
                    f"UPDATE {table} SET {col} = ? WHERE {link} = ?",
                    (v, r[0]))
    conn.commit()


def probe_source(
    db_path: str,
    *,
    scorer: Optional[Callable[[str, Sequence[str]], Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Run the counterfactual fiduciary probe on one source.

    `scorer(db_path, item_ids) -> {id: score}` defaults to the production
    ordering (-price through the canonical projection). It is injectable
    so a regression can prove the probe DETECTS a leak (a biased scorer
    must come back VIOLATION), not merely rubber-stamps.
    """
    out: Dict[str, Any] = {
        "status": "UNVERIFIED",
        "probe": "fiduciary_probe/1.0",
        "channels": [],
        "evaluated_items": 0,
        "scope": ("one permutation per detected commercial channel, over "
                  "the production price ordering; does not cover LLM "
                  "prompt-context leaks or undetected spellings"),
    }
    if not db_path or not os.path.exists(db_path):
        out["reason"] = "source file missing"
        return out
    try:
        out["source_version"] = list(source_version(db_path))
    except Exception:
        # The version stamp is enrichment; the probe still runs without it.
        pass

    try:
        proj = build_source_projection(db_path)
    except Exception as exc:
        out["reason"] = f"projection failed: {type(exc).__name__}"
        return out
    if proj is None or proj.entity_table is None:
        out["reason"] = "no resolvable entity table"
        return out

    ident = proj.resolve("identity")
    price_res = proj.resolve("price")
    stock_res = proj.resolve("stock")
    cat_res = proj.resolve("category")
    if not ident.resolved or not price_res.column:
        out["reason"] = "entity id or price axis unresolved"
        return out

    conn = _connect(db_path)
    try:
        channels = detect_commercial_channels(
            conn, proj.entity_table, ident.column)
        out["channels"] = [
            {"table": c["table"], "column": c["column"],
             "concept": c["concept"]} for c in channels]
        if not channels:
            out["status"] = "NO_COMMERCIAL_CHANNEL"
            out["reason"] = ("no commercial-typed column was detected "
                             "among the declared concept spellings")
            return out

        pids = _evaluation_set(conn, proj.entity_table, ident.column,
                               price_res.column, stock_res.column,
                               cat_res.column)
        out["evaluated_items"] = len(pids)
        if len(pids) < 2:
            out["reason"] = ("fewer than two rankable items; a flip "
                             "cannot move a rank")
            return out
    finally:
        conn.close()

    with tempfile.TemporaryDirectory(prefix="fic_probe_") as tmp:
        permuted_path = os.path.join(tmp, "permuted.db")
        shutil.copyfile(db_path, permuted_path)
        pconn = _connect(permuted_path)
        try:
            _permute_channels(pconn, channels)
        finally:
            pconn.close()

        if scorer is not None:
            score_true = scorer(db_path, pids)
            score_perm = scorer(permuted_path, pids)
        else:
            tconn = _connect(db_path)
            try:
                score_true = _production_scores(
                    tconn, proj.entity_table, ident.column,
                    price_res.column, pids)
            finally:
                tconn.close()
            pconn = _connect(permuted_path)
            try:
                score_perm = _production_scores(
                    pconn, proj.entity_table, ident.column,
                    price_res.column, pids)
            finally:
                pconn.close()

    audit = audit_ranking(score_true, score_perm)
    cert = certify(score_true=score_true, score_permuted=score_perm)
    out["status"] = cert.status
    out["fid"] = audit.fid
    out["evaluable"] = audit.evaluable
    out["moved_items"] = [
        {"item_id": d.item_id, "rank_true": d.rank_true,
         "rank_permuted": d.rank_permuted}
        for d in audit.moved_items
    ]
    out["violations"] = cert.violations()
    if cert.status != "CERTIFIED" and not out.get("reason"):
        out["reason"] = "; ".join(out["violations"]) or "audit not evaluable"
    return out
