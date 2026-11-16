"""
Independent bundle-compatibility verifier (F26 fix, R0).

Status: EVALUATION INFRASTRUCTURE. This module exists so that agreement
between the offline agent and the benchmark oracle means something: until
now `offline_agent.py` imported the oracle's own `check_bundle`
(benchmark_cases.py:246), so a high agreement score established consistency
of a SHARED implementation, not an independent physical truth. This verifier
is a second implementation with its own raw-value parsing -- it reads the
EAV rows straight from SQLite and normalises them with rules written here,
never importing attribute_algebra.

Honest limits:
  * Independence is at the CODE level. Both implementations read the same
    catalogue, so a wrong value stored in the database fools both. True
    source-independent labels need a second data provenance (R5).
  * The parsing rules below are written from the catalogue's own surface
    forms (see data/generate_marketplace_scale.py): sockets like "AM5",
    "Socket AM5", "sAM5"; bolt patterns like "5x112", "5*112 мм"; rim
    diameters like "R18", '18"'; hub clearances like "18 дюймів". A catalogue
    with different surface forms needs its own parsing profile -- that is
    the adaptation problem, not something this verifier hides.
  * Undecidable bundles are ACCEPTED here too (absence of evidence is not
    evidence of incompatibility), matching the oracle's stated semantics so
    the differential test compares like with like.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import List, Sequence

_MM_PER_INCH = 25.4

_SOCKET_KEY = re.compile(r"сокет|socket|гніздо|роз'?єм", re.I)
_PCD_KEY = re.compile(r"розболтовк|pcd|кріпленн|болт", re.I)
_DIAMETER_KEY = re.compile(r"діаметр|diameter|радіус|розмір", re.I)
_CLEARANCE_KEY = re.compile(r"допуск|clearance", re.I)

_SOCKET_VAL = re.compile(r"(?:^|(?<=\s))(?:socket\s*)?(s?)(am[45]|lga ?\d{4})(?:$|(?=\s))", re.I)
_PCD_VAL = re.compile(r"(\d{1,2})\s*[*xXх]\s*(\d{2,3}(?:[.,]\d+)?)")
_RIM_VAL = re.compile(r"(?:^R|\br|R?)(\d{1,2}(?:[.,]\d+)?)\s*(?:\"|дюйм|мм)?\s*$", re.I)
_NUM_VAL = re.compile(r"(\d{1,2}(?:[.,]\d+)?)\s*(дюйм\w*|мм|\"|inch)?", re.I)


@dataclass
class IndependentVerdict:
    ok: bool
    reasons: List[str] = field(default_factory=list)


def _norm_socket(raw: str) -> str:
    """'Socket AM5'/'sAM5'/'am5' -> 'AM5'; 'Socket LGA1700' -> 'LGA1700'."""
    m = _SOCKET_VAL.search(raw)
    if not m:
        return raw.strip().upper()
    _, core = m.group(1), m.group(2).replace(" ", "").upper()
    return core  # the leading 's' marker is decoration, dropped either way


def _norm_pcd(raw: str) -> str:
    m = _PCD_VAL.search(raw)
    if not m:
        return ""
    return f"{m.group(1)}x{m.group(2).replace(',', '.')}"


def _to_mm(raw: str) -> float:
    """Parse a length value; inches (дюйм, ") convert, bare numbers are mm."""
    m = _NUM_VAL.search(raw)
    if not m:
        return 0.0
    val = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "").lower()
    if unit.startswith("дюйм") or unit == '"' or unit == "inch":
        return val * _MM_PER_INCH
    return val


def _attrs(db_path: str, product_id: str) -> List[tuple]:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    try:
        return conn.execute(
            "SELECT meta_key, meta_value FROM product_eav_meta WHERE product_id=?",
            (product_id,),
        ).fetchall()
    finally:
        conn.close()


def check_bundle_independent(db_path: str, product_ids: Sequence[str]) -> IndependentVerdict:
    """
    Second implementation of the bundle verdict: socket equality, bolt-pattern
    equality, rim-inside-hub-clearance -- from raw EAV rows, own parsing.
    """
    reasons: List[str] = []
    sockets, pcds = {}, {}
    rim_mm, hub_mm = {}, {}

    for pid in product_ids:
        for key, val in _attrs(db_path, pid):
            k, v = str(key), str(val)
            if _CLEARANCE_KEY.search(k):
                hub_mm[pid] = _to_mm(v)
            elif _DIAMETER_KEY.search(k):
                rim_mm[pid] = _to_mm(v)
            elif _PCD_KEY.search(k):
                p = _norm_pcd(v)
                if p:
                    pcds[pid] = p
            elif _SOCKET_KEY.search(k):
                s = _norm_socket(v)
                if s:
                    sockets[pid] = s

    distinct_sockets = sorted(set(sockets.values()))
    if len(distinct_sockets) > 1:
        reasons.append(f"різні сокети: {sorted(sockets.values())}")

    distinct_pcds = sorted(set(pcds.values()))
    if len(distinct_pcds) > 1:
        reasons.append(f"різна розболтовка: {distinct_pcds}")

    for hub_id, limit in hub_mm.items():
        for rim_id, d in rim_mm.items():
            if rim_id != hub_id and d > limit:
                reasons.append(
                    f"діаметр {d:.0f} мм перевищує допуск {limit:.0f} мм")

    return IndependentVerdict(ok=not reasons, reasons=reasons)