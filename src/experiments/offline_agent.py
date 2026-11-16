"""
Py-Buddy: deterministic agent (no LLM).

This is the *deterministic core* of the architecture running on its own:
the Value Algebra, the constraint checker and the catalogue queries, with
the LLM planner removed.

It exists for two reasons.

  1. Reproducibility. `python src/benchmark.py --offline --selftest` proves
     the metric is satisfiable, and the deterministic layer can be scored
     without an API key, a network, or a model. A reviewer can therefore
     separate "does the verification layer work" from "is the LLM any good".

  2. Honesty about attribution. Reported under the name `pybuddy_offline`,
     never as a live-LLM result. It measures exactly one thing: how much of
     the factual grounding comes from determinism rather than from the
     language model.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional

from benchmark_cases import (
    Snapshot,
)
from independent_verifier import check_bundle_independent

_NUM_RE = re.compile(r"((?:\d{1,3}(?:[ \u00a0']\d{3})+|\d+)(?:[.,]\d{1,2})?)\s*грн", re.I)
_PRICE_Q = re.compile(r"скільки коштує|яка ціна|яка точна ціна|ціна", re.I)
_CHEAPEST = re.compile(r"найдешевш|дешеви|бюджетн", re.I)
_EXPENSIVE = re.compile(r"найдорожч|дорожч|преміум|максимальн", re.I)
_AVAILABILITY = re.compile(r"[вy у] наявності|в наявності|наявні|доступн|є в stock|залиш", re.I)
_REV_SUFFIX = re.compile(r"\s*\(Rev \d+\)\s*$", re.I)


def _dedupe_by_base(names: List[str]) -> List[str]:
    """
    One entry per base product: "X" and "X (Rev N)" are LISTINGS of the same
    product, and a compat bundle built from both variants checks a product
    against itself. The longest variant (the one the query actually named)
    wins, so the bundle is the pair of distinct products the user asked about.
    """
    best: Dict[str, str] = {}
    for n in names:
        base = _REV_SUFFIX.sub("", n)
        if base not in best or len(n) > len(best[base]):
            best[base] = n
    return list(best.values())
_COMPAT = re.compile(r"сумісн|підійд|підход|збір|комплект|набір|постав|встанов", re.I)
_OVERCON = re.compile(r"порадь|до \d[\d \u00a0']*\s*грн", re.I)


def _parse_budget(query: str) -> Optional[float]:
    m = _NUM_RE.search(query)
    if not m:
        return None
    raw = re.sub(r"[ \u00a0']", "", m.group(1)).replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


class OfflineDeterministicAgent:
    """Deterministic planner + tools, no language model."""

    name = "pybuddy_offline"

    def __init__(self, snap: Snapshot):
        self.snap = snap
        self.conn = sqlite3.connect("file:%s?mode=ro" % snap.db_path.replace("\\", "/"), uri=True)
        self._names = {n.lower(): n for n in snap.names()}
        # name -> product id, for answering "what does X cost".
        #
        # In these catalogues one name can be sold by several merchants,
        # so a name does not identify a row. This picks the cheapest
        # listing, which is the one a customer asking a price question
        # means, and the answer quotes that row's id so the figure is
        # traceable to a specific seller.
        self._name_to_id: Dict[str, str] = {}
        for pid, name in snap.id_to_name.items():
            current = self._name_to_id.get(name)
            if current is None:
                self._name_to_id[name] = pid
                continue
            try:
                cheaper = float(snap.products[pid]["price"]) < \
                    float(snap.products[current]["price"])
            except (KeyError, TypeError, ValueError):
                continue
            if cheaper:
                self._name_to_id[name] = pid

    # -- retrieval helpers -------------------------------------------------

    def _find_products(self, query: str, limit: int = 5) -> List[str]:
        """
        Match catalogue names mentioned verbatim in the question, most
        specific first.

        The catalogue carries both "X" and "X (Rev N)" as distinct names, and
        a query naming "X (Rev 1)" contains BOTH keys. Taking the first dict
        hit silently resolved the question to a different listing than the
        one asked about -- the stricter F25 metric surfaced this as a
        recall-0 answer on direct_price cases.
        """
        low = query.lower()
        hits = [orig for low_key, orig in self._names.items() if low_key in low]
        hits.sort(key=len, reverse=True)
        return hits[:limit]

    def _by_category(self, category: str, price_desc: bool,
                     budget: Optional[float] = None) -> List[str]:
        # No stock filter here: the superlative questions ask about the
        # CATALOGUE ("Який процесор у каталозі найдешевший?"), and the oracle
        # is computed over the whole category. Filtering to in-stock rows
        # made the agent answer a different question than the one asked.
        rows = self.conn.execute(
            "SELECT id, price FROM products WHERE category=?",
            (category,),
        ).fetchall()
        rows = [(pid, float(p)) for pid, p in rows]
        if budget is not None:
            rows = [r for r in rows if r[1] <= budget]
        rows.sort(key=lambda r: r[1], reverse=price_desc)
        return [pid for pid, _ in rows[:5]]

    @staticmethod
    def _detect_category(query: str) -> Optional[str]:
        low = query.lower()
        for cat in ("cpu", "motherboard", "ram", "gpu", "psu", "ssd", "rim", "wheel_hub"):
            if cat in low:
                return cat
        words = {
            "процесор": "cpu", "плата": "motherboard", "материнк": "motherboard",
            # "пам'ят" stems both "пам'ять" and "память": the apostrophe form
            # is the Ukrainian spelling and used to fall through to
            # "Уточніть..." -- a non-answer the lenient metric scored as correct.
            "пам'ят": "ram", "памят": "ram", "озп": "ram",
            "відеокарт": "gpu", "графічн": "gpu",
            "живлен": "psu", "бж": "psu", "ssd": "ssd", "накопичувач": "ssd",
            "диск": "rim", "колесо": "rim", "маточин": "wheel_hub", "ступиц": "wheel_hub",
        }
        for w, cat in words.items():
            if w in low:
                return cat
        return None

    def _price_line(self, pid: str) -> str:
        row = self.conn.execute(
            "SELECT name, price FROM products WHERE id=?", (pid,)
        ).fetchone()
        return f"{row[0]} за {float(row[1]):.2f} грн"

    # -- tools -------------------------------------------------------------

    def _answer_price(self, query: str) -> str:
        names = self._find_products(query)
        if names:
            # `id_to_name_inv` did not exist. The Py-Buddy arm of the benchmark
            # raised AttributeError on its first price question, which is
            # why every number in the paper for that arm had to be taken on
            # faith: the harness had never actually run it.
            pid = next((self._name_to_id.get(n) for n in names
                        if self._name_to_id.get(n)), None)
            if pid:
                return f"Ціна: {self._price_line(pid)}."

        cat = self._detect_category(query)
        budget = _parse_budget(query)
        if cat and (_CHEAPEST.search(query) or _EXPENSIVE.search(query)):
            desc = bool(_EXPENSIVE.search(query))
            rows = self._by_category(cat, desc, None if _CHEAPEST.search(query) else budget)
            if not rows:
                return (f"У категорії {cat} немає товарів у наявності "
                        f"за вказаним бюджетом.")
            word = "найдорожчий" if desc else "найдешевший"
            return f"Ціна {word} товару в категорії {cat}: {self._price_line(rows[0])}."
        return "Уточніть, будь ласка, який саме товар вас цікавить."

    def _answer_availability(self, query: str) -> str:
        cat = self._detect_category(query)
        if not cat:
            return "Уточніть категорію товарів, будь ласка."
        rows = self.conn.execute(
            "SELECT name, price, stock_quantity FROM products "
            "WHERE category=? AND stock_quantity>0 ORDER BY price ASC LIMIT 5",
            (cat,),
        ).fetchall()
        if not rows:
            return f"У категорії {cat} наразі немає товарів у наявності."
        items = "; ".join(f"{n} — {float(p):.2f} грн (залишок {int(s)})" for n, p, s in rows)
        return f"У наявності в категорії {cat}: {items}."

    @staticmethod
    def _clean_reasons(reasons: List[str]) -> List[str]:
        """
        Client-facing reasons name the violated invariant class, not the raw
        values behind it. Pasting "['AM5', 'LGA1700']" into the answer emits
        model-like tokens the grounding check cannot attribute to any
        catalogue product -- and a shopper does not need the raw list, only
        the reason.
        """
        out: List[str] = []
        for r in reasons:
            if "сокет" in r:
                out.append("різні сокети у товарів комплекту")
            elif "розболтовка" in r:
                out.append("різна розболтовка")
            elif "діаметр" in r:
                out.append("діаметр диска перевищує допуск маточини")
            else:
                out.append(r)
        return out

    def _answer_compat(self, query: str):
        """
        Returns (text, verdict). F48: the compatibility decision is a
        structured fact this arm computed -- it is emitted as a verdict
        field so the evaluator classifies the stance from the system's
        own output, not by grepping the prose for Ukrainian cue words.
        """
        deduped = _dedupe_by_base(self._find_products(query, limit=12))
        pids = [self._name_to_id[n] for n in deduped if n in self._name_to_id]
        if len(pids) >= 2:
            # F26: the agent verifies with its OWN implementation
            # (independent_verifier), not with the oracle's check_bundle --
            # importing the oracle's verifier made agent-vs-oracle agreement
            # a tautology of shared code, not evidence.
            v = check_bundle_independent(self.snap.db_path, pids)
            names = [self.snap.id_to_name[p] for p in pids[:2]]
            why = (" " + "; ".join(self._clean_reasons(v.reasons))) if v.reasons else ""
            if v.ok:
                # The verdict must name what it verdicts about: an unnamed
                # "так, сумісно" carries recall 0 and is unfalsifiable.
                return (f"Так, {names[0]} сумісний з {names[1]} — "
                        f"ця конфігурація підходить." + why), "accept"
            return (f"Ні, {names[0]} не сумісний з {names[1]} — "
                    f"конфігурація відхилена." + why), "reject"

        # Two-item form: "Чи підійде X на Y?" — look for an explicit pairing.
        for pid in pids[:1]:
            cat = self.snap.products[pid].get("category")
            # Ordered by price, not RANDOM(): a benchmark that returns a
            # different answer on every run cannot be compared with
            # anything, including itself.
            partner = self.conn.execute(
                "SELECT id FROM products WHERE category=? AND id<>? "
                "AND stock_quantity>0 ORDER BY price ASC LIMIT 1",
                (cat, pid)).fetchone()
            if partner:
                pids = [pid, partner[0]]
            break
        if len(pids) < 2:
            return ("Уточніть, будь ласка, які саме два товари потрібно перевірити.",
                    None)

        v = check_bundle_independent(self.snap.db_path, pids)
        names = [self.snap.id_to_name[p] for p in pids]
        why = (" " + "; ".join(self._clean_reasons(v.reasons))) if v.reasons else ""
        if v.ok:
            return (f"Так, {names[0]} сумісний з {names[1]}." + why), "accept"
        return (f"Ні, {names[0]} не сумісний з {names[1]}." + why), "reject"

    def _answer_overconstrained(self, query: str) -> str:
        cat = self._detect_category(query)
        budget = _parse_budget(query)
        if not cat or budget is None:
            return "Уточніть категорію та бюджет, будь ласка."
        cheapest = self.conn.execute(
            "SELECT MIN(price) FROM products WHERE category=? AND stock_quantity>0",
            (cat,),
        ).fetchone()[0]
        if cheapest is None:
            return f"У категорії {cat} немає товарів у наявності."
        if float(cheapest) > budget:
            return (f"Повністю виконати запит неможливо: найдешевший товар у категорії "
                    f"{cat} коштує {float(cheapest):.2f} грн, що перевищує ваш бюджет. "
                    f"Найдешевший доступний варіант: "
                    f"{self._price_line(self._by_category(cat, False)[0])}.")
        return f"У межах бюджету {budget:.2f} грн є товари: " \
               f"{self._price_line(self._by_category(cat, False, budget)[0])}."

    def _answer_price_trap(self, query: str) -> str:
        return ("Такого товару в нашому каталозі немає. "
                "Я не можу назвати ціну того, чого в базі немає.")

    def _answer_conversational(self, query: str) -> str:
        # The catalogue size is read, not hardcoded: a tier corpus in the
        # scale-envelope protocol must not report the full set.
        return ("Вітаю! Я консультант Py-Buddy. Працюю з каталогом із "
                f"{len(self.snap.products)} товарів, перевіряю сумісність "
                "детерміновано і не вигадую цін чи характеристик. "
                "Чим допомогти?")

    # -- planner -----------------------------------------------------------

    def answer(self, query: str) -> Dict[str, object]:
        low = query.lower()
        steps = 1
        verdict = None
        if _COMPAT.search(low):
            text, verdict = self._answer_compat(query)
        elif _OVERCON.search(low):
            text = self._answer_overconstrained(query)
        elif _AVAILABILITY.search(low):
            text = self._answer_availability(query)
        elif _PRICE_Q.search(low) or _CHEAPEST.search(low) or _EXPENSIVE.search(low):
            text = self._answer_price(query)
        elif any(w in low for w in ("привіт", " віта", "хто ти", "як працюєш",
                                     "розкажи про", "що вмієш")):
            text = self._answer_conversational(query)
        else:
            text = ("Не вдалося однозначно інтерпретувати запит. "
                    "Уточніть, будь ласка, товар або категорію.")
            steps = 0
        return {"text": text, "steps": steps, "cost_usd": 0.0,
                "verdict": verdict}


# `id_to_name_inv` is attached here rather than in Snapshot to keep the
# snapshot dataclass focused on the data it caches.
def _attach_inverse(snap: Snapshot) -> Snapshot:
    if not hasattr(snap, "id_to_name_inv"):
        snap.id_to_name_inv = {v: k for k, v in snap.id_to_name.items()}
    return snap


def build(snap: Snapshot) -> OfflineDeterministicAgent:
    return OfflineDeterministicAgent(_attach_inverse(snap))
