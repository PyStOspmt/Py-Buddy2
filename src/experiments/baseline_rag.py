"""
Py-Buddy: honest retrieval baseline (Phase 1, replacing the old strawman).

What the previous benchmark called a "naive Vector RAG baseline" was a
hardcoded string of eight products pasted into a prompt, scored by substring
search. That is not a baseline, and comparing against it produced the
meaningless "40% vs 100%" headline in the paper.

This module implements an actual retrieval-augmented baseline:

    BM25 lexical retrieval over the real catalogue
        -> top-k product cards (name, category, price, attributes)
        -> single LLM call that answers strictly from those cards
        -> no constraint engine, no verification, no abstention policy

That is the strongest *fair* opponent for a plain RAG assistant, and it is
the configuration a competent engineer would actually ship first.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Ukrainian + English function words that carry no retrieval signal.
_STOP = {
    "the", "and", "for", "with", "you", "your", "this", "that", "from", "have",
    "what", "which", "price", "how", "much", "does", "can", "tell",
    "і", "й", "та", "а", "але", "що", "як", "яка", "який", "які", "чи", "не",
    "в", "у", "на", "до", "з", "із", "за", "для", "від", "це",
    "цей", "ця", "мені", "мне", "мене", "будь", "якщо", "при",
}

_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [
        t for t in (x.lower() for x in _TOKEN_RE.findall(text or ""))
        if len(t) > 1 and t not in _STOP
    ]


@dataclass
class Document:
    doc_id: str
    name: str
    category: str
    price: float
    text: str
    attrs: Dict[str, str] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)
    length: int = 0


class BM25Index:
    """Standard Okapi BM25 over the catalogue's own text fields."""

    K1 = 1.5
    B = 0.75

    def __init__(self, docs: Sequence[Document]):
        self.docs: List[Document] = list(docs)
        self.n = len(self.docs)
        self.df: Counter = Counter()
        for d in self.docs:
            d.tokens = tokenize(d.text)
            d.length = len(d.tokens)
            for t in set(d.tokens):
                self.df[t] += 1
        self.avg_len = (sum(d.length for d in self.docs) / self.n) if self.n else 0.0
        self._index: Dict[str, Dict[int, int]] = {}
        for i, d in enumerate(self.docs):
            for t, c in Counter(d.tokens).items():
                self._index.setdefault(t, {})[i] = c

    def search(self, query: str, k: int = 8) -> List[Tuple[Document, float]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores: Dict[int, float] = {}
        for t in q_tokens:
            postings = self._index.get(t)
            if not postings:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            for i, tf in postings.items():
                d = self.docs[i]
                norm = self.K1 * (1 - self.B + self.B * d.length / (self.avg_len or 1))
                scores[i] = scores.get(i, 0.0) + idf * (tf * (self.K1 + 1)) / (tf + norm)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
        return [(self.docs[i], s) for i, s in ranked]


def build_index(db_path: str) -> BM25Index:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.replace("\\", "/"), uri=True)
    try:
        meta: Dict[str, Dict[str, str]] = {}
        try:
            for pid, k, v in conn.execute(
                "SELECT product_id, meta_key, meta_value FROM product_eav_meta"
            ):
                meta.setdefault(str(pid), {})[str(k)] = str(v)
        except sqlite3.OperationalError:
            pass
        docs: List[Document] = []
        for pid, name, cat, price in conn.execute(
            "SELECT id, name, category, price FROM products"
        ):
            attrs = meta.get(str(pid), {})
            attr_text = " ".join(f"{k} {v}" for k, v in attrs.items())
            text = " ".join([name, cat, attr_text])
            docs.append(Document(str(pid), name, cat, float(price), text, attrs=attrs))
    finally:
        conn.close()
    return BM25Index(docs)


def render_cards(hits: Iterable[Tuple[Document, float]], limit: int = 8) -> str:
    lines = []
    for i, (d, _score) in enumerate(list(hits)[:limit], start=1):
        spec_str = ""
        if d.attrs:
            pairs = [f"{k}={v}" for k, v in sorted(d.attrs.items())[:8]]
            spec_str = " | характеристики: " + ", ".join(pairs)
        lines.append(f"[{i}] {d.name} | категорія: {d.category} | ЦІНА: {d.price:.2f} грн{spec_str}")
    return "\n".join(lines)


BASELINE_SYSTEM_PROMPT = (
    "Ти — асистент інтернет-магазину. Відповідай УКРАЇНСЬКОЮ мовою, "
    "коротко і конкретно. Використовуй ТІЛЬКИ надані картки товарів. "
    "Не вигадуй товарів, цін або характеристик, яких немає в картках. "
    "Якщо відповіді немає в картках — прямо скажи, що такого товару немає. "
    "Відповідь має бути звичайним текстом, без JSON."
)


class RetrievalBaseline:
    """
    BM25 + one LLM call. No constraint engine, no verification gate.

    Deliberately *stronger* than the strawman it replaces: it sees real
    catalogue text. If the neuro-symbolic system cannot beat this, the
    complexity is not earning its keep — and the benchmark will say so.
    """

    name = "bm25_rag"

    def __init__(self, db_path: str, runner=None, top_k: int = 8):
        self.index = build_index(db_path)
        self.top_k = top_k
        self.runner = runner

    def retrieve(self, query: str) -> str:
        return render_cards(self.index.search(query, self.top_k), self.top_k)

    def answer(
        self,
        query: str,
        *,
        conversation_history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> Tuple[str, int, float]:
        """
        Returns (answer_text, llm_calls, cost_usd).

        With no runner configured the suite still runs offline: every
        retrieved card is rendered into the answer, so the baseline gets
        the same evidence opportunity the live LLM arm would have. It is
        a deterministic retrieval upper bound, reported as such, and must
        never be quoted as an LLM result.
        """
        retrieval_q = query
        if conversation_history:
            prior_users = [str(t.get("content") or "") for t in conversation_history if t.get("role") == "user"]
            if prior_users:
                retrieval_q = " ".join(prior_users[-2:] + [query])
        cards = self.retrieve(retrieval_q)
        if self.runner is None:
            # Offline arm: render EVERY retrieved card, not a top-1
            # template. A purpose-built planner vs one card is not a
            # comparison (F26); this gives the baseline the same answer
            # opportunity the live arm's LLM would have over the same
            # retrieval -- deterministically, at zero model spend. It is
            # a retrieval upper bound, never quotable as an LLM result.
            hits = self.index.search(retrieval_q, self.top_k)
            if not hits:
                return ("У наявності немає товарів за вашим запитом.", 0, 0.0)
            lines = [f"{d.name} — {d.price:.2f} грн" for d, _s in hits]
            return ("Знайдені товари: " + "; ".join(lines) + ".", 0, 0.0)

        history_block = ""
        if conversation_history:
            turns = [f"{t.get('role', 'user')}: {str(t.get('content') or '')[:400]}"
                     for t in list(conversation_history)[-4:]]
            history_block = "Попередній діалог:\n" + "\n".join(turns) + "\n\n"
        prompt = (
            f"{history_block}Картки товарів:\n{cards or '(не знайдено)'}\n\n"
            f"Запит клієнта: {query}\n\nВідповідь:"
        )
        text, cost = self.runner.answer_once(BASELINE_SYSTEM_PROMPT, prompt)
        return (text, 1, cost)
