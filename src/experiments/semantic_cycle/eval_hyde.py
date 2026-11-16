"""Exploratory HyDE ablation (docs/18, exploratory; NOT a registered endpoint).

An LLM writes ONE hypothetical record in the source's own format (3 real
records of the index serve as format exemplars) for the user's request;
that text is embedded as a DOCUMENT and searched densely. The model only
proposes a hypothesis; retrieval against the real index decides what is
returned, so nothing the model writes can enter an answer by itself.
Generated texts are cached on disk and every call is logged.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

SYSTEM = ("You write one hypothetical catalogue record. Output exactly one "
          'JSON object {"record": "..."} and nothing else.')


def exemplar_docs(store, n: int = 3) -> list[str]:
    ids = [eid for eid, _h in store.hashes()]
    picks = [ids[len(ids) * (i + 1) // (n + 1)] for i in range(n)]
    texts = store.get_many(picks)
    return [texts[i].text[:400] for i in picks if i in texts]


def build_prompt(exemplars: list[str], query: str) -> str:
    ex = "\n".join(f"Example {i + 1}:\n{t}\n" for i, t in
                   enumerate(exemplars))
    return ("Below are 3 example records from a catalogue. They only show "
            "the data format and style; never copy their content.\n\n"
            f"{ex}\nUser request (any language): {query!r}\n\n"
            "Write ONE hypothetical record in the same format and style as "
            "the examples that would be the perfect answer to the request. "
            "Invent plausible concrete details. Keep it under 400 "
            'characters. Return JSON {"record": "<record text>"}.')


def rrf(rankings: list[list[str]], weights: list[float], k: int = 60,
        top: int = 50) -> list[str]:
    """Evaluator-owned weighted Reciprocal Rank Fusion (1-based ranks)."""
    score: dict[str, float] = {}
    for ids, w in zip(rankings, weights):
        for rank, eid in enumerate(ids, 1):
            score[eid] = score.get(eid, 0.0) + w / (k + rank)
    return [e for e, _s in sorted(score.items(),
                                  key=lambda kv: (-kv[1], kv[0]))][:top]


class Hyde:
    def __init__(self, store, emb, llm, cache_path: Path, depth: int = 50,
                 lex_weights: tuple = (0.3,)):
        self.store, self.emb, self.llm = store, emb, llm
        self.depth = depth
        self.lex_weights = lex_weights
        self.exemplars = exemplar_docs(store)
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                d = json.loads(line)
                self.cache[d["key"]] = d["record"]

    def _record(self, query: str) -> str:
        key = hashlib.sha256(query.encode()).hexdigest()
        if key in self.cache:
            return self.cache[key]
        out = self.llm.complete_json(SYSTEM, build_prompt(
            self.exemplars, query), temperature=0.3)
        text = str(out.get("record", "")).strip()
        if not text:
            raise RuntimeError("empty hypothetical record")
        self.cache[key] = text
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "query": query,
                                "record": text}, ensure_ascii=False) + "\n")
        return text

    def _rank(self, vec: np.ndarray) -> list[str]:
        return self.store.search_dense(vec.astype(np.float32),
                                       self.depth).ids

    def run(self, query: str) -> dict:
        out = {}
        t0 = time.perf_counter()
        try:
            text = self._record(query)
            h = self.emb.embed_documents([text])[0]
            q = self.emb.embed_queries([query])[0]
            mix = q + h
            mix = mix / (np.linalg.norm(mix) or 1.0)
            ms = round((time.perf_counter() - t0) * 1000, 2)
            mix_ids = self.store.search_dense(mix.astype(np.float32),
                                              250).ids
            lex_ids = [r[0] for r in self.store.search_lexical(
                query, 250, prefix_truncate=True)]
            out["hyde_dense"] = {"ids": self._rank(h), "error": None,
                                 "ms": ms, "hyde_text": text}
            out["hyde_mix"] = {"ids": mix_ids[:self.depth], "error": None,
                               "ms": ms}
            for w in self.lex_weights:
                out[f"hyde_hybrid_l{w}"] = {
                    "ids": rrf([mix_ids, lex_ids], [1.0, w],
                               top=self.depth),
                    "error": None, "ms": ms}
        except Exception as exc:  # noqa: BLE001 -- counted as failure
            err = f"{type(exc).__name__}: {exc}"
            names = ["hyde_dense", "hyde_mix"] + [
                f"hyde_hybrid_l{w}" for w in self.lex_weights]
            for m in names:
                out[m] = {"ids": [], "error": err, "ms": 0.0}
        return out
