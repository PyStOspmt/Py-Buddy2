"""
Py-Buddy: morphological intent matching by character n-grams.

What this is
------------
A nearest-neighbour cache over *character n-gram* vectors, used to reuse a
verified SQL query for a question that is worded differently but shares
its stem. It generalises over INFLECTION -- "найдешевший диск" and
"найдешевші диски" land in the same place -- and over nothing else.

What it is not
--------------
It is not a semantic encoder, and the previous version of this file said
it was:

    "Projects Ukrainian and multi-lingual queries into a normalized
     256-dimensional semantic space."

There is no semantic space here. `NgramVectorizer` is an FNV-1a hash of
3-grams and 4-grams folded into 256 buckets with sublinear term
frequency, L2-normalised. Consequences, all of them real:

  * "Яка ціна на процесор?" and "Скільки коштує чіп?" have no character
    n-grams in common beyond noise, and will not match. A synonym is
    invisible to it, always, by construction.
  * Hash collisions are guaranteed, not rare. 256 buckets against a
    vocabulary of thousands of distinct n-grams means unrelated features
    share a coordinate, and the cosine is computed over the collision.
    Raising `dimensions` reduces this and costs memory; it does not make
    the encoder semantic.
  * It is English-agnostic and language-agnostic only in the sense that it
    does not care what language you feed it. It has no idea what any of
    it means, in any language.

The name changed because the old one was a claim the code could not
support, and ADR-011 already recorded that decision. `SemanticIntentRouter`
and `VectorEncoder` survive as aliases so existing imports do not break;
they should be removed, not relied on.

Is it still worth having
------------------------
Yes, for what it is. It has no dependencies, runs in about a
millisecond, and catches the common case of a customer rephrasing a
question they have already asked. It is a cache keyed by surface form
with a little morphological tolerance -- which is exactly what a TTL
cache with a fuzzy key does, and a good deal more honest than calling it
a latent space.

The alternative -- a real sentence embedding model -- is a genuine
upgrade and a genuine dependency, and is the obvious next step. See
docs/06_MASTER_PLAN.md, Phase 5.

What is NOT the same thing
--------------------------
`attribute_algebra` unifies attribute VALUES across spellings and units
by canonicalising them. That is a real algebraic operation with a proven
associativity property. Do not cite this module next to it as if the
two were siblings.
"""
from __future__ import annotations

import collections
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

#: Bucket count. 256 is what this always was; see the module docstring for
#: why it is a parameter and not a claim.
DEFAULT_DIMENSIONS = 256

#: Similarity above which a cached query is reused. 0.70 in practice admits
#: mostly-inflected rewrites; measure before raising it.
DEFAULT_THRESHOLD = 0.70


class NgramVectorizer:
    """
    Character n-gram vectoriser: sublinear TF over hashed n-grams, L2
    normalised. Fast, dependency-free, and morphological.

    It is a surface-form representation. Two texts land close together
    when they share character sequences, and for no other reason.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS,
                 ngrams: Tuple[int, ...] = (3, 4)) -> None:
        if dimensions < 8:
            raise ValueError("dimensions below 8 make collisions dominate")
        self.dimensions = dimensions
        self.ngrams = ngrams

    def _hash_token(self, token: str) -> int:
        """
        FNV-1a, folded into `dimensions` buckets.

        Not a cryptographic hash and not trying to be: it needs to be fast
        and stable across processes, which FNV-1a is.
        """
        h = 2166136261
        for ch in token:
            h = (h ^ ord(ch)) * 16777619
        return (h & 0x7FFFFFFF) % self.dimensions

    def features(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
        tokens = [t for t in cleaned.split() if len(t) > 1]
        out: List[str] = []
        for token in tokens:
            out.append(token)
            for n in self.ngrams:
                if len(token) > n:
                    out.extend(token[i:i + n] for i in range(len(token) - n + 1))
        return out

    def encode(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        features = self.features(text)
        if not features:
            return vec
        counts = collections.Counter(features)
        total = len(features)
        for feature, count in counts.items():
            vec[self._hash_token(feature)] += (1.0 + math.log(count)) / total
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        return max(-1.0, min(1.0, dot))


@dataclass
class CachedIntent:
    """One verified query, its vector, and how long it stays reusable."""

    query: str
    vector: List[float]
    intent_name: str
    target_sql: str
    created_at: float
    ttl_seconds: float
    last_accessed: float

    def is_expired(self, now: float) -> bool:
        # `>=`, not `>`: a zero TTL means "dead on arrival", and with `>` an
        # entry registered in the same clock tick survived its own expiry.
        return (now - self.created_at) >= self.ttl_seconds


class IntentCacheStore(ABC):
    """Where verified queries are kept between turns."""

    @abstractmethod
    def find_best_match(self, query_vec: List[float],
                        threshold: float) -> Optional[Tuple[str, str, float]]:
        ...

    @abstractmethod
    def register_intent(self, query: str, query_vec: List[float],
                        intent_name: str, target_sql: str,
                        ttl_seconds: float) -> None:
        ...

    @abstractmethod
    def cleanup_expired(self) -> int:
        ...


@dataclass
class LocalMemoryVectorStore(IntentCacheStore):
    """
    Bounded in-process store with TTL and LRU eviction.

    Not thread-safe, and not persistent: restarting the process loses
    every learned query. That is a deliberate choice for a cache whose
    contents are re-derivable from conversation, and Redis is the answer
    when it stops being one.
    """

    max_size: int = 512
    entries: "collections.OrderedDict[str, CachedIntent]" = field(
        default_factory=collections.OrderedDict)

    def find_best_match(self, query_vec: List[float],
                        threshold: float = DEFAULT_THRESHOLD
                        ) -> Optional[Tuple[str, str, float]]:
        now = time.time()
        best: Optional[CachedIntent] = None
        best_score = 0.0
        for entry in list(self.entries.values()):
            score = NgramVectorizer.cosine_similarity(query_vec, entry.vector)
            if score > best_score:
                best, best_score = entry, score
        if best is None or best_score < threshold:
            return None
        self.entries.move_to_end(best.query, last=True)
        best.last_accessed = now
        return best.intent_name, best.target_sql, best_score

    def register_intent(self, query: str, query_vec: List[float],
                        intent_name: str, target_sql: str,
                        ttl_seconds: float = 86400.0) -> None:
        now = time.time()
        if len(self.entries) >= self.max_size:
            self.entries.popitem(last=False)      # least recently used
        self.entries[query] = CachedIntent(
            query=query, vector=query_vec, intent_name=intent_name,
            target_sql=target_sql, created_at=now, ttl_seconds=ttl_seconds,
            last_accessed=now)

    def cleanup_expired(self) -> int:
        now = time.time()
        expired = [k for k, v in self.entries.items() if v.is_expired(now)]
        for key in expired:
            del self.entries[key]
        return len(expired)


class RedisVectorStore(IntentCacheStore):
    """
    Redis adapter.

    Delegates to the in-memory store. It does NOT use Redis search
    vectors: the previous version implied a vector index it never built,
    and a Redis that was never reached (no REDIS_URL, or no redis-py)
    silently served local matches under a Redis name. What it now does is
    what it always did, and says so.
    """

    def __init__(self, redis_url: Optional[str] = None,
                 fallback: Optional[LocalMemoryVectorStore] = None) -> None:
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self.fallback = fallback or LocalMemoryVectorStore()
        self.client = None
        if self.redis_url:
            try:
                import redis
                self.client = redis.from_url(self.redis_url)
            except Exception:
                self.client = None

    def find_best_match(self, query_vec: List[float],
                        threshold: float = DEFAULT_THRESHOLD
                        ) -> Optional[Tuple[str, str, float]]:
        return self.fallback.find_best_match(query_vec, threshold)

    def register_intent(self, query: str, query_vec: List[float],
                        intent_name: str, target_sql: str,
                        ttl_seconds: float = 86400.0) -> None:
        self.fallback.register_intent(query, query_vec, intent_name,
                                     target_sql, ttl_seconds)

    def cleanup_expired(self) -> int:
        return self.fallback.cleanup_expired()


class FuzzyNgramRouter:
    """
    Reuses a verified SQL query for a morphologically similar question.

    "Neuro-symbolic intent router with zero keyword dictionaries" was the
    old docstring. It has no keyword dictionaries, true, but it is not
    neuro-symbolic and it is not what the name implied. What it does is
    worth having; what it is called was not.
    """

    #: Prototypes are seeded once and never expire. They are the fallback
    #: when the learned cache is cold, not a learned artefact.
    PROTOTYPE_TTL = 31536000.0

    def __init__(self, store: Optional[IntentCacheStore] = None,
                 encoder: Optional[NgramVectorizer] = None,
                 db_path: Optional[str] = None) -> None:
        self.encoder = encoder or NgramVectorizer()
        self.store = store or LocalMemoryVectorStore()
        self.db_path = db_path
        self._seed_prototypes(db_path=db_path)

    def _prototype_sql_pair(self, db_path: Optional[str] = None) -> Optional[Tuple[str, str]]:
        if not db_path:
            return (
                "SELECT name, price FROM products ORDER BY price DESC LIMIT 1;",
                "SELECT name, price FROM products ORDER BY price ASC LIMIT 1;",
            )
        from source_projection import build_source_projection, quote_ident
        proj = build_source_projection(db_path)
        table = proj.entity_table
        ident = proj.resolve("identity")
        price_res = proj.resolve("price")
        if not table or not ident.resolved or not price_res.resolved or not price_res.column:
            return None
        name_col = proj.resolve("name").column or ident.column
        q = quote_ident
        return (
            f"SELECT {q(name_col)} AS name, {q(price_res.column)} AS price FROM {q(table)} WHERE {q(price_res.column)} IS NOT NULL ORDER BY {q(price_res.column)} DESC LIMIT 1;",
            f"SELECT {q(name_col)} AS name, {q(price_res.column)} AS price FROM {q(table)} WHERE {q(price_res.column)} IS NOT NULL ORDER BY {q(price_res.column)} ASC LIMIT 1;",
        )

    def _seed_prototypes(self, db_path: Optional[str] = None) -> None:
        """
        Two representative superlative queries, each with the SQL it maps to
        for the resolved schema (`F10`: omitted when source has no price role).
        """
        sql_pair = self._prototype_sql_pair(db_path=db_path)
        if sql_pair is None:
            return
        max_sql, min_sql = sql_pair
        prototypes = [
            (
                "EXTREMA_PRICE_MAX",
                max_sql,
                [
                    "Який найдорожчий товар у вас в магазині",
                    "Який із ваших товарів передбачає найбільше вкладення",
                    "Покажи найдорожчу позицію в каталозі",
                    "Товар з максимальним цінником",
                    "Найбільша ціна товару в асортименті",
                ],
            ),
            (
                "EXTREMA_PRICE_MIN_ALL",
                min_sql,
                [
                    "Який найдешевший товар у магазині",
                    "Що можна купити найдешевше",
                    "Найбільш бюджетна позиція в асортименті",
                ],
            ),
        ]
        for intent_name, sql, exemplars in prototypes:
            for exemplar in exemplars:
                self.store.register_intent(
                    exemplar, self.encoder.encode(exemplar), intent_name, sql,
                    ttl_seconds=self.PROTOTYPE_TTL)

    def route_query(self, query: str,
                    threshold: float = DEFAULT_THRESHOLD
                    ) -> Optional[Tuple[str, str, float]]:
        """
        (intent_name, target_sql, similarity) above the threshold, else None.

        None is a real answer and must stay one. A cache that never misses
        is not a cache.
        """
        return self.store.find_best_match(self.encoder.encode(query), threshold)

    def learn_from_feedback(self, query: str, intent_name: str,
                            target_sql: str,
                            ttl_seconds: float = 86400.0) -> None:
        """
        Register a query the ReAct loop already resolved and verified.

        The learned entries carry a day of life and the prototypes carry a
        year, so a bad learning is forgotten and a good one is not.
        """
        self.store.register_intent(query, self.encoder.encode(query),
                                   intent_name, target_sql, ttl_seconds)


# --------------------------------------------------------------------------
# Backwards-compatible names. Remove these once no import refers to them.
# --------------------------------------------------------------------------
VectorEncoder = NgramVectorizer
SemanticIntentRouter = FuzzyNgramRouter
CachedIntentEntry = CachedIntent
