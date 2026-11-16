"""Retrieval metrics with hand-checkable definitions (docs/18 section 5).

`ranked` is the ordered list of returned entity ids (best first, no
duplicates). `relevant` is the data owner's relevant set. Nothing here
depends on the system under test.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """|top-k ∩ R| / min(|R|, k): 1.0 means 'as complete as k allows'."""
    if not relevant:
        raise ValueError("empty relevant set")
    hits = len(set(ranked[:k]) & relevant)
    return hits / min(len(relevant), k)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    top = ranked[:k]
    return len(set(top) & relevant) / k if k > 0 else 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2)
              for i, e in enumerate(ranked[:k]) if e in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, e in enumerate(ranked):
        if e in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranked: Sequence[str], target: str, k: int) -> float:
    return 1.0 if target in ranked[:k] else 0.0


def bootstrap_ci(values: Sequence[float], n: int = 10000,
                 seed: int = 20261112, alpha: float = 0.05
                 ) -> tuple[float, float, float]:
    """Mean with a percentile-bootstrap interval; (nan,nan,nan) if empty."""
    vals = list(values)
    if not vals:
        return (math.nan, math.nan, math.nan)
    rng = random.Random(seed)
    m = len(vals)
    means = sorted(sum(vals[rng.randrange(m)] for _ in range(m)) / m
                   for _ in range(n))
    lo = means[int((alpha / 2) * n)]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return (sum(vals) / m, lo, hi)


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float],
                          n: int = 10000, seed: int = 20261112,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean(a-b) over paired requests with a percentile interval."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci([x - y for x, y in zip(a, b)], n, seed, alpha)


def bootstrap_ci_np(values: Sequence[float], n: int = 10000,
                    seed: int = 20261112, alpha: float = 0.05
                    ) -> tuple[float, float, float]:
    """Vectorised percentile bootstrap (numpy); same contract as
    `bootstrap_ci`, used for the large aggregate tables."""
    import numpy as np
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    step = max(1, 4_000_000 // a.size)
    for s in range(0, n, step):
        m = min(step, n - s)
        means[s:s + m] = a[rng.integers(0, a.size, (m, a.size))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(a.mean()), float(lo), float(hi))


def paired_bootstrap_diff_np(a: Sequence[float], b: Sequence[float],
                             n: int = 10000, seed: int = 20261112,
                             alpha: float = 0.05
                             ) -> tuple[float, float, float]:
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    return bootstrap_ci_np([x - y for x, y in zip(a, b)], n, seed, alpha)


def auroc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie): Mann-Whitney U / (m*n)."""
    if not pos or not neg:
        return math.nan
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))
