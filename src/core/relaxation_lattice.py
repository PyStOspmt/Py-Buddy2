"""
Py-Buddy: the relaxation lattice.

Why a lattice
-------------
`nvfg.solve` relaxes exactly ONE constraint, and when no single move can
rescue a query it reports UNSATISFIABLE. That is a real limitation, and it
is not obvious that relaxing two constraints fixes it. This module makes

the space of relaxations an object with structure, so the question becomes

a question about a lattice instead of a hunch.

Representation
--------------
A relaxation assigns, per constraint, either

    None    keep the bound exactly as stated, or
    a float a new bound, possibly the infinite one (= drop the bound).

`None` means *unchanged* and the per-operator infinity means *no bound*.
Conflating the two is not a stylistic choice: if "unchanged" and "dropped"
share a value, the top of the lattice is indistinguishable from the bottom,
and every existence claim collapses. That bug shipped in the first version
of this file and is why the two are separate now.

Order
-----
    L(K) = { r : r is a relaxation of the constraint system K }

ordered by `r ⪯ r'` iff r' is at least as loose as r. The order is
componentwise, and each component ranges over a chain (the catalogue's
values, plus "no bound"), so L(K) is a finite product of chains: a bounded
lattice with a least element (the identity) and a greatest element (every
bound dropped).

Looseness is made monotone by a single scalar, so that "which of these two
is looser" has exactly one implementation in the file:

    rank(x <= v) =  v      bigger is looser
    rank(x >= v) = -v      smaller is looser
    rank(no bound) = +inf

Theorem 1 (existence)
---------------------
Let C != {} be a finite catalogue and K a finite system of inequality
constraints on numeric attributes. Let

    R = { r in L(K) : F(r) = { x in C : K_r(x) } != {} }

Then

  (a) R != {}.                                    -- because TOP in R
  (b) R has a greatest element, and it is TOP.
  (c) R is closed under join.
  (d) min { Regret(r) : r in R } is attained.

Consequently: **if the category is non-empty, a grounded, non-vacuous
answer exists.** That is the existence half of the Non-Vacuous Factual
Guarantee, and (b) says exactly what it is: drop every bound, and the
answer is the whole catalogue. The interesting question is not whether one
exists but how much it costs -- which is Theorem 2.

When the caller also carries KEPT constraints (equalities and other
non-relaxable requirements the lattice must not move), the theorem's
premise tightens honestly: an admissible relaxation exists iff the kept
constraints alone are jointly satisfiable over C, since TOP reduces the
joint system to exactly them. A query whose kept constraints already
exclude every item is therefore UNSATISFIABLE, not relaxable -- hard
requirements are not traded away to manufacture usefulness (F21).

Proof. (a)-(b) Each coordinate's domain is {unchanged} u {catalogue
values strictly looser than the bound} u {no bound}, a finite chain with a
top; a product of finite chains is a finite lattice with bottom BOTTOM and
top TOP, where TOP drops every bound. If C != {} then F(TOP) = C != {}, so
TOP is in R and, being the lattice's greatest element, is R's greatest
element too. (c) F(r join r') contains F(r), because a point satisfying r
satisfies every bound that r' relaxes no further; hence r in R implies
r join r' is in R. (d) R is a finite non-empty set and Regret is real, so
the minimum is attained. QED

Theorem 2 (no free lunch)
-------------------------
Regret(BOTTOM) = 0, and Regret(r) = 0 iff r = BOTTOM. So a regret-minimal
relaxation performs no work at all, and every non-trivial answer costs
strictly something. The interesting part is WHICH something.

Note what Theorem 1 does not give: that TOP is a good answer. It is the
coarsest relaxation and its Regret is the sum of every bound given up. A
configurator that always falls back to TOP is correct and useless -- which
is the vacuous output `nvfg` exists to prevent, arriving by another road.

Theorem 3 (what the single-move search is missing)
--------------------------------------------------
Theorem 1(a) guarantees some admissible relaxation exists. If none of the
single-coordinate moves is admissible, then at least two bounds must move
together, and the cheapest joint move is a genuine optimisation over a
finite set. `minimal` performs it exhaustively and reports truncation
rather than presenting a truncated search as an answer -- a result that
might not be the cheapest, presented as if it were, is a lie told to a
customer, which is the one thing this project does not do.

Complexity
----------
With k constraints and n distinct catalogue values per attribute, the
candidate set has at most (n + 3)^k points and evaluating one costs
O(|C| * k), so exhaustive minimisation is O(|C| * k * (n + 3)^k). Fine for
the k <= 4 a dialogue ever produces, hopeless in general -- which is why
`minimal` takes a `budget` and says when it ran out.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from nvfg import Constraint, NVFGResult, NVFGStatus, feasible

#: A relaxation assigns a new bound to each constraint, or None to keep the
#: stated one. A float infinity means the bound is dropped. The ordering of a
#: relaxation is part of its value, so these are hashable and can live in
#: sets.
Relaxation = Tuple[Optional[float], ...]

_MISSING = -math.inf - 1.0     # sort key for None, below every real value


def _rank(c: Constraint, value: float) -> float:
    """
    How loose a bound is, as one monotone scalar.

    "Looser" is direction-dependent and getting it backwards would invert
    the entire lattice, so it is defined once, here, and nowhere else:

        x <= v   loosens by v INCREASING
        x >= v   loosens by v DECREASING

    +inf is loosest for <= and -inf is loosest for >=, which is exactly what
    `Constraint.satisfied_by` does with those values -- so "no bound" needs
    no special case anywhere downstream.
    """
    return value if c.op.symbol == "<=" else -value


def _unrank(c: Constraint, rank: float) -> float:
    return rank if c.op.symbol == "<=" else -rank


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a minimisation, including whether it was exhaustive."""

    relaxation: Optional[Relaxation]
    regret: float
    complete: bool
    considered: int = 0
    admissible: int = 0

    @property
    def truncated(self) -> bool:
        return not self.complete


@dataclass(frozen=True)
class RelaxationLattice:
    """
    The relaxations of one constraint system, with their order.

    Immutable: every operation returns a new value. A structure that mutates
    in place is not a lattice, it is a state machine with a nice name.
    """

    constraints: Tuple[Constraint, ...]

    def __post_init__(self) -> None:
        for c in self.constraints:
            if not c.relaxable:
                raise ValueError(
                    f"constraint {c!r} is not relaxable; the lattice is defined "
                    "over inequality constraints only"
                )

    @property
    def size(self) -> int:
        return len(self.constraints)

    # -- construction ----------------------------------------------------

    def bottom(self) -> Relaxation:
        """The least element: change nothing."""
        return tuple([None] * self.size)

    def top(self) -> Relaxation:
        """The greatest element: drop every bound.

        Admissible whenever the catalogue is non-empty -- that is
        Theorem 1(b), and the reason a grounded answer always exists.
        """
        return tuple([_rank(c, math.inf) for c in self.constraints])

    def _check(self, r: Relaxation) -> Relaxation:
        """Validate a relaxation and return its canonical encoding.

        Two things used to slip through here:

        * an infinity in the TIGHTENING direction (``x <= -inf``,
          ``x >= +inf``) was accepted as if it dropped the bound -- it
          actually makes the constraint unsatisfiable, which is the exact
          opposite of a relaxation; and
        * a bound equal to the stated one produced a second encoding of
          the same lattice point, so ``moved``/``regret``/``join`` had to
          special-case it everywhere. The canonical form is `None`, and
          `_check` now returns it.
        """
        if len(r) != self.size:
            raise ValueError(f"relaxation of arity {len(r)}, expected {self.size}")
        out: List[Optional[float]] = []
        for c, v in zip(self.constraints, r):
            if v is None or v == c.value:
                out.append(None)
                continue
            if not math.isinf(v) and _rank(c, v) <= _rank(c, c.value):
                raise ValueError(
                    f"{c!r} -> {v} tightens the constraint; that is not a relaxation"
                )
            if math.isinf(v) and _rank(c, v) != math.inf:
                raise ValueError(
                    f"{c!r} -> {v} makes the constraint unsatisfiable; "
                    "only the infinity in the loosening direction is a relaxation"
                )
            out.append(v)
        return tuple(out)

    def canonicalize(self, r: Relaxation) -> Relaxation:
        """The unique encoding of a relaxation: `None` for an unmoved bound."""
        return self._check(r)

    # -- order and structure --------------------------------------------

    def _r(self, c: Constraint, v: Optional[float]) -> float:
        """
        Looseness rank of one coordinate.

        `None` -- "keep the bound as stated" -- is a concrete position on the
        chain, namely the rank of the stated bound, NOT the top. Treating it
        as the top is the same confusion as before in a new place: the
        bottom of the lattice would become its own top.
        """
        return _rank(c, c.value) if v is None else _rank(c, v)

    def leq(self, a: Relaxation, b: Relaxation) -> bool:
        """
        a <= b: a is NO LOOSER than b.

        Reading <= in the usual direction -- the bottom of the lattice is
        below everything -- rather than in the "at least as loose" direction,
        which would make bottom its own top and quietly invert every
        subsequent use of the word "minimal".
        """
        a = self._check(a)
        b = self._check(b)
        return all(self._r(c, x) <= self._r(c, y)
                   for c, x, y in zip(self.constraints, a, b))

    def join(self, a: Relaxation, b: Relaxation) -> Relaxation:
        """
        The looser of the two, pointwise.

        A coordinate whose result coincides with the stated bound comes back
        as `None` rather than as that number. Otherwise `join(r, r)` would
        not equal `r` -- the same constraint system in a different encoding --
        and idempotence, which the algebra tests, would be a coincidence of
        numbers rather than a law of the structure.
        """
        a = self._check(a)
        b = self._check(b)
        out = []
        for c, x, y in zip(self.constraints, a, b):
            hi = max(self._r(c, x), self._r(c, y))
            out.append(None if hi == self._r(c, None) else _unrank(c, hi))
        return tuple(out)

    def meet(self, a: Relaxation, b: Relaxation) -> Relaxation:
        """The tighter of the two, pointwise; `None` when it is the stated bound."""
        a = self._check(a)
        b = self._check(b)
        out = []
        for c, x, y in zip(self.constraints, a, b):
            lo = min(self._r(c, x), self._r(c, y))
            out.append(None if lo == self._r(c, None) else _unrank(c, lo))
        return tuple(out)

    def is_relaxation(self, r: Relaxation) -> bool:
        """Would the lattice accept this as one of its points?"""
        try:
            self._check(r)
        except ValueError:
            return False
        return True

    # -- semantics -------------------------------------------------------

    def applied(self, r: Relaxation) -> List[Constraint]:
        """The constraint system as relaxed by r. `None` keeps a bound."""
        out: List[Constraint] = []
        for c, v in zip(self.constraints, self._check(r)):
            out.append(c if v is None
                       else Constraint(c.name, c.attribute, c.op, v, c.unit,
                                       c.protected))
        return out

    def feasible(self, items: Sequence[Dict[str, object]],
                 r: Relaxation) -> List[Dict[str, object]]:
        return feasible(items, self.applied(r))

    def moved(self, r: Relaxation) -> List[Tuple[Constraint, float]]:
        """Which constraints this relaxation actually moves, and to what."""
        return [(c, v) for c, v in zip(self.constraints, self._check(r))
                if v is not None]

    def describe(self, r: Relaxation) -> str:
        parts = []
        for c, v in self.moved(r):
            unit = f" {c.unit}" if c.unit else ""
            if math.isinf(v):
                parts.append(f"{c.name}: {c.value:.8g}{unit} → без обмеження")
            else:
                parts.append(f"{c.name}: {c.value:.8g}{unit} → {v:.8g}{unit}")
        return "; ".join(parts) or "жодну вимогу не змінено"

    # -- cost ------------------------------------------------------------

    def regret(
        self,
        r: Relaxation,
        weights: Optional[Sequence[float]] = None,
    ) -> float:
        """
        Weighted sum of the SHARE of each requirement given up.

        The share, not the raw difference: 170 W and 170 UAH are not the
        same sacrifice, and only a ratio is comparable across units. Weights
        let a caller say which requirements matter more; they default to
        equal, and equal is the only default that is not an opinion.

        F21: weights must be finite and strictly positive. A negative
        weight turns "cost" into "reward" and the optimiser then prefers
        giving more away; a zero weight declares a sacrifice free, which
        is a lie by omission -- and Theorem 2's `Regret(r) = 0 iff
        r = BOTTOM` stops being true. Both are rejected rather than
        silently reinterpreted.
        """
        w = self._weights(weights)
        total = 0.0
        for weight, c, v in zip(w, self.constraints, self._check(r)):
            if v is None or v == c.value:
                continue
            if math.isinf(v):
                share = math.inf           # an unbounded requirement is gone
            else:
                share = abs(v - c.value) / abs(c.value) if c.value else math.inf
            total += weight * share
        return total

    def _weights(self, weights: Optional[Sequence[float]]) -> List[float]:
        """Validated regret weights: one finite, positive number per
        constraint; equal by default. See `regret` for why zeros and
        negatives are refused rather than clipped."""
        w = list(weights) if weights is not None else [1.0] * self.size
        if len(w) != self.size:
            raise ValueError("one weight per constraint")
        for x in w:
            if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0.0):
                raise ValueError(
                    f"regret weight {x!r} is not a positive finite number; "
                    "a zero or negative weight would hide a real sacrifice")
        return w

    def _order_key(self, r: Relaxation) -> Tuple:
        """A total, None-free ordering -- so ties are broken deterministically."""
        return tuple(
            (_MISSING, 0.0) if v is None else (_rank(c, v), v)
            for c, v in zip(self.constraints, r)
        )

    # -- the optimisation ------------------------------------------------

    def candidates(
        self, items: Sequence[Dict[str, object]]
    ) -> List[List[Optional[float]]]:
        """
        The finite set of bound values worth considering, per constraint.

        {keep} u {catalogue values that are genuine relaxations} u {no
        bound}. A value the catalogue never takes cannot make an empty
        feasible set non-empty, so including it would only inflate the
        search -- and an inflated search is a slower one that finds the same
        answer, which is the worst of both.
        """
        per: List[List[Optional[float]]] = []
        for c in self.constraints:
            # `None` already means "keep the stated bound", so listing the
            # bound as its own option would enumerate the same constraint
            # system twice, in two encodings, for no gain.
            options: List[Optional[float]] = [None, _rank(c, math.inf)]
            for item in items:
                raw = item.get(c.attribute)
                if raw is None:
                    continue
                try:
                    v = float(raw)
                except (TypeError, ValueError):
                    continue
                if v in options or _rank(c, v) <= _rank(c, c.value):
                    continue
                options.append(v)
            per.append(sorted(
                options,
                key=lambda v: (v is not None,
                               self._r(c, v) if v is not None else 0.0),
            ))
        return per

    def admissible(
        self,
        items: Sequence[Dict[str, object]],
        budget: Optional[int] = None,
        kept: Sequence[Constraint] = (),
    ) -> Tuple[List[Relaxation], bool]:
        """
        Every relaxation that leaves the feasible set non-empty.

        F21/P18: `kept` constraints -- the ones the lattice cannot move --
        are enforced DURING the search, not re-checked against its winner.
        Optimising without them picks the cheapest relaxation that serves
        only items the kept constraints reject, after which the bridge
        could do nothing but lie: a pricier feasible relaxation existed.
        """
        found: List[Relaxation] = []
        complete = True
        for seen, combo in enumerate(product(*self.candidates(items)), start=1):
            if budget is not None and seen > budget:
                complete = False
                break
            r = tuple(combo)
            if feasible(items, list(kept) + self.applied(r)):
                found.append(r)
        return found, complete

    def minimal(
        self,
        items: Sequence[Dict[str, object]],
        *,
        weights: Optional[Sequence[float]] = None,
        budget: Optional[int] = 200_000,
        kept: Sequence[Constraint] = (),
    ) -> SearchResult:
        """
        The cheapest relaxation that leaves a usable answer.

        Exhaustive over the candidate set, and it says so: `complete` is
        False when `budget` cut the enumeration short. `weights` are
        validated before the search begins, so a bad weight fails fast
        rather than mid-enumeration.
        """
        self._weights(weights)
        found, complete = self.admissible(items, budget, kept)
        if not found:
            return SearchResult(None, 0.0, complete)
        best = min(found, key=lambda r: (self.regret(r, weights), self._order_key(r)))
        return SearchResult(best, self.regret(best, weights), complete,
                            admissible=len(found))

    def pareto(
        self,
        items: Sequence[Dict[str, object]],
        kept: Sequence[Constraint] = (),
    ) -> List[Relaxation]:
        """
        Regret-minimal relaxations: those no other admissible relaxation
        beats on every moved constraint at once.

        Returned instead of a single winner when the costs are genuinely
        incomparable -- "raise the budget by 40% or drop the power floor by
        60%" is not a question with an answer, and pretending otherwise is
        how a configurator talks a customer into a worse machine.

        The frontier is the full pairwise non-dominated set: a streaming
        filter that only checks a newcomer against what is already kept
        misses that the newcomer can itself evict an earlier entry.
        """
        found, _ = self.admissible(items, kept=kept)
        costs = {r: cv for r in found
                 if (cv := self._cost_vector(r)) is not None}
        frontier = [r for r, cv in costs.items()
                    if not any(o != r and self._dominates(costs[o], cv)
                               for o in costs)]
        return sorted(frontier, key=lambda r: (self.regret(r), self._order_key(r)))

    def _cost_vector(self, r: Relaxation) -> Optional[Dict[int, float]]:
        """Per-constraint absolute movement, or None if the bound is gone."""
        out: Dict[int, float] = {}
        for i, (c, v) in enumerate(zip(self.constraints, self._check(r))):
            if v is None:
                continue
            if math.isinf(v):
                return None
            out[i] = abs(v - c.value)
        return out

    @staticmethod
    def _dominates(a: Dict[int, float], b: Dict[int, float]) -> bool:
        """
        a Pareto-dominates b: costs are minimised, so a must be no more
        expensive on EVERY axis and strictly cheaper on at least one.

        A missing axis means the constraint was never moved -- cost zero,
        not infinity. The previous version compared in the wrong
        direction and treated an absent axis as infinitely expensive, so
        a strictly dominated bound stayed on the frontier and was offered
        to the customer as a choice (P20).
        """
        keys = set(a) | set(b)
        if not keys:
            return False
        no_worse = all(a.get(i, 0.0) <= b.get(i, 0.0) for i in keys)
        strictly_better = any(a.get(i, 0.0) < b.get(i, 0.0) for i in keys)
        return no_worse and strictly_better


# --------------------------------------------------------------------------
# Bridge to the Non-Vacuous Factual Guarantee
# --------------------------------------------------------------------------


def solve(
    items: Sequence[Dict[str, object]],
    constraints: Sequence[Constraint],
    *,
    weights: Optional[Sequence[float]] = None,
    budget: Optional[int] = 200_000,
    max_items: int = 3,
    item_label: Callable[[Dict[str, object]], str] = lambda i: str(i.get("name", "")),
) -> NVFGResult:
    """
    `nvfg.solve` with the lattice in place of the single-move heuristic.

    The only observable difference is the corner `nvfg.solve` gives up on:
    queries where no ONE bound can be moved far enough, but two together
    can. Those now get a grounded answer naming every requirement that
    moved, instead of an UNSATISFIABLE that was a fact about a one-move
    search and not about the catalogue.
    """
    if not items:
        return NVFGResult(NVFGStatus.EMPTY_QUERY, reason="Каталог порожній.")

    strict = feasible(items, constraints)
    relaxable = tuple(c for c in constraints if c.relaxable)
    if not relaxable:
        if strict:
            return NVFGResult(NVFGStatus.FEASIBLE,
                              items=[item_label(i) for i in strict[:max_items]],
                              solved_items=list(strict))
        return NVFGResult(
            NVFGStatus.UNSATISFIABLE,
            reason="Жодну з вимог не можна послабити, а запит нерозв'язний.",
        )

    lattice = RelaxationLattice(relaxable)
    # F21/P18: the kept constraints stay inside the optimisation. Taking
    # them out and re-checking the winner afterwards is how a solver
    # reports UNSATISFIABLE while a perfectly feasible -- just slightly
    # more expensive -- relaxation is sitting in the search space.
    kept = [c for c in constraints if not c.relaxable]
    found = lattice.minimal(items, weights=weights, budget=budget, kept=kept)
    if found.relaxation is None:
        if not found.complete:
            # F21/P19: the budget cut the enumeration short. Nothing was
            # proven about the unexamined space, so neither infeasibility
            # nor satisfiability may be claimed.
            return NVFGResult(
                NVFGStatus.UNKNOWN,
                reason=("Пошук можливих послаблень обрізано за бюджетом "
                        "перебору — недосяжність не доведена й відповідь "
                        "не знайдена."),
            )
        if not feasible(items, kept):
            return NVFGResult(
                NVFGStatus.UNSATISFIABLE,
                reason=("Обмеження, яке не можна послабити, не виконується "
                        "жодним товаром: "
                        + "; ".join(str(c) for c in kept) + "."),
            )
        return NVFGResult(
            NVFGStatus.UNSATISFIABLE,
            reason="Жоден товар не відповідає цим вимогам навіть без верхніх меж.",
        )

    r = found.relaxation
    solved = feasible(items, kept + lattice.applied(r))

    if not lattice.moved(r):
        return NVFGResult(NVFGStatus.FEASIBLE,
                          items=[item_label(i) for i in solved[:max_items]],
                          solved_items=list(solved))

    regret_out = round(found.regret, 6)
    if regret_out == 0.0 and found.regret > 0.0:
        regret_out = found.regret
    caveat = "" if found.complete else (
        " (пошук обмежено за бюджетом, відповідь може бути не найдешевшою)")
    return NVFGResult(
        NVFGStatus.RELAXED,
        items=[item_label(i) for i in solved[:max_items]],
        solved_items=list(solved),
        # `Relaxation` holds a single constraint; a multi-constraint
        # description travels as text, which is what the customer reads.
        relaxation=None,
        regret=regret_out,
        relative_regret=regret_out,
        relaxation_text=("Жоден товар не задовольняє всі вимоги одночасно. "
                         "Найменша зміна, яка робить запит розв'язним: "
                         + lattice.describe(r) + caveat
                         + ". За точними вимогами відповідників немає; "
                         "найближча альтернатива при послабленні: "
                         + ", ".join(item_label(i) for i in solved[:max_items]) + "."),
    )
