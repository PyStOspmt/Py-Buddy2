"""
Py-Buddy: Non-Vacuous Factual Guarantee (NVFG).

The open problem
----------------
arXiv:2603.16817 ("Is Conformal Factuality for RAG-based LLMs Robust?")
established a trade-off and did not resolve it:

    "conformal filtering suffers from low usefulness at high factuality
     levels due to vacuous outputs"

Every way of buying perfect factuality by filtering is a way of buying
nothing. A shop assistant that answers "I am unable to provide that
information" has hallucinated nothing and helped no one.

The claim
---------
Given a catalogue C and hard constraints K, a response exists that is

    grounded   every claim is a projection of a row we queried, and
    informative  it either surfaces a catalogue item, or names the
                 specific constraint that is infeasible together with
               its numeric margin

              Grounded(r) AND Informative(r)

Define:

    NVFG = |{ r : Grounded(r) AND Informative(r) }| / |K|

The algorithm
-------------
The dead end this removes is the *empty-set deadlock*: a customer asks
for a configuration that does not exist, and the only two available
behaviours are "0 results" or a relaxation nobody asked for.

    1. Compute the feasible set F = { x in C : K(x) }.
    2. F non-empty -> answer from F. Nothing to relax.
    3. F empty -> for each relaxable constraint c_i, search for the least
       amount by which c_i must be relaxed so that F becomes NON-EMPTY
       under the remaining constraints taken together.
    4. Drop the cheapest of those, and answer from the relaxed set F',
       stating which constraint moved, in which direction, and by how much.

Two rules in step 3 matter, and both were wrong in the first draft.

Relaxation is validated JOINTLY, never in isolation
    "Lower the power bound to the quietest cooler's 30 W" looks minimal if
    you look at the power requirement alone -- and then yields nothing,
    because that cooler costs 2600 and the budget says 1000. A candidate
    bound counts only if `feasible(items, others + [relaxed])` is
    non-empty. Hence the honest bound is 200 -> 45 W, the quietest cooler
    the budget can still afford. The catalogue is finite, so this is a
    search over distinct item values, not an approximation.

"Cheapest" is measured as the SHARE of the requirement given up
    170 W and 170 UAH are not the same sacrifice; comparing raw magnitudes
    across UAH, watts and millimetres is meaningless, and so is any
    abstract score that hides the unit. The order is therefore

        (|new - old| / |old|)  first,  absolute amount  second,
        constraint name  third   (for a deterministic tie-break)

    while the number reported to the customer stays in their own unit.

Every number in step 4 comes from the Value Algebra, so it is grounded by
construction rather than by checking afterwards.

Existence
---------
If at least one constraint can be relaxed while keeping the feasible set
non-empty, a non-vacuous grounded answer exists, and the algorithm above
finds one -- that is the guarantee.
If NO constraint can be relaxed that far -- the category is genuinely
empty, or the constraints are mutually binding (see
`relaxation_lattice` for the multi-constraint case) -- then no such answer
exists, and the correct output is an explicit refusal that names what was
searched. That case is reported as UNSATISFIABLE, never dressed up as a
recommendation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------


class Op(Enum):
    """Comparison operators. `AT_MOST`/`AT_LEAST` are the relaxing ones."""

    EQ = "eq"
    AT_MOST = "<="      # relaxing upward
    AT_LEAST = ">="      # relaxing downward

    @property
    def relaxable(self) -> bool:
        # Defined on the member rather than as a class attribute: a set of
        # members inside an Enum body is evaluated before those members
        # exist, and raises "argument of type 'Op' is not iterable".
        return self in (Op.AT_MOST, Op.AT_LEAST)

    @property
    def symbol(self) -> str:
        return {Op.EQ: "=", Op.AT_MOST: "<=", Op.AT_LEAST: ">="}[self]


@dataclass(frozen=True)
class Constraint:
    """
    One hard requirement on one canonical attribute.

    `unit` is a display unit only; comparison happens on `value`, already
    canonical (see attribute_algebra). Keeping the unit lets the relaxation
    be reported in the customer's own terms instead of an abstract score.

    `protected` marks a requirement the caller forbids the solver to
    trade away -- safety, identity, dosage, legal limits. An inequality
    that would normally be a relaxation candidate stays fixed when
    protected, which is exactly the link the existence theorem states:
    an admissible relaxation exists iff the protected constraints alone
    are jointly satisfiable.
    """

    name: str                     # "budget", "socket", "diameter"
    attribute: str                # canonical attribute key
    op: Op
    value: float
    unit: str = ""
    protected: bool = False

    @property
    def relaxable(self) -> bool:
        return self.op.relaxable and not self.protected

    def satisfied_by(self, item_value: Optional[float]) -> bool:
        # F21/P21: unknown data does not satisfy a hard requirement. The old
        # `None -> True` let an item that omits the attribute walk through
        # every hard filter, including safety and identity constraints.
        if item_value is None:
            return False
        if self.op is Op.EQ:
            return item_value == self.value
        if self.op is Op.AT_MOST:
            return item_value <= self.value
        return item_value >= self.value

    def violated_by(self, item_value: Optional[float]) -> bool:
        # Unknown is not a violation either: it is unknown. Relaxing a bound
        # cannot rescue an item that states no value at all, so an unknown
        # must be excluded from the feasible set without being counted as a
        # violation that relaxation could repair.
        if item_value is None:
            return False
        return not self.satisfied_by(item_value)

    def __str__(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        tag = " [захищене]" if self.protected else ""
        return f"{self.name} {self.op.symbol} {self.value}{unit}{tag}"


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


class NVFGStatus:
    """A non-vacuous answer was produced."""
    FEASIBLE = "FEASIBLE"                    # no relaxation needed
    RELAXED = "RELAXED"                      # minimal relaxation applied
    UNSATISFIABLE = "UNSATISFIABLE"          # no non-vacuous answer exists
    EMPTY_QUERY = "EMPTY_QUERY"              # nothing to do at all
    #: F21/P19: the admissible-space search was cut short by its budget, so
    #: neither feasibility nor infeasibility is proven. Reporting
    #: UNSATISFIABLE here would be a claim the evidence does not support.
    UNKNOWN = "UNKNOWN"


@dataclass
class Relaxation:
    """The single constraint that moved, and by how much."""

    constraint: str
    from_value: float
    to_value: float
    unit: str
    direction: str                # "up" or "down"

    def describe(self) -> str:
        return (f"якщо послабити {self.constraint} з {self.from_value:g} "
                f"до {self.to_value:g} {self.unit} ({self.direction})")

    def to_dict(self) -> Dict[str, object]:
        return {
            "constraint": self.constraint,
            "from": self.from_value,
            "to": self.to_value,
            "unit": self.unit,
            "direction": self.direction,
        }


@dataclass
class NVFGResult:
    status: str
    items: List[str] = field(default_factory=list)
    relaxation: Optional[Relaxation] = None
    #: Regret of the relaxation, in the relaxed constraint's own unit.
    #: 0.0 when nothing was relaxed.
    regret: float = 0.0
    #: The same regret as a share of the requirement. Dimensionless, and
    #: therefore the only regret that may be compared across constraints
    #: measured in different units.
    relative_regret: float = 0.0
    #: Why no non-vacuous answer exists, when one does not.
    reason: str = ""
    #: Explanation of a relaxation that moves SEVERAL constraints at once,
    #: which a single `Relaxation` cannot express. Set by the lattice
    #: solver; `explain()` prefers it over the single-constraint wording.
    relaxation_text: str = ""
    #: The raw item dicts of the COMPLETE (possibly relaxed) feasible set.
    #: `items` holds display labels capped at `max_items`; downstream
    #: callers that must rank, ground, or re-check the surviving rows need
    #: the rows themselves, untruncated.
    solved_items: List[Dict[str, object]] = field(default_factory=list)

    @property
    def informative(self) -> bool:
        """Did we say something that helps?

        Both a recommendation and a specific, grounded reason count. A bare
        refusal does not — that is the vacuous case this module exists to
        remove.
        """
        if self.status in (NVFGStatus.UNSATISFIABLE, NVFGStatus.UNKNOWN):
            return bool(self.reason)
        if self.status == NVFGStatus.EMPTY_QUERY:
            return False
        return bool(self.items)

    @property
    def grounded(self) -> bool:
        """Every claim in `items` came from a query result by construction."""
        return self.status in (NVFGStatus.FEASIBLE, NVFGStatus.RELAXED)

    @property
    def non_vacuous(self) -> bool:
        return self.grounded and self.informative

    def explain(self) -> str:
        if self.status == NVFGStatus.FEASIBLE:
            return ("Знайдено " + str(len(self.items)) +
                    " товар(ів), що задовольняють усі вимоги.")
        if self.status == NVFGStatus.RELAXED:
            if self.relaxation_text:
                return self.relaxation_text
            return ("Жоден товар не задовольняє всі вимоги одночасно. "
                    "Найменша зміна, яка робить запит розв'язним: "
                    + (self.relaxation.describe() if self.relaxation else "?") +
                    ". Ось що доступно: " + ", ".join(self.items) + ".")
        if self.status == NVFGStatus.UNSATISFIABLE:
            return ("Не знайдено жодного товару за цими критеріями. " + self.reason)
        if self.status == NVFGStatus.UNKNOWN:
            return ("Пошук не завершено, тому достеменно відповісти неможливо. "
                    + self.reason)
        return "Немає чого шукати: запит не містить ані вимог, ані кандидатів."

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "items": list(self.items),
            "relaxation": self.relaxation.to_dict() if self.relaxation else None,
            "regret": self.regret,
            "relative_regret": self.relative_regret,
            "relaxation_text": self.relaxation_text,
            "solved_items": list(self.solved_items),
            "informative": self.informative,
            "grounded": self.grounded,
            "non_vacuous": self.non_vacuous,
            "explanation": self.explain(),
        }


# --------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------


def feasible(
    items: Sequence[Dict[str, object]],
    constraints: Sequence[Constraint],
) -> List[Dict[str, object]]:
    """The feasible set F = { x in C : K(x) }. Pure predicate, no model."""
    out: List[Dict[str, object]] = []
    for item in items:
        ok = True
        for c in constraints:
            v = _attr(item, c.attribute)
            # F21/P21: an item that states no value for a hard requirement is
            # not provably feasible -- excluded, but not counted as a
            # violation (relaxation cannot manufacture the missing data).
            if v is None or c.violated_by(v):
                ok = False
                break
        if ok:
            out.append(item)
    return out


def _attr(item: Dict[str, object], attribute: str) -> Optional[float]:
    raw = item.get(attribute)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _candidate_bounds(
    items: Sequence[Dict[str, object]], c: Constraint
) -> List[float]:
    """
    Every bound worth considering when relaxing `c`.

    Only a value the catalogue actually exhibits can ever leave a non-empty
    feasible set, so the search space is exactly the distinct item values --
    finite, and no wider than the catalogue.
    """
    values = {_attr(i, c.attribute) for i in items}
    return sorted(v for v in values if v is not None)


def _minimal_relaxation(
    items: Sequence[Dict[str, object]],
    constraints: Sequence[Constraint],
    c: Constraint,
) -> Optional[Tuple[float, float, Constraint, Constraint, List[Dict[str, object]]]]:
    """
    The smallest change to `c` that makes the JOINT feasible set non-empty.

    Returns (relative_regret, absolute_regret, original, relaxed, feasible_set),
    or None when relaxing `c` alone cannot rescue the query.

    A candidate bound is accepted only if `feasible` returns something, which
    is the whole point: an isolated minimum can be worthless. Relaxing
    {power >= 200} down to 30 W -- the quietest cooler in the catalogue --
    is the cheapest move in isolation and yields an empty set, because that
    cooler costs 2600 against a budget of 1000. The honest bound is 45 W.
    """
    others = [x for x in constraints if x is not c]
    best: Optional[Tuple[float, float, str, Constraint, Constraint, List[Dict[str, object]]]] = None
    for v in _candidate_bounds(items, c):
        if c.op is Op.AT_MOST:
            if v <= c.value:
                continue                        # tighter, not a relaxation
            relaxed = Constraint(c.name, c.attribute, Op.AT_MOST, v, c.unit,
                                 c.protected)
        else:
            if v >= c.value:
                continue
            relaxed = Constraint(c.name, c.attribute, Op.AT_LEAST, v, c.unit,
                                 c.protected)
        got = feasible(items, others + [relaxed])
        if not got:
            continue
        amount = abs(relaxed.value - c.value)
        rel = amount / abs(c.value) if c.value else float("inf")
        key = (rel, amount, c.name)
        if best is None or key < best[:3]:
            best = (rel, amount, c.name, c, relaxed, got)
    if best is None:
        return None
    return best[0], best[1], best[3], best[4], best[5]


def _deadlock_reason(
    items: Sequence[Dict[str, object]], constraints: Sequence[Constraint]
) -> str:
    """
    An honest, specific refusal: which requirements could not be met, and
    that no single one of them could be relaxed far enough to help.

    Vague refusals are the vacuous output this module exists to remove, so
    the reason always enumerates the constraints rather than shrugging.
    """
    listed = "; ".join(str(c) for c in constraints) or "(без вимог)"
    return ("Жоден товар не відповідає цим вимогам, і жодну з них не можна "
            "послабити достатньо, щоб зʼявився хоч один варіант: "
            + listed + ".")


def solve(
    items: Sequence[Dict[str, object]],
    constraints: Sequence[Constraint],
    *,
    max_items: int = 3,
    item_label: Callable[[Dict[str, object]], str] = lambda i: str(i.get("name", "")),
) -> NVFGResult:
    """
    Produce the best possible non-vacuous, grounded answer.

    Never blocks, never invents, and when it cannot satisfy the request it
    says exactly which constraint stands in the way and by how much.

    Only *single*-constraint relaxations are considered here. The
    multi-constraint case -- two mutually binding requirements that no one
    of them can rescue -- is handled by `relaxation_lattice`.
    """
    if not items:
        return NVFGResult(NVFGStatus.EMPTY_QUERY, reason="Каталог порожній.")
    if not constraints:
        return NVFGResult(NVFGStatus.FEASIBLE,
                          items=[item_label(i) for i in items[:max_items]],
                          solved_items=list(items), regret=0.0)

    strict = feasible(items, constraints)
    if strict:
        return NVFGResult(
            NVFGStatus.FEASIBLE,
            items=[item_label(i) for i in strict[:max_items]],
            solved_items=list(strict),
            regret=0.0,
        )

    # Empty-set deadlock. Rank every single-constraint rescue by the share of
    # the requirement it costs, so the order of `constraints` in the query
    # cannot change the answer.
    ranked: List[Tuple[float, float, str, Constraint, Constraint, List[Dict[str, object]]]] = []
    for c in constraints:
        if not c.relaxable:
            continue
        found = _minimal_relaxation(items, constraints, c)
        if found is not None:
            rel, amount, original, relaxed, got = found
            ranked.append((rel, amount, c.name, original, relaxed, got))
    ranked.sort(key=lambda t: (t[0], t[1], t[2]))

    if not ranked:
        return NVFGResult(
            NVFGStatus.UNSATISFIABLE, reason=_deadlock_reason(items, constraints),
        )

    rel, amount, _name, original, relaxed, relaxed_feasible = ranked[0]
    return NVFGResult(
        NVFGStatus.RELAXED,
        items=[item_label(i) for i in relaxed_feasible[:max_items]],
        solved_items=list(relaxed_feasible),
        relaxation=Relaxation(
            constraint=original.name,
            from_value=original.value,
            to_value=relaxed.value,
            unit=original.unit,
            direction="вгору" if relaxed.op is Op.AT_MOST else "вниз",
        ),
        regret=round(amount, 4),
        relative_regret=round(rel, 6),
    )


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class NVFGReport:
    """Aggregate over a query set."""

    n: int
    feasible: int
    relaxed: int
    unsatisfiable: int
    empty: int
    non_vacuous: int
    mean_regret: float
    #: Share of *infeasible* queries that still got a useful answer.
    #: This is the number the paper leaves open.
    recovery_rate: float
    #: Share of all queries that produced a vacuous answer -- one that
    #: neither recommends a catalogue item nor names a specific, grounded
    #: reason. An honest refusal with a reason is informative, not vacuous.
    vacuity_rate: float
    #: Mean share of the requirement given up, over relaxed queries.
    mean_relative_regret: float = 0.0
    #: F21/P19: queries whose relaxation search was truncated by budget --
    #: neither satisfiable nor unsatisfiable is proven for these.
    unknown: int = 0

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


def report(results: Sequence[NVFGResult]) -> NVFGReport:
    n = len(results)
    if n == 0:
        return NVFGReport(0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    counts = {s: sum(1 for r in results if r.status == s) for s in
              (NVFGStatus.FEASIBLE, NVFGStatus.RELAXED,
               NVFGStatus.UNSATISFIABLE, NVFGStatus.EMPTY_QUERY,
               NVFGStatus.UNKNOWN)}
    infeasible = counts[NVFGStatus.RELAXED] + counts[NVFGStatus.UNSATISFIABLE]
    relaxed = [r for r in results if r.status == NVFGStatus.RELAXED]
    return NVFGReport(
        n=n,
        feasible=counts[NVFGStatus.FEASIBLE],
        relaxed=counts[NVFGStatus.RELAXED],
        unsatisfiable=counts[NVFGStatus.UNSATISFIABLE],
        empty=counts[NVFGStatus.EMPTY_QUERY],
        non_vacuous=sum(1 for r in results if r.non_vacuous),
        mean_regret=round(
            sum(r.regret for r in relaxed) / max(1, counts[NVFGStatus.RELAXED]), 4),
        recovery_rate=round(
            100.0 * counts[NVFGStatus.RELAXED] / infeasible, 2) if infeasible else 0.0,
        # Vacuity is measured against INFORMATIVENESS, not against
        # `non_vacuous`. An unsatisfiable query answered with a specific,
        # grounded reason has said something true and useful; counting it as
        # vacuous would make the metric reward silence.
        vacuity_rate=round(100.0 * (n - sum(1 for r in results if r.informative)) / n, 2),
        mean_relative_regret=round(
            sum(r.relative_regret for r in relaxed) / max(1, counts[NVFGStatus.RELAXED]), 6),
        unknown=counts[NVFGStatus.UNKNOWN],
    )
