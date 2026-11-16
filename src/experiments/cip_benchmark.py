"""
Py-Buddy: Constraint Induction Precision & Recall (CIP / CIR).

Benchmarks structural domain-agnostic constraint induction against naive
attribute-name substring baselines (English hints only):

    ["tdp", "watt", "draw", "weight", ...]      # demand indicators
    ["psu_wattage", "wattage", "capacity", ...]  # capacity providers
    ["length", "width", "height", "diameter"]    # dimensions
    ["max_length", "clearance", "arch_clearance"]# their limits

so it is blind to every Ukrainian key, every synonym, and every merchant that
spells a key differently -- and simultaneously confident about keys that
merely *contain* the substring. Both failures are invisible until you measure
them, and "we are domain-agnostic" is exactly the sort of claim a reviewer
checks first.

What this measures
------------------
Three domains, each with a hand-declared ground truth, each with the same
logical attributes spelled several different ways. Two induction systems are
scored with identical arithmetic:

  baseline    `ContractInductor._induce_deterministic` -- keyword matching
  proposed    `induce_structural` -- no domain words anywhere

The proposed system
-------------------
Two structural criteria, both derived from the data:

  1. EQUIVALENCE, from the Value Algebra (attribute_algebra). Keys that
     canonicalise onto overlapping vocabularies AND share product scope are
     the *same* logical attribute, so the constraint is emitted once over
     the whole group. `Сокет` and `socket` are one matching key, not two.

  2. CONTAINMENT, from rank-ladder dominance. Two attribute groups in the
     same physical dimension, living in different product categories, form
     a capacity constraint when the provider's value ladder covers the
     consumer's position by position. A 4080 at 320 W is covered by an 850 W
     PSU; it is not covered by a 150 W board allowance. No word list, no
     unit-by-hand.

Matching is exact on the *group*, not on the string. A system that reports
`eq(Сокет)` and `eq(socket)` separately has not found the matching key, it has
found half of it twice -- and the arithmetic below says so.

Modelling decisions that are assumptions, logged in docs/07_RESEARCH_LOG.md
  A7. Ground truth is declared per domain and is checkable: `verify()` proves
      every declared rule is entailed by the catalogue data through an
      independent path, so a dataset cannot be rigged by declaring a rule the
      data does not support.
  A8. Orientation of a containment rule is read off the ladders: the side
      with the higher ceiling is the provider. Counter-example: two
      attributes whose ranges merely interleave are not a constraint in
      either direction, and this criterion will not see them.
  A9. Only attributes whose values are *measured* (a recognised dimension)
      take part in containment. A bare number with no unit is skipped, which
      is why tyre width never becomes a constraint here.

Measured (`python src/cip_benchmark.py`)

    dataset      n  |truth|     baseline P/R/F1     proposed P/R/F1
    pc_build    15      6       25.0/ 16.7/ 20.0       60.0/100.0/ 75.0
    auto_parts   9      3       50.0/ 33.3/ 40.0       75.0/100.0/ 85.7
    pharma       5      2        0.0/  0.0/  0.0       66.7/100.0/ 80.0
    TOTAL                     25.0/ 18.2/ 21.1       64.7/100.0/ 78.6

Negative controls and anonymisation (F12)
------------------------------------------
Each dataset also declares *negative controls*: rules that carry the
structural signature of a constraint but are declared non-rules (shared
colour, a demand that numerically covers a sibling demand, a shared key
name with different semantics). The report names which system emits
which negative instead of letting them vanish inside the FP count.
Under `anonymized()` -- every key, category and product name replaced by
an opaque token -- the proposed method's tp/fp/fn are identical on all
three datasets (invariance tested, not asserted); the name-matching
baseline cannot improve and degrades where its vocabulary carried it.

Scope of the claim
------------------
Three hand-declared datasets and two evaluator classes (equivalence and
containment) demonstrate that the method reads structure rather than
names. They do not constitute universal rule discovery: the precision
ceiling documented below is real, and no measured claim beyond these
datasets is made.

The baseline finds not one containment constraint in any dataset: its
word lists are English, the catalogues are not, and it reports `eq(socket)`
while leaving `Сокет` (a zero-variance key in the pc_build sample) unmerged --
the two spellings of one matching key are never unified.

The proposed method is at 100% recall and 64.7% precision. The six false
positives are two mistakes, and both are declared negative controls rather
than tuned away:

    1. worst-case coverage cannot tell a LIMIT from another DEMAND.

A board's CPU ceiling (130..150 W) and a graphics card's draw (170..320 W)
have disjoint ranges, so the criterion concludes the card "covers" the
board. Both are consumers of the same supply; neither is a provider. A
numerical criterion has no way to know that, and the keyword lists this
method replaced were not solving the problem either -- they were guessing
it differently. Distinguishing sibling categories needs a signal about the
relation between categories, not between numbers; that is the open problem
this benchmark exists to make visible, and
`test_two_consumers_of_one_provider_are_mistaken_for_a_pair` keeps it
visible.

    2. a shared key name is not a shared attribute.

`Колір` spans three categories in pc_build and two in pharma and is
emitted as a matching key in both; colour agreement is not compatibility.
Conversely the baseline emits `eq(Ширина)` in auto_parts -- rim width in
J vs tyre width in mm -- which the proposed vocabulary-grouping correctly
rejects.

Equivalence is not subject to this: it is decided by vocabulary overlap and
scope, and is exact.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from attribute_algebra import canonical_form
from contract_inductor import (
    EQ,
    LE,
    ContractInductor,
    StructuralRule as InducedRule,
    _key_scopes,
    _key_values,
    _structural_covers as _covers,
    _structural_measure as _measure,
    induce_structural_rules as induce_structural,
)
from hypergraph import (
    CumulativeCapacityConstraint,
    EquivalenceConstraint,
    ProductNode,
    SpatialBoundingConstraint,
)

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Dataset:
    name: str
    products: Tuple[ProductNode, ...]
    truth: Tuple[InducedRule, ...]
    #: Rules that carry the same structural signature as real constraints
    #: (a symbolic key spanning categories, a numeric ladder that covers)
    #: but are declared NOT to be compatibility rules. A shared colour, a
    #: sibling demand that numerically covers another demand -- observed
    #: correlation must not be promoted to a physical rule (F12). They are
    #: reported separately instead of being silently absorbed into the FP
    #: count, so the failure mode stays visible.
    negatives: Tuple[InducedRule, ...] = ()


# --------------------------------------------------------------------------
# Structural induction (now unified in src/core/contract_inductor.py)
# --------------------------------------------------------------------------

def rules_from_baseline(products: Sequence[ProductNode]) -> List[InducedRule]:
    """The same rules, read off the keyword-matching inducer."""
    out: List[InducedRule] = []
    for c in ContractInductor()._induce_deterministic(list(products)):
        if isinstance(c, EquivalenceConstraint):
            key = (c.attr_name,)
            out.append(InducedRule(EQ, key, key))
        elif isinstance(c, CumulativeCapacityConstraint):
            out.append(InducedRule(LE, (c.demand_attr,), (c.capacity_attr,)))
        elif isinstance(c, SpatialBoundingConstraint):
            out.append(InducedRule(LE, (c.dimension_attr,), (c.max_dimension_attr,)))
    return sorted(set(out))


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return 100.0 * self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return 100.0 * self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> Dict[str, object]:
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn,
                "precision": round(self.precision, 2),
                "recall": round(self.recall, 2), "f1": round(self.f1, 2)}


def score(predicted: Iterable[InducedRule], truth: Iterable[InducedRule]) -> PRF:
    """
    Exact set matching on the identity triple (kind, subject group, object
    group).

    Strict on purpose. A rule that mentions two of the three keys in a
    matching group has not found the constraint -- it has found something
    that will mis-validate real baskets -- and a lenient matcher would hide
    exactly the defect this benchmark exists to expose.
    """
    p, t = set(predicted), set(truth)
    return PRF(tp=len(p & t), fp=len(p - t), fn=len(t - p))


# --------------------------------------------------------------------------
# Ground-truth verification
# --------------------------------------------------------------------------


def verify(dataset: Dataset) -> List[str]:
    """
    Problems with the declared ground truth. Empty list = the truth is sound.

    Deliberately independent of `induce_structural`: it re-derives every
    declared rule from the raw catalogue rather than from the inducer, so a
    dataset cannot flatter itself by declaring whatever the inducer happens
    to emit. A benchmark whose ground truth is generated by the system under
    test has a recall of 1.0 by construction and measures nothing.
    """
    problems: List[str] = []
    products = dataset.products
    scopes = _key_scopes(products)
    values = _key_values(products)
    known = set(values)

    def measure_of(key: str) -> List[float]:
        out = []
        for v in values.get(key, ()):
            m = _measure(v)
            if m is not None:
                out.append(m[0])
        return out

    for rule in dataset.truth:
        missing = (set(rule.subject) | set(rule.obj)) - known
        if missing:
            problems.append(f"{rule}: unknown key(s) {sorted(missing)}")
            continue
        if rule.kind == EQ:
            if set(rule.subject) != set(rule.obj):
                problems.append(f"{rule}: equivalence must constrain one group")
            scope: Set[str] = set()
            for k in rule.subject:
                scope |= scopes.get(k, set())
            if len(scope) < 2:
                problems.append(f"{rule}: group spans one category only")
            vocab = [
                canonical_form(v, key=" ".join(rule.subject))
                for k in rule.subject for v in values[k]
            ]
            vocab = {v for v in vocab if v}
            if len(vocab) < 2:
                problems.append(f"{rule}: group vocabulary is a single value")
        else:
            subject = [x for k in rule.subject for x in measure_of(k)]
            obj = [x for k in rule.obj for x in measure_of(k)]
            if not subject or not obj:
                problems.append(f"{rule}: side without comparable measurements")
            elif not _covers(obj, subject):
                problems.append(
                    f"{rule}: not entailed -- the weakest {rule.obj} does not "
                    f"cover the strongest {rule.subject}")
    return problems


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def _p(pid: str, name: str, category: str, price: float, **attrs: str) -> ProductNode:
    return ProductNode(id=pid, name=name, category=category, price=price,
                       attributes=dict(attrs))


def pc_build() -> Dataset:
    """
    Desktop build. Every logical attribute is spelled two or three ways
    across languages and cases, exactly as the live Rozetka catalogue does.

    Ground truth: sockets and memory type are matching keys; the board's CPU
    allowance covers TDP; both the CPU and the graphics card are covered by
    the power supply.
    """
    products = [
        # cpu1 and mb1 carry BOTH socket spellings, exactly as a scraped
        # listing does: the vendor column and the normalised column coexist.
        # That overlap is what lets the Value Algebra tell one attribute from
        # two -- see the limitation noted in the module docstring.
        _p("cpu1", "Ryzen 5 5600X", "cpu", 4200, tdp="105W", **{
            "Споживана потужність": "105 Вт", "Сокет": "AM4", "socket": "AM4"}),
        _p("cpu2", "Core i5-12400F", "cpu", 3100, TDP="65 Вт", **{
            "Споживана потужність": "65 Вт", "socket": "LGA1700"}),
        _p("cpu3", "Ryzen 7 5800X", "cpu", 5100, tdp="105W", **{
            "Споживана потужність": "105 Вт", "Сокет": "AM4"}),
        _p("cpu4", "Core i7-13700K", "cpu", 8900, TDP="125 Вт", **{
            "Споживана потужність": "125 Вт", "socket": "LGA1700"}),
        _p("mb1", "B550M Pro", "motherboard", 2900, **{
            "Сокет": "AM4", "socket": "AM4", "Потужність CPU (макс.)": "130W",
            "Підтримка ОЗП": "DDR4", "Форм-фактор": "ATX", "Колір": "чорний"}),
        _p("mb2", "Z690 D4", "motherboard", 6400, **{
            "socket": "LGA1700", "Потужність CPU (макс.)": "150W",
            "Підтримка ОЗП": "DDR5", "Форм-фактор": "ATX"}),
        _p("mb3", "B650M", "motherboard", 4400, **{
            "Сокет": "AM4", "Потужність CPU (макс.)": "130W",
            "Підтримка ОЗП": "DDR5", "Форм-фактор": "mATX"}),
        _p("gpu1", "GeForce RTX 3060", "gpu", 9500, **{
            "Потужність": "170W", "Колір": "чорний"}),
        _p("gpu2", "GeForce RTX 4070 Ti", "gpu", 22000, **{
            "Потужність": "285W"}),
        _p("gpu3", "GeForce RTX 4080", "gpu", 31000, **{
            "Потужність": "320W"}),
        _p("psu1", "БЖ 650W", "psu", 1800, **{"Потужність БЖ": "650W"}),
        _p("psu2", "БЖ 850W Gold", "psu", 2600, **{"Потужність БЖ": "850W"}),
        _p("ram1", "DDR4 16GB", "ram", 1200, **{
            "Тип пам'яті": "DDR4", "Підтримка ОЗП": "DDR4"}),
        _p("ram2", "DDR5 32GB", "ram", 2400, **{
            "Тип пам'яті": "DDR5", "Підтримка ОЗП": "DDR5"}),
        _p("case1", "Корпус Mid Tower", "case", 2600, **{
            "Форм-фактор": "ATX", "Колір": "білий"}),
    ]
    power = ("TDP", "tdp", "Споживана потужність")
    truth = (
        InducedRule(EQ, ("socket", "Сокет"), ("socket", "Сокет")),
        InducedRule(EQ, ("Підтримка ОЗП", "Тип пам'яті"),
                     ("Підтримка ОЗП", "Тип пам'яті")),
        InducedRule(EQ, ("Форм-фактор",), ("Форм-фактор",)),
        InducedRule(LE, power, ("Потужність CPU (макс.)",)),
        InducedRule(LE, power, ("Потужність БЖ",)),
        InducedRule(LE, ("Потужність",), ("Потужність БЖ",)),
    )
    negatives = (
        # eq(Колір): colour is symbolic and spans three categories, so it
        # looks exactly like a matching key -- and is not one. Colour
        # agreement is not a compatibility requirement.
        InducedRule(EQ, ("Колір",), ("Колір",)),
        # le(power <= Потужність): the GPU's draw numerically covers the
        # CPU's draw. Both are demands on the same supply; coverage
        # between siblings is correlation, not a constraint.
        InducedRule(LE, power, ("Потужність",)),
        # le(Потужність CPU (макс.) <= Потужність): the board's own CPU
        # ceiling is covered by the GPU's draw for the same reason.
        InducedRule(LE, ("Потужність CPU (макс.)",), ("Потужність",)),
    )
    return Dataset("pc_build", tuple(products), truth, negatives)


def auto_parts() -> Dataset:
    """
    Wheels, hubs and brakes.

    Ground truth: the bolt pattern is a matching key; the rim diameter must
    clear the hub's maximum; a brake disc must fit inside the rim. The hub's
    maximum and the brake diameter share a unit and a catalogue but no
    physical relation, which is what makes this dataset worth scoring.
    """
    products = [
        _p("rim1", 'Диск 19" R19', "rim", 5400, **{
            "Діаметр": "R19", "Розболтовка": "5*112", "Ширина": "7.0J"}),
        _p("rim2", 'Диск 19" чорний', "rim", 6100, **{
            "diameter": '19.0"', "Розболтовка": "5*112"}),
        _p("bolt1", "Болти 5*112", "bolt", 1400, **{"Розболтовка": "5*112"}),
        _p("bolt2", "Болти 5*120", "bolt", 1600, **{"Розболтовка": "5x120"}),
        _p("hub1", "Ступиця R19", "hub", 3200, **{"max_diameter": "R19"}),
        _p("hub2", "Ступиця 21\"", "hub", 3900, **{"max_diameter": '21.0"'}),
        _p("brake1", "Диск гальмівний 300мм", "brake", 1900, **{
            "Діаметр диска": "300мм"}),
        _p("tire1", "Шина 205/55R16", "tire", 2400, **{"Ширина": "205"}),
        _p("tire2", "Шина 225/45R17", "tire", 2900, **{"Ширина": "225"}),
    ]
    rim = ("diameter", "Діаметр")
    truth = (
        InducedRule(EQ, ("Розболтовка",), ("Розболтовка",)),
        InducedRule(LE, rim, ("max_diameter",)),
        InducedRule(LE, ("Діаметр диска",), rim),
    )
    negatives = (
        # le(Діаметр диска <= max_diameter): the hub's rim limit
        # numerically covers the brake disc. The hub limit constrains the
        # rim, not the brake; coverage across unrelated categories is a
        # coincidence the data alone cannot disprove.
        InducedRule(LE, ("Діаметр диска",), ("max_diameter",)),
        # eq(Ширина): the same key name means rim width in J ("7.0J") and
        # tyre width in mm ("205") -- a shared identifier is not a shared
        # attribute, and the baseline's structural equivalence check
        # actually emits this non-rule while the proposed grouping (which
        # requires a common canonical vocabulary) does not.
        InducedRule(EQ, ("Ширина",), ("Ширина",)),
    )
    return Dataset("auto_parts", tuple(products), truth, negatives)


def pharma() -> Dataset:
    """
    A single active substance prescribed in courses.

    Ground truth: dosage form is a matching key between medicine and course;
    the course's daily mass ceiling covers the per-unit mass. The colour
    attribute is present in both categories with disjoint vocabularies -- the
    case a keyword matcher cannot tell from a real matching key.
    """
    products = [
        _p("med1", "Парацетамол 0.5", "medicine", 60, **{
            "Форма": "таблетки", "Вид": "таблетки", "Маса": "0.5кг",
            "Колір": "білий"}),
        _p("med2", "Ібупрофен 0.25", "medicine", 85, **{
            "Форма": "капсули", "Вид": "капсули", "Маса": "0.25кг"}),
        _p("med3", "Амоксицилін 1", "medicine", 140, **{
            "Форма": "таблетки", "Вид": "таблетки", "Маса": "1кг"}),
        _p("crs1", "Курс 1", "course", 400, **{
            "Форма": "таблетки", "Вид": "таблетки",
            "Максимальна добова маса": "3кг", "Колір": "синій"}),
        _p("crs2", "Курс 2", "course", 520, **{
            "Форма": "капсули", "Вид": "капсули",
            "Максимальна добова маса": "5кг"}),
    ]
    truth = (
        InducedRule(EQ, ("Вид", "Форма"), ("Вид", "Форма")),
        InducedRule(LE, ("Маса",), ("Максимальна добова маса",)),
    )
    negatives = (
        # eq(Колір): a shared attribute across medicine and course with
        # disjoint vocabularies -- the weakest possible signal, still
        # enough for a symbolic multi-category rule.
        InducedRule(EQ, ("Колір",), ("Колір",)),
    )
    return Dataset("pharma", tuple(products), truth, negatives)


def datasets() -> List[Dataset]:
    return [pc_build(), auto_parts(), pharma()]


# --------------------------------------------------------------------------
# Identifier anonymisation (bilingual column/key invariance check)
# --------------------------------------------------------------------------


def anonymized(dataset: Dataset) -> Dataset:
    """
    The same dataset with every business identifier replaced by an opaque
    token: category names, attribute keys, product names. Attribute
    *values* are left untouched.

    An inducer that reads structure must score identically; an inducer
    that reads names cannot. The keyword baseline is expected to collapse
    -- that collapse is the measurement, not a bug: it demonstrates that
    the baseline's competence lives in its word lists. The structural
    method claims invariance and the benchmark now checks the claim.
    """
    key_map = {
        k: f"attr_{i:03d}"
        for i, k in enumerate(sorted(_key_scopes(dataset.products)), 1)
    }
    cat_map = {
        c: f"cat_{i:03d}"
        for i, c in enumerate(
            sorted({p.category for p in dataset.products}), 1)
    }

    def remap(keys: Sequence[str]) -> Tuple[str, ...]:
        return tuple(sorted(key_map[k] for k in keys))

    products = tuple(
        ProductNode(
            id=p.id, name=f"item_{i:03d}", category=cat_map[p.category],
            price=p.price,
            attributes={key_map[k]: v for k, v in p.attributes.items()},
        )
        for i, p in enumerate(dataset.products, 1)
    )
    truth = tuple(
        InducedRule(r.kind, remap(r.subject), remap(r.obj), r.headroom)
        for r in dataset.truth
    )
    negatives = tuple(
        InducedRule(r.kind, remap(r.subject), remap(r.obj), r.headroom)
        for r in dataset.negatives
    )
    return Dataset(dataset.name + "_anon", products, truth, negatives)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def compare(datasets_: Sequence[Dataset] = ()) -> Dict[str, object]:
    """Run both systems over every dataset and aggregate."""
    if not datasets_:
        datasets_ = datasets()
    rows: List[Dict[str, object]] = []
    for d in datasets_:
        predicted_structural = induce_structural(d.products)
        predicted_baseline = rules_from_baseline(d.products)

        anon = anonymized(d)
        anon_structural = induce_structural(anon.products)
        anon_baseline = rules_from_baseline(anon.products)

        neg = set(d.negatives)
        row_proposed = score(predicted_structural, d.truth)
        rows.append({
            "dataset": d.name,
            "n_products": len(d.products),
            "truth": len(set(d.truth)),
            "problems": verify(d),
            "baseline": score(predicted_baseline, d.truth).to_dict(),
            "proposed": row_proposed.to_dict(),
            "baseline_rules": [str(r) for r in predicted_baseline],
            "proposed_rules": [str(r) for r in predicted_structural],
            # Negative controls: which declared non-rules each system
            # emits anyway (F12). A hit is also an FP, but naming it keeps
            # the specific correlation-promoted-to-rule failure visible.
            "negative_controls": len(neg),
            "baseline_negative_hits": sorted(
                str(r) for r in neg if r in set(predicted_baseline)),
            "proposed_negative_hits": sorted(
                str(r) for r in neg if r in set(predicted_structural)),
            # Anonymisation check: identical tp/fp/fn on the renamed copy
            # means the method read structure, not names.
            "anonymized": {
                "baseline": score(anon_baseline, anon.truth).to_dict(),
                "proposed": score(anon_structural, anon.truth).to_dict(),
                "proposed_invariant":
                    score(anon_structural, anon.truth) == row_proposed,
            },
        })

    def total(system: str) -> PRF:
        agg = PRF()
        for row in rows:
            cell = row[system]
            agg = PRF(agg.tp + cell["tp"], agg.fp + cell["fp"], agg.fn + cell["fn"])
        return agg

    return {
        "rows": rows,
        "baseline": total("baseline").to_dict(),
        "proposed": total("proposed").to_dict(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    report = compare()
    print(f"{'dataset':<12} {'n':>3} {'|truth|':>7} "
          f"{'base P/R/F1':>22} {'proposed P/R/F1':>22}")
    print("-" * 72)
    for row in report["rows"]:
        b, p = row["baseline"], row["proposed"]
        print(f"{row['dataset']:<12} {row['n_products']:>3} {row['truth']:>7} "
              f"{b['precision']:>6.1f}/{b['recall']:>5.1f}/{b['f1']:>5.1f} "
              f"{p['precision']:>6.1f}/{p['recall']:>5.1f}/{p['f1']:>5.1f}")
    b, p = report["baseline"], report["proposed"]
    print("-" * 72)
    print(f"{'TOTAL':<12} {'':>3} {'':>7} "
          f"{b['precision']:>6.1f}/{b['recall']:>5.1f}/{b['f1']:>5.1f} "
          f"{p['precision']:>6.1f}/{p['recall']:>5.1f}/{p['f1']:>5.1f}")

    print("\nnegative controls (declared non-rules still emitted):")
    for row in report["rows"]:
        for system in ("baseline", "proposed"):
            hits = row[f"{system}_negative_hits"]
            if hits:
                print(f"  {row['dataset']:<12} {system:<9} {' | '.join(hits)}")

    print("\nanonymised identifiers (keys/categories -> opaque tokens):")
    for row in report["rows"]:
        a = row["anonymized"]
        inv = "invariant" if a["proposed_invariant"] else "CHANGED"
        print(f"  {row['dataset']:<12} proposed {inv}, "
              f"baseline F1 {a['baseline']['f1']:.1f}")

    broken = [r for r in report["rows"] if r["problems"]]
    if broken:
        print("\nGROUND TRUTH NOT DERIVABLE FROM THE DATA:")
        for row in broken:
            for problem in row["problems"]:
                print(f"  {row['dataset']}: {problem}")
        return 1

    if "-v" in (argv or sys.argv[1:]):
        print()
        for row in report["rows"]:
            print(f"[{row['dataset']}]")
            print("  truth     " + " | ".join(row["proposed_rules"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
