"""
Per-key unit profiling: what a catalogue's units say about the catalogue.

Why this module exists, and what it deliberately does NOT do
-----------------------------------------------------------
The obvious way to make the algebra adapt to a store without a dictionary is to
*learn* unit aliases from the data: two values under one key carrying the same
number in different tokens must denote the same thing. Measured on
`data/real_off.db`, that heuristic is not merely weak, it is actively dangerous:

    'g' == 'mg'    "corroborated" by 915 (key, number) cells
    'kJ' == 'kcal'                                  52 cells
    'g' == 'ml'                                     28 cells

`g` and `mg` are 1000x apart. Those 915 cells are not evidence of an alias; they
are evidence that **Open Food Facts itself writes the same number under two
units in one key**: `nutrition_calcium_per100g` holds 3999 values in `g` and
1468 in `mg`, and the mg ones (`0.0614 mg`, `0.112 mg`) are gram-scale numbers
typed with the wrong unit. A learner would have "corroborated" that 1000x error
915 times and reinstated exactly the defect removed in docs/08 section 4.1.

So no conversion factor is inferred here. What *is* sound is a per-key profile:
which unit dominates a key, and which tokens are rare enough in it to be
data-entry accidents. That asserts nothing about two values being equal -- it
only labels a key. A wrong label costs a human thirty seconds; a wrong equality
silently corrupts every constraint built on top of it.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Optional

from attribute_algebra import Shape, dimension_for, spec_of

#: A key needs at least this many unit-bearing values before its unit
#: distribution says anything: among five values every token is a minority.
MIN_VALUES_FOR_PROFILE = 20

#: A token below this share of a key is a minority unit -- a probable
#: unit-entry error rather than a legitimate second unit of that key.
MINORITY_SHARE = 0.05


@dataclass(frozen=True)
class KeyUnitProfile:
    """How one EAV key uses units. Descriptive only: nothing is merged."""

    key: str
    values: int = 0                    # values carrying a unit token
    unitless: int = 0                  # measures with no unit at all
    dominant: Optional[str] = None
    dominant_share: float = 0.0
    units: Dict[str, int] = field(default_factory=dict)
    minority: Dict[str, int] = field(default_factory=dict)
    mixed: bool = False                # two or more units in normal use
    too_small: bool = False            # below MIN_VALUES_FOR_PROFILE

    @property
    def dominant_dimension(self) -> Optional[str]:
        return dimension_for(self.dominant) if self.dominant else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "values": self.values,
            "unitless": self.unitless,
            "dominant": self.dominant,
            "dominant_share": round(self.dominant_share, 4),
            "dominant_dimension": self.dominant_dimension,
            "units": self.units,
            "minority_units": self.minority,
            "mixed_units": self.mixed,
            "too_small_to_profile": self.too_small,
        }


def profile_key(
    key: str,
    values: Iterable[object],
    *,
    min_values: int = MIN_VALUES_FOR_PROFILE,
    minority_share: float = MINORITY_SHARE,
) -> KeyUnitProfile:
    """Profile one key's unit usage from its raw surface values."""
    counts: Counter = Counter()
    unitless = 0
    for raw in values:
        sp = spec_of(str(raw))
        if sp.shape is not Shape.MEASURE:
            continue
        if not sp.unit:
            unitless += 1
            continue
        counts[sp.unit] += 1

    total = sum(counts.values())
    if not counts:
        return KeyUnitProfile(key=key, values=0, unitless=unitless, too_small=True)

    # Count first, name second: a tie must not depend on dictionary ordering.
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    dominant, dominant_count = ranked[0]
    share = dominant_count / total
    minority = {unit: count for unit, count in ranked
                if unit != dominant and count / total < minority_share}
    in_normal_use = sum(1 for _unit, count in ranked if count / total >= minority_share)
    return KeyUnitProfile(
        key=key,
        values=total,
        unitless=unitless,
        dominant=dominant,
        dominant_share=share,
        units=dict(ranked),
        minority=minority,
        mixed=in_normal_use > 1,
        too_small=total < min_values,
    )


def profile_key_units(
    values_by_key: Mapping[str, Iterable[object]],
    **kwargs,
) -> Dict[str, KeyUnitProfile]:
    """Profile every key of a catalogue. Order-stable, no hidden state."""
    return {key: profile_key(key, values, **kwargs)
            for key, values in sorted(values_by_key.items())}


def summarise(profiles: Mapping[str, KeyUnitProfile], *, top: int = 15
              ) -> Dict[str, object]:
    """Report shape: how many keys were profiled, which ones look broken."""
    considered = [p for p in profiles.values() if p.values and not p.too_small]
    with_minority = [p for p in considered if p.minority]
    mixed = [p for p in considered if p.mixed]
    unitless_only = [p for p in profiles.values() if not p.values and p.unitless]
    offenders = sorted(with_minority,
                       key=lambda p: (-sum(p.minority.values()), p.key))[:top]
    return {
        "keys_total": len(profiles),
        "keys_profiled": len(considered),
        "keys_skipped_too_small": sum(1 for p in profiles.values() if p.too_small),
        "keys_with_minority_unit": len(with_minority),
        "keys_mixing_units_in_normal_use": len(mixed),
        "keys_with_no_unit_at_all": len(unitless_only),
        "minority_values_flagged": sum(sum(p.minority.values()) for p in with_minority),
        "unit_hygiene_pct": hygiene_pct(profiles),
        "unitless_values": sum(p.unitless for p in profiles.values()),
        "minority_share_threshold": MINORITY_SHARE,
        "min_values_for_profile": MIN_VALUES_FOR_PROFILE,
        "top_minority_unit_keys": [p.to_dict() for p in offenders],
        "note": ("descriptive only: no value is merged and no conversion factor is "
                 "inferred -- see this module's docstring for the measurement that "
                 "rules the naive learner out"),
    }


def unit_caveat(
    key: str,
    value: object,
    profiles: Mapping[str, KeyUnitProfile],
) -> Optional[str]:
    """Caveat to attach when a value of a flagged key is quoted to a user.

    Returns None unless this very value carries the key's minority unit, so a
    normal value costs nothing. The wording names the key and the counts, so
    the reader can judge it rather than trust it.

    Deliberately not wired into the answering path: no endpoint currently
    quotes a unit-bearing attribute (the grounding set carries names, prices
    and ids only), and inventing a call site would be a fake integration. This
    is the sanctioned way to do it when such a path exists.
    """
    profile = profiles.get(key)
    if profile is None or not profile.minority:
        return None
    sp = spec_of(str(value))
    if sp.shape is not Shape.MEASURE or not sp.unit or sp.unit not in profile.minority:
        return None
    return (f"unit `{sp.unit}` is rare in `{key}` "
            f"({profile.minority[sp.unit]} of {profile.values} values; "
            f"dominant `{profile.dominant}`) — likely a unit-entry error")


def hygiene_pct(profiles: Mapping[str, KeyUnitProfile]) -> float:
    """Share of profiled values whose unit agrees with their key's dominant one.

    A store-facing number: 100.0 means every unit-bearing value under a profiled
    key uses that key's normal unit. It is a data-hygiene score, not a
    correctness score -- the algebra stays correct whatever this returns.
    """
    considered = [p for p in profiles.values() if p.values and not p.too_small]
    total = sum(p.values for p in considered)
    if not total:
        return 100.0
    flagged = sum(sum(p.minority.values()) for p in considered)
    return round(100.0 * (1.0 - flagged / total), 2)


def to_json(profiles: Mapping[str, KeyUnitProfile], **kwargs) -> str:
    return json.dumps(summarise(profiles, **kwargs), ensure_ascii=False, indent=2)

