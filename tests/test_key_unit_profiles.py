"""
What the catalogue says about its own units -- and what it must not be taken to say.

These tests exist mostly to pin a *negative*: profiling may label a key, but it
may never make two values equal. The Open Food Facts nutrition fields contain
915 (key, number) cells where `g` and `mg` carry the same number, which is a
1000x unit-entry error in the source data, not an alias. A learner that treated
those as corroboration would rebuild the exact defect this project removed.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from attribute_algebra import CONFLICT, canonical_form, fold_all
from key_unit_profiles import (
    MIN_VALUES_FOR_PROFILE,
    hygiene_pct,
    profile_key,
    profile_key_units,
    summarise,
    to_json,
    unit_caveat,
)


def test_a_rare_unit_in_a_dominated_key_is_flagged_as_minority():
    values = ["1.0 g"] * 40 + ["0.05 mg"] * 2
    profile = profile_key("nutrition_calcium_per100g", values)
    assert profile.dominant == "g"
    assert profile.dominant_share == pytest.approx(40 / 42)
    assert profile.minority == {"mg": 2}
    assert profile.mixed is False
    assert profile.dominant_dimension == "M"


def test_two_units_in_normal_use_are_a_mix_not_an_error():
    values = ["1 kg"] * 30 + ["500 g"] * 30
    profile = profile_key("weight", values)
    assert profile.minority == {}
    assert profile.mixed is True


def test_a_tiny_key_is_not_profiled():
    profile = profile_key("serving_size", ["1 g", "2 mg"])
    assert profile.too_small is True
    assert profile.values == 2


def test_unitless_measures_are_counted_separately():
    profile = profile_key("count", ["12", "13", "14"] * 10 + ["15 g"] * 5)
    assert profile.unitless == 30
    assert profile.values == 5
    assert profile.dominant == "g"


def test_profiling_never_creates_an_equality():
    """The whole point: a profile is a label, not a merge.

    915 same-number (g, mg) cells in Open Food Facts are unit-entry errors in
    the source. If profiling inferred `g == mg` from them it would reinstate the
    1000x defect, so the algebra must be untouched afterwards.
    """
    noisy = ["0.5 g"] * 900 + ["0.5 mg"] * 15
    profile_key("nutrition_caffeine_value", noisy)
    assert canonical_form("0.5 g") != canonical_form("0.5 mg")
    assert fold_all(["0.5 g", "0.5 mg"]) is CONFLICT
    assert canonical_form("500 mg") == canonical_form("0.5 g")


def test_a_profiled_catalogue_raises_no_conversion_factor():
    profiles = profile_key_units({
        "weight": ["1.0 g"] * 40 + ["0.05 mg"] * 2,
        "serving_quantity": ["250 ml"] * 12 + ["0.25 l"] * 12,
    })
    assert profiles["weight"].minority == {"mg": 2}
    # 250 ml and 0.25 l are both in normal use: merged by the *table*, not
    # learned, and the profile never claims a factor between them.
    assert profiles["serving_quantity"].minority == {}
    assert profiles["serving_quantity"].mixed is True


def test_summary_is_json_serialisable_and_ordered():
    profiles = profile_key_units({
        "b_key": ["1.0 g"] * 30 + ["0.05 mg"] * 1,
        "a_key": ["250 ml"] * 30,
    })
    report = summarise(profiles)
    assert report["keys_total"] == 2
    assert report["keys_profiled"] == 2
    assert report["keys_with_minority_unit"] == 1
    assert report["minority_values_flagged"] == 1
    assert report["top_minority_unit_keys"][0]["key"] == "b_key"
    json.loads(to_json(profiles))


def test_empty_and_absurd_inputs_do_not_raise():
    profiles = profile_key_units({
        "empty": [],
        "no_measures": ["AM5", "ATX"],
        "tie": ["1 g", "1 kg"],
    })
    report = summarise(profiles)
    assert report["keys_total"] == 3
    assert profiles["empty"].values == 0
    # A 50/50 tie is neither dominant nor minority: no claim either way.
    assert profiles["tie"].minority == {}


def test_threshold_is_reported_so_the_reader_can_judge_it():
    report = summarise({})
    assert report["minority_share_threshold"] == 0.05
    assert report["min_values_for_profile"] == MIN_VALUES_FOR_PROFILE


# --------------------------------------------------------------------------
# Q6: what the profile is allowed to do to an answer
# --------------------------------------------------------------------------

def _flagged():
    return profile_key_units({
        "quantity": ["500 g"] * 40 + ["1 kg", "8 oz"],
        "colour": ["red", "blue"] * 20,
    })


def test_caveat_fires_only_for_the_rare_unit_of_that_key():
    profiles = _flagged()
    caveat = unit_caveat("quantity", "1 kg", profiles)
    assert caveat is not None
    assert "kg" in caveat and "quantity" in caveat
    assert "dominant `g`" in caveat and "1 of 42" in caveat
    # The ordinary value of the same key needs no caveat.
    assert unit_caveat("quantity", "500 g", profiles) is None


def test_caveat_is_silent_where_there_is_nothing_to_say():
    profiles = _flagged()
    assert unit_caveat("colour", "red", profiles) is None      # no units at all
    assert unit_caveat("no_such_key", "1 kg", profiles) is None
    assert unit_caveat("quantity", "AM5", profiles) is None    # not a measurement


def test_caveat_does_not_change_the_value_it_describes():
    """A caveat is a warning, never a correction: the algebra is untouched."""
    profiles = _flagged()
    unit_caveat("quantity", "1 kg", profiles)
    assert canonical_form("1 kg") == canonical_form("1000 g")
    assert fold_all(["1 kg", "500 g"]) is CONFLICT


def test_hygiene_score_is_a_store_facing_number():
    profiles = _flagged()
    # 42 unit-bearing values in `quantity`, 2 of them minority units.
    assert hygiene_pct(profiles) == pytest.approx(100.0 * (1 - 2 / 42), abs=0.01)
    assert summarise(profiles)["unit_hygiene_pct"] == hygiene_pct(profiles)
    assert hygiene_pct({}) == 100.0
    assert hygiene_pct(profile_key_units({"colour": ["red"] * 30})) == 100.0
