"""Tests for scripts/power_closed_loop_eval.py — the McNemar power calculation.

The script exists to say whether a planned experiment could have detected its own effect, and its
conclusion ("9 % power against a quadrupling") is only worth as much as its arithmetic. Two of the
tests below check that arithmetic against values computable by hand; the rest pin the structural
properties that make the conclusion robust to the exact numbers.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "power_closed_loop_eval", _REPO_ROOT / "scripts" / "power_closed_loop_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


pw = _load()


@pytest.mark.parametrize(
    "b, c, expected",
    [
        (0, 0, 1.0),        # no discordant seeds at all: the test has nothing to say
        (0, 1, 1.0),        # 2 * 0.5
        (0, 5, 0.0625),     # 2 * 0.5**5 -- famously just misses
        (0, 6, 0.03125),    # 2 * 0.5**6 -- the floor
        (1, 5, 0.21875),    # 2 * (C(6,0)+C(6,1))/2**6 = 2 * 7/64
    ],
)
def test_mcnemar_exact_matches_hand_computation(b, c, expected):
    assert pw.mcnemar_exact_p(b, c) == pytest.approx(expected)


def test_the_test_is_symmetric_in_its_two_arms():
    """A two-sided test that preferred one direction would silently favour the treatment."""
    for b, c in ((0, 6), (1, 5), (2, 9), (3, 3)):
        assert pw.mcnemar_exact_p(b, c) == pytest.approx(pw.mcnemar_exact_p(c, b))


def test_six_discordant_seeds_is_the_floor():
    """The load-bearing number: fewer than six disagreements cannot reach 0.05, ever."""
    assert pw.min_discordant_for_significance(0.05) == 6
    assert pw.mcnemar_exact_p(0, 5) > 0.05
    assert pw.mcnemar_exact_p(0, 6) <= 0.05


def test_a_run_with_too_few_discordant_seeds_cannot_be_significant_however_it_splits():
    """This is what makes concordance so expensive: five disagreements are unusable outright."""
    for d in range(1, 6):
        assert pw._max_minority(d, 0.05) < 0, d
    assert pw._max_minority(6, 0.05) == 0


def test_power_against_a_quadrupling_at_twenty_seeds_is_under_ten_percent():
    """The headline of docs/comparison-power-analysis.md, pinned so a refactor cannot move it."""
    p = pw.power(1 / 20, 4 / 20, 20)
    assert 0.05 < p < 0.10, p


def test_power_is_zero_when_the_arms_are_identical():
    assert pw.power(0.2, 0.2, 20) == pytest.approx(0.0, abs=0.05)


def test_power_increases_with_effect_size_and_with_seeds():
    """Two monotonicities. A power function violating either is wrong regardless of its values."""
    base = 1 / 20
    by_effect = [pw.power(base, t / 20, 20) for t in (2, 4, 6, 8, 10)]
    assert by_effect == sorted(by_effect)
    by_seeds = [pw.power(base, 6 / 20, n) for n in (10, 20, 40, 80)]
    assert by_seeds == sorted(by_seeds)


def test_detecting_a_quadrupling_needs_far_more_than_the_frozen_twenty_seeds():
    """The actionable half: the fix is seeds, and the protocol's 20 are not close to enough."""
    n = pw.seeds_for_power(1 / 20, 4 / 20, 0.8)
    assert n is not None and 60 < n < 120, n
    assert pw.power(1 / 20, 4 / 20, n) >= 0.8
    assert pw.power(1 / 20, 4 / 20, n - 1) < 0.8


def test_a_higher_baseline_needs_more_seeds_not_fewer():
    """Stated in the document; a reader would plausibly assume the opposite."""
    low = pw.seeds_for_power(1 / 20, 4 / 20, 0.8)
    high = pw.seeds_for_power(5 / 20, 8 / 20, 0.8)
    assert high > low, (low, high)
