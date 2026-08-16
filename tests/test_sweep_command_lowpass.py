"""T-45 / PR-11 §8 — the filter has to be a filter, and it has to reach the data.

§8 names four things, and three of them are tests. The one that matters most is the second:

    A test that fails when the filter is removed. Not "a test that the filter is called" — one
    that fails if the filtered array equals the raw array, at every swept cutoff.

That is the offline half of gate G0.3, and it exists because a filter threaded through the call
chain but never applied yields a flat grid and a confident verdict that the jerk is irreducible —
the same shape as a real **R**, and indistinguishable from it in the output. The peer who ran
`PR-10-anchor-delay-sweep.md` reported paying a mutation test to catch the analogous defect in the
delay knob.

The fourth is the one that keeps this honest against itself: the filter is checked against an
*analytic* signal — a sinusoid above the cutoff must be attenuated and one below it must not — so
"low-pass" is a measured property rather than a claim in a docstring.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lp = _load("sweep_command_lowpass")

FS = 30.0


# --------------------------------------------------------------------------- the kernel


@pytest.mark.parametrize("fc", lp.CUTOFFS_HZ)
def test_kernel_is_symmetric_and_sums_to_one(fc):
    """PR-11 §8.3. Zero phase is a property of symmetry, and unit sum is what leaves DC alone.

    Both are load-bearing and neither is obvious from reading the code: a kernel that summed to
    something other than 1 would rescale every filtered cell, and a scale change is exactly the
    shrinkage confound §4 exists to keep out — it would arrive disguised as a filtering effect.
    """
    k = lp.lowpass_kernel(fc, FS)
    assert k.shape[0] % 2 == 1, "an even-length kernel has no centre and therefore non-zero phase"
    np.testing.assert_allclose(k, k[::-1], atol=1e-15)
    assert k.sum() == pytest.approx(1.0)


def test_kernel_refuses_a_cutoff_at_or_above_nyquist():
    """Nyquist is the no-op and is expressed as `fc=None`, not as `fc=15`."""
    for bad in (0.0, -1.0, FS / 2, FS):
        with pytest.raises(SystemExit):
            lp.lowpass_kernel(bad, FS)


def test_lower_cutoff_means_a_wider_kernel():
    """The half-width formula is `ceil(2 fs / fc)`, so it must grow as the cutoff falls. This is
    the same monotonicity G0.3 checks at runtime, one level down."""
    widths = [lp.lowpass_kernel(fc, FS).shape[0] for fc in sorted(lp.CUTOFFS_HZ)]
    assert widths == sorted(widths, reverse=True)


# --------------------------------------------------------------------------- it is a low-pass


def _sine(freq_hz: float, n: int = 600) -> np.ndarray:
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * freq_hz * t).reshape(-1, 1).astype(np.float32)


def test_a_tone_above_the_cutoff_is_attenuated_and_one_below_is_not():
    """PR-11 §8.4 — checked against an analytic signal, not against the filter's own output.

    The interior is measured rather than the whole array: the edge-clamped padding is a deliberate
    boundary artifact (§3) and the chunks at both ends are dropped anyway by the inherited trim
    rule, so scoring the clamp here would test the padding rather than the passband.
    """
    fc = 3.0
    keep = 60
    for freq, expected in ((1.0, "pass"), (10.0, "stop")):
        raw = _sine(freq)
        out = lp.lowpass(raw, fc, FS)
        ratio = float(np.std(out[keep:-keep]) / np.std(raw[keep:-keep]))
        if expected == "pass":
            assert ratio > 0.9, f"{freq} Hz is below fc={fc} and must survive, got {ratio:.3f}"
        else:
            assert ratio < 0.1, f"{freq} Hz is above fc={fc} and must be cut, got {ratio:.3f}"


def test_the_filter_has_zero_phase():
    """A symmetric kernel must not shift the signal in time — which is the entire reason this
    experiment can be read separately from PR-10's delay finding.

    A causal filter would move the peak; the assertion is that this one does not, to within a
    sample. If it ever did, the grid would be re-measuring the offset T-44 already measured and
    reporting it as a filtering effect.
    """
    raw = _sine(1.0)
    out = lp.lowpass(raw, 5.0, FS)
    keep = 80
    a = raw[keep:-keep, 0]
    b = out[keep:-keep, 0]
    lags = np.arange(-5, 6)
    scores = [float(np.dot(a, np.roll(b, int(k)))) for k in lags]
    assert lags[int(np.argmax(scores))] == 0, "best alignment is not at lag 0 — the filter has phase"


def test_dc_passes_untouched():
    const = np.full((200, 3), 0.37, dtype=np.float32)
    for fc in lp.CUTOFFS_HZ:
        np.testing.assert_allclose(lp.lowpass(const, fc, FS), const, atol=1e-5)


def test_channels_are_filtered_independently():
    """A bug that leaked one joint into another would still produce a smooth, plausible grid."""
    n = 400
    arr = np.zeros((n, 2), dtype=np.float32)
    arr[:, 0] = _sine(10.0, n)[:, 0]
    out = lp.lowpass(arr, 3.0, FS)
    assert np.abs(out[:, 1]).max() < 1e-6, "channel 1 was silent and did not stay silent"


# --------------------------------------------------------------------------- G0.3, offline half


def test_every_swept_cutoff_actually_changes_the_array():
    """**PR-11 §8.2, and the one that fails if the filter is removed.**

    Not "the filter was called": the assertion is on the array. A `lowpass` that returned its input
    — the exact shape of the defect G0.3 guards at runtime — fails here at every cutoff.
    """
    rng = np.random.default_rng(0)
    raw = np.cumsum(rng.normal(scale=0.01, size=(500, 43)), axis=0).astype(np.float32)
    for fc in lp.CUTOFFS_HZ:
        out = lp.lowpass(raw, fc, FS)
        assert not np.array_equal(out, raw), f"fc={fc} Hz left the array untouched"
        assert float(np.sqrt(np.mean((out - raw) ** 2))) > 0.0


def test_rms_change_grows_monotonically_as_the_cutoff_falls():
    """The runtime gate G0.3 asserts exactly this on real episodes. Pinning it on synthetic data
    means a violation is a filter bug rather than a property of the corpus."""
    rng = np.random.default_rng(1)
    raw = np.cumsum(rng.normal(scale=0.01, size=(600, 8)), axis=0).astype(np.float32)
    rms = [
        float(np.sqrt(np.mean((lp.lowpass(raw, fc, FS) - raw) ** 2)))
        for fc in sorted(lp.CUTOFFS_HZ, reverse=True)
    ]
    assert rms == sorted(rms), f"RMS change did not grow as fc fell: {rms}"


def test_the_no_op_cell_is_the_identity_and_reports_zero_change():
    """`fc=None` must return the SAME object's contents untouched — that is what makes the no-op
    cell bit-identical to a T-44 cell rather than merely close to one, and G0.1 compares them to
    ±0.5 pp."""
    rng = np.random.default_rng(2)
    raw = {"action": rng.normal(size=(50, 43)).astype(np.float32), "state": np.zeros((50, 43))}
    out, rms = lp.filtered_raw(raw, None, FS)
    assert rms == 0.0
    np.testing.assert_array_equal(out["action"], raw["action"])


def test_filtered_raw_does_not_mutate_its_input():
    """The driver filters the same cached episode once per cutoff. In-place mutation would make
    every cell after the first a filter of a filter, and the grid would still look smooth."""
    rng = np.random.default_rng(3)
    action = rng.normal(size=(200, 4)).astype(np.float32)
    raw = {"action": action, "state": np.zeros((200, 4), dtype=np.float32)}
    before = action.copy()
    out, rms = lp.filtered_raw(raw, 3.0, FS)
    np.testing.assert_array_equal(raw["action"], before)
    assert rms > 0.0
    assert not np.array_equal(out["action"], before)


# --------------------------------------------------------------------------- the rule


def _cell(l1: float, vs_zero: float = 0.0) -> dict:
    return {"skill_vs_repeat_pct": l1, "skill_vs_zero_pct": vs_zero, "smoothness_ratio": 5.0}


def test_verdict_R_when_no_cutoff_clears_l1():
    """A plausible R shape: filtering helps a little, monotonically, and never enough."""
    grid = {fc: _cell(-260.0 + fc) for fc in lp.CUTOFFS_HZ} | {None: _cell(-224.89)}
    out = lp._verdict(grid, grid, any_l1_anywhere=False, lowest_cutoff=1.0)
    assert out["verdict"] == "R"
    assert "collection spec" in out["reading"]


def test_a_perfectly_flat_grid_is_not_read_as_over_smoothing():
    """Regression for a bug this file caught before any real cell existed.

    A naive `max` over tied cells returns whichever key it met first — the lowest cutoff — and the
    lowest cutoff winning is precisely the trigger for verdict **E**, "over-smoothing wins
    monotonically". So a grid where filtering changed *nothing* would have reported the one reading
    it cannot support. Ties break toward less filtering; see `_verdict`'s docstring.
    """
    flat = {fc: _cell(-200.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(-200.0)}
    out = lp._verdict(flat, flat, any_l1_anywhere=False, lowest_cutoff=1.0)
    assert out["verdict"] == "R"
    assert out["fc_star"] is None, "a flat grid must prefer the no-op, not the most aggressive cell"

    # Same tie among the cutoffs only, with the no-op slightly worse: the widest passband wins.
    tied = {fc: _cell(-200.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(-201.0)}
    assert lp._verdict(tied, tied, any_l1_anywhere=False, lowest_cutoff=1.0)["fc_star"] == max(
        lp.CUTOFFS_HZ
    )


def test_verdict_E_when_the_lowest_cutoff_wins():
    grid = {fc: _cell(-fc) for fc in lp.CUTOFFS_HZ} | {None: _cell(-100.0)}
    out = lp._verdict(grid, grid, any_l1_anywhere=True, lowest_cutoff=1.0)
    assert out["verdict"] == "E"
    assert out["fc_star"] == 1.0


def test_verdict_S_when_the_gain_is_shrinkage():
    """Clears L1 with a material L1 gain, but `skill_vs_zero_pct` barely moves — which is what
    shrinking predictions toward the zero baseline looks like."""
    grid_a = {fc: _cell(-50.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(-60.0)}
    grid_a[3.0] = _cell(40.0)
    grid_b = {fc: _cell(-50.0, -50.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(-60.0, -60.0)}
    grid_b[3.0] = _cell(20.0, -57.0)
    out = lp._verdict(grid_a, grid_b, any_l1_anywhere=True, lowest_cutoff=1.0)
    assert out["verdict"] == "S"
    assert out["zero_gain_pp"] == pytest.approx(3.0)


def test_verdict_F_needs_both_gains():
    grid_a = {fc: _cell(-50.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(-60.0)}
    grid_a[3.0] = _cell(40.0)
    grid_b = {fc: _cell(-50.0, -50.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(-60.0, -60.0)}
    grid_b[3.0] = _cell(20.0, -20.0)
    out = lp._verdict(grid_a, grid_b, any_l1_anywhere=True, lowest_cutoff=1.0)
    assert out["verdict"] == "F"
    assert out["fc_star"] == 3.0
    assert out["l1_gain_pp"] == pytest.approx(80.0)
    assert out["zero_gain_pp"] == pytest.approx(40.0)


def test_verdict_the_no_op_winning_cannot_be_F():
    """If unfiltered beats every cutoff, filtering found nothing — the analogue of T-44's
    `d* = 0 cannot produce T`."""
    grid = {fc: _cell(-50.0, -50.0) for fc in lp.CUTOFFS_HZ} | {None: _cell(80.0, 80.0)}
    out = lp._verdict(grid, grid, any_l1_anywhere=True, lowest_cutoff=1.0)
    assert out["fc_star"] is None
    assert out["verdict"] != "F"
