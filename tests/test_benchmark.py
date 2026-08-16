"""Tests for wam.evaluation.benchmark: the WAM-Bench offline ladder (T-27)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wam.evaluation import (
    BENCH_SPEC_DEFAULT,
    BENCH_SPECS,
    MAX_HORIZON_RATIO,
    MAX_SMOOTHNESS_RATIO,
    MIN_SMOOTHNESS_RATIO,
    RUNG_NAMES,
    BenchReport,
    ChunkPrediction,
    bench_metrics,
    compare_bench,
    save_predictions_jsonl,
)
from wam.evaluation.benchmark import _causal_previous_action, _shift_tolerant_sq, _smoothness_rung
from wam.interfaces import ActionChunk, ActionMode

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_bench.py"

DT = 0.05
STEPS = 8
DIM = 3


def chunk(targets: np.ndarray, gripper: float | np.ndarray = 0.5) -> ActionChunk:
    t = targets.shape[0]
    g = np.full(t, gripper, dtype=np.float32) if np.isscalar(gripper) else np.asarray(gripper)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=targets.astype(np.float32),
        gripper_target=g.astype(np.float32),
        dt_s=DT,
    )


def episode(
    predicted: list[np.ndarray],
    target: list[np.ndarray],
    episode_id: str = "ep0",
    stride: int = STEPS,
    gripper: float = 0.5,
) -> list[ChunkPrediction]:
    """Chunk sequence with a realistic emission cadence: one chunk every ``stride`` steps."""
    dt_ns = int(DT * 1e9)
    return [
        ChunkPrediction(
            predicted=chunk(p, gripper),
            target=chunk(t, gripper),
            episode_id=episode_id,
            t_ns=i * stride * dt_ns,
        )
        for i, (p, t) in enumerate(zip(predicted, target))
    ]


def constant(value: float, steps: int = STEPS, dim: int = DIM) -> np.ndarray:
    return np.full((steps, dim), value, dtype=np.float32)


@pytest.mark.parametrize("spec_version", sorted(BENCH_SPECS))
def test_perfect_prediction_tops_the_accuracy_rungs(spec_version: str) -> None:
    """Must hold under EVERY spec: it is the guard that a rule change did not move the anchor."""
    targets = [constant(0.1), constant(0.2), constant(0.3)]
    report = bench_metrics(episode(targets, targets), spec_version=spec_version)

    assert report.mse == pytest.approx(0.0)
    assert report.skill_vs_zero_pct == pytest.approx(100.0)
    assert report.skill_vs_repeat_pct == pytest.approx(100.0)
    assert [r.passed for r in report.rungs[:3]] == [True, True, True]
    assert report.score == pytest.approx(100.0)


def test_zero_prediction_scores_zero_skill_and_fails_l0() -> None:
    targets = [constant(0.1), constant(0.2), constant(0.3)]
    preds = [np.zeros_like(t) for t in targets]
    report = bench_metrics(episode(preds, targets))

    # Predicting zero IS the zero-delta baseline, so skill against it is exactly 0 — and the
    # gate is strict (> 0), which is the point: tying with "hold still" is not passing.
    assert report.skill_vs_zero_pct == pytest.approx(0.0)
    assert report.rungs[0].passed is False
    assert report.level == -1
    assert report.level_name == "none — below L0"


def test_repeat_baseline_is_causal_and_uses_the_step_before_the_chunk() -> None:
    """The reference action must be the one executed at t-dt, never a step from the future."""
    a, b = constant(1.0), constant(2.0)
    preds = episode([a, b], [a, b], stride=STEPS)

    # First chunk of an episode has no predecessor -> rest posture, i.e. zeros.
    assert _causal_previous_action(preds, 0, STEPS).tolist() == [0.0] * DIM
    # Non-overlapping chunks (stride == T): the predecessor's last step.
    assert _causal_previous_action(preds, 1, STEPS).tolist() == [1.0] * DIM


def test_repeat_baseline_does_not_reach_into_the_future_when_chunks_overlap() -> None:
    """With stride < T the previous chunk extends past the current start — index, don't take [-1]."""
    ramp = np.arange(STEPS, dtype=np.float32)[:, None] * np.ones((1, DIM), dtype=np.float32)
    preds = episode([ramp, ramp], [ramp, ramp], stride=3)

    # Chunk 1 starts 3 control steps after chunk 0, so the action at t-dt is index 2, value 2.0.
    # Taking the predecessor's last step would return 7.0 — four steps of the future.
    assert _causal_previous_action(preds, 1, STEPS).tolist() == [2.0] * DIM


def test_critical_intervals_select_the_high_motion_chunks() -> None:
    # Distinct magnitudes: a quantile over TIED energies would select every tied chunk, which is
    # correct quantile behaviour but says nothing about the selection rule under test.
    quiet = [constant(0.001 * (i + 1)) for i in range(8)]
    loud = [constant(1.0), constant(2.0)]
    targets = quiet + loud
    preds = [np.zeros_like(t) for t in targets]
    report = bench_metrics(episode(preds, targets), critical_quantile=0.8)

    assert report.num_predictions == 10
    assert report.num_critical == 2  # top 20% of 10 chunks
    # Predicting zero makes each chunk's error its own squared magnitude, so CI-MSE must be
    # exactly the mean over the two loud chunks — the quiet 80% is excluded, not down-weighted.
    assert report.ci_mse == pytest.approx((1.0**2 + 2.0**2) / 2)
    assert report.ci_mse > report.mse


def test_level_is_the_highest_contiguous_rung() -> None:
    """Clearing a later rung while failing an earlier one must not be credited as progress."""
    targets = [constant(0.1) for _ in range(4)]
    # Predict the exact opposite: worse than zero (fails L0/L1/L2) but perfectly flat, so the
    # horizon and smoothness rungs (L3/L4) both pass on their own terms.
    preds = [-t for t in targets]
    report = bench_metrics(episode(preds, targets))

    passed = [r.passed for r in report.rungs]
    assert passed[0] is False
    assert passed[3] is True
    assert report.level == -1
    assert 0.0 < report.score < 100.0


def test_inertia_warning_fires_when_only_the_weak_baseline_is_beaten() -> None:
    """A model between the two baselines is the exact case a raw-MSE dashboard misreports."""
    rng = np.random.default_rng(0)
    targets = [
        constant(0.5) + rng.normal(0, 0.01, (STEPS, DIM)).astype(np.float32) for _ in range(6)
    ]
    # Halfway between zero and the truth: beats holding still, loses to repeating the last action.
    preds = [t * 0.5 for t in targets]
    report = bench_metrics(episode(preds, targets))

    assert report.skill_vs_zero_pct > 0.0
    assert report.skill_vs_repeat_pct < 0.0
    assert report.level == 0  # L0 passes, L1 fails -> the ladder stops at L0
    assert report.rungs[1].passed is False
    assert any("inertia" in w for w in report.warnings)


def test_degenerate_gripper_channel_is_flagged() -> None:
    targets = [constant(0.1) for _ in range(4)]
    report = bench_metrics(episode(targets, targets, gripper=0.48))

    assert report.gripper_dynamic_range == pytest.approx(0.0)
    assert any("degenerate" in w.lower() for w in report.warnings)
    assert any("grasp-success proxy" in w for w in report.warnings)


# --- gripper admissibility (T-31) ----------------------------------------------------------------


def gripper_episode(
    values: np.ndarray, episode_id: str = "ep0", stride: int = STEPS
) -> list[ChunkPrediction]:
    """One episode whose demonstrated AND predicted gripper follow ``values``, 8 steps per chunk.

    ``stride`` is the emission cadence, as in :func:`episode`. At the default it equals the
    horizon and the chunks tile the episode; below it they overlap, which is the regime the
    closed loop actually runs in and the only one in which a step can be read twice.
    """
    dt_ns = int(DT * 1e9)
    return [
        ChunkPrediction(
            predicted=chunk(constant(0.1), values[start : start + STEPS]),
            target=chunk(constant(0.1), values[start : start + STEPS]),
            episode_id=episode_id,
            t_ns=start * dt_ns,
        )
        for start in range(0, values.shape[0] - STEPS + 1, stride)
    ]


def test_gripper_accuracy_is_withheld_below_the_dynamic_range_floor() -> None:
    """A number that cannot be believed must not be renderable as a number."""
    targets = [constant(0.1) for _ in range(4)]
    report = bench_metrics(episode(targets, targets, gripper=0.48))

    assert report.gripper_accuracy is None
    assert "0.25" in report.gripper_accuracy_withheld_reason
    markdown = report.render_markdown()
    assert "| gripper_accuracy | n/a — withheld" in markdown
    assert "| gripper_accuracy | 1 |" not in markdown


def test_gripper_majority_baseline_is_reported_even_when_accuracy_is_withheld() -> None:
    """Withholding must not lose information — the majority rate IS what the accuracy measured."""
    values = np.full(STEPS * 10, 0.48, dtype=np.float32)
    values[: STEPS * 2] = 0.52  # 20% above the binarization threshold, all within 0.04 p2p
    report = bench_metrics(gripper_episode(values))

    assert report.gripper_accuracy is None
    assert report.gripper_majority_pct == pytest.approx(80.0)
    assert "80.0%" in report.gripper_accuracy_withheld_reason
    assert "| gripper_majority_pct | 80.0% |" in report.render_markdown()


def test_gripper_transitions_are_debounced_and_counted_per_episode() -> None:
    """A channel with real open/close events keeps its accuracy and reports the event count."""
    values = np.where((np.arange(STEPS * 8) // STEPS) % 2 == 0, 0.05, 0.95).astype(np.float32)
    report = bench_metrics(gripper_episode(values))

    assert report.gripper_accuracy == pytest.approx(1.0)
    assert report.gripper_accuracy_withheld_reason == ""
    assert report.gripper_dynamic_range == pytest.approx(0.9)
    assert report.gripper_transitions_per_episode == pytest.approx(7.0)


def test_the_transition_count_does_not_change_when_the_same_episode_is_emitted_faster() -> None:
    """The event count is a property of the robot's timeline, not of the emission cadence.

    With stride < T every overlapped step is commanded by several chunks, so concatenating whole
    chunks replays each open/close once per chunk that saw it: 35 transitions here instead of 7,
    on 232 steps instead of the 64 the episode has. That inflates the very number the audit's
    transition clause reads, and always upwards — so the same demonstration, re-emitted at a
    faster cadence, would look like a busier gripper.
    """
    values = np.where((np.arange(STEPS * 8) // STEPS) % 2 == 0, 0.05, 0.95).astype(np.float32)
    tiled = bench_metrics(gripper_episode(values))
    overlapping = bench_metrics(gripper_episode(values, stride=2))

    assert len(gripper_episode(values, stride=2)) > len(gripper_episode(values))
    assert tiled.gripper_transitions_per_episode == pytest.approx(7.0)
    assert overlapping.gripper_transitions_per_episode == pytest.approx(
        tiled.gripper_transitions_per_episode
    )


def test_withholding_the_gripper_number_moves_no_rung_points() -> None:
    """The claim that justifies NOT spec-versioning the withholding change.

    The gripper is a diagnostic and earns no rung credit, so suppressing it cannot move a score
    or a level — and the degenerate-vs-live pair is the only way to check that rather than assert
    it. Every scored quantity is built from ``targets``, which is identical in both runs.
    """
    dead = np.full(STEPS * 8, 0.48, dtype=np.float32)
    live = np.where((np.arange(STEPS * 8) // STEPS) % 2 == 0, 0.05, 0.95).astype(np.float32)
    withheld = bench_metrics(gripper_episode(dead))
    reported = bench_metrics(gripper_episode(live))

    assert withheld.gripper_accuracy is None and reported.gripper_accuracy is not None
    assert withheld.score == pytest.approx(reported.score)
    assert withheld.level == reported.level
    for a, b in zip(withheld.rungs, reported.rungs):
        assert (a.points, a.passed, a.value) == (b.points, b.passed, b.value)


# --- versioned bench specs (I-10) ----------------------------------------------------------------


def test_min_smoothness_ratio_is_the_reciprocal_of_the_max() -> None:
    """Derived, never literal: the floor cannot be quietly tuned to flatter a run it scored."""
    assert MIN_SMOOTHNESS_RATIO == 1.0 / MAX_SMOOTHNESS_RATIO
    assert BENCH_SPECS["0.2.0"].min_smoothness_ratio == MIN_SMOOTHNESS_RATIO
    assert BENCH_SPECS["0.1.0"].min_smoothness_ratio is None
    assert BENCH_SPEC_DEFAULT == "0.1.0"


def test_smoothness_band_is_one_sided_under_spec_0_1_0_and_two_sided_under_0_2_0() -> None:
    """Archived runs stay reproducible; the new rule is a real change, not cosmetics."""
    one, two = BENCH_SPECS["0.1.0"], BENCH_SPECS["0.2.0"]

    # A prediction as smooth as the demonstrations tops the rung under BOTH — the anchor did
    # not move, which is what makes 0.2.0 a completion of 0.1.0 rather than a new rule.
    for spec in (one, two):
        assert _smoothness_rung(1.0, spec)[2:] == (True, 20.0)

    # T-16's 0.29: 3.4x SMOOTHER than a demonstration. Full marks under 0.1.0, a fail under 0.2.0.
    assert _smoothness_rung(0.29, one)[2:] == (True, 20.0)
    assert _smoothness_rung(0.29, two)[2:] == (False, 0.0)

    # The one-sided failure is unchanged, and both band edges score zero points but pass.
    assert _smoothness_rung(2.35, one)[2] is False
    assert _smoothness_rung(2.35, two)[2] is False
    assert _smoothness_rung(2.0, two)[2:] == (True, 0.0)
    assert _smoothness_rung(0.5, two)[2:] == (True, 0.0)

    # Degenerate ratios: inf is a diverging run, 0.0 a perfectly flat one. Neither is demo-like.
    assert _smoothness_rung(float("inf"), two)[2:] == (False, 0.0)
    assert _smoothness_rung(0.0, two)[2:] == (False, 0.0)


def test_a_run_that_fails_l1_keeps_its_level_whatever_l4_says() -> None:
    """Why "no archived run's level moves" was NOT a test of the 0.2.0 adoption.

    ``level`` is the highest CONTIGUOUS rung passed, so a run that fails L1 is capped at L0 no
    matter what L3 and L4 decide. All three archived runs fail L1 (skill_vs_repeat_pct −20.9 /
    −129.0 / −32.4, all three tiled, tabulated in docs/benchmark.md before the rule was written;
    T-16's −21.8 in its real window fails L1 too, so the re-score does not disturb this either),
    so an L4-only change could not have moved any of their levels. This is the shape of that run:
    L4 flips from PASS to FAIL and the level does not move a rung.
    """
    rng = np.random.default_rng(0)
    targets = [
        constant(0.5) + rng.normal(0, 0.01, (STEPS, DIM)).astype(np.float32) for _ in range(6)
    ]
    preds = [t * 0.5 for t in targets]  # beats zero-delta, loses to repeat-last-action

    old = bench_metrics(episode(preds, targets), spec_version="0.1.0")
    new = bench_metrics(episode(preds, targets), spec_version="0.2.0")

    assert old.rungs[1].passed is False  # L1 fails, so L2-L4 cannot be reached
    assert old.rungs[4].passed is True and new.rungs[4].passed is False
    assert old.level == new.level == 0
    assert new.score < old.score  # the score is the only thing that could have moved


def test_no_ratio_scores_more_under_spec_0_2_0_than_under_0_1_0() -> None:
    """Adoption rule for spec 0.2.0, restated so it could have failed: no score may INCREASE.

    Unlike the level, an L4 rule change moves the score directly, and nothing about adding a
    floor forces the move to be downward: two of the three archived runs score 0/20 on L4 under
    the one-sided gate (t18 at r = 5.10, d1-full-gen at r = 2.35), so a 0.2.0 that re-anchored
    the points function while adding the floor — a linear penalty on |r - 1|, or a wider ceiling
    to make room for the new side — would have raised them. The shipped rule satisfies the
    constraint for EVERY ratio and not only for the three archived ones, which is what makes it a
    constraint on the rule change rather than an observation about three files.

    "Every ratio" is swept on the same dense grid the sibling adoption test uses, plus the
    archived ratios and the two band edges as named points. An earlier version of this test
    checked 11 hand-picked ratios and the prose still said "every" — 11 points do not establish
    a claim over the axis, and the grid was already available one test down.
    """
    grid = np.concatenate(
        [
            np.linspace(1e-6, 6.0, 6000),
            np.geomspace(1e-6, 1e3, 2000),
            np.array([0.29, 2.35, 5.098, MIN_SMOOTHNESS_RATIO, 2.0, 1.0]),
        ]
    )
    for ratio in grid:
        old_points = _smoothness_rung(float(ratio), BENCH_SPECS["0.1.0"])[3]
        new_points = _smoothness_rung(float(ratio), BENCH_SPECS["0.2.0"])[3]
        assert new_points <= old_points + 1e-12, ratio


def test_the_two_specs_disagree_exactly_below_the_derived_floor() -> None:
    """Second adoption rule: 0.2.0 may change an L4 verdict only where it intends to.

    The intended change is "too bland is also a failure", i.e. r < MIN_SMOOTHNESS_RATIO. A
    verdict that differed anywhere else — at the ceiling, or at r == 1.0 — would mean the band
    moved rather than gained a second side, and that would be a re-litigation of the three
    recorded L4 results rather than a completion of the rule.
    """
    one, two = BENCH_SPECS["0.1.0"], BENCH_SPECS["0.2.0"]
    grid = np.linspace(1e-3, 6.0, 6000)
    for ratio in grid:
        differs = _smoothness_rung(ratio, one)[2] != _smoothness_rung(ratio, two)[2]
        assert differs == (ratio < MIN_SMOOTHNESS_RATIO), ratio


def test_a_bland_run_loses_points_without_gaining_a_rung_under_the_new_spec() -> None:
    """The adoption rules end to end on a scored run, not only on the rung helper."""
    rng = np.random.default_rng(7)
    targets = [
        constant(0.2) + rng.normal(0, 0.05, (STEPS, DIM)).astype(np.float32) for _ in range(4)
    ]
    # A prediction that follows the demo at 80% of its jerk: r = 0.64, inside 0.2.0's band, so
    # BOTH specs pass L4 and only the points differ. The boundary case the second rule is about.
    preds = [constant(0.2) + (t - constant(0.2)) * 0.8 for t in targets]
    old = bench_metrics(episode(preds, targets), spec_version="0.1.0")
    new = bench_metrics(episode(preds, targets), spec_version="0.2.0")

    assert MIN_SMOOTHNESS_RATIO < new.smoothness_ratio < 1.0
    assert old.rungs[4].passed is new.rungs[4].passed is True
    assert new.rungs[4].points < old.rungs[4].points
    assert new.score < old.score
    assert new.level == old.level


def test_scoring_a_run_under_a_newer_spec_leaves_the_default_untouched() -> None:
    rng = np.random.default_rng(3)
    smooth = [constant(0.2) for _ in range(4)]
    jerky = [constant(0.2) + rng.normal(0, 0.05, (STEPS, DIM)).astype(np.float32) for _ in range(4)]
    preds = episode(jerky, smooth)

    default = bench_metrics(preds)
    explicit = bench_metrics(preds, spec_version="0.1.0")
    newer = bench_metrics(preds, spec_version="0.2.0")

    assert default.spec_version == "0.1.0"
    assert default.model_dump() == explicit.model_dump()
    assert newer.spec_version == "0.2.0"
    assert newer.smoothness_ratio == pytest.approx(default.smoothness_ratio)
    assert "bench spec 0.2.0" in newer.render_markdown()
    with pytest.raises(ValueError, match="unknown bench spec"):
        bench_metrics(preds, spec_version="9.9.9")
    with pytest.raises(ValueError, match="bench spec"):
        compare_bench(default, newer)


def test_archived_bench_json_parses_as_spec_0_1_0() -> None:
    """Backward compatibility of runs/*/bench.json: no new field may be required."""
    targets = [constant(0.1) for _ in range(4)]
    report = bench_metrics(episode(targets, targets, gripper=0.48))
    payload = json.loads(report.to_json())
    for field in (
        "spec_version",
        "gripper_majority_pct",
        "gripper_transitions_per_episode",
        "gripper_accuracy_withheld_reason",
    ):
        payload.pop(field)
    payload["gripper_accuracy"] = 0.8533653846153846  # what t18's archived report carries

    restored = BenchReport.from_json(json.dumps(payload))
    assert restored.spec_version == "0.1.0"
    assert restored.gripper_accuracy == pytest.approx(0.8533653846153846)
    assert restored.gripper_majority_pct == 0.0
    assert restored.gripper_accuracy_withheld_reason == ""


def test_shift_tolerant_error_rewards_a_pure_phase_offset() -> None:
    ramp = np.arange(STEPS, dtype=np.float64)[:, None] * np.ones((1, DIM))
    shifted = np.roll(ramp, 1, axis=0)
    assert _shift_tolerant_sq(shifted, ramp, 1) < _shift_tolerant_sq(shifted, ramp, 0)
    # Same signal, no shift needed: tolerance cannot make a perfect prediction worse.
    assert _shift_tolerant_sq(ramp, ramp, 1) == pytest.approx(0.0)


def test_timing_gain_is_reported_for_a_lagging_policy() -> None:
    ramp = np.arange(STEPS, dtype=np.float32)[:, None] * np.ones((1, DIM), dtype=np.float32) * 0.1
    lagged = np.roll(ramp, 1, axis=0)
    report = bench_metrics(episode([lagged] * 4, [ramp] * 4))

    assert report.shift_tolerant_mse < report.mse
    assert report.timing_gain_pct > 0.0


def test_horizon_ratio_detects_a_chunk_that_falls_apart() -> None:
    target = constant(0.0)
    growing = np.linspace(0.0, 1.0, STEPS, dtype=np.float32)[:, None] * np.ones(
        (1, DIM), np.float32
    )
    report = bench_metrics(episode([growing] * 4, [target] * 4))

    assert report.horizon_ratio == float("inf")  # first step is exact, last is not
    assert report.rungs[3].passed is False
    assert report.rungs[3].points == 0.0


def test_smoothness_ratio_flags_a_jerkier_policy() -> None:
    rng = np.random.default_rng(1)
    smooth = [constant(0.2) for _ in range(4)]
    jerky = [constant(0.2) + rng.normal(0, 0.05, (STEPS, DIM)).astype(np.float32) for _ in range(4)]
    report = bench_metrics(episode(jerky, smooth))

    assert report.smoothness_ratio > MAX_SMOOTHNESS_RATIO
    assert report.rungs[4].passed is False


def test_multiple_episodes_do_not_leak_across_the_repeat_baseline() -> None:
    """Each episode restarts from rest; a chunk must never reference the previous episode."""
    targets = [constant(1.0), constant(1.0)]
    a = episode(targets, targets, episode_id="a")
    b = episode(targets, targets, episode_id="b")
    report = bench_metrics([*a, *b])

    assert report.num_episodes == 2
    # Both episodes contribute one leading chunk with a zeros reference and one with 1.0, so the
    # repeat baseline error is the average of (1.0 - 0.0)^2 and 0.0 -> 0.5.
    assert report.baselines.repeat_mse == pytest.approx(0.5)


def test_rejects_empty_and_out_of_range_quantile() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        bench_metrics([])
    targets = [constant(0.1)]
    with pytest.raises(ValueError, match="critical_quantile"):
        bench_metrics(episode(targets, targets), critical_quantile=1.0)


def test_report_round_trips_and_renders() -> None:
    targets = [constant(0.1) for _ in range(4)]
    report = bench_metrics(episode(targets, targets), run_name="demo")
    restored = BenchReport.from_json(report.to_json())

    assert restored == report
    markdown = report.render_markdown()
    assert "WAM-Bench — `demo`" in markdown
    for name in RUNG_NAMES:
        assert name in markdown
    assert f"<= {MAX_HORIZON_RATIO:g}" in markdown
    assert "# WAM-Bench comparison" in compare_bench(report, restored)


def test_script_scores_a_run_directory(tmp_path: Path) -> None:
    """The default invocation scores every registered spec, each into its OWN pair of files.

    ``bench.json`` is the artifact recorded verdicts were read from, so it has to keep meaning
    the default spec forever. Naming the files is the whole mechanism: if a later spec's report
    landed on ``bench.json`` the run's headline would silently change to a rule it was never
    scored under, and neither the exit code, the stdout nor ``run_name`` would look any
    different — which is why the filename set and each file's ``spec_version`` are asserted here
    rather than the fact that something was written.
    """
    targets = [constant(0.1), constant(0.2), constant(0.3)]
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    save_predictions_jsonl(episode(targets, targets), run_dir / "predictions.jsonl")

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), str(run_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "WAM-Bench" in result.stdout

    others = [v for v in sorted(BENCH_SPECS) if v != BENCH_SPEC_DEFAULT]
    assert others, "with one registered spec this test cannot see the collision it guards"
    assert {p.name for p in run_dir.iterdir()} == {
        "predictions.jsonl",
        "bench.json",
        "bench.md",
        *(f"bench-{v}.{ext}" for v in others for ext in ("json", "md")),
    }

    written = BenchReport.from_json((run_dir / "bench.json").read_text(encoding="utf-8"))
    assert written.run_name == "run-a"
    assert written.spec_version == BENCH_SPEC_DEFAULT
    assert (run_dir / "bench.md").read_text(encoding="utf-8").startswith("# WAM-Bench")
    for version in others:
        report = BenchReport.from_json(
            (run_dir / f"bench-{version}.json").read_text(encoding="utf-8")
        )
        assert report.spec_version == version


def test_script_refuses_to_compare_different_holdouts(tmp_path: Path) -> None:
    targets = [constant(0.1), constant(0.2)]
    dirs = []
    for name, ep_id in (("run-a", "ep0"), ("run-b", "ep1")):
        d = tmp_path / name
        d.mkdir()
        save_predictions_jsonl(episode(targets, targets, episode_id=ep_id), d / "predictions.jsonl")
        dirs.append(str(d))

    result = subprocess.run(
        [sys.executable, str(_SCRIPT), *dirs, "--compare", "--no-write"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "holdout mismatch" in result.stderr


# ---------------------------------------------------------------------------------------
# WHAT smoothness_ratio ACTUALLY REPORTS ON AN ANCHORED CHUNK (2026-08-16).
#
# `jerk` is `sum(d2**2)` over the second differences WITHIN one chunk, and `d2[0]` is the only
# term containing `targets[0]`. When the chunk is built by anchoring on an observed state —
# eval_t39_baseline.commanded_to_chunk, and every policy arm that reuses it — `targets[0]` is the
# standing TRACKING ERROR (command minus current state) while `targets[t>0]` are per-step command
# increments. Two different physical quantities in one array, and the sum is over squares, so the
# larger one takes the statistic.
#
# Measured on the real PR-10 chunks by scripts/audit_smoothness_ratio.py: index 0 carries 96.8 %
# of the predicted jerk sum and 6.6 % of the target's (which is 1/14, i.e. flat, as a uniform first
# difference should be). These two tests pin the MECHANISM on synthetic data, where the command's
# high-frequency content can be held exactly fixed while the anchor moves.
#
# This is not a defect in commanded_to_chunk — `action[t] - q[t]` is the correct commanded
# displacement over step t, and under perfect tracking there is no discontinuity at all. It is a
# statement about what the METRIC reports on such a chunk, and it is why PR-10 and PR-10-RESULT-T-44
# both read "8x jerkier" as high-frequency content when PR-11's 1 Hz low-pass then moved it 5 %.
# ---------------------------------------------------------------------------------------

_ANCHORED_STEPS = 16
_INCREMENT = 0.002        # a per-step command increment, rad — the corpus's order of magnitude
_TRACKING_ERROR = 0.03    # a standing command-minus-state offset, rad — likewise


def _anchored(anchor_error: float, *, increments: np.ndarray | None = None) -> np.ndarray:
    """A JOINT_DELTA chunk whose element 0 is a tracking error and the rest command increments."""
    if increments is None:
        t = np.arange(_ANCHORED_STEPS, dtype=np.float64)
        increments = _INCREMENT * (1.0 + 0.3 * np.sin(t))
    out = np.repeat(np.asarray(increments, dtype=np.float64)[:, None], DIM, axis=1)
    out[0] += anchor_error
    return out.astype(np.float32)


def _smoothness(predicted: np.ndarray, target: np.ndarray) -> float:
    preds = episode([predicted], [target], stride=_ANCHORED_STEPS)
    return bench_metrics(preds).smoothness_ratio


def test_a_tracking_offset_alone_inflates_smoothness_ratio() -> None:
    """The command is byte-identical in both arms; only the anchor moved."""
    target = _anchored(0.0)
    perfect = _anchored(0.0)
    offset = _anchored(_TRACKING_ERROR)

    # Everything the word "jerk" could mean is unchanged between `perfect` and `offset`: they
    # differ in exactly one element, and it is the one the anchor sets.
    assert np.array_equal(perfect[1:], offset[1:])

    assert _smoothness(perfect, target) == pytest.approx(1.0)
    inflated = _smoothness(offset, target)
    assert inflated > 4 * MAX_SMOOTHNESS_RATIO, inflated


def test_low_passing_the_command_barely_moves_an_offset_dominated_ratio() -> None:
    """PR-11's result, as a controlled fact: filtering cannot fix what filtering does not touch."""
    target = _anchored(0.0)
    offset = _anchored(_TRACKING_ERROR)
    # The most violent low-pass available: every increment replaced by their mean, so steps 1..15
    # carry NO high-frequency content whatsoever. The anchor term is untouched, as a filter over
    # the commanded column leaves it.
    flat = np.full(_ANCHORED_STEPS, _INCREMENT, dtype=np.float64)
    filtered = _anchored(_TRACKING_ERROR, increments=flat)

    before = _smoothness(offset, target)
    after = _smoothness(filtered, target)
    # A metric reporting high-frequency content would collapse here. This one moves single digits.
    assert abs(after - before) / before < 0.05, (before, after)
    assert after > 4 * MAX_SMOOTHNESS_RATIO, after


_AUDIT = Path(__file__).resolve().parent.parent / "scripts" / "audit_smoothness_ratio.py"


def test_the_smoothness_audit_reproduces_the_metric_it_audits(tmp_path: Path) -> None:
    """An audit that computes its own slightly-different number audits nothing.

    The script's job is to split `bench_metrics`' jerk sum by within-chunk index. It earns the
    right to say where the sum comes from only by first reproducing the sum, so that is asserted
    here against a `bench.json` this test wrote — the same check the script prints as `drift`.
    """
    target = _anchored(0.0)
    offset = _anchored(_TRACKING_ERROR)
    preds = episode([offset], [target], stride=_ANCHORED_STEPS)

    run = tmp_path / "cell"
    run.mkdir()
    save_predictions_jsonl(preds, run / "predictions.jsonl")
    report = bench_metrics(preds)
    (run / "bench.json").write_text(report.to_json() + "\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(_AUDIT), str(run), "--json-out", str(tmp_path / "audit.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "MISMATCH" not in result.stdout

    audited = json.loads((tmp_path / "audit.json").read_text())[0]
    assert audited["smoothness_ratio_recomputed"] == pytest.approx(
        report.smoothness_ratio, rel=1e-12
    )
    # And the decomposition has to be load-bearing: on a chunk whose only anomaly is the anchor,
    # index 0 carries essentially the whole predicted sum and none of the target's.
    assert audited["index0_share_pred"] > 0.9
    assert audited["index0_share_target"] < 0.2
