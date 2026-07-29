"""Tests for wam.evaluation.benchmark: the WAM-Bench offline ladder (T-27)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wam.evaluation import (
    MAX_HORIZON_RATIO,
    MAX_SMOOTHNESS_RATIO,
    RUNG_NAMES,
    BenchReport,
    ChunkPrediction,
    bench_metrics,
    compare_bench,
    save_predictions_jsonl,
)
from wam.evaluation.benchmark import _causal_previous_action, _shift_tolerant_sq
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


def test_perfect_prediction_tops_the_accuracy_rungs() -> None:
    targets = [constant(0.1), constant(0.2), constant(0.3)]
    report = bench_metrics(episode(targets, targets))

    assert report.mse == pytest.approx(0.0)
    assert report.skill_vs_zero_pct == pytest.approx(100.0)
    assert report.skill_vs_repeat_pct == pytest.approx(100.0)
    assert [r.passed for r in report.rungs[:3]] == [True, True, True]


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
    assert any("NOT a grasp-success proxy" in w for w in report.warnings)


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
    written = BenchReport.from_json((run_dir / "bench.json").read_text(encoding="utf-8"))
    assert written.run_name == "run-a"
    assert (run_dir / "bench.md").read_text(encoding="utf-8").startswith("# WAM-Bench")


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
