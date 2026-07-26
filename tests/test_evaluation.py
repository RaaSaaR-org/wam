"""Tests for wam.evaluation: E1 offline metrics (T-14) + ablation harness (T-18, AC-07)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wam.evaluation import (
    VERDICT_HELPS,
    VERDICT_HURTS,
    VERDICT_NO_DIFF,
    ChunkPrediction,
    compare_runs,
    e1_metrics,
    evaluate_policy,
    holdout_split,
    load_predictions_jsonl,
    prediction_from_dict,
    prediction_to_dict,
    save_predictions_jsonl,
)
from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    Observation,
    Policy,
    RobotState,
)

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_offline.py"

DT = 0.05


def make_chunk(
    targets: np.ndarray, gripper: np.ndarray | None = None, dt_s: float = DT
) -> ActionChunk:
    t = targets.shape[0]
    if gripper is None:
        gripper = np.zeros(t, dtype=np.float32)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=targets.astype(np.float32),
        gripper_target=gripper.astype(np.float32),
        dt_s=dt_s,
    )


def make_pred(
    pred_targets: np.ndarray,
    tgt_targets: np.ndarray,
    episode_id: str = "ep0",
    t_ns: int = 0,
    pred_grip: np.ndarray | None = None,
    tgt_grip: np.ndarray | None = None,
) -> ChunkPrediction:
    return ChunkPrediction(
        predicted=make_chunk(pred_targets, pred_grip),
        target=make_chunk(tgt_targets, tgt_grip),
        episode_id=episode_id,
        t_ns=t_ns,
    )


def make_state(n: int = 2, t_ns: int = 1_000) -> RobotState:
    return RobotState(
        timestamp_ns=t_ns,
        q=np.zeros(n, dtype=np.float32),
        dq=np.zeros(n, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.zeros(1, dtype=np.float32),
    )


# --- e1_metrics: hand-verified values -----------------------------------------------------------


def test_e1_metrics_known_mse_mae() -> None:
    # T=4, D=2; error = +0.1 in dim0, -0.3 in dim1, constant over steps.
    tgt = np.zeros((4, 2), dtype=np.float32)
    prd = np.tile(np.array([0.1, -0.3], dtype=np.float32), (4, 1))
    report = e1_metrics([make_pred(prd, tgt)])
    assert report.mse == pytest.approx((0.1**2 + 0.3**2) / 2, rel=1e-6)
    assert report.mae == pytest.approx((0.1 + 0.3) / 2, rel=1e-6)
    assert report.num_predictions == 1
    assert report.num_episodes == 1
    assert report.horizon_steps == 4
    assert report.target_dim == 2
    assert report.mode == "joint_delta"


def test_e1_metrics_per_joint_and_labels() -> None:
    tgt = np.zeros((4, 2), dtype=np.float32)
    prd = np.tile(np.array([0.1, -0.3], dtype=np.float32), (4, 1))
    spec = CanonicalSpaceSpec(joint_names=("shoulder", "elbow"))
    report = e1_metrics([make_pred(prd, tgt)], spec)
    assert list(report.per_joint_mse) == ["shoulder", "elbow"]
    assert report.per_joint_mse["shoulder"] == pytest.approx(0.01, rel=1e-6)
    assert report.per_joint_mse["elbow"] == pytest.approx(0.09, rel=1e-6)
    assert report.per_joint_mae["elbow"] == pytest.approx(0.3, rel=1e-6)
    # Without a spec: generic dim labels.
    report2 = e1_metrics([make_pred(prd, tgt)])
    assert list(report2.per_joint_mse) == ["dim_0", "dim_1"]


def test_e1_metrics_per_step_error_growth() -> None:
    # T=5, D=1; error at step t is 0.1*t -> per-step MSE grows quadratically.
    t_steps = 5
    tgt = np.zeros((t_steps, 1), dtype=np.float32)
    prd = (0.1 * np.arange(t_steps, dtype=np.float32)).reshape(-1, 1)
    report = e1_metrics([make_pred(prd, tgt)])
    assert len(report.per_step_mse) == t_steps
    assert len(report.per_step_mae) == t_steps
    expected = [(0.1 * t) ** 2 for t in range(t_steps)]
    assert report.per_step_mse == pytest.approx(expected, rel=1e-5, abs=1e-12)
    assert all(a < b for a, b in zip(report.per_step_mse, report.per_step_mse[1:]))


def test_e1_metrics_gripper_accuracy() -> None:
    tgt = np.zeros((4, 1), dtype=np.float32)
    prd = np.zeros((4, 1), dtype=np.float32)
    tgt_grip = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    prd_grip = np.array([0.4, 0.6, 0.4, 0.2], dtype=np.float32)  # binarized: 0,1,0,0
    report = e1_metrics([make_pred(prd, tgt, pred_grip=prd_grip, tgt_grip=tgt_grip)])
    assert report.gripper_accuracy == pytest.approx(0.75)


def test_e1_metrics_smoothness_second_diff() -> None:
    # pred[t] = 0.1*t^2 -> second difference is constant 0.2 -> mean square 0.04.
    t_vals = 0.1 * np.arange(4, dtype=np.float32) ** 2
    prd = t_vals.reshape(-1, 1)
    tgt = np.zeros((4, 1), dtype=np.float32)
    report = e1_metrics([make_pred(prd, tgt)])
    assert report.smoothness_pred == pytest.approx(0.04, rel=1e-5)
    assert report.smoothness_target == pytest.approx(0.0, abs=1e-12)


def test_e1_metrics_per_episode_breakdown() -> None:
    tgt = np.zeros((3, 1), dtype=np.float32)
    preds = [
        make_pred(np.full((3, 1), 0.1, dtype=np.float32), tgt, episode_id="ep_a"),
        make_pred(np.full((3, 1), 0.1, dtype=np.float32), tgt, episode_id="ep_a"),
        make_pred(np.full((3, 1), 0.2, dtype=np.float32), tgt, episode_id="ep_b"),
    ]
    report = e1_metrics(preds)
    assert report.num_episodes == 2
    assert set(report.per_episode) == {"ep_a", "ep_b"}
    assert report.per_episode["ep_a"].num_chunks == 2
    assert report.per_episode["ep_a"].mse == pytest.approx(0.01, rel=1e-5)
    assert report.per_episode["ep_b"].mse == pytest.approx(0.04, rel=1e-5)
    # Overall = weighted element mean: (2*3*0.01 + 3*0.04) / 9
    assert report.mse == pytest.approx((6 * 0.01 + 3 * 0.04) / 9, rel=1e-5)


def test_e1_metrics_variable_chunk_lengths() -> None:
    tgt3 = np.zeros((3, 1), dtype=np.float32)
    tgt5 = np.zeros((5, 1), dtype=np.float32)
    preds = [
        make_pred(np.full((3, 1), 0.1, dtype=np.float32), tgt3),
        make_pred(np.full((5, 1), 0.3, dtype=np.float32), tgt5),
    ]
    report = e1_metrics(preds)
    assert report.horizon_steps == 5
    # Steps 0-2 average both predictions, steps 3-4 only the long one.
    assert report.per_step_mse[0] == pytest.approx((0.01 + 0.09) / 2, rel=1e-5)
    assert report.per_step_mse[4] == pytest.approx(0.09, rel=1e-5)


def test_e1_metrics_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="no predictions"):
        e1_metrics([])
    tgt = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        e1_metrics([make_pred(np.zeros((3, 2), dtype=np.float32), tgt)])
    bad = np.zeros((4, 2), dtype=np.float32)
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        e1_metrics([make_pred(bad, tgt)])


def test_e1_report_json_and_markdown() -> None:
    tgt = np.zeros((4, 2), dtype=np.float32)
    prd = np.tile(np.array([0.1, -0.3], dtype=np.float32), (4, 1))
    report = e1_metrics([make_pred(prd, tgt)])
    data = json.loads(report.to_json())
    assert data["mse"] == pytest.approx(report.mse)
    md = report.render_markdown()
    for heading in (
        "# E1 offline evaluation",
        "## Overall",
        "## Per joint",
        "## Per horizon step",
        "## Per episode",
    ):
        assert heading in md
    assert "| MSE |" in md and "| ep0 |" in md


# --- evaluate_policy (policy-agnostic) ----------------------------------------------------------


class _OffsetPolicy:
    """Minimal Policy: returns the stored chunk + fixed offset. No model internals."""

    def __init__(self, chunk: ActionChunk, offset: float) -> None:
        self._chunk = chunk
        self._offset = offset

    def predict(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            mode=self._chunk.mode,
            targets=self._chunk.targets + np.float32(self._offset),
            gripper_target=self._chunk.gripper_target.copy(),
            dt_s=self._chunk.dt_s,
        )


def test_evaluate_policy_collects_predictions() -> None:
    target = make_chunk(np.zeros((4, 2), dtype=np.float32))
    policy = _OffsetPolicy(target, offset=0.1)
    assert isinstance(policy, Policy)
    obs1 = Observation(images={}, state=make_state(t_ns=111), instruction="pick")
    obs2 = Observation(images={}, state=make_state(t_ns=222), instruction="pick")
    preds = evaluate_policy(policy, [(obs1, target, "ep_x"), (obs2, target, "ep_y")])
    assert [p.episode_id for p in preds] == ["ep_x", "ep_y"]
    assert [p.t_ns for p in preds] == [111, 222]
    report = e1_metrics(preds)
    assert report.mse == pytest.approx(0.1**2, rel=1e-5)
    # Pairs without explicit episode id fall back to the keyword default.
    preds2 = evaluate_policy(policy, [(obs1, target)], episode_id="fallback")
    assert preds2[0].episode_id == "fallback"


# --- holdout_split ------------------------------------------------------------------------------


def test_holdout_split_deterministic_and_partitioning() -> None:
    ids = [f"ep{i:03d}" for i in range(20)]
    train1, hold1 = holdout_split(ids, ratio=0.25, seed=7)
    train2, hold2 = holdout_split(list(reversed(ids)), ratio=0.25, seed=7)  # order-independent
    assert (train1, hold1) == (train2, hold2)
    assert len(hold1) == 5 and len(train1) == 15
    assert sorted(train1 + hold1) == sorted(ids)
    assert set(train1).isdisjoint(hold1)
    _train3, hold3 = holdout_split(ids, ratio=0.25, seed=8)
    assert hold3 != hold1  # fixed seeds chosen to differ


def test_holdout_split_edges() -> None:
    with pytest.raises(ValueError, match="ratio"):
        holdout_split(["a", "b"], ratio=1.0, seed=0)
    # Tiny ratio on >= 2 ids still yields a non-empty holdout.
    train, hold = holdout_split(["a", "b", "c"], ratio=0.01, seed=0)
    assert len(hold) == 1 and len(train) == 2
    assert holdout_split(["only"], ratio=0.5, seed=0) == (["only"], [])
    assert holdout_split([], ratio=0.5, seed=0) == ([], [])
    # Duplicates are deduplicated before splitting.
    train_d, hold_d = holdout_split(["a", "a", "b", "b"], ratio=0.5, seed=1)
    assert sorted(train_d + hold_d) == ["a", "b"]


# --- ablation (T-18, AC-07) ---------------------------------------------------------------------


def _report_with_error(error: float, grip_acc_bad_steps: int = 0):
    tgt = np.zeros((4, 2), dtype=np.float32)
    prd = np.full((4, 2), error, dtype=np.float32)
    tgt_grip = np.ones(4, dtype=np.float32)
    prd_grip = np.ones(4, dtype=np.float32)
    prd_grip[:grip_acc_bad_steps] = 0.0
    return e1_metrics([make_pred(prd, tgt, pred_grip=prd_grip, tgt_grip=tgt_grip)])


def test_ablation_verdict_helps() -> None:
    reports = {
        "action_only": _report_with_error(0.2),
        "world_action": _report_with_error(0.1),
    }
    ablation = compare_runs(reports)
    assert ablation.baseline_name == "action_only"
    assert ablation.candidate_name == "world_action"
    assert ablation.verdict == VERDICT_HELPS
    mse = ablation.metrics["mse"]
    assert mse.baseline == pytest.approx(0.04, rel=1e-5)
    assert mse.candidate == pytest.approx(0.01, rel=1e-5)
    assert mse.delta == pytest.approx(-0.03, rel=1e-5)
    assert mse.improvement_pct == pytest.approx(75.0, rel=1e-5)


def test_ablation_verdict_hurts() -> None:
    reports = {
        "action_only": _report_with_error(0.1),
        "world_action": _report_with_error(0.2),
    }
    ablation = compare_runs(reports)
    assert ablation.verdict == VERDICT_HURTS
    assert ablation.metrics["mse"].improvement_pct == pytest.approx(-300.0, rel=1e-5)


def test_ablation_verdict_no_difference_within_threshold() -> None:
    # 2% MSE gap < default 5% threshold -> not significant.
    base = _report_with_error(0.1)
    cand = _report_with_error(0.1 * np.sqrt(0.98))
    ablation = compare_runs({"action_only": base, "world_action": cand})
    assert ablation.verdict == VERDICT_NO_DIFF


def test_ablation_higher_is_better_metric_direction() -> None:
    # Candidate has worse gripper accuracy -> negative improvement for that metric.
    reports = {
        "action_only": _report_with_error(0.1, grip_acc_bad_steps=0),
        "world_action": _report_with_error(0.1, grip_acc_bad_steps=2),
    }
    ablation = compare_runs(reports)
    grip = ablation.metrics["gripper_accuracy"]
    assert grip.higher_is_better
    assert grip.baseline == pytest.approx(1.0)
    assert grip.candidate == pytest.approx(0.5)
    assert grip.improvement_pct == pytest.approx(-50.0, rel=1e-5)


def test_ablation_explicit_names_and_errors() -> None:
    a, b = _report_with_error(0.2), _report_with_error(0.1)
    ablation = compare_runs({"runA": a, "runB": b}, baseline="runA", candidate="runB")
    assert ablation.verdict == VERDICT_HELPS
    with pytest.raises(ValueError, match="cannot infer"):
        compare_runs({"runA": a, "runB": b})  # no baseline hint in names
    with pytest.raises(ValueError, match="at least two"):
        compare_runs({"runA": a})
    md = ablation.render_markdown()
    assert "**Verdict: video branch helps**" in md and "| mse |" in md


# --- serialization + CLI wiring -----------------------------------------------------------------


def test_prediction_jsonl_roundtrip(tmp_path: Path) -> None:
    tgt = np.zeros((4, 2), dtype=np.float32)
    prd = np.tile(np.array([0.1, -0.3], dtype=np.float32), (4, 1))
    original = [make_pred(prd, tgt, episode_id="ep_r", t_ns=42)]
    restored = prediction_from_dict(prediction_to_dict(original[0]))
    np.testing.assert_array_equal(restored.predicted.targets, original[0].predicted.targets)
    assert restored.episode_id == "ep_r" and restored.t_ns == 42
    assert restored.predicted.mode is ActionMode.JOINT_DELTA

    path = tmp_path / "preds.jsonl"
    save_predictions_jsonl(original, path)
    loaded = load_predictions_jsonl(path)
    assert len(loaded) == 1
    assert e1_metrics(loaded).mse == pytest.approx(e1_metrics(original).mse)

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"episode_id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        load_predictions_jsonl(bad)


def test_eval_offline_script_dashboard_and_compare(tmp_path: Path) -> None:
    tgt = np.zeros((4, 2), dtype=np.float32)
    base_preds = [make_pred(np.full((4, 2), 0.2, dtype=np.float32), tgt)]
    cand_preds = [make_pred(np.full((4, 2), 0.1, dtype=np.float32), tgt)]
    base_jsonl = tmp_path / "action_only.jsonl"
    cand_jsonl = tmp_path / "world_action.jsonl"
    save_predictions_jsonl(base_preds, base_jsonl)
    save_predictions_jsonl(cand_preds, cand_jsonl)

    base_json = tmp_path / "action_only.json"
    cand_json = tmp_path / "world_action.json"
    for jsonl, out in ((base_jsonl, base_json), (cand_jsonl, cand_json)):
        proc = subprocess.run(
            [sys.executable, str(_SCRIPT), str(jsonl), "--joint-names", "j0,j1", "--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "# E1 offline evaluation" in proc.stdout
        assert "| j0 |" in proc.stdout
        assert out.exists()

    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--compare", str(base_json), str(cand_json)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "**Verdict: video branch helps**" in proc.stdout
