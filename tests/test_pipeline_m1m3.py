"""M1-M3 integration regression (miniature D1 overfit gate pipeline).

Miniature version of ``scripts/overfit_d1.py``: synthetic D1 recording -> T-11 validation
gates -> action-only training (loss must drop) -> checkpoint traceability (AC-04) -> E1
offline eval -> ablation scaffold (AC-07). Tiny dims + short episodes keep it fast, CPU-only
and deterministic.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wam.backbones.tiny import TinyBackboneConfig
from wam.data import ValidationThresholds, list_episodes, validate_dataset
from wam.data.validation import EPISODE_GATES
from wam.decoders import ActionHeadConfig
from wam.encoders import StateMLPConfig
from wam.evaluation import (
    VERDICT_NO_DIFF,
    E1Report,
    build_eval_pairs,
    compare_runs,
    e1_metrics,
    evaluate_policy,
    holdout_split,
    load_predictions_jsonl,
    save_predictions_jsonl,
)
from wam.interfaces import ActionMode, CanonicalSpaceSpec, load_config
from wam.safety import SafetyConfig
from wam.training import (
    ActionLossWeights,
    ActionOnlyConfig,
    ActionOnlyTrainer,
    EpisodeDataset,
    TrainingMonitor,
    load_action_only_checkpoint,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "overfit_d1.py"

_spec_loader = importlib.util.spec_from_file_location("overfit_d1", _SCRIPT)
assert _spec_loader is not None and _spec_loader.loader is not None
overfit_d1 = importlib.util.module_from_spec(_spec_loader)
_spec_loader.loader.exec_module(overfit_d1)

MS = 1_000_000
EPISODES = 2
ITERATIONS = 6  # capture steps per episode -> 6 chunks, 1.2 s duration
CHUNK_STEPS = 8
IMAGE_HW = 32
TRAIN_STEPS = 50


@pytest.fixture(scope="module")
def spec() -> CanonicalSpaceSpec:
    robot_cfg = load_config(_REPO_ROOT / "configs" / "robot" / "mock.yaml")
    return CanonicalSpaceSpec(**robot_cfg["robot"]["canonical_space"])


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory, spec: CanonicalSpaceSpec) -> Path:
    """Record the miniature synthetic D1 set once for the whole module."""
    robot_cfg = load_config(_REPO_ROOT / "configs" / "robot" / "mock.yaml")
    safety_cfg = SafetyConfig.model_validate(
        load_config(_REPO_ROOT / "configs" / "safety" / "default.yaml")
    )
    root = tmp_path_factory.mktemp("d1-mini")
    ids = overfit_d1.record_d1(
        root,
        spec,
        robot_cfg["robot"].get("limits", {}),
        safety_cfg,
        episodes=EPISODES,
        iterations=ITERATIONS,
        prefix_steps=4,
        chunk_steps=CHUNK_STEPS,
        dt_s=0.05,
        image_hw=IMAGE_HW,
        seed=0,
        sync_tolerance_ns=20 * MS,
    )
    assert ids == [f"d1-{i:04d}" for i in range(EPISODES)]
    return root


def _tiny_config(steps: int = TRAIN_STEPS) -> ActionOnlyConfig:
    return ActionOnlyConfig(
        state=StateMLPConfig(embedding_dim=16, hidden_dims=(32,), num_joints=6, gripper_dims=1),
        backbone=TinyBackboneConfig(
            feature_dim=32,
            patch_size=8,
            depth=1,
            num_heads=4,
            num_frames=2,
            image_hw=(IMAGE_HW, IMAGE_HW),
            max_text_tokens=8,
            state_embedding_dim=16,
        ),
        head=ActionHeadConfig(
            feature_dim=32,
            num_steps=CHUNK_STEPS,
            target_dim=6,
            gripper_dims=1,
            mode=ActionMode.JOINT_DELTA,
            dt_s=0.05,
            hidden_dims=(32,),
        ),
        seed=0,
        lr=5e-3,
        batch_size=ITERATIONS,
        steps=steps,
        weights=ActionLossWeights(action=1.0, gripper=0.5, smoothness=0.0, limit=0.0),
    )


@pytest.fixture(scope="module")
def trained(dataset_root: Path) -> tuple[ActionOnlyTrainer, list[dict], TrainingMonitor]:
    """Train the tiny action-only model on the first episode (module-level, reused)."""
    torch.manual_seed(0)
    data = EpisodeDataset(
        dataset_root / "d1-0000",
        camera="front",
        num_frames=2,
        chunk_steps=CHUNK_STEPS,
    )
    trainer = ActionOnlyTrainer(_tiny_config())
    monitor = TrainingMonitor()
    history = trainer.train(data, monitor=monitor)
    return trainer, history, monitor


# -- 1. validation gates (T-11) ------------------------------------------------------------


def test_validation_gates_pass(dataset_root: Path) -> None:
    thresholds = ValidationThresholds(sync_tolerance_ns=20 * MS, min_episodes=EPISODES)
    report = validate_dataset(dataset_root, thresholds)
    assert report.passed, f"failed gates: {report.failed_gates()}"
    assert len(report.episodes) == EPISODES
    for episode in report.episodes:
        assert episode.passed, f"{episode.episode_id}: {episode.failed_gates()}"
        assert tuple(g.name for g in episode.gates) == EPISODE_GATES


# -- 2. action-only training overfits (T-13 trend) + traceable checkpoint (AC-04) -----------


def test_training_loss_drops(trained: tuple) -> None:
    _trainer, history, monitor = trained
    assert len(history) == TRAIN_STEPS
    initial, final = history[0], history[-1]
    assert final["total"] < 0.5 * initial["total"]
    assert final["action"] < 0.5 * initial["action"]
    assert np.isfinite([h["total"] for h in history]).all()
    assert len(monitor.history) == TRAIN_STEPS
    assert monitor.ema is not None and np.isfinite(monitor.ema)


def test_overfit_gate_helper() -> None:
    # relative criterion (5 % of initial) dominates for a big initial loss
    ok, threshold = overfit_d1.overfit_gate(0.2, 0.009, rel_pct=5.0, abs_threshold=1e-5)
    assert ok and threshold == pytest.approx(0.01)
    ok, _ = overfit_d1.overfit_gate(0.2, 0.011, rel_pct=5.0, abs_threshold=1e-5)
    assert not ok
    # absolute floor covers a degenerate (already tiny) initial loss
    ok, threshold = overfit_d1.overfit_gate(1e-6, 5e-6, rel_pct=5.0, abs_threshold=1e-5)
    assert ok and threshold == pytest.approx(1e-5)


def test_checkpoint_roundtrip_and_metadata(
    trained: tuple, dataset_root: Path, tmp_path: Path, spec: CanonicalSpaceSpec
) -> None:
    trainer, _history, monitor = trained
    snapshot_ref = overfit_d1.dataset_snapshot_hash(dataset_root)
    assert snapshot_ref.startswith("sha256:")
    path = tmp_path / "checkpoint.safetensors"
    metadata = trainer.save_checkpoint(path, run_id="m1m3-test", dataset_snapshot_ref=snapshot_ref)
    assert metadata.config_hash and metadata.checkpoint_ref == str(path)
    assert metadata.dataset_snapshot_ref == snapshot_ref

    log_path = monitor.to_jsonl(tmp_path / "training_log.jsonl", metadata)
    lines = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(lines) == TRAIN_STEPS + 1  # run_metadata line + one per step
    assert all(line["run_id"] == "m1m3-test" for line in lines)

    model, loaded_meta = load_action_only_checkpoint(path)
    assert loaded_meta.config_hash == metadata.config_hash
    pairs = overfit_d1.build_eval_pairs(dataset_root / "d1-0001", "front", CHUNK_STEPS)
    obs = pairs[0][0]
    chunk = model.predict(obs)
    assert chunk.validate(spec) == []
    reference = trainer.model.eval().predict(obs)
    np.testing.assert_allclose(chunk.targets, reference.targets, rtol=0, atol=0)


# -- 3. E1 offline eval on the holdout episode (T-14) + ablation scaffold (AC-07) -----------


def test_e1_eval_and_ablation_scaffold(
    trained: tuple, dataset_root: Path, tmp_path: Path, spec: CanonicalSpaceSpec
) -> None:
    trainer, _history, _monitor = trained
    train_ids, holdout_ids = holdout_split(
        [d.name for d in list_episodes(dataset_root)], ratio=0.5, seed=0
    )
    assert len(train_ids) == 1 and len(holdout_ids) == 1
    pairs = overfit_d1.build_eval_pairs(dataset_root / holdout_ids[0], "front", CHUNK_STEPS)
    assert len(pairs) == ITERATIONS

    trainer.model.eval()
    predictions = evaluate_policy(trainer.model, pairs)
    report = e1_metrics(predictions, spec)
    assert report.num_predictions == ITERATIONS
    assert report.num_episodes == 1
    assert report.horizon_steps == CHUNK_STEPS
    assert report.target_dim == spec.num_joints
    assert np.isfinite([report.mse, report.mae, report.gripper_accuracy]).all()
    assert set(report.per_joint_mse) == set(spec.joint_names)
    assert "# E1 offline evaluation" in report.render_markdown()
    assert E1Report.from_json(report.to_json()) == report

    jsonl = tmp_path / "predictions.jsonl"
    save_predictions_jsonl(predictions, jsonl)
    loaded = load_predictions_jsonl(jsonl)
    assert len(loaded) == len(predictions)
    np.testing.assert_allclose(loaded[0].predicted.targets, predictions[0].predicted.targets)

    ablation = compare_runs({"action_only": report, "world_action_candidate": report})
    assert ablation.baseline_name == "action_only"
    assert ablation.verdict == VERDICT_NO_DIFF
    assert ablation.metrics["mse"].delta == 0.0


# -- 4. CLI smoke: scripts/overfit_d1.py end to end -----------------------------------------


def test_overfit_d1_main_smoke(tmp_path: Path) -> None:
    out = tmp_path / "d1"
    run_dir = tmp_path / "runs"
    rc = overfit_d1.main(
        [
            "--out",
            str(out),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "smoke",
            "--episodes",
            "2",
            "--holdout",
            "1",
            "--iterations",
            str(ITERATIONS),
            "--steps",
            "8",
            "--batch-size",
            "4",
            "--image-hw",
            str(IMAGE_HW),
            "--gate-abs",
            "1.0",  # relaxed gate: this smoke test checks plumbing, not overfit
        ]
    )
    assert rc == 0
    produced = run_dir / "smoke"
    for name in (
        "checkpoint.safetensors",
        "run_metadata.json",
        "training_log.jsonl",
        "validation_report.json",
        "predictions.jsonl",
        "e1_action_only.json",
        "e1_action_only.md",
        "ablation_scaffold.json",
    ):
        assert (produced / name).is_file(), f"missing artifact {name}"
    metadata = json.loads((produced / "run_metadata.json").read_text())
    assert metadata["run_id"] == "smoke"
    assert metadata["dataset_snapshot_ref"] == overfit_d1.dataset_snapshot_hash(out)
    e1 = E1Report.from_json((produced / "e1_action_only.json").read_text())
    assert e1.num_episodes == 1


# -- 5. T-29 / I-7: the eval window IS the training window -----------------------------------
#
# The defect these tests guard against shipped and produced a recorded verdict: EpisodeDataset fed
# the real num_frames window ending at the chunk while predict() tiled a single frame, so every
# world-action number was measured on a freeze-frame. Nothing compared the two paths.

_HISTORY_FRAMES = 3


@pytest.fixture
def moving_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stamp frame ``i`` with the value ``i`` so a window's identity is visible in its content.

    Necessary, not decorative: the synthetic D1 recorder writes a **constant** image for every
    frame (max abs difference between consecutive frames is 0), so on the stock fixture a window
    misaligned in time is byte-identical to a correct one. A test that cannot fail is part of how
    T-29 survived to a published number, so these tests substitute frames that actually move.
    """
    from wam.data.episode import EpisodeReader

    original = EpisodeReader.read_frames

    def stamped(self: EpisodeReader, camera: str) -> np.ndarray:
        frames = np.array(original(self, camera), copy=True)
        for i in range(frames.shape[0]):
            frames[i] = i % 256
        return frames

    monkeypatch.setattr(EpisodeReader, "read_frames", stamped)


def _training_windows(episode_dir: Path) -> list[np.ndarray]:
    dataset = EpisodeDataset(
        [episode_dir],
        camera="front",
        num_frames=_HISTORY_FRAMES,
        chunk_steps=CHUNK_STEPS,
        verify_checksums=False,
    )
    return [np.asarray(dataset[i]["frames"]) for i in range(len(dataset))]


def _eval_windows(episode_dir: Path) -> list[np.ndarray]:
    pairs = build_eval_pairs(episode_dir, "front", CHUNK_STEPS, num_frames=_HISTORY_FRAMES)
    return [np.asarray(obs.image_history["front"]) for obs, _target, _ep in pairs]


def test_eval_frame_window_is_byte_identical_to_the_training_window(
    dataset_root: Path, moving_frames: None
) -> None:
    """THE regression guard for T-29: same chunk, same frames. Otherwise the eval measures a
    different input than the weights were fitted on, and the resulting number means nothing."""
    training = _training_windows(dataset_root / "d1-0001")
    evaluation = _eval_windows(dataset_root / "d1-0001")

    assert len(training) == len(evaluation) == ITERATIONS
    for i, (train_window, eval_window) in enumerate(zip(training, evaluation, strict=True)):
        assert train_window.shape == (_HISTORY_FRAMES, IMAGE_HW, IMAGE_HW, 3)
        np.testing.assert_array_equal(eval_window, train_window, err_msg=f"chunk {i}")

    # and the windows genuinely differ from one another, i.e. the comparison above had content
    # to disagree about
    assert len({w[-1].flat[0] for w in evaluation}) > 1


def test_eval_history_is_the_frames_ending_at_the_observation_clamped_at_the_start(
    dataset_root: Path, moving_frames: None
) -> None:
    """Content == index, so the selected indices can be asserted directly: the window ends AT the
    observed frame (never after — that would read the future) and repeats frame 0 at the start."""
    pairs = build_eval_pairs(
        dataset_root / "d1-0001", "front", CHUNK_STEPS, num_frames=_HISTORY_FRAMES
    )
    for obs, _target, _ep in pairs:
        window = np.asarray(obs.image_history["front"])
        selected = [int(frame.flat[0]) for frame in window]
        last = int(np.asarray(obs.images["front"]).flat[0])
        assert selected[-1] == last, "history must end at the observed frame"
        expected = [max(last - (len(window) - 1 - k), 0) for k in range(len(window))]
        assert selected == expected

    first = np.asarray(pairs[0][0].image_history["front"])
    np.testing.assert_array_equal(first[0], first[1])  # clamped at the episode start


def test_history_off_by_default_so_archived_runs_stay_reproducible(dataset_root: Path) -> None:
    """Every result recorded before 2026-07-30 was measured on the tiled path. The default keeps
    reproducing them; the real window is opt-in and the A/B between the two is the experiment."""
    pairs = build_eval_pairs(dataset_root / "d1-0001", "front", CHUNK_STEPS)
    assert all(obs.image_history is None for obs, _t, _e in pairs)


def test_tiled_and_windowed_reach_the_policy_as_different_clips(
    dataset_root: Path, moving_frames: None
) -> None:
    """The premise of the experiment, on the real predict() path: without history the model sees
    one still N times (no motion at all); with it, N distinct frames."""
    from wam.training._utils import resolve_frame_context

    tiled = resolve_frame_context(
        build_eval_pairs(dataset_root / "d1-0001", "front", CHUNK_STEPS)[-1][0],
        "front",
        _HISTORY_FRAMES,
    )
    assert tiled.shape[0] == _HISTORY_FRAMES
    for i in range(1, _HISTORY_FRAMES):
        np.testing.assert_array_equal(np.asarray(tiled[i]), np.asarray(tiled[0]))

    windowed = resolve_frame_context(
        build_eval_pairs(
            dataset_root / "d1-0001", "front", CHUNK_STEPS, num_frames=_HISTORY_FRAMES
        )[-1][0],
        "front",
        _HISTORY_FRAMES,
    )
    assert not np.array_equal(np.asarray(windowed[0]), np.asarray(windowed[-1]))
