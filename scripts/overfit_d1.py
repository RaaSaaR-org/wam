#!/usr/bin/env python3
"""D1 overfit gate (T-13, M2 exit criterion): full M1-M3 pipeline on synthetic data.

Pipeline (PRD §14 M2, principle P6 "overfit first"):
1. Record a synthetic D1 dataset: N episodes via MockCaptureSession (MockRobot + per-seed
   DummyPolicy variants, a few seconds each).
2. Run the T-11 dataset validation gates — ALL must pass.
3. Train the action-only baseline (T-13) on the train split with a TrainingMonitor (R-07).
   Overfit gate: final action loss <= max(rel% of initial action loss, absolute floor).
   The absolute floor (default 1e-5, i.e. RMSE ~3e-3 in normalized units) sits an order of
   magnitude below the ~2e-2 per-step delta signal of the demo policy, so passing it means
   the model genuinely memorized D1 rather than predicting zeros.
4. E1 offline eval (T-14) on held-out episodes; E1Report JSON + markdown under runs/.
   The report is named ``e1_action_only.json`` so the M3 ablation (AC-07) can consume it as
   the auto-detected baseline via ``scripts/eval_offline.py --compare``.
5. RunMetadata with config hash, checkpoint_ref and dataset_snapshot_ref (AC-04).
6. One-line summary; exit 0 iff validation gates + overfit gate pass.

Usage: .venv/bin/python scripts/overfit_d1.py --steps 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wam.backbones.tiny import TinyBackboneConfig
from wam.data import (
    MANIFEST_FILENAME,
    EpisodeReader,
    MockCaptureSession,
    ValidationThresholds,
    list_episodes,
    validate_dataset,
)
from wam.data.validation import DatasetReport
from wam.decoders import ActionHeadConfig
from wam.encoders import StateMLPConfig
from wam.evaluation import (
    E1Report,
    compare_runs,
    e1_metrics,
    evaluate_policy,
    holdout_split,
    save_predictions_jsonl,
)
from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    Observation,
    load_config,
)
from wam.robot import MockRobot
from wam.runtime.mock_loop import DummyPolicy
from wam.safety import SafetyConfig, SafetyLayer
from wam.training import (
    ActionLossWeights,
    ActionOnlyConfig,
    ActionOnlyTrainer,
    EpisodeDataset,
    TrainingMonitor,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Rotating short imperative instructions (German is primary, PRD §6 Sprachumfang).
_INSTRUCTIONS = (
    "Greife die rote Tasse.",
    "Stelle die Tasse auf die Markierung.",
    "Greife den blauen Würfel.",
    "Lege den Würfel in die Zielzone.",
)

# Simulated wrist-camera clock skew (2 ms): exercises sync accounting, stays in tolerance.
_WRIST_SKEW_NS = 2_000_000


def overfit_gate(
    initial_action: float, final_action: float, *, rel_pct: float, abs_threshold: float
) -> tuple[bool, float]:
    """M2 overfit gate: final action loss <= max(rel% of initial, absolute floor).

    Returns ``(passed, threshold_used)``. The relative criterion is the primary one (T-13:
    "final action loss < rel_pct% of initial"); the absolute floor covers the degenerate
    case of an already-tiny initial loss.
    """
    threshold = max(initial_action * rel_pct / 100.0, abs_threshold)
    return final_action <= threshold, threshold


def dataset_snapshot_hash(root: Path) -> str:
    """Stable content hash of a dataset directory (AC-04 dataset_snapshot_ref).

    Hashes every episode manifest (relative path + bytes). Manifests embed sha256 checksums
    of all data files, so this digest pins the full dataset content.
    """
    digest = hashlib.sha256()
    for episode_dir in list_episodes(root):
        digest.update(str(episode_dir.relative_to(root)).encode("utf-8"))
        digest.update((episode_dir / MANIFEST_FILENAME).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def record_d1(
    out: Path,
    spec: CanonicalSpaceSpec,
    limits: dict[str, Any],
    safety_cfg: SafetyConfig,
    *,
    episodes: int,
    iterations: int,
    prefix_steps: int,
    chunk_steps: int,
    dt_s: float,
    image_hw: int,
    seed: int,
    sync_tolerance_ns: int,
) -> list[str]:
    """Record the synthetic D1 set; returns the episode ids (== directory names)."""
    out.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / (dt_s * prefix_steps)  # one frame per capture step
    episode_ids: list[str] = []
    for i in range(episodes):
        episode_seed = seed + i
        rng = np.random.default_rng(episode_seed)
        # Per-seed variation, well inside the default safety limits (D1: 1 task, variants).
        amplitude_rad = float(0.05 + 0.15 * rng.random())
        period_s = float(1.6 + 1.4 * rng.random())
        gripper_period_s = float(4.0 + 2.0 * rng.random())
        initial_q = rng.uniform(-0.3, 0.3, spec.num_joints)

        robot_kwargs: dict[str, Any] = {
            "spec": spec,
            "seed": episode_seed,
            "initial_q": initial_q,
            "image_hw": (image_hw, image_hw),
        }
        for key in ("q_min", "q_max", "dq_max"):
            if key in limits:
                robot_kwargs[key] = np.asarray(limits[key], dtype=np.float64)
        robot = MockRobot(**robot_kwargs)
        policy = DummyPolicy(
            spec,
            steps=chunk_steps,
            dt_s=dt_s,
            amplitude_rad=amplitude_rad,
            period_s=period_s,
            gripper_period_s=gripper_period_s,
        )
        safety = SafetyLayer(safety_cfg, spec=spec)
        cameras = tuple(robot.render_frames(1))
        session = MockCaptureSession(
            robot,
            policy,
            safety,
            fps=fps,
            sync_tolerance_ns=sync_tolerance_ns,
            camera_offsets_ns={cam: _WRIST_SKEW_NS for cam in cameras if cam == "wrist"},
            instruction=_INSTRUCTIONS[i % len(_INSTRUCTIONS)],
        )
        episode_id = f"d1-{i:04d}"
        if (out / episode_id / MANIFEST_FILENAME).is_file():
            shutil.rmtree(out / episode_id)  # deterministic re-record on repeated runs
        result = session.record_episode(
            out / episode_id,
            episode_id,
            iterations=iterations,
            prefix_steps=prefix_steps,
            extra={
                "d_phase": "D1",
                "seed": episode_seed,
                "policy": {
                    "name": "DummyPolicy",
                    "amplitude_rad": amplitude_rad,
                    "period_s": period_s,
                    "gripper_period_s": gripper_period_s,
                    "steps": chunk_steps,
                    "dt_s": dt_s,
                },
            },
        )
        episode_ids.append(episode_id)
        print(
            f"recorded {episode_id}: iterations={result.iterations} "
            f"max_sync_error_ns={result.max_sync_error_ns} "
            f"interventions={result.interventions_total}"
        )
    return episode_ids


def build_training_config(spec: CanonicalSpaceSpec, args: argparse.Namespace) -> ActionOnlyConfig:
    """Tiny action-only config matched to the recorded D1 (dims from the canonical spec)."""
    hw = (args.image_hw, args.image_hw)
    state = StateMLPConfig(
        embedding_dim=32,
        hidden_dims=(64, 64),
        num_joints=spec.num_joints,
        gripper_dims=spec.gripper_dims,
    )
    backbone = TinyBackboneConfig(
        feature_dim=64,
        patch_size=8,
        depth=2,
        num_heads=4,
        num_frames=4,
        image_hw=hw,
        state_embedding_dim=32,
    )
    head = ActionHeadConfig(
        feature_dim=64,
        num_steps=args.chunk_steps,
        target_dim=spec.target_dim(ActionMode.JOINT_DELTA),
        gripper_dims=1,
        mode=ActionMode.JOINT_DELTA,
        dt_s=args.dt_s,
        hidden_dims=(64,),
    )
    return ActionOnlyConfig(
        state=state,
        backbone=backbone,
        head=head,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        steps=args.steps,
        loss="l2",
        weights=ActionLossWeights(action=1.0, gripper=0.5, smoothness=0.0, limit=0.0),
        camera="front",
    )


def build_eval_pairs(
    episode_dir: Path, camera: str, chunk_steps: int
) -> list[tuple[Observation, ActionChunk, str]]:
    """(Observation, target chunk, episode_id) pairs for evaluate_policy from one episode."""
    reader = EpisodeReader(episode_dir)
    frames = reader.read_frames(camera)
    frame_ts = reader.frame_timestamps(camera)
    states = reader.read_states()
    state_ts = np.asarray([s.timestamp_ns for s in states], dtype=np.int64)
    instruction = reader.manifest.instruction
    episode_id = reader.manifest.episode_id

    pairs: list[tuple[Observation, ActionChunk, str]] = []
    for chunk, _executed_prefix, ts in reader.read_actions():
        if chunk.num_steps < chunk_steps:
            continue  # same contract as EpisodeDataset: shorter chunks are skipped
        target = (
            chunk
            if chunk.num_steps == chunk_steps
            else ActionChunk(
                mode=chunk.mode,
                targets=np.asarray(chunk.targets[:chunk_steps], dtype=np.float32),
                gripper_target=np.asarray(chunk.gripper_target[:chunk_steps], dtype=np.float32),
                dt_s=chunk.dt_s,
            )
        )
        frame_idx = max(int(np.searchsorted(frame_ts, ts, side="right")) - 1, 0)
        state = states[max(int(np.searchsorted(state_ts, ts, side="right")) - 1, 0)]
        obs = Observation(images={camera: frames[frame_idx]}, state=state, instruction=instruction)
        pairs.append((obs, target, episode_id))
    return pairs


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "mock-d1")
    parser.add_argument("--run-dir", type=Path, default=_REPO_ROOT / "runs")
    parser.add_argument("--run-id", type=str, default=None, help="default: d1-overfit-seed<seed>")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=2, help="episodes held out for E1")
    parser.add_argument("--iterations", type=int, default=25, help="capture steps per episode")
    parser.add_argument("--prefix-steps", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-hw", type=int, default=64, help="square frame size (px)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sync-tolerance-ms", type=float, default=20.0)
    parser.add_argument("--gate-rel-pct", type=float, default=5.0)
    parser.add_argument("--gate-abs", type=float, default=1e-5)
    parser.add_argument(
        "--robot-config", type=Path, default=_REPO_ROOT / "configs" / "robot" / "mock.yaml"
    )
    parser.add_argument(
        "--safety-config", type=Path, default=_REPO_ROOT / "configs" / "safety" / "default.yaml"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.episodes < 2 or not (1 <= args.holdout <= args.episodes - 1):
        raise SystemExit("need --episodes >= 2 and 1 <= --holdout <= episodes-1")

    robot_cfg = load_config(args.robot_config)
    safety_cfg = SafetyConfig.model_validate(load_config(args.safety_config))
    robot_section = robot_cfg["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    limits: dict[str, Any] = robot_section.get("limits", {})
    args.dt_s = float(robot_section.get("control", {}).get("dt_s", 0.05))
    tolerance_ns = round(args.sync_tolerance_ms * 1e6)
    run_id = args.run_id or f"d1-overfit-seed{args.seed}"
    run_dir = args.run_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Record synthetic D1 ---------------------------------------------------------------
    episode_ids = record_d1(
        args.out,
        spec,
        limits,
        safety_cfg,
        episodes=args.episodes,
        iterations=args.iterations,
        prefix_steps=args.prefix_steps,
        chunk_steps=args.chunk_steps,
        dt_s=args.dt_s,
        image_hw=args.image_hw,
        seed=args.seed,
        sync_tolerance_ns=tolerance_ns,
    )

    # 2. Validation gates (T-11) -----------------------------------------------------------
    thresholds = ValidationThresholds(sync_tolerance_ns=tolerance_ns, min_episodes=args.episodes)
    report: DatasetReport = validate_dataset(args.out, thresholds)
    (run_dir / "validation_report.json").write_text(report.to_json() + "\n")
    gates_ok = report.passed
    print(f"validation gates: {'PASS' if gates_ok else 'FAIL ' + ','.join(report.failed_gates())}")
    if not gates_ok:
        for ep in report.episodes:
            if not ep.passed:
                print(f"  FAIL {ep.episode_id}: {','.join(ep.failed_gates())}")

    # 3. Train/holdout split + action-only overfit (T-13) -----------------------------------
    train_ids, holdout_ids = holdout_split(episode_ids, args.holdout / args.episodes, args.seed)
    train_dirs = [args.out / eid for eid in train_ids]
    holdout_dirs = [args.out / eid for eid in holdout_ids]
    config = build_training_config(spec, args)
    dataset = EpisodeDataset(
        train_dirs,
        camera=config.camera,
        num_frames=config.backbone.num_frames,
        chunk_steps=config.head.num_steps,
    )
    trainer = ActionOnlyTrainer(config)
    monitor = TrainingMonitor()
    history = trainer.train(dataset, monitor=monitor)
    initial_action = history[0]["action"]
    final_action = float(np.mean([h["action"] for h in history[-5:]]))
    overfit_ok, gate_threshold = overfit_gate(
        initial_action, final_action, rel_pct=args.gate_rel_pct, abs_threshold=args.gate_abs
    )
    print(
        f"train: {len(dataset)} samples from {len(train_dirs)} episodes, "
        f"{len(history)} steps | action loss {initial_action:.6g} -> {final_action:.6g} "
        f"(gate <= {gate_threshold:.3g}: {'PASS' if overfit_ok else 'FAIL'})"
    )

    # 5. Traceability first (AC-04): checkpoint + RunMetadata + training log ----------------
    snapshot_ref = dataset_snapshot_hash(args.out)
    metadata = trainer.save_checkpoint(
        run_dir / "checkpoint.safetensors",
        run_id=run_id,
        dataset_snapshot_ref=snapshot_ref,
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    monitor.to_jsonl(run_dir / "training_log.jsonl", metadata)

    # 4. E1 offline eval on the holdout episodes (T-14) -------------------------------------
    pairs: list[tuple[Observation, ActionChunk, str]] = []
    for episode_dir in holdout_dirs:
        pairs.extend(build_eval_pairs(episode_dir, config.camera, config.head.num_steps))
    trainer.model.eval()
    predictions = evaluate_policy(trainer.model, pairs)
    e1: E1Report = e1_metrics(predictions, spec)
    save_predictions_jsonl(predictions, run_dir / "predictions.jsonl")
    (run_dir / "e1_action_only.json").write_text(e1.to_json() + "\n")
    (run_dir / "e1_action_only.md").write_text(e1.render_markdown())

    # Ablation scaffold (AC-07): the action-only report doubles as its own placeholder
    # candidate until the M3 joint model produces a real one (eval_offline.py --compare).
    ablation = compare_runs({"action_only": e1, "world_action_placeholder": e1})
    (run_dir / "ablation_scaffold.json").write_text(ablation.to_json() + "\n")

    # 6. One-line summary + exit code --------------------------------------------------------
    ok = gates_ok and overfit_ok
    print(
        f"{'PASS' if ok else 'FAIL'} D1 overfit gate | episodes={args.episodes} "
        f"(train {len(train_ids)} / holdout {len(holdout_ids)}) | "
        f"gates={'OK' if gates_ok else 'FAIL'} | "
        f"action {initial_action:.6g}->{final_action:.6g} "
        f"({100.0 * final_action / max(initial_action, 1e-12):.2f}% of initial) | "
        f"E1 mse={e1.mse:.6g} mae={e1.mae:.6g} gripper_acc={e1.gripper_accuracy:.3f} | "
        f"run={run_dir}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
