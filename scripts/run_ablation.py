#!/usr/bin/env python3
"""T-18 real-data ablation (AC-07): world-action vs. action-only on converted real episodes.

Trains the JointTrainer (T-16 world-action model) against an EXISTING action-only baseline
run so the two runs differ only in the ablated component (the video/world branch):

- state/backbone/head configs, seed, lr, weight decay, grad clip, batch size, steps, camera
  and device are copied from the baseline checkpoint's embedded config;
- action-side loss weights match the baseline (action 1.0 / gripper 0.5, no smoothness or
  limit terms); the joint run only ADDS the world-branch terms (video, action_flow,
  action_recon, alignment) — that is the component under test;
- the train/holdout split is recovered from the baseline's predictions.jsonl (the holdout
  episode ids it was actually scored on), and the dataset content hash must match the
  baseline's dataset_snapshot_ref, or the run aborts (same-split contract of compare_runs).

Inference for E1 mirrors the baseline policy path: the single observation frame is tiled to
the backbone's num_frames context and features are taken from forward_flow at t=1 (the clean
end of the rectified flow — the same feature pathway the action head was trained behind).

Usage: .venv/bin/python scripts/run_ablation.py [--baseline-run runs/d1-full-gen-seed0]
                                                [--dataset datasets/gr00t-apple-full]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from overfit_d1 import build_eval_pairs, dataset_snapshot_hash

from wam.backbones.registry import build_backbone_config
from wam.decoders import ActionHeadConfig
from wam.encoders import ActionChunkEncoderConfig, StateMLPConfig
from wam.evaluation import (
    E1Report,
    compare_runs,
    e1_metrics,
    evaluate_policy,
    load_predictions_jsonl,
    save_predictions_jsonl,
)
from wam.evaluation.ablation import DEFAULT_THRESHOLD_PCT
from wam.interfaces import ActionChunk, Observation
from wam.training import EpisodeDataset, JointTrainer, TrainingMonitor
from wam.training._utils import load_checkpoint_raw
from wam.training.joint import JointLossWeights, JointTrainingConfig, JointWorldActionModel

_REPO_ROOT = Path(__file__).resolve().parent.parent


class JointPolicy:
    """Policy-protocol wrapper for a trained JointWorldActionModel.

    Mirrors ActionOnlyModel.predict: one observation frame tiled to num_frames. Features come
    from forward_flow at t=1 — video_t == the clean tiled context there, and the action head
    was trained on pooled forward_flow features, so this stays on the training pathway.
    """

    def __init__(self, model: JointWorldActionModel) -> None:
        self.model = model

    @torch.no_grad()
    def predict(self, observation: Observation) -> ActionChunk:
        cfg = self.model.config
        camera = cfg.camera
        if camera not in observation.images:
            raise KeyError(
                f"observation has no camera {camera!r}; have {sorted(observation.images)}"
            )
        image = torch.as_tensor(observation.images[camera])
        frames = image.unsqueeze(0).expand(cfg.backbone.num_frames, -1, -1, -1)
        state_emb = self.model.state_encoder.encode(observation.state)
        text_ctx = self.model.backbone.condition_text(observation.instruction)
        state_ctx = self.model.backbone.condition_state(state_emb)
        t = torch.ones(1)
        _, feats = self.model.backbone.forward_flow(frames.unsqueeze(0), t, text_ctx, state_ctx)
        pooled = feats.mean(dim=1)  # same pooling as co_denoise -> action_head
        return self.model.action_head.decode(pooled[0])


def build_joint_config(base_cfg: dict, args: argparse.Namespace) -> JointTrainingConfig:
    """Joint config that copies every baseline choice and only adds the world branch."""
    head = ActionHeadConfig(**base_cfg["head"])
    return JointTrainingConfig(
        state=StateMLPConfig(**base_cfg["state"]),
        # Dispatch on the embedded `kind` instead of assuming tiny: backbone configs are a
        # discriminated union with extra="forbid", so a checkpoint written by a Wan run would
        # otherwise blow up inside TinyBackboneConfig on its own fields.
        backbone=build_backbone_config(base_cfg["backbone"]),
        action_encoder=ActionChunkEncoderConfig(
            latent_dim=32,
            hidden_dims=(64,),
            target_dim=head.target_dim,
            gripper_dims=1,
            max_steps=max(32, head.num_steps),
        ),
        head=head,
        velocity_hidden_dims=(64,),
        seed=int(base_cfg["seed"]),
        device=args.device or base_cfg["device"],
        lr=float(base_cfg["lr"]),
        weight_decay=float(base_cfg["weight_decay"]),
        grad_clip=float(base_cfg["grad_clip"]),
        batch_size=int(base_cfg["batch_size"]),
        steps=args.steps or int(base_cfg["steps"]),
        weights=JointLossWeights(
            video=1.0,
            action_flow=1.0,
            action_recon=1.0,
            action_reg=float(base_cfg["weights"]["action"]),
            gripper=float(base_cfg["weights"]["gripper"]),
            alignment=0.1,
            smoothness=float(base_cfg["weights"]["smoothness"]),
            limit=float(base_cfg["weights"]["limit"]),
        ),
        limit_margin=float(base_cfg["limit_margin"]),
        camera=base_cfg["camera"],
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-run", type=Path, default=_REPO_ROOT / "runs" / "d1-full-gen-seed0"
    )
    parser.add_argument(
        "--dataset", type=Path, default=_REPO_ROOT / "datasets" / "gr00t-apple-full"
    )
    parser.add_argument("--run-dir", type=Path, default=_REPO_ROOT / "runs")
    parser.add_argument("--run-id", type=str, default="t18-real-ablation-seed0")
    parser.add_argument("--steps", type=int, default=None, help="override baseline step count")
    parser.add_argument("--device", type=str, default=None, help="override baseline device")
    parser.add_argument("--threshold-pct", type=float, default=DEFAULT_THRESHOLD_PCT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_dir = args.run_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Baseline artifacts: embedded config, E1 report, holdout ids, dataset hash ---------
    _, base_cfg, base_meta = load_checkpoint_raw(args.baseline_run / "checkpoint.safetensors")
    baseline_e1 = E1Report.from_json(
        (args.baseline_run / "e1_action_only.json").read_text(encoding="utf-8")
    )
    baseline_preds = load_predictions_jsonl(args.baseline_run / "predictions.jsonl")
    holdout_ids = sorted({p.episode_id for p in baseline_preds})

    snapshot_ref = dataset_snapshot_hash(args.dataset)
    if base_meta.dataset_snapshot_ref != snapshot_ref:
        raise SystemExit(
            f"dataset hash mismatch: baseline trained on {base_meta.dataset_snapshot_ref}, "
            f"{args.dataset} is {snapshot_ref} — same-split contract violated"
        )

    episode_ids = sorted(d.name for d in args.dataset.iterdir() if (d / "manifest.json").is_file())
    missing = set(holdout_ids) - set(episode_ids)
    if missing:
        raise SystemExit(f"baseline holdout episodes missing from dataset: {sorted(missing)}")
    train_ids = [eid for eid in episode_ids if eid not in set(holdout_ids)]
    print(
        f"baseline {base_meta.run_id}: split train {len(train_ids)} / holdout {len(holdout_ids)}, "
        f"dataset hash OK ({snapshot_ref[:18]}...)"
    )

    # 2. Train the world-action candidate on the identical split ---------------------------
    config = build_joint_config(base_cfg, args)
    dataset = EpisodeDataset(
        [args.dataset / eid for eid in train_ids],
        camera=config.camera,
        num_frames=config.backbone.num_frames,
        chunk_steps=config.head.num_steps,
    )
    trainer = JointTrainer(config)
    monitor = TrainingMonitor()
    print(
        f"train world-action: {len(dataset)} samples, {config.steps} steps, "
        f"batch {config.batch_size}, device {config.device}"
    )
    history = trainer.train(dataset, monitor=monitor)
    for i in range(0, len(history), max(1, len(history) // 20)):
        h = history[i]
        print(
            f"  step {int(h['step']):5d} total {h['total']:.5g} video {h['video']:.5g} "
            f"action_reg {h['action_reg']:.5g} gripper {h['gripper']:.5g}"
        )
    initial_action, final_action = (
        history[0]["action_reg"],
        float(np.mean([h["action_reg"] for h in history[-5:]])),
    )
    print(f"action_reg {initial_action:.6g} -> {final_action:.6g}")

    # 3. Traceability (AC-04) --------------------------------------------------------------
    git_commit = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or None
    )
    metadata = trainer.save_checkpoint(
        run_dir / "checkpoint.safetensors",
        run_id=args.run_id,
        dataset_snapshot_ref=snapshot_ref,
        git_commit=git_commit,
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    monitor.to_jsonl(run_dir / "training_log.jsonl", metadata)

    # 4. E1 on the baseline's holdout episodes ---------------------------------------------
    first = args.dataset / holdout_ids[0]
    from wam.data import EpisodeReader

    spec = EpisodeReader(first).manifest.spec
    pairs = []
    for eid in holdout_ids:
        pairs.extend(build_eval_pairs(args.dataset / eid, config.camera, config.head.num_steps))
    trainer.model.eval()
    predictions = evaluate_policy(JointPolicy(trainer.model), pairs)
    e1 = e1_metrics(predictions, spec)
    save_predictions_jsonl(predictions, run_dir / "predictions.jsonl")
    (run_dir / "e1_world_action.json").write_text(e1.to_json() + "\n")
    (run_dir / "e1_world_action.md").write_text(e1.render_markdown())

    # 5. AC-07 verdict ---------------------------------------------------------------------
    ablation = compare_runs(
        {"action_only": baseline_e1, "world_action": e1},
        baseline="action_only",
        candidate="world_action",
        threshold_pct=args.threshold_pct,
    )
    (run_dir / "ablation.json").write_text(ablation.to_json() + "\n")
    (run_dir / "ablation.md").write_text(ablation.render_markdown())
    print(ablation.render_markdown())
    print(
        f"T-18 | holdout {len(holdout_ids)} episodes / {len(pairs)} chunks | "
        f"baseline mse={baseline_e1.mse:.6g} candidate mse={e1.mse:.6g} | "
        f"verdict: {ablation.verdict} | run={run_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
