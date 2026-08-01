#!/usr/bin/env python3
"""How much action information does the trained latent actually carry? (T-30 pre-flight, I-3)

The T-30 question is whether the joint model's *flow* branch predicts better chunks than its
single-shot regression head. Answering it costs a GPU eval pass per readout. This script answers
the half of it that needs no GPU at all, in seconds, on a laptop:

  ceiling  encode the DEMONSTRATED chunks with ``action_encoder`` and decode them straight back
           through ``action_recon``. That round trip is the best any sampler can do, because the
           sampler's whole job is to land on the latent the encoder would have produced. If the
           round trip is not far better than the run's own error, no sampler helps and I-3 is a
           retrain, not a decode change.

  floor    decode the per-step latent CENTROIDS instead. The velocity head has no step-index
           input — it sees ``[z_t | pooled | t]`` and nothing else — so a sampler that recovers
           only *which step* a latent belongs to, and not the per-chunk content on top of it,
           bottoms out here. Between floor and ceiling is everything the A/B can win.

Both numbers come from one checkpoint plus one archived ``predictions.jsonl``; no backbone is
built (that would mean tens of GB of Wan weights to run two MLPs), nothing touches a GPU, and the
predictions file is read for its ``target`` chunks — the demonstrations — plus the run's own
``predicted`` chunks as the reference to beat.

    scripts/check_action_latent.py \\
        --checkpoint runs/t16-lora-seed0/checkpoints/step-020000 \\
        --predictions runs/t16-lora-seed0/eval-latest/predictions.jsonl

Reference values for that exact pair, measured 2026-08-01: round-trip target MSE 8.10372e-07
against the run's 1.21027e-05 and repeat-last-action's 9.13766e-06; content-free floor 1.68201e-05,
which is *worse* than zero-delta (1.63276e-05). Re-run it on any future checkpoint for free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from wam.encoders import ActionChunkEncoder
from wam.evaluation import bench_metrics, load_predictions_jsonl
from wam.training._utils import load_checkpoint_raw
from wam.training.joint import JointTrainingConfig, build_action_recon

MODEL_FILENAME = "model.safetensors"

CEILING_MARGIN = 4.0
"""How much better than the run's own MSE the round trip must be for a sampler to be worth a GPU.

Fixed before the T-30 A/B was run. Below this the decoder, not the readout, is the binding
constraint, and swapping the readout cannot move the verdict — the honest next step is then a
retrain of the action branch, which is a different task than I-3 promises.
"""


def _load_modules(
    model_path: Path,
) -> tuple[JointTrainingConfig, ActionChunkEncoder, torch.nn.Module]:
    """``(config, action_encoder, action_recon)`` restored from a checkpoint, CPU only.

    Only the two action-branch modules are rebuilt. Loading the whole ``JointWorldActionModel``
    would construct its backbone, and for every checkpoint this script is aimed at that means
    materializing a frozen multi-GB Wan tower to run a 32->64->16 MLP.
    """
    state_dict, config_dict, _ = load_checkpoint_raw(model_path)
    config = JointTrainingConfig.model_validate(config_dict)

    encoder = ActionChunkEncoder(config.action_encoder)
    recon = build_action_recon(config.action_encoder)
    for name, module in (("action_encoder.", encoder), ("action_recon.", recon)):
        slice_ = {k[len(name) :]: v for k, v in state_dict.items() if k.startswith(name)}
        if not slice_:
            raise SystemExit(
                f"{model_path}: no '{name}*' tensors in the checkpoint. The flow readout decodes "
                f"through action_recon, so a checkpoint without it cannot be sampled at all."
            )
        module.load_state_dict(slice_)
        module.eval()
    return config, encoder, recon


def _rms(chunks: np.ndarray) -> float:
    """Root-mean-square magnitude of a stack of chunks — the mean-seeking symptom, not the error."""
    return float(np.sqrt((chunks**2).mean()))


def _smoothness(chunks: np.ndarray) -> float:
    """Mean squared second temporal difference — the same jerk measure WAM-Bench ratios."""
    if chunks.shape[1] < 3:
        return 0.0
    second = chunks[:, 2:] - 2.0 * chunks[:, 1:-1] + chunks[:, :-2]
    return float((second**2).mean())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=f"a step dir or the {MODEL_FILENAME} inside it",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="an archived predictions.jsonl — its target chunks are the demonstrations",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    model_path = args.checkpoint
    if model_path.is_dir():
        model_path = model_path / MODEL_FILENAME
    if not model_path.is_file():
        raise SystemExit(f"{model_path} missing — not a restorable checkpoint")

    config, encoder, recon = _load_modules(model_path)
    enc = config.action_encoder
    predictions = load_predictions_jsonl(args.predictions)
    if not predictions:
        raise SystemExit(f"{args.predictions} holds no predictions")

    latents, round_trip, round_trip_gripper = [], [], []
    with torch.no_grad():
        for pred in predictions:
            z = encoder.encode(pred.target)  # [T, L]
            decoded = recon(z)
            latents.append(z.numpy())
            round_trip.append(decoded[:, : enc.target_dim].numpy())
            round_trip_gripper.append(decoded[:, enc.target_dim :].mean(dim=-1).numpy())

    # float64 from here on, matching wam.evaluation — these are differences of small numbers.
    latents = np.stack(latents).astype(np.float64)  # [N, T, L]
    round_trip = np.stack(round_trip).astype(np.float64)  # [N, T, D]
    round_trip_gripper = np.stack(round_trip_gripper).astype(np.float64)
    demos = np.stack([p.target.targets for p in predictions]).astype(np.float64)
    demo_gripper = np.stack([p.target.gripper_target for p in predictions]).astype(np.float64)
    deployed = np.stack([p.predicted.targets for p in predictions]).astype(np.float64)

    # The ladder's own definitions rather than a second dialect of them, so the numbers printed
    # here can be compared against a bench.json line for line.
    bench = bench_metrics(predictions, run_name="check_action_latent")

    round_trip_mse = float(((round_trip - demos) ** 2).mean())
    round_trip_gripper_mse = float(((round_trip_gripper - demo_gripper) ** 2).mean())
    demo_jerk = _smoothness(demos)

    # Content-free floor: decode the per-step centroid, i.e. "the sampler knew which step this
    # was and nothing else". Centroids are taken over these same chunks, which makes the floor
    # OPTIMISTIC — a sampler cannot know the holdout's own per-step means.
    with torch.no_grad():
        centroid_decoded = recon(torch.as_tensor(latents.mean(axis=0), dtype=torch.float32))
    centroids = centroid_decoded[:, : enc.target_dim].numpy().astype(np.float64)
    floor_mse = float(((centroids[None] - demos) ** 2).mean())

    # How separable are the steps in latent space? The encoder adds a learned positional
    # embedding, so if the per-step clusters are far apart relative to their spread, then the
    # latent's dominant content is WHICH STEP it is — and an unconditional sampler drawing i.i.d.
    # noise per step lands in a random cluster.
    flat = latents.reshape(-1, latents.shape[-1])
    per_step = latents.mean(axis=0)
    distances = ((flat[:, None, :] - per_step[None]) ** 2).sum(axis=-1)
    nearest = distances.argmin(axis=1)
    truth = np.tile(np.arange(latents.shape[1]), latents.shape[0])
    step_accuracy = float((nearest == truth).mean())
    within = float((latents - per_step[None]).std())
    between = float(per_step.std())

    print(f"checkpoint  {model_path}")
    print(f"predictions {args.predictions}  ({len(predictions)} chunks, {bench.num_episodes} eps)")
    print(f"latent      {enc.latent_dim}-d, {latents.shape[1]} steps, std {latents.std():.4g}")
    print()
    print("target MSE against the demonstrated chunks")
    print(f"  round trip (encoder -> action_recon)   {round_trip_mse:.6g}   <- the CEILING")
    print(f"  content-free floor (step centroid)     {floor_mse:.6g}   <- the FLOOR")
    print(f"  this run's deployed readout            {bench.mse:.6g}")
    print(f"  repeat-last-action (the L1 bar)        {bench.baselines.repeat_mse:.6g}")
    print(f"  zero-delta (hold still)                {bench.baselines.zero_mse:.6g}")
    print(f"  round-trip gripper MSE                 {round_trip_gripper_mse:.6g}")
    print()
    print("jerk (mean squared second difference), as a ratio to the demonstrations")
    print(f"  round trip   {_smoothness(round_trip) / demo_jerk:.4g}")
    print(f"  deployed     {bench.smoothness_ratio:.4g}")
    print()
    # Magnitude, separately from error: a mean-seeking regressor is not merely wrong, it is
    # SMALL, and the shortfall is the symptom that says so without touching the error at all.
    print("RMS |targets| (a mean-seeking readout under-shoots the demonstrations)")
    print(
        f"  demonstrations {_rms(demos):.5g} | deployed {_rms(deployed):.5g} | "
        f"round trip {_rms(round_trip):.5g}"
    )
    print()
    print("step identifiability of the latent (why it matters: the velocity head sees no step)")
    chance = 1.0 / latents.shape[1]
    print(f"  nearest-centroid step accuracy  {step_accuracy:.4f}  (chance {chance:.4f})")
    print(f"  within-step std {within:.4g}  vs  between-step centroid std {between:.4g}")
    print()

    if round_trip_mse >= bench.mse / CEILING_MARGIN:
        print(
            f"GATE FAILED: the round trip is not {CEILING_MARGIN:g}x better than the run's own\n"
            "  error, so the latent/decoder pair is the binding constraint and no readout change\n"
            "  can move the verdict. I-3 would be a retrain of the action branch, not a decode\n"
            "  change — do not spend the GPU pass."
        )
        return 1
    if round_trip_mse >= bench.baselines.repeat_mse:
        print(
            "GATE FAILED: even a perfect sampler lands above repeat-last-action, so the flow\n"
            "  readout cannot clear WAM-Bench L1 no matter how well it is integrated."
        )
        return 1

    print(
        f"GATE PASSED: the ceiling is {bench.mse / round_trip_mse:.1f}x below the run's own error "
        f"and {bench.baselines.repeat_mse / round_trip_mse:.1f}x below\n"
        "  the L1 bar. The latent is not the bottleneck, so whatever the A/B scores is a\n"
        "  statement about the velocity head's conditioning — not about the representation and\n"
        "  not about the decoder."
    )
    if step_accuracy > 0.5 and floor_mse >= bench.baselines.zero_mse:
        print()
        print(
            f"  Read the floor before booking the GPU, though. Step index is recoverable from the\n"
            f"  latent at {step_accuracy:.0%} accuracy, and a sampler that recovers only that "
            f"scores {floor_mse:.6g}\n"
            "  — worse than holding still. The velocity head takes no step index, so the content\n"
            "  it has to hit is a small perturbation riding on a much larger positional signal.\n"
            "  Expect the A/B near the floor; a result near the ceiling is the surprise worth\n"
            "  writing up."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
