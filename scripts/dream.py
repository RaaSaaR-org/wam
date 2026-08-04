#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "diffusers>=0.35.0",
#   "transformers>=4.51.0",
#   "accelerate",
#   "peft",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
#   "ftfy",
#   "numpy",
#   "pydantic>=2",
#   "pyyaml",
#   "pyarrow",
#   "opencv-python-headless",
# ]
# ///
"""Dream a short future out of the video branch and measure whether it moves (T-35).

Five arms, one report. The point of the set is that no single one of them is readable alone:

    gt          recorded frames, untouched                  (no model — runs on a laptop)
    recon       decode(encode(gt)) — the VAE round-trip      (the reference every ratio divides by)
    lora        sampled with the fine-tuned adapter          (the dream)
    base        sampled with the adapter DISABLED            (what training changed, if anything)
    base_seed1  base again at a different sampling seed      (the null d(lora, base) is compared to)

Why this exists next to the "generate future" tab (``scripts/hf_job_wan_probe.py``): that route
runs the exported LoRA inside a stock ``WanImageToVideoPipeline``, which has no state input, so
every clip measured on 2026-07-30 was generated **without the proprioception token the DiT was
trained with** (``wan_i2v.py:605``). This route integrates WAM's own flow with the text and state
context ``co_denoise`` builds — the model as trained. That is the whole reason to spend a GPU
minute on it; see ``src/wam/evaluation/dream.py`` for what it still cannot answer.

    # the reference number, no GPU and no Wan weights needed:
    .venv/bin/python scripts/dream.py --gt-only --dataset datasets/gr00t-apple-grip --episodes 12

    # the full comparison (needs the base weights):
    uv run scripts/dream.py --checkpoint runs/t16-lora-seed0/checkpoints/step-020000 \
        --dataset datasets/gr00t-apple-grip --episodes 8 --backbone-source /model --device cuda
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

# The job mounts this repo's `src/` — make `wam` importable before anything else.
for candidate in ("/wam-src", str(Path(__file__).resolve().parents[1] / "src")):
    if Path(candidate).is_dir() and candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np

DEFAULT_DATASET = "datasets/gr00t-apple-grip"
DEFAULT_CAMERA = "ego"
DEFAULT_STEPS = 32
DEFAULT_SEED = 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = p.add_argument_group("data")
    data.add_argument("--dataset", default=DEFAULT_DATASET, help="episode root directory")
    data.add_argument("--episodes", type=int, default=8, help="how many episodes to read")
    data.add_argument("--windows-per-episode", type=int, default=2)
    data.add_argument("--camera", default=DEFAULT_CAMERA)
    data.add_argument(
        "--keep-padded",
        action="store_true",
        help="keep start-of-episode windows whose frames are all identical (clamped by "
        "frame_window_indices); they are dropped by default — see drop_padded_windows",
    )
    data.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="pixel frames per clip; default = the checkpoint's trained geometry (or 9)",
    )

    model = p.add_argument_group("model")
    model.add_argument(
        "--checkpoint",
        default=None,
        help="WAM joint checkpoint: the step-NNNNNN directory or its model.safetensors",
    )
    model.add_argument("--backbone-source", default=None, help="local Wan snapshot dir")
    model.add_argument("--device", default=None)
    model.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Euler steps")
    model.add_argument("--seed", type=int, default=DEFAULT_SEED)
    model.add_argument(
        "--anchor",
        type=int,
        default=0,
        help="pin this many leading LATENT frames to the observation (an intervention — the "
        "faithful arm is 0; anchored frames are excluded from motion scoring)",
    )

    out = p.add_argument_group("output")
    out.add_argument("--gt-only", action="store_true", help="reference arms only: no model, no GPU")
    out.add_argument("--no-base-arm", action="store_true", help="skip the two adapter-off arms")
    out.add_argument("--out", default="runs/dream", help="output directory")
    out.add_argument("--contact-sheet", action="store_true", help="write a PNG per arm")
    out.add_argument(
        "--video",
        action="store_true",
        help="write an mp4 per arm (native 30 fps and a slow 6 fps copy) plus a stacked "
        "recon/lora/base comparison — 9 frames is 0.30 s, so the slow copy is the watchable one",
    )
    return p.parse_args(argv)


# ---- data ------------------------------------------------------------------------------


def build_batch(args: argparse.Namespace, num_frames: int) -> tuple[dict[str, Any], list[str]]:
    """Evenly-spaced windows from the first ``--episodes`` episodes, collated into one batch.

    Evenly spaced rather than random: the arms must see the SAME windows, and a seeded shuffle
    would be one more thing to keep in sync between a laptop run and a Space run for no gain.
    Episode order is ``list_episodes``' (sorted), which is a fixed, inspectable set — this is a
    diagnostic over a handful of clips, not a holdout score, so nothing here is a split claim.
    """
    import torch

    from wam.data.episode import list_episodes
    from wam.training.datasets import EpisodeDataset, collate_episode_batch

    root = Path(args.dataset)
    episode_dirs = list_episodes(root)[: args.episodes]
    if not episode_dirs:
        raise FileNotFoundError(f"no episodes under {root}")

    samples: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        dataset = EpisodeDataset(
            [episode_dir], camera=args.camera, num_frames=num_frames, verify_checksums=False
        )
        total = len(dataset)
        take = min(args.windows_per_episode, total)
        indices = np.linspace(0, total - 1, num=take, dtype=int) if take else []
        samples.extend(dataset[int(i)] for i in indices)

    if not args.keep_padded:
        samples, dropped = drop_padded_windows(samples)
        if dropped:
            # `frame_window_indices` clamps at an episode's start, so a window ending at frame 0
            # is nine copies of frame 0 and its motion is exactly 0. That is the startup
            # behaviour of a rolling buffer and it is correct — but as a reference for "how much
            # does real data move" it is an artifact, and it drags the mean down hard: on the
            # first 12 GR00T episodes it turned 1.66 into 1.11.
            print(f"  dropped {dropped} start-of-episode windows (all frames identical)")

    batch = collate_episode_batch(samples)
    return {
        key: (value.to(torch.float32) if torch.is_floating_point(value) else value)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }, [d.name for d in episode_dirs]


def drop_padded_windows(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Remove windows whose frames are all identical; returns ``(kept, dropped_count)``.

    Exact equality, not a threshold: a genuinely still clip (the arm resting after a place —
    ~0.07 on this corpus) is real data and must survive, while a clamped window is bit-identical
    by construction. Anything in between is the recording, and the reference arm's whole job is
    to include it.
    """
    kept = [s for s in samples if not _all_frames_identical(s["frames"])]
    return kept, len(samples) - len(kept)


def _all_frames_identical(frames: Any) -> bool:
    array = frames.numpy() if hasattr(frames, "numpy") else np.asarray(frames)
    return bool(array.shape[0] > 1 and np.all(array == array[0]))


CHECKPOINT_FILENAME = "model.safetensors"


def resolve_checkpoint(path: str | Path) -> str:
    """A checkpoint directory or file -> the safetensors file ``load_joint_policy`` wants.

    ``load_checkpoint_raw`` opens a FILE (the config and provenance ride in the safetensors
    metadata), but every other way of naming a checkpoint in this project is a directory: the
    trainer writes ``checkpoints/step-020000/``, and ``snapshot_download`` on the Space returns
    a repo root. Resolving in one place keeps the Space and a laptop pointing at the same tensor
    file instead of failing differently.

    ``trainer_state.pt`` (660 MB of optimizer state) is deliberately not looked for: nothing here
    resumes training, and it is the reason a checkpoint is 945 MB rather than 330.
    """
    source = Path(path)
    if source.is_dir():
        candidate = source / CHECKPOINT_FILENAME
        if not candidate.is_file():
            found = sorted(p.name for p in source.glob("*.safetensors"))
            raise FileNotFoundError(
                f"{source} has no {CHECKPOINT_FILENAME}"
                + (f" (found: {found})" if found else " and no safetensors at all")
            )
        return str(candidate)
    return str(source)


def batch_from_windows(
    windows: list[dict[str, Any]], instruction: str, *, keep_padded: bool = False
) -> dict[str, Any]:
    """``hf_job_wan_probe.build_windows`` output -> the batch :func:`sample_video` consumes.

    The ZeroGPU Space never materializes WAM episode directories: the probe path converts raw
    LeRobot parquet + mp4 straight into windows in memory. Rather than teach the Space a second
    way to assemble a batch, that representation is mapped onto the SAME keys ``EpisodeDataset``
    emits — so the tensors the Space samples from and the ones a local run samples from are
    assembled by one collate, and a discrepancy between the two would be a shape error rather
    than a quietly different number.

    ``targets``/``gripper_target`` are the chunk labels; the video branch never reads them, but
    ``collate_episode_batch`` does, and dropping them here would make the batch a different type
    depending on where it came from.
    """
    import torch

    from wam.training.datasets import collate_episode_batch

    samples: list[dict[str, Any]] = []
    for window in windows:
        state = window["state"]
        imu = np.concatenate(
            [
                np.asarray(state.imu.orientation_wxyz, dtype=np.float32).reshape(-1),
                np.asarray(state.imu.angular_velocity, dtype=np.float32).reshape(-1),
                np.asarray(state.imu.linear_acceleration, dtype=np.float32).reshape(-1),
            ]
        )
        validity = state.validity.as_dict()
        label = np.asarray(window["label"], dtype=np.float32)
        num_joints = int(np.asarray(state.q).shape[0])
        steps = label.shape[0] // (num_joints + 1)
        samples.append(
            {
                "frames": torch.from_numpy(np.ascontiguousarray(window["frames"])),
                "q": torch.as_tensor(np.asarray(state.q, dtype=np.float32)),
                "dq": torch.as_tensor(np.asarray(state.dq, dtype=np.float32)),
                "imu": torch.as_tensor(imu),
                "gripper": torch.as_tensor(np.asarray(state.gripper_state, dtype=np.float32)),
                "validity": torch.as_tensor(
                    np.asarray(
                        [validity["q"], validity["dq"], validity["imu"], validity["gripper"]],
                        dtype=bool,
                    )
                ),
                "targets": torch.as_tensor(label[: steps * num_joints].reshape(steps, num_joints)),
                "gripper_target": torch.as_tensor(label[steps * num_joints :]),
                "instruction": instruction,
            }
        )
    if not keep_padded:
        samples, _ = drop_padded_windows(samples)
    if not samples:
        raise ValueError("no usable windows left after dropping clamped start-of-episode ones")
    return collate_episode_batch(samples)


# ---- arms ------------------------------------------------------------------------------


def strip_anchor(frames: Any, anchor_latent_frames: int, temporal_stride: int) -> Any:
    """Drop the pixel frames an anchored sample copied from the observation.

    Wan's causal VAE maps latent frame 0 to pixel frame 0 and every later latent frame to
    ``temporal_stride`` pixel frames, so ``a`` anchors cover ``1 + (a-1)*stride`` pixel frames
    when ``a >= 1``. Scoring those would mix the recording's real motion into the dream's number
    — the single easiest way to make this whole measurement say the opposite of the truth.
    """
    if anchor_latent_frames <= 0:
        return frames
    covered = 1 + (anchor_latent_frames - 1) * temporal_stride
    return frames[:, covered:]


def vae_temporal_stride(model: Any) -> int:
    """The backbone's VAE temporal stride, asked of the backbone rather than assumed.

    Only :func:`strip_anchor` needs it, and only when anchoring — but a wrong stride there
    silently leaves recorded frames inside a "dream" arm, so it is read from the loaded adapter
    (Wan2.2: 4) instead of being written down twice.
    """
    adapter = getattr(model.backbone, "_adapter", model.backbone)
    return int(getattr(adapter, "_vae_temporal", 4) or 4)


def run_arms(args: argparse.Namespace, batch: dict[str, Any], model: Any) -> dict[str, Any]:
    """Sample every model arm; returns ``{arm_name: frames}`` ready for :func:`build_report`."""
    from wam.evaluation.dream import sample_video, vae_roundtrip

    stride = vae_temporal_stride(model)
    arms: dict[str, Any] = {"recon": vae_roundtrip(model, batch["frames"])}

    def sampled(seed: int) -> Any:
        return strip_anchor(
            sample_video(
                model, batch, steps=args.steps, seed=seed, anchor_latent_frames=args.anchor
            ),
            args.anchor,
            stride,
        )

    has_lora = bool(getattr(model.backbone, "lora_enabled", False))
    t0 = time.perf_counter()
    arms["lora" if has_lora else "sample"] = sampled(args.seed)
    print(f"  sampled {'lora' if has_lora else 'sample'} in {time.perf_counter() - t0:.1f}s")

    if has_lora and not args.no_base_arm:
        model.backbone.set_lora_enabled(False)
        try:
            t0 = time.perf_counter()
            arms["base"] = sampled(args.seed)
            # The null: same weights, same conditioning, different sampling noise. Without it a
            # nonzero d(lora, base) says only that the sampler is stochastic.
            arms["base_seed1"] = sampled(args.seed + 1)
            print(f"  sampled base x2 in {time.perf_counter() - t0:.1f}s")
        finally:
            model.backbone.set_lora_enabled(True)

    arms["recon"] = strip_anchor(arms["recon"], args.anchor, stride)
    return arms


# ---- artifacts -------------------------------------------------------------------------


def write_contact_sheet(frames: Any, path: Path) -> bool:
    """One clip per row, frames left to right — the eyeball check next to the numbers."""
    try:
        import cv2
    except ImportError:
        return False
    from wam.evaluation.dream import as_frames_255

    array = as_frames_255(frames)
    rows = [np.concatenate(list(clip), axis=1) for clip in array[:4]]
    sheet = np.concatenate(rows, axis=0).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)))


#: The corpus was recorded at 30.05 fps, so a 9-frame clip is 0.30 s of real time. Writing it at
#: 30 fps is honest and unwatchable; the slow mp4 exists to be looked at and says so in its name.
NATIVE_FPS = 30.0
SLOW_FPS = 6.0


def write_clip_video(frames: Any, path: Path, *, fps: float = NATIVE_FPS, gap: int = 2) -> bool:
    """Every clip of one arm, end to end, as an mp4. Returns False without cv2.

    ``gap`` blank frames separate clips: 9 frames is 0.3 s, so several clips run together read as
    one continuous shot and a viewer would see cuts as motion the model predicted.
    """
    try:
        import cv2
    except ImportError:
        return False
    from wam.evaluation.dream import as_frames_255

    array = as_frames_255(frames).clip(0, 255).astype(np.uint8)
    _, _, height, width, _ = array.shape
    spacer = np.zeros((gap, height, width, 3), dtype=np.uint8)
    strip = np.concatenate([np.concatenate([clip, spacer]) for clip in array])
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        return False
    try:
        for frame in strip:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


def write_comparison_video(arms: dict[str, Any], path: Path, *, fps: float = SLOW_FPS) -> bool:
    """The arms stacked in one frame, same clip and timestep in every panel.

    The comparison this whole task exists to make, and it only works side by side: `lora` against
    `recon` is the ratio in G1 made visible, and `base` in the same frame is what the adapter is
    doing to a prior that cannot generate at this geometry at all.
    """
    try:
        import cv2
    except ImportError:
        return False
    from wam.evaluation.dream import as_frames_255

    order = [name for name in ("recon", "lora", "base") if name in arms]
    if len(order) < 2:
        return False
    stacks = {name: as_frames_255(arms[name]).clip(0, 255).astype(np.uint8) for name in order}
    clips = min(s.shape[0] for s in stacks.values())
    frames_n, height, width, _ = stacks[order[0]].shape[1:]
    label_h = 16
    panels_h = (height + label_h) * len(order)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, panels_h))
    if not writer.isOpened():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        for clip in range(clips):
            for step in range(frames_n):
                panels = []
                for name in order:
                    bar = np.zeros((label_h, width, 3), dtype=np.uint8)
                    cv2.putText(
                        bar, name, (4, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
                    )
                    panels += [bar, cv2.cvtColor(stacks[name][clip, step], cv2.COLOR_RGB2BGR)]
                writer.write(np.concatenate(panels, axis=0))
    finally:
        writer.release()
    return path.is_file() and path.stat().st_size > 0


# ---- entry -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from wam.evaluation.dream import DREAM_VERSION, build_report, measure_clips

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "args": vars(args).copy(),
        "dream_version": DREAM_VERSION,
    }

    # ---- gt-only: the reference number, no torch model and no Wan weights ----------------
    if args.gt_only:
        num_frames = args.num_frames or 9
        batch, episode_names = build_batch(args, num_frames)
        info["episodes"] = episode_names
        gt = measure_clips(batch["frames"], arm="gt")
        report = {
            "version": DREAM_VERSION,
            "arms": {"gt": gt.model_dump()},
            "reference_arm": None,
            "info": info,
        }
        (out_dir / "dream_gt.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["arms"], indent=2))
        print(f"\nreport -> {out_dir / 'dream_gt.json'}")
        return 0

    if not args.checkpoint:
        print("--checkpoint is required unless --gt-only", file=sys.stderr)
        return 2

    import torch

    from wam.runtime.policies import load_joint_policy

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    checkpoint = resolve_checkpoint(args.checkpoint)
    policy = load_joint_policy(checkpoint, device=device, backbone_source=args.backbone_source)
    model = policy.model
    model.eval()
    print(f"loaded {checkpoint} on {device} in {time.perf_counter() - t0:.1f}s")

    num_frames = args.num_frames or int(model.config.backbone.num_frames)
    if args.num_frames and args.num_frames != int(model.config.backbone.num_frames):
        # The archived generate-tab table's headline finding was that this adapter is
        # geometry-bound: at 4x its trained resolution it destroys the prior, and the base
        # produces noise at the trained one. A geometry override is legitimate but never neutral.
        print(
            f"WARNING: sampling at {args.num_frames} frames, trained at "
            f"{model.config.backbone.num_frames} — out-of-distribution, label the result",
            file=sys.stderr,
        )
    batch, episode_names = build_batch(args, num_frames)
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    info["episodes"] = episode_names
    info["clips"] = int(batch["frames"].shape[0])
    info["num_frames"] = num_frames
    info["trained_num_frames"] = int(model.config.backbone.num_frames)
    info["image_hw"] = list(getattr(model.config.backbone, "image_hw", ()) or ())
    info["checkpoint"] = checkpoint
    info["run_id"] = getattr(policy.metadata, "run_id", None)
    info["config_hash"] = getattr(policy.metadata, "config_hash", None)
    info["anchor_latent_frames"] = args.anchor
    info["steps"] = args.steps
    info["state_conditioned"] = True  # the difference from the generate-future tab

    arms = run_arms(args, batch, model)
    # Raw recording, for context only: it sits on the dataset's own pixel grid (GR00T 120x160),
    # not the DiT-legal one the decoded arms come back on, so its ratio is not a like-for-like
    # comparison — that is what `recon` is for.
    arms["gt"] = strip_anchor(batch["frames"], args.anchor, vae_temporal_stride(model))

    pairs: dict[str, tuple[str, str]] = {}
    if "lora" in arms and "base" in arms:
        pairs["lora_vs_base"] = ("lora", "base")
    if "base" in arms and "base_seed1" in arms:
        pairs["base_seed_null"] = ("base", "base_seed1")

    if torch.cuda.is_available():
        info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report = build_report(arms, reference_arm="recon", pairs=pairs, info=info)

    (out_dir / "dream.json").write_text(report.model_dump_json(indent=2))
    if args.contact_sheet:
        for name, frames in arms.items():
            if write_contact_sheet(frames, out_dir / f"{name}.png"):
                print(f"  contact sheet -> {out_dir / f'{name}.png'}")
    if args.video:
        for name, frames in arms.items():
            for fps, tag in ((NATIVE_FPS, ""), (SLOW_FPS, "_slow")):
                target = out_dir / f"{name}{tag}.mp4"
                if write_clip_video(frames, target, fps=fps):
                    print(f"  video -> {target}")
        if write_comparison_video(arms, out_dir / "comparison_slow.mp4"):
            print(f"  video -> {out_dir / 'comparison_slow.mp4'}")

    print("\n===== DREAM =====")
    for name, metrics in report.arms.items():
        ratio = report.motion_ratio.get(name)
        print(
            f"  {name:<12} motion {metrics.motion:8.3f}  ratio {ratio:6.3f}  "
            f"static {metrics.static_fraction:5.2f}  std {metrics.pixel_std:6.2f}"
        )
    for label, value in report.pair_distance.items():
        print(f"  {label:<12} {value:.4f}")
    for name, verdict in report.verdicts.items():
        print(f"  VERDICT {name}: {verdict}")
    print(f"\nreport -> {out_dir / 'dream.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
