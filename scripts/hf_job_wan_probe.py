# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "diffusers>=0.35.0",
#   "transformers>=4.51.0",
#   "accelerate",
#   "safetensors",
#   "sentencepiece",
#   "protobuf",
#   "ftfy",
#   "numpy",
#   "pydantic>=2",
#   "pyyaml",
#   "pyarrow",
#   "opencv-python-headless",
#   "imageio",
#   "imageio-ffmpeg",
#   "huggingface_hub>=0.34",
# ]
# ///
"""Wan readout probes on real robot data + future-video generation (T-16 prep, OD-04).

Two jobs, one script — both run free on the ZeroGPU Space (``deploy/wan-smoke-space``):

1. **Probe mode** (default). The 2026-07-26 readout ablation ranked DiT blocks by
   *label-free* sensitivity and picked (20, 29). This validates that pick against real
   action labels: windows of real GR00T-G1 ego frames + robot states go through the frozen
   Wan DiT once each, every block's features are ridge-regressed onto the BC-relabeled
   action chunk, and blocks are ranked by held-out-**episode** R². A state-only ridge (raw
   q/dq/gripper) is the floor the video features must beat to prove they carry
   action-relevant signal beyond proprioception.

   ``--readout`` (T-26 / I-1) decides how a ``[B, S, D]`` activation becomes one vector, and
   scores several choices on the *same* forward passes: ``mean`` (the historical mean-pool,
   kept byte-for-byte so ``runs/wan_probe/`` stays reproducible), ``grid<R>x<C>`` (keep the
   token geometry), ``rand<N>`` (same width, geometry removed — the control that separates
   "position matters" from "more dimensions").

2. **--generate**. Sample an actual future video from the same checkpoint, conditioned on
   a real episode's first frame + an instruction (``WanImageToVideoPipeline``). Nothing WAM
   trains uses this sampling path — it exists to *see* what the backbone imagines for the
   robot's scene, and to produce presentation material.

Usage (HF Jobs / local GPU; the Space wraps the same functions):

    uv run scripts/hf_job_wan_probe.py --source /model --data-dir data/raw/gr00t_apple
    uv run scripts/hf_job_wan_probe.py --source /model --data-dir data/raw/gr00t_apple \
        --generate --gen-prompt "the robot picks up the apple and places it on the plate"
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Side-by-side layouts: scripts/ locally and on HF Jobs, flat on the Space.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for candidate in ("/wam-src", str(_HERE.parent / "src")):
    if Path(candidate).is_dir() and candidate not in sys.path:
        sys.path.insert(0, candidate)

import convert_lerobot_g1 as conv
import numpy as np

try:  # deployed side-by-side as smoke.py on the Space
    import smoke
except ImportError:
    import hf_job_wan_smoke as smoke

DEFAULT_MODEL_ID = smoke.DEFAULT_MODEL_ID
DEFAULT_INSTRUCTION = "move the apple to the plate"
NUM_JOINTS = 15
GRIPPER_DIMS = 2
# Mean-pool (the historical readout, so the recorded numbers stay the anchor), one coarse
# spatial grid, and the same-width random control that tells the two apart — see I-1 below.
DEFAULT_READOUTS = "mean,grid2x2,rand4"
# The standard Wan quality negative prompt (from the model card); generation only.
DEFAULT_NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Last report payload for in-process callers (the ZeroGPU Space); CLI runs ignore it.
LAST_REPORT: dict[str, Any] = {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_argument_group("model source")
    src.add_argument("--source", default=None, help="local snapshot dir (e.g. mounted /model)")
    src.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="hub repo id (needs --download)")
    src.add_argument("--download", action="store_true", help="allow fetching from the Hub")
    src.add_argument("--dtype", default="bfloat16")
    src.add_argument("--device", default=None, help="default: cuda if available else cpu")
    src.add_argument("--device-map", default=None, help="accelerate device_map (e.g. 'cuda')")

    data = p.add_argument_group("real data (LeRobot GR00T-G1 snapshot)")
    data.add_argument("--data-dir", required=True, help="snapshot root with data/ + videos/")
    data.add_argument("--episodes", type=int, default=12, help="use the first N episodes")
    data.add_argument("--start", type=int, default=0, help="first source episode index")
    data.add_argument("--windows-per-episode", type=int, default=8)
    data.add_argument(
        "--window-select",
        choices=("linspace", "motion"),
        default="linspace",
        help="which chunks of each episode become windows. 'linspace' spreads them evenly and "
        "at 2 per episode returns the first and last — on GR00T-AppleToPlate the two moments "
        "the arm is not in frame. 'motion' takes the highest-motion windows (a selected "
        "subpopulation, not a corpus sample) — see wam.evaluation.dream.select_windows_by_motion",
    )
    data.add_argument("--frames", type=int, default=5, help="context frames per window")
    data.add_argument("--height", type=int, default=192, help="probe frame height (mult. of 32)")
    data.add_argument("--width", type=int, default=256, help="probe frame width (mult. of 32)")
    data.add_argument("--chunk-steps", type=int, default=16, help="action label chunk length")
    data.add_argument("--instruction", default=None, help="default: task string from meta/")

    probe = p.add_argument_group("probe")
    probe.add_argument("--alphas", default="1,10,100,1000,10000", help="ridge alpha grid")
    probe.add_argument("--measured-blocks", default="20,29", help="ablation pick under test")
    probe.add_argument("--heuristic-blocks", default="15,22", help="depth-heuristic reference")
    probe.add_argument("--state-seed", type=int, default=0, help="StateMLP init seed")
    probe.add_argument("--embedding-dim", type=int, default=32, help="StateMLP embedding dim")
    probe.add_argument(
        "--readout",
        default=DEFAULT_READOUTS,
        help="comma list of token->vector readouts: 'mean' | 'grid<R>x<C>' | 'rand<N>' (I-1). "
        f"The first one is the primary (reported under info.probe). Default: {DEFAULT_READOUTS}",
    )
    probe.add_argument(
        "--readout-seed", type=int, default=0, help="permutation seed for rand<N> readouts"
    )

    gen = p.add_argument_group("generation (--generate)")
    gen.add_argument("--generate", action="store_true", help="sample a future video instead")
    gen.add_argument("--gen-prompt", default=None, help="default: the probe instruction")
    gen.add_argument("--gen-episode", type=int, default=0, help="episode for the start frame")
    gen.add_argument("--gen-frame", type=int, default=0, help="frame index for the start frame")
    gen.add_argument("--gen-num-frames", type=int, default=49, help="(F-1) %% 4 == 0")
    gen.add_argument("--gen-steps", type=int, default=35, help="denoising steps")
    gen.add_argument("--gen-height", type=int, default=480)
    gen.add_argument("--gen-width", type=int, default=640)
    gen.add_argument("--gen-seed", type=int, default=0)
    gen.add_argument("--gen-out", default="wan_future.mp4")
    gen.add_argument(
        "--gen-lora",
        default=None,
        help="HF repo id or local dir holding a scripts/export_lora.py adapter; applies a WAM "
        "fine-tune to the frozen prior. Omit for the base model.",
    )
    gen.add_argument(
        "--gen-lora-scale",
        type=float,
        default=1.0,
        help="adapter strength. 1.0 = as trained (the saved alpha/r is honoured), 0.0 = base "
        "model. Sweep it to isolate what the fine-tune changed.",
    )

    p.add_argument("--out", default="wan_probe_report.json")
    return p.parse_args(argv)


# ---- data: real windows -----------------------------------------------------------------


def _zero_imu() -> Any:
    from wam.interfaces.schema import IMUState

    return IMUState(
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        linear_acceleration=np.zeros(3, dtype=np.float32),
    )


def build_windows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Real episodes -> probe windows. CPU-only; keep it outside the GPU call.

    One window per action chunk start: ``frames`` context frames ending at the chunk start,
    the canonical RobotState there, and the flattened chunk as the label
    (``chunk_steps * 15`` joint deltas + ``chunk_steps`` gripper synergies).
    """
    from wam.interfaces.schema import RobotState, ValidityMask

    source = Path(args.data_dir)
    instructions = conv.load_instructions(source)
    episode_indices = list(range(args.start, args.start + args.episodes))
    instruction = args.instruction or instructions.get(episode_indices[0], DEFAULT_INSTRUCTION)

    windows: list[dict[str, Any]] = []
    for ep in episode_indices:
        parquet = source / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        video = (
            source
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{ep:06d}.mp4"
        )
        data = conv.read_source_episode(parquet)
        frames = conv.read_video_frames(video, (args.height, args.width))
        n = min(data["state"].shape[0], frames.shape[0])
        state43, ts_ns = data["state"][:n], data["ts_ns"][:n]
        q, grip = conv.canonical_q(state43), conv.gripper_state(state43)
        dt_s = float(np.diff(ts_ns).mean() / 1e9) if n > 1 else 1.0 / 30.0
        chunks = conv.relabel_chunks(q, grip, chunk_steps=args.chunk_steps, dt_s=dt_s)
        eligible = [(c, s) for c, s in chunks if s >= args.frames - 1]
        if not eligible:
            continue
        count = min(args.windows_per_episode, len(eligible))
        if getattr(args, "window_select", "linspace") == "motion":
            from wam.evaluation.dream import select_windows_by_motion

            # One pass over the episode's frame diffs, then a window's score is the mean of the
            # diffs it spans — identical to differencing each window separately, at 1/frames the
            # work. int16 because np.diff on uint8 wraps and a wrapped 255->0 reads as no motion.
            diffs = np.abs(np.diff(frames.astype(np.int16), axis=0)).mean(axis=(1, 2, 3))
            scores = [float(diffs[s - args.frames + 1 : s].mean()) for _, s in eligible]
            picks = select_windows_by_motion(scores, count)
        else:
            picks = np.unique(np.linspace(0, len(eligible) - 1, count).round().astype(int))
        for k in picks:
            chunk, start = eligible[int(k)]
            dq = (q[start] - q[start - 1]) / dt_s if start > 0 else np.zeros(NUM_JOINTS)
            state = RobotState(
                timestamp_ns=int(ts_ns[start]),
                q=q[start],
                dq=dq.astype(np.float32),
                imu=_zero_imu(),
                gripper_state=grip[start],
                validity=ValidityMask(q=True, dq=start > 0, imu=False, gripper=True),
            )
            label = np.concatenate(
                [
                    np.asarray(chunk.targets, dtype=np.float32).reshape(-1),
                    np.asarray(chunk.gripper_target, dtype=np.float32),
                ]
            )
            windows.append(
                {
                    "frames": frames[start - args.frames + 1 : start + 1],
                    "state": state,
                    "label": label,
                    "episode": ep,
                    "start": int(start),
                }
            )
    info = {
        "dataset": str(source),
        "episodes": episode_indices,
        "windows": len(windows),
        "window_select": getattr(args, "window_select", "linspace"),
        "frames": args.frames,
        "resize": [args.height, args.width],
        "chunk_steps": args.chunk_steps,
        "label_dim": args.chunk_steps * (NUM_JOINTS + 1),
        "instruction": instruction,
    }
    return windows, instruction, info


# ---- readouts: how a [B, S, D] activation collapses to one feature vector ------------------
#
# I-1 (docs/improvements.md). Everything recorded so far — T-15's block ranking, T-24's
# "frozen Cosmos3 does not beat state-only either" — measured the backbone *through a
# mean-pool*. Averaging the token grid deletes *where* things are, and "the cube is at token
# (4, 6)" is the single most useful item in that map for pick-and-place. So those runs did not
# show the spatial information is absent from the backbone; they showed it does not survive
# averaging. Two different claims, and the weaker one is currently written into TASKS.md as
# the stronger one. These readouts are the difference.
#
# `rand<N>` is what makes the comparison worth anything: it pools the same tokens into N groups
# of the same size, chosen by a seeded permutation instead of by geometry, so it has the
# *identical* feature width as the grid it controls. grid > rand means geometry carries signal;
# grid ~= rand means we only bought dimensions and the mean-pool verdict stands.

Readout = tuple[str, Any]


def parse_readouts(spec: str) -> tuple[Readout, ...]:
    """``"mean,grid2x2,rand4"`` -> ``(("mean", None), ("grid", (2, 2)), ("rand", 4))``.

    Order is preserved and duplicates are dropped; the first entry is the primary readout.
    """
    out: list[Readout] = []
    seen: set[str] = set()
    for raw in spec.split(","):
        name = raw.strip().lower()
        if not name or name in seen:
            continue
        seen.add(name)
        if name == "mean":
            out.append(("mean", None))
        elif name.startswith("grid"):
            parts = name[4:].split("x")
            if len(parts) != 2 or not all(p.isdigit() and int(p) > 0 for p in parts):
                raise ValueError(f"bad grid readout {raw!r} — expected e.g. 'grid2x2'")
            out.append(("grid", (int(parts[0]), int(parts[1]))))
        elif name.startswith("rand"):
            groups = name[4:]
            if not groups.isdigit() or int(groups) < 1:
                raise ValueError(f"bad rand readout {raw!r} — expected e.g. 'rand4'")
            out.append(("rand", int(groups)))
        else:
            raise ValueError(
                f"unknown readout {raw!r} — expected 'mean', 'grid<R>x<C>' or 'rand<N>'"
            )
    if not out:
        raise ValueError("--readout must name at least one readout")
    return tuple(out)


def readout_label(kind: str, param: Any) -> str:
    """Inverse of :func:`parse_readouts` for one entry — the key used in reports."""
    if kind == "mean":
        return "mean"
    if kind == "grid":
        return f"grid{param[0]}x{param[1]}"
    return f"rand{param}"


def readout_width(kind: str, param: Any, feature_dim: int) -> int:
    """Feature width of one readout: ``cells * feature_dim`` (mean has one cell)."""
    if kind == "mean":
        return feature_dim
    cells = param[0] * param[1] if kind == "grid" else int(param)
    return cells * feature_dim


def _spatial_tokens(tokens: Any, grid: tuple[int, int, int]) -> Any:
    """``[B, S, D]`` -> ``[B, R*C, D]`` in fp32, averaged over the latent-frame axis.

    Every geometry-preserving readout collapses time first: with the probe's 5 context frames
    the Wan VAE leaves F'=2, so keeping that axis would double the feature width to test a
    claim I-1 does not make. What is under test is *space*.
    """
    frames, rows, cols = grid
    batch, tokens_n, dim = tokens.shape
    if tokens_n != frames * rows * cols:
        raise ValueError(
            f"activation has {tokens_n} tokens but the grid {grid} implies {frames * rows * cols}"
        )
    collapsed = tokens.float().reshape(batch, frames, rows, cols, dim).mean(dim=1)
    return collapsed.reshape(batch, rows * cols, dim)


def apply_readouts(
    tokens: Any, grid: tuple[int, int, int], readouts: tuple[Readout, ...], *, seed: int = 0
) -> dict[str, Any]:
    """``[B, S, D]`` -> ``{label: [B, width]}``, sharing the temporal collapse across readouts."""
    import torch

    out: dict[str, Any] = {}
    flat: Any = None
    for kind, param in readouts:
        label = readout_label(kind, param)
        if kind == "mean":
            # Byte-for-byte the historical readout, so `mean` reproduces runs/wan_probe/.
            out[label] = tokens.float().mean(dim=1)
            continue
        if flat is None:
            flat = _spatial_tokens(tokens, grid)
        batch, positions, dim = flat.shape
        if kind == "grid":
            cell_rows, cell_cols = param
            _, rows, cols = grid
            if rows % cell_rows or cols % cell_cols:
                raise ValueError(
                    f"{label} does not divide the {rows}x{cols} token grid evenly — pick "
                    f"cell counts that divide it, or a different --height/--width"
                )
            cells = flat.reshape(
                batch, cell_rows, rows // cell_rows, cell_cols, cols // cell_cols, dim
            )
            out[label] = cells.mean(dim=(2, 4)).reshape(batch, cell_rows * cell_cols * dim)
            continue
        groups = int(param)
        if groups > positions:
            raise ValueError(f"{label} wants {groups} groups but only {positions} tokens exist")
        # Same group count and group size as the grid it controls, geometry replaced by chance.
        order = np.random.default_rng(seed).permutation(positions)[: positions // groups * groups]
        index = torch.as_tensor(
            np.ascontiguousarray(order.reshape(groups, -1)), device=flat.device, dtype=torch.long
        )
        out[label] = flat[:, index, :].mean(dim=2).reshape(batch, groups * dim)
    return out


# ---- GPU: per-block features under every readout ------------------------------------------


def extract_features(
    args: argparse.Namespace,
    windows: list[dict[str, Any]],
    instruction: str,
    report: smoke.Report,
) -> dict[str, np.ndarray]:
    """One frozen DiT forward per window -> ``{readout: [N, num_blocks, width]}``."""
    import torch

    from wam.backbones.registry import get_backbone
    from wam.encoders.state_mlp import StateMLP, StateMLPConfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapter = get_backbone(
        "wan_i2v",
        checkpoint_path=args.source,
        model_id=None if args.source else args.model_id,
        device=device,
        dtype=args.dtype,
        allow_download=args.download,
        device_map=args.device_map,
    )
    t0 = time.perf_counter()
    adapter.load()
    report.check("probe.load", adapter.is_loaded, f"{time.perf_counter() - t0:.1f}s")
    geometry = adapter.describe()
    blocks = tuple(range(geometry["num_layers"]))
    readouts = parse_readouts(args.readout)
    grid = adapter.token_grid(args.frames, args.height, args.width)
    geometry["token_grid"] = list(grid)
    report.info["geometry"] = geometry

    torch.manual_seed(args.state_seed)
    encoder = StateMLP(
        StateMLPConfig(
            embedding_dim=args.embedding_dim, num_joints=NUM_JOINTS, gripper_dims=GRIPPER_DIMS
        )
    )
    text_ctx = adapter.condition_text(instruction)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    features = {
        readout_label(kind, param): np.zeros(
            (len(windows), len(blocks), readout_width(kind, param, geometry["feature_dim"])),
            dtype=np.float32,
        )
        for kind, param in readouts
    }
    expected_tokens = adapter.expected_token_count(args.frames, args.height, args.width)
    checked_tokens = False
    t0 = time.perf_counter()
    for i, window in enumerate(windows):
        video_ctx = adapter.condition_video(window["frames"][None])
        state_ctx = adapter.condition_state(encoder.encode(window["state"]))
        with torch.no_grad():
            per_block = adapter.features_by_block(video_ctx, text_ctx, state_ctx, blocks=blocks)
        for j, block in enumerate(blocks):
            activation = per_block[block]
            if not checked_tokens:
                # The grid reshape is only meaningful if S is exactly F'*H'*W'. Assert it once,
                # loudly: a mismatch means every spatial readout below is nonsense.
                report.check(
                    "probe.token_count_matches_grid",
                    int(activation.shape[1]) == expected_tokens,
                    f"S={int(activation.shape[1])}, grid {grid} implies {expected_tokens}",
                )
                checked_tokens = True
            for label, value in apply_readouts(
                activation, grid, readouts, seed=args.readout_seed
            ).items():
                features[label][i, j] = value[0].cpu().numpy()
    wall = time.perf_counter() - t0
    timings = {"forwards_s": round(wall, 2), "per_window_s": round(wall / max(len(windows), 1), 3)}
    if torch.cuda.is_available():
        timings["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report.info["timings"] = timings
    report.info["readouts"] = {
        label: {"width": int(arr.shape[2]), "cells": int(arr.shape[2] // geometry["feature_dim"])}
        for label, arr in features.items()
    }

    shapes = ", ".join(f"{label}{arr.shape}" for label, arr in features.items())
    report.check(
        "probe.features_finite",
        all(bool(np.isfinite(arr).all()) for arr in features.values()),
        shapes,
    )
    spreads = {label: float(arr[:, 0].std(axis=0).mean()) for label, arr in features.items()}
    report.check(
        "probe.features_vary_across_windows",
        min(spreads.values()) > 1e-6,
        "block-0 spread " + ", ".join(f"{k}={v:.4f}" for k, v in spreads.items()),
    )
    return features


# ---- CPU: ridge probes -------------------------------------------------------------------


def _ridge_predictor(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Any:
    """Closed-form ridge via thin SVD (n << d); features standardized with train stats."""
    mu_x, sd = x_train.mean(axis=0), x_train.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    mu_y = y_train.mean(axis=0)
    xs = (x_train - mu_x) / sd
    u, s, vt = np.linalg.svd(xs, full_matrices=False)
    coef = (vt.T * (s / (s * s + alpha))) @ (u.T @ (y_train - mu_y))

    def predict(x: np.ndarray) -> np.ndarray:
        return ((x - mu_x) / sd) @ coef + mu_y

    return predict


def _r2(y_true: np.ndarray, y_pred: np.ndarray, mu_train: np.ndarray) -> float:
    """Pooled multi-output R^2 against the train-mean predictor."""
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - mu_train) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def probe_r2(
    x: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    alphas: tuple[float, ...],
) -> dict[str, float]:
    """Alpha picked on val episodes, refit on train+val, scored on test episodes."""
    tr, va, te = split["train"], split["val"], split["test"]
    scored = [
        (_r2(y[va], _ridge_predictor(x[tr], y[tr], a)(x[va]), y[tr].mean(axis=0)), a)
        for a in alphas
    ]
    val_r2, alpha = max(scored, key=lambda t: t[0])
    trval = np.concatenate([tr, va])
    predict = _ridge_predictor(x[trval], y[trval], alpha)
    test_r2 = _r2(y[te], predict(x[te]), y[trval].mean(axis=0))
    return {"val_r2": round(val_r2, 4), "test_r2": round(test_r2, 4), "alpha": alpha}


def episode_split(episode_of: np.ndarray) -> dict[str, np.ndarray]:
    """Deterministic by-episode split (no window of a test episode is ever trained on)."""
    episodes = sorted({int(e) for e in episode_of})
    if len(episodes) < 4:
        raise ValueError(f"need >= 4 episodes for a train/val/test split, got {len(episodes)}")
    n_test = max(2, round(0.25 * len(episodes)))
    n_val = max(1, round(0.15 * len(episodes)))
    test_eps, val_eps = episodes[-n_test:], episodes[-n_test - n_val : -n_test]
    train_eps = episodes[: -n_test - n_val]
    if not train_eps:
        raise ValueError("split left no training episodes — use more episodes")

    def pick(eps):
        return np.flatnonzero(np.isin(episode_of, eps))

    return {
        "train": pick(train_eps),
        "val": pick(val_eps),
        "test": pick(test_eps),
        "train_eps": np.asarray(train_eps),
        "val_eps": np.asarray(val_eps),
        "test_eps": np.asarray(test_eps),
    }


def _score_one_readout(
    pooled: np.ndarray,
    score: Any,
    measured: tuple[int, ...],
    heuristic: tuple[int, ...],
    state_row: dict[str, Any],
) -> dict[str, Any]:
    """Block ranking + candidate pairs for ONE readout's ``[N, num_blocks, width]`` features."""
    num_blocks = pooled.shape[1]
    singles = {b: score(pooled[:, b]) for b in range(num_blocks)}
    ranked = sorted(singles, key=lambda b: singles[b]["joints"]["val_r2"], reverse=True)
    suggested = tuple(sorted(ranked[:2]))

    def pair_features(blocks: tuple[int, ...]) -> np.ndarray:
        return np.concatenate([pooled[:, b] for b in blocks], axis=1)

    candidates = {
        f"measured_{'_'.join(map(str, measured))}": score(pair_features(measured)),
        f"heuristic_{'_'.join(map(str, heuristic))}": score(pair_features(heuristic)),
        f"suggested_{'_'.join(map(str, suggested))}": score(pair_features(suggested)),
        "state_only": state_row,
    }
    measured_key = f"measured_{'_'.join(map(str, measured))}"
    heuristic_key = f"heuristic_{'_'.join(map(str, heuristic))}"
    return {
        "per_block": {str(b): singles[b] for b in range(num_blocks)},
        "ranking_by_val_r2": ranked,
        "suggested_blocks": list(suggested),
        "candidates": candidates,
        "verdict": {
            "measured_beats_heuristic": candidates[measured_key]["joints"]["test_r2"]
            >= candidates[heuristic_key]["joints"]["test_r2"],
            "video_beats_state_only": candidates[measured_key]["joints"]["test_r2"]
            > candidates["state_only"]["joints"]["test_r2"],
        },
    }


def analyze_probes(
    features: np.ndarray | dict[str, np.ndarray],
    windows: list[dict[str, Any]],
    args: argparse.Namespace,
    report: smoke.Report,
) -> None:
    """Rank every block by held-out-episode R^2 against the real action labels, per readout.

    Joint deltas (~1e-2 rad) and absolute gripper synergies (~1e-1..1) live on different
    scales, so a pooled R^2 would mostly measure gripper prediction. The two groups are
    scored separately; the block ranking uses the joint-delta score — that is the quantity
    the action head must actually get right.

    ``features`` is either the ``{readout: [N, num_blocks, width]}`` dict from
    :func:`extract_features` or a bare ``[N, num_blocks, D]`` array, which is treated as a
    single ``mean`` readout (the Cosmos3 probe still hands one over). The FIRST readout is the
    primary: it lands in ``info.probe`` unchanged, so `mean` keeps reproducing the numbers in
    ``runs/wan_probe/`` while the spatial readouts are reported alongside it (I-1).
    """
    readouts = {"mean": features} if isinstance(features, np.ndarray) else dict(features)
    if not readouts:
        raise ValueError("analyze_probes needs at least one readout")
    alphas = tuple(float(a) for a in args.alphas.split(",") if a.strip())
    measured = tuple(int(b) for b in args.measured_blocks.split(","))
    heuristic = tuple(int(b) for b in args.heuristic_blocks.split(","))
    y = np.stack([w["label"] for w in windows])
    joint_dim = min(args.chunk_steps * NUM_JOINTS, y.shape[1])
    y_joint, y_grip = y[:, :joint_dim], y[:, joint_dim:]
    episode_of = np.asarray([w["episode"] for w in windows])
    split = episode_split(episode_of)
    report.check(
        "probe.split_by_episode",
        min(len(split["train"]), len(split["val"]), len(split["test"])) > 0,
        f"windows train/val/test = {len(split['train'])}/{len(split['val'])}/{len(split['test'])}",
    )
    report.check("probe.labels_nonconstant", float(y.std()) > 1e-8, f"label std {y.std():.5f}")

    def score(x: np.ndarray) -> dict[str, Any]:
        row: dict[str, Any] = {"joints": probe_r2(x, y_joint, split, alphas)}
        if y_grip.size:
            row["gripper"] = probe_r2(x, y_grip, split, alphas)
        return row

    # Proprioception is the floor every readout must clear, and it does not depend on the
    # backbone — fit it once so the comparison cannot drift between readouts.
    state_x = np.stack(
        [np.concatenate([w["state"].q, w["state"].dq, w["state"].gripper_state]) for w in windows]
    ).astype(np.float32)
    state_row = score(state_x)

    results = {
        label: _score_one_readout(pooled, score, measured, heuristic, state_row)
        for label, pooled in readouts.items()
    }
    primary = next(iter(readouts))
    singles = {int(b): row for b, row in results[primary]["per_block"].items()}
    suggested = tuple(results[primary]["suggested_blocks"])

    def fmt(row: dict[str, Any]) -> str:
        joints = f"joints val={row['joints']['val_r2']:.4f} test={row['joints']['test_r2']:.4f}"
        if "gripper" in row:
            return f"{joints} | gripper test={row['gripper']['test_r2']:.4f}"
        return joints

    print(f"\n[readout: {primary}]\nblock  joints_val_R2  joints_test_R2  gripper_test_R2")
    for b in sorted(singles):
        row = singles[b]
        grip = f"{row['gripper']['test_r2']:15.4f}" if "gripper" in row else " " * 15
        marker = (
            "  <- suggested"
            if b in suggested
            else (
                "  <- measured" if b in measured else ("  <- heuristic" if b in heuristic else "")
            )
        )
        print(
            f"{b:5d}  {row['joints']['val_r2']:13.4f}  {row['joints']['test_r2']:14.4f}  "
            f"{grip}{marker}"
        )
    print()
    for name, row in results[primary]["candidates"].items():
        print(f"{name}: {fmt(row)}")

    comparison = _compare_readouts(results, state_row)
    if len(results) > 1:
        print(
            "\nreadout comparison (best suggested pair, joints test R^2; "
            f"state-only floor {comparison['state_only_joints_test_r2']:.4f})"
        )
        for label, row in comparison["per_readout"].items():
            flag = "beats state-only" if row["beats_state_only"] else "below state-only"
            print(f"  {label:>10}: {row['suggested_joints_test_r2']:.4f}  ({flag})")
        for control in comparison["grid_vs_random_control"]:
            verdict = "geometry helps" if control["geometry_helps"] else "no geometry gain"
            print(
                f"  {control['grid']} {control['grid_r2']:.4f} vs same-width "
                f"{control['control']} {control['control_r2']:.4f} -> {verdict}"
            )

    values = [
        group[k]
        for result in results.values()
        for row in result["per_block"].values()
        for group in row.values()
        for k in ("val_r2", "test_r2")
    ]
    report.check("probe.r2_finite", bool(np.isfinite(values).all()), f"{len(values)} scores")
    best = max(
        max(group["test_r2"] for group in row.values())
        for result in results.values()
        for row in result["per_block"].values()
    )
    report.check(
        "probe.features_predict_actions",
        best > 0.0,
        f"best single-block held-out R2 {best:.4f} (0 = train-mean predictor)",
    )
    report.check(
        "probe.readouts_scored",
        set(results) == set(readouts),
        f"{len(results)} readout(s): {', '.join(results)}",
    )
    report.info["probe"] = {
        "metric": "multi-output R^2 on held-out episodes (alpha chosen on val episodes); "
        "joint deltas and gripper synergies scored separately, ranking by joints",
        "label": f"flattened BC chunk: {args.chunk_steps}x{NUM_JOINTS} joint deltas "
        f"+ {args.chunk_steps} gripper synergies",
        "split_episodes": {k: split[f"{k}_eps"].tolist() for k in ("train", "val", "test")},
        "alphas": list(alphas),
        "readout": primary,
        **results[primary],
        "readouts": results,
        "readout_comparison": comparison,
    }


def _compare_readouts(
    results: dict[str, dict[str, Any]], state_row: dict[str, Any]
) -> dict[str, Any]:
    """The I-1 headline: does any geometry-preserving readout clear the state-only floor, and
    does it clear its own same-width random control?

    Both questions are read off the *suggested* block pair — the pair each readout's own
    ranking picks — because that is what a real action head would consume.
    """
    floor = state_row["joints"]["test_r2"]
    per_readout: dict[str, Any] = {}
    for label, result in results.items():
        pair = next(k for k in result["candidates"] if k.startswith("suggested_"))
        r2 = result["candidates"][pair]["joints"]["test_r2"]
        blocks = [int(b) for b in result["per_block"]]
        best_block = max(blocks, key=lambda b: result["per_block"][str(b)]["joints"]["test_r2"])
        per_readout[label] = {
            "suggested_blocks": result["suggested_blocks"],
            "suggested_joints_test_r2": r2,
            "best_single_block": best_block,
            "best_single_block_joints_test_r2": result["per_block"][str(best_block)]["joints"][
                "test_r2"
            ],
            "beats_state_only": r2 > floor,
        }

    controls = []
    for label, row in per_readout.items():
        if not label.startswith("grid"):
            continue
        rows, cols = (int(v) for v in label[4:].split("x"))
        control = f"rand{rows * cols}"
        if control not in per_readout:
            continue
        controls.append(
            {
                "grid": label,
                "control": control,
                "grid_r2": row["suggested_joints_test_r2"],
                "control_r2": per_readout[control]["suggested_joints_test_r2"],
                "geometry_helps": row["suggested_joints_test_r2"]
                > per_readout[control]["suggested_joints_test_r2"],
            }
        )

    spatial = [lbl for lbl in per_readout if lbl.startswith("grid")]
    return {
        "primary": next(iter(results)),
        "state_only_joints_test_r2": floor,
        "per_readout": per_readout,
        "grid_vs_random_control": controls,
        "any_spatial_beats_state_only": any(
            per_readout[lbl]["beats_state_only"] for lbl in spatial
        ),
        "any_geometry_gain_over_control": any(c["geometry_helps"] for c in controls),
    }


# ---- generation (presentation material, not a training path) ----------------------------


def load_gen_frame(args: argparse.Namespace) -> np.ndarray:
    """Native-resolution RGB start frame from a raw source episode."""
    import cv2

    video = (
        Path(args.data_dir)
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / f"episode_{args.gen_episode:06d}.mp4"
    )
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.gen_frame)
        ok, bgr = cap.read()
    finally:
        cap.release()
    if ok:
        return np.ascontiguousarray(bgr[:, :, ::-1])
    # cv2's FFmpeg build may lack the codec (GR00T ships AV1) — imageio fallback.
    import imageio.v3 as iio

    for i, rgb in enumerate(iio.imiter(str(video), plugin="FFMPEG")):
        if i == args.gen_frame:
            return np.ascontiguousarray(np.asarray(rgb))
    raise ValueError(f"cannot read frame {args.gen_frame} from {video}")


_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def apply_lora(pipe: Any, args: argparse.Namespace, report: smoke.Report) -> dict[str, Any]:
    """Attach a WAM fine-tune to the pipeline's DiT and set its strength.

    Goes through ``pipe.transformer.load_lora_adapter``, not the pipeline-level
    ``load_lora_weights``: the export writes model-relative keys (``blocks.0.attn1...``, no
    ``transformer.`` prefix), which is exactly the layout ``WanI2VAdapter.load_lora`` reads back
    with ``prefix=None``. Same file, same loader, so what the Space shows is what a resume would
    restore.

    The saved metadata carries ``lora_alpha=64, r=32``; without it the loader would infer
    ``alpha = r`` and apply the adapter at half strength (see ``scripts/export_lora.py``). This
    logs the rebuilt scaling so a wrong-strength run is visible in the report rather than being
    mistaken for a weak fine-tune.
    """
    import json as _json

    from safetensors import safe_open

    source = str(args.gen_lora)
    if Path(source).is_dir():
        lora_dir = Path(source)
    else:  # a Hub repo id — the download costs no GPU quota when it happens outside @spaces.GPU
        from huggingface_hub import snapshot_download

        lora_dir = Path(snapshot_download(source, repo_type="model"))

    weight_file = lora_dir / _LORA_WEIGHT_NAME
    if not weight_file.is_file():
        raise FileNotFoundError(f"{weight_file} not found — run scripts/export_lora.py first")

    with safe_open(str(weight_file), framework="pt") as handle:
        meta = handle.metadata() or {}
        tensor_count = len(handle.keys())
    saved = _json.loads(meta.get("lora_adapter_metadata", "{}"))
    if not saved:
        report.check(
            "generate.lora_metadata",
            False,
            "no lora_adapter_metadata — alpha will be inferred as r, halving the trained strength",
        )

    t0 = time.perf_counter()
    pipe.transformer.load_lora_adapter(
        str(lora_dir), adapter_name="wam", prefix=None, weight_name=_LORA_WEIGHT_NAME
    )
    pipe.transformer.set_adapters("wam", args.gen_lora_scale)
    attach_s = time.perf_counter() - t0

    config = pipe.transformer.peft_config["wam"]
    report.check(
        "generate.lora",
        True,
        f"{tensor_count} tensors, r={config.r} alpha={config.lora_alpha} "
        f"(scaling {config.lora_alpha / config.r:g}) x scale {args.gen_lora_scale:g}, {attach_s:.1f}s",
    )
    info = {
        "source": source,
        "dir": str(lora_dir),
        "tensors": tensor_count,
        "scale": args.gen_lora_scale,
        "rank": config.r,
        "alpha": config.lora_alpha,
        "effective_scaling": (config.lora_alpha / config.r) * args.gen_lora_scale,
        "attach_s": round(attach_s, 1),
    }
    provenance = lora_dir / "wam_provenance.json"
    if provenance.is_file():
        # Which checkpoint these pixels came from, carried through to the report (AC-04).
        info["provenance"] = _json.loads(provenance.read_text())
    return info


def generate_future(
    args: argparse.Namespace, image_rgb: np.ndarray, report: smoke.Report
) -> dict[str, Any]:
    """Sample a future clip from the start frame + prompt; writes mp4 (+ start-frame png)."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video
    from PIL import Image

    if (args.gen_num_frames - 1) % 4:
        raise ValueError(f"--gen-num-frames must satisfy (F-1) % 4 == 0, got {args.gen_num_frames}")
    if args.gen_height % 32 or args.gen_width % 32:
        raise ValueError("--gen-height/--gen-width must be multiples of 32")
    source = args.source or args.model_id
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    prompt = args.gen_prompt or DEFAULT_INSTRUCTION

    t0 = time.perf_counter()
    # The Wan VAE wants fp32; casting it with the rest of the pipeline degrades decode.
    vae = AutoencoderKLWan.from_pretrained(source, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(source, vae=vae, torch_dtype=torch.bfloat16)
    pipe.to(device)
    load_s = time.perf_counter() - t0
    report.check("generate.load", True, f"{load_s:.1f}s")

    lora_info = apply_lora(pipe, args, report) if args.gen_lora else None

    start = Image.fromarray(image_rgb).resize((args.gen_width, args.gen_height), Image.LANCZOS)
    t0 = time.perf_counter()
    result = pipe(
        image=start,
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE,
        height=args.gen_height,
        width=args.gen_width,
        num_frames=args.gen_num_frames,
        guidance_scale=5.0,
        num_inference_steps=args.gen_steps,
        generator=torch.Generator(device).manual_seed(args.gen_seed),
    )
    sample_s = time.perf_counter() - t0

    out_path = Path(args.gen_out)
    export_to_video(result.frames[0], str(out_path), fps=24)
    start.save(out_path.with_suffix(".start.png"))
    ok = out_path.is_file() and out_path.stat().st_size > 0
    report.check("generate.wrote_mp4", ok, f"{out_path} ({sample_s:.0f}s sampling)")
    info = {
        "prompt": prompt,
        "episode": args.gen_episode,
        "start_frame": args.gen_frame,
        "num_frames": args.gen_num_frames,
        "steps": args.gen_steps,
        "size": [args.gen_height, args.gen_width],
        "seed": args.gen_seed,
        "fps": 24,
        "load_s": round(load_s, 1),
        "sample_s": round(sample_s, 1),
        "mp4": str(out_path),
        "start_png": str(out_path.with_suffix(".start.png")),
        "lora": lora_info,
    }
    if torch.cuda.is_available():
        info["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    report.info["generate"] = info
    return info


# ---- entry -------------------------------------------------------------------------------


def finalize(report: smoke.Report, out_path: str) -> int:
    payload = {"ok": not report.failed, "checks": report.checks, "info": report.info}
    LAST_REPORT.clear()
    LAST_REPORT.update(payload)
    serialized = json.dumps(payload, indent=2, default=str)
    print(f"\n===== REPORT =====\n{serialized}", flush=True)
    try:
        Path(out_path).write_text(serialized)
        print(f"report -> {out_path}", flush=True)
    except OSError as err:  # a read-only mount must not sink a passing run
        print(f"could not write {out_path}: {err}", flush=True)
    if report.failed:
        print(f"FAILED: {', '.join(report.failed)}", flush=True)
        return 1
    print(f"ALL {len(report.checks)} CHECKS PASSED", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = smoke.Report()
    import torch

    import wam

    report.info.update(
        {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "wam_version": getattr(wam, "__version__", "unknown"),
            "args": vars(args),
        }
    )
    if args.generate:
        generate_future(args, load_gen_frame(args), report)
    else:
        windows, instruction, data_info = build_windows(args)
        report.info["data"] = data_info
        report.check("probe.windows_built", len(windows) >= 24, f"{len(windows)} windows")
        features = extract_features(args, windows, instruction, report)
        analyze_probes(features, windows, args, report)
    return finalize(report, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
