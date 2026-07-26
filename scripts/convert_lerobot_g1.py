#!/usr/bin/env python3
"""Convert LeRobot v2.1 Unitree G1 episodes (GR00T-N1.7 layout) to WAM episode format.

Source layout (e.g. ``nvidia/GR00T-N1.7-AppleToPlate``, CC-BY-4.0): per-episode parquet with
43-dim ``observation.state``/``action`` vectors plus one ego-view mp4, meta files per LeRobot
v2.1. Fetch a subset with ``huggingface_hub.snapshot_download(..., allow_patterns=[...])``.

Mapping into the canonical G1 space (``configs/robot/g1.yaml``, 15 joints + 2 grippers):

- ``q`` = [waist_yaw (state[12]), left_arm (state[15:22]), right_arm (state[22:29])].
  GR00T's waist block is [yaw, roll, pitch] (Unitree joint order); roll/pitch and both legs
  are dropped — the canonical space is upper-body, the task is static loco-manipulation.
- ``dq`` = finite-difference of ``q`` (the source has no velocities); first row is zero.
- IMU is absent in the source -> zeros with ``validity.imu=False`` (FR-02 missing-group path).
- ``gripper_state`` = per-hand grasp synergy: mean of the 7 Dex3-class hand joints, affinely
  mapped to [0, 1] via (mean+1)/2 and clipped. MVP placeholder synergy (OD-01): consistent,
  monotone in closure, NOT calibrated to physical aperture.
- Actions are relabeled from executed states (standard BC on demonstrations): per-step
  ``targets[t] = q[t+1] - q[t]`` (JOINT_DELTA), ``gripper_target[t]`` = mean of both hands'
  synergy at t+1. Steps are grouped into non-overlapping chunks of ``--chunk-steps``;
  ``executed_prefix`` = chunk length (demonstrations are fully executed). The 43-dim
  ``action`` column (absolute position targets for the vendor controller) is not used.
- Frames: ego mp4 decoded, resized to ``--resize H W`` (even, backbone-patch friendly),
  written as camera ``ego`` with the parquet timestamps (mp4 pts are not sync-grade).

Usage:
  .venv/bin/python scripts/convert_lerobot_g1.py --source <snapshot_dir> \
      --out datasets/gr00t-apple --episodes 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wam.data import MANIFEST_FILENAME
from wam.data.episode import EpisodeWriter
from wam.interfaces import load_config
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_STATE_DIM = 43
# GR00T G1 43-dim layout (meta/modality.json): legs [0:12], waist [12:15] = [yaw, roll,
# pitch] per Unitree joint order, arms 7 DoF each, hands 7 DoF each (Dex3-class).
WAIST_YAW = 12
LEFT_ARM = slice(15, 22)
RIGHT_ARM = slice(22, 29)
LEFT_HAND = slice(29, 36)
RIGHT_HAND = slice(36, 43)

CAMERA_NAME = "ego"


def canonical_q(state: np.ndarray) -> np.ndarray:
    """[..., 43] source state -> [..., 15] canonical q (waist_yaw, left arm, right arm)."""
    state = np.asarray(state, dtype=np.float32)
    if state.shape[-1] != SOURCE_STATE_DIM:
        raise ValueError(f"expected last dim {SOURCE_STATE_DIM}, got {state.shape[-1]}")
    return np.concatenate(
        [state[..., WAIST_YAW : WAIST_YAW + 1], state[..., LEFT_ARM], state[..., RIGHT_ARM]],
        axis=-1,
    )


def hand_synergy(hand: np.ndarray) -> np.ndarray:
    """[..., 7] hand joints -> scalar grasp synergy in [0, 1] (MVP placeholder, see module doc)."""
    mean = np.asarray(hand, dtype=np.float32).mean(axis=-1)
    return np.clip((mean + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def gripper_state(state: np.ndarray) -> np.ndarray:
    """[..., 43] source state -> [..., 2] canonical gripper [left, right] synergies."""
    state = np.asarray(state, dtype=np.float32)
    return np.stack(
        [hand_synergy(state[..., LEFT_HAND]), hand_synergy(state[..., RIGHT_HAND])], axis=-1
    )


def relabel_chunks(
    q: np.ndarray, grip: np.ndarray, *, chunk_steps: int, dt_s: float
) -> list[tuple[ActionChunk, int]]:
    """Executed states -> non-overlapping JOINT_DELTA chunks; returns (chunk, start_index).

    ``targets[t] = q[start+t+1] - q[start+t]``; ``gripper_target[t]`` = mean over both hands
    at ``start+t+1``. The trailing remainder shorter than ``chunk_steps`` is dropped.
    """
    n = q.shape[0]
    chunks: list[tuple[ActionChunk, int]] = []
    for start in range(0, n - chunk_steps, chunk_steps):
        deltas = np.diff(q[start : start + chunk_steps + 1], axis=0).astype(np.float32)
        grip_next = grip[start + 1 : start + chunk_steps + 1].mean(axis=-1).astype(np.float32)
        chunks.append(
            (
                ActionChunk(
                    mode=ActionMode.JOINT_DELTA,
                    targets=deltas,
                    gripper_target=grip_next,
                    dt_s=dt_s,
                ),
                start,
            )
        )
    return chunks


def read_source_episode(parquet_path: Path) -> dict[str, np.ndarray]:
    """Read one LeRobot episode parquet -> {state [n, 43], ts_ns [n]}."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["observation.state", "timestamp"])
    state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
    ts_s = table["timestamp"].to_numpy(zero_copy_only=False).astype(np.float64)
    if state.ndim != 2 or state.shape[1] != SOURCE_STATE_DIM:
        raise ValueError(
            f"{parquet_path}: expected [n, {SOURCE_STATE_DIM}] states, got {state.shape}"
        )
    return {"state": state, "ts_ns": np.round(ts_s * 1e9).astype(np.int64)}


def read_video_frames(video_path: Path, resize_hw: tuple[int, int]) -> np.ndarray:
    """Decode an mp4 to uint8 RGB [n, H, W, 3], resized to ``resize_hw`` (INTER_AREA).

    cv2 first; if its FFmpeg build lacks the codec (GR00T ships **AV1**, which the pip
    ``opencv-python-headless`` wheel cannot decode), fall back to imageio's bundled ffmpeg.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    h, w = resize_hw
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if (bgr.shape[0], bgr.shape[1]) != (h, w):
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            frames.append(np.ascontiguousarray(bgr[:, :, ::-1]))
    finally:
        cap.release()
    if not frames:
        frames = _read_video_frames_imageio(video_path, resize_hw)
    if not frames:
        raise ValueError(f"no frames decoded from {video_path}")
    return np.stack(frames)


def _read_video_frames_imageio(video_path: Path, resize_hw: tuple[int, int]) -> list[np.ndarray]:
    """Codec fallback via imageio-ffmpeg (RGB already); empty if imageio is unavailable."""
    try:
        import imageio.v3 as iio
    except ImportError:
        return []
    import cv2

    h, w = resize_hw
    frames: list[np.ndarray] = []
    for rgb in iio.imiter(str(video_path), plugin="FFMPEG"):
        rgb = np.asarray(rgb)
        if rgb.shape[:2] != (h, w):
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(np.ascontiguousarray(rgb))
    return frames


def load_instructions(source: Path) -> dict[int, str]:
    """meta/episodes.jsonl -> {episode_index: first task string} (empty dict if absent)."""
    path = source / "meta" / "episodes.jsonl"
    if not path.is_file():
        return {}
    out: dict[int, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tasks = row.get("tasks") or []
        if tasks:
            out[int(row["episode_index"])] = str(tasks[0])
    return out


def convert_episode(
    source: Path,
    episode_index: int,
    out_dir: Path,
    spec: CanonicalSpaceSpec,
    *,
    instruction: str,
    resize_hw: tuple[int, int],
    chunk_steps: int,
    dq_from_diff: bool = True,
) -> dict[str, Any]:
    """Convert one source episode into ``out_dir``; returns summary stats."""
    parquet_path = source / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    video_path = (
        source
        / "videos"
        / "chunk-000"
        / "observation.images.ego_view"
        / f"episode_{episode_index:06d}.mp4"
    )
    data = read_source_episode(parquet_path)
    frames = read_video_frames(video_path, resize_hw)

    n = min(data["state"].shape[0], frames.shape[0])
    state, ts_ns = data["state"][:n], data["ts_ns"][:n]
    q = canonical_q(state)
    grip = gripper_state(state)
    dt_s = float(np.diff(ts_ns).mean() / 1e9) if n > 1 else 1.0 / 30.0
    dq = np.zeros_like(q)
    if dq_from_diff and n > 1:
        dq[1:] = np.diff(q, axis=0) / dt_s

    episode_id = out_dir.name
    if (out_dir / MANIFEST_FILENAME).is_file():
        shutil.rmtree(out_dir)  # deterministic re-convert
    zero_imu = IMUState(
        orientation_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
        angular_velocity=np.zeros(3, dtype=np.float32),
        linear_acceleration=np.zeros(3, dtype=np.float32),
    )
    chunks = relabel_chunks(q, grip, chunk_steps=chunk_steps, dt_s=dt_s)
    with EpisodeWriter(
        out_dir,
        episode_id,
        spec,
        fps=1.0 / dt_s,
        instruction=instruction,
        extra={
            "d_phase": "D1-real",
            "source": {
                "dataset": "nvidia/GR00T-N1.7-AppleToPlate",
                "license": "CC-BY-4.0",
                "robot_type": "unitree_g1",
                "episode_index": episode_index,
            },
            "mapping": {
                "converter": Path(__file__).name,
                "action_relabel": "executed-state deltas (BC)",
                "dq": "finite-difference" if dq_from_diff else "invalid",
                "gripper_synergy": "clip((mean(hand_7dof)+1)/2, 0, 1)",
                "dropped": ["legs", "waist_roll", "waist_pitch", "per-finger hand joints"],
            },
        },
    ) as writer:
        for t in range(n):
            writer.add_frame(CAMERA_NAME, frames[t], int(ts_ns[t]))
            writer.add_state(
                RobotState(
                    timestamp_ns=int(ts_ns[t]),
                    q=q[t],
                    dq=dq[t],
                    imu=zero_imu,
                    gripper_state=grip[t],
                    validity=ValidityMask(q=True, dq=dq_from_diff, imu=False, gripper=True),
                )
            )
        for chunk, start in chunks:
            writer.add_action(
                chunk, executed_prefix=chunk.num_steps, timestamp_ns=int(ts_ns[start])
            )

    max_delta = max((float(np.abs(np.asarray(c.targets)).max()) for c, _ in chunks), default=0.0)
    return {"steps": n, "chunks": len(chunks), "dt_s": dt_s, "max_abs_delta": max_delta}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="LeRobot snapshot root")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "datasets" / "gr00t-apple")
    parser.add_argument("--episodes", type=int, default=10, help="convert the first N episodes")
    parser.add_argument("--start", type=int, default=0, help="first source episode index")
    parser.add_argument("--chunk-steps", type=int, default=16)
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=(120, 160),
        metavar=("H", "W"),
        help="output frame size; must be even (yuv420) and patch-friendly",
    )
    parser.add_argument(
        "--robot-config", type=Path, default=_REPO_ROOT / "configs" / "robot" / "g1.yaml"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    robot_cfg = load_config(args.robot_config)
    spec = CanonicalSpaceSpec(**robot_cfg["robot"]["canonical_space"])
    instructions = load_instructions(args.source)
    args.out.mkdir(parents=True, exist_ok=True)

    for i in range(args.start, args.start + args.episodes):
        out_dir = args.out / f"gr00t-apple-{i:06d}"
        stats = convert_episode(
            args.source,
            i,
            out_dir,
            spec,
            instruction=instructions.get(i, "move the apple to the plate"),
            resize_hw=tuple(args.resize),
            chunk_steps=args.chunk_steps,
        )
        print(
            f"converted episode {i}: steps={stats['steps']} chunks={stats['chunks']} "
            f"dt_s={stats['dt_s']:.4f} max|delta|={stats['max_abs_delta']:.4f} -> {out_dir}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
