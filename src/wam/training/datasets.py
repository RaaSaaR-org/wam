"""Episode-backed torch datasets for trainers (T-13/T-16).

``EpisodeDataset`` reads episode directories written by ``wam.data.episode`` (T-07) and
windows them into supervised samples: one observation (frame window + robot state) plus the
next commanded action chunk.

Contracts:
- ``wam.data`` is imported lazily INSIDE methods — this module must import even if the data
  API is still shifting (integration reconciles); construction stores paths only, decoding
  happens on first ``len()``/indexing.
- One sample per recorded commanded chunk. The observation is the last camera frame and last
  state at or before the chunk's command timestamp; the frame window is the ``num_frames``
  frames ending there (left-padded by repeating the first frame).
- Sample dict (torch tensors unless noted):
  ``frames`` uint8 [F, H, W, 3] · ``q`` [N] · ``dq`` [N] · ``imu`` [10] · ``gripper`` [G] ·
  ``validity`` bool [4] (q, dq, imu, gripper) · ``targets`` float32 [T, D] ·
  ``gripper_target`` float32 [T] · ``instruction`` str.
- ``chunk_steps`` fixes T: longer chunks are truncated, shorter ones skipped (documented,
  deterministic); ``None`` keeps native lengths (batching then requires equal T per batch).
- Targets are consumed as-recorded in PHYSICAL canonical units (identity normalization —
  the MVP convention across the whole stack). Two guards fail LOUDLY instead of silently
  training on wrong-unit data: (1) an episode manifest declaring a non-identity
  ``NormalizationSpec`` for action targets is rejected (nothing in the pipeline would apply
  it), and (2) per-step |targets| must stay < 1.0 — the shipped decoders are tanh-bounded
  to (-1, 1), so larger deltas are silently unlearnable.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["EpisodeDataset", "collate_episode_batch"]

_STACK_KEYS = ("frames", "q", "dq", "imu", "gripper", "validity", "targets", "gripper_target")

# Manifest normalization keys that would describe action targets. The MVP pipeline applies
# NO normalization anywhere (identity end-to-end), so a non-identity spec under one of these
# keys means the recorded stats would be silently ignored -> hard error instead.
_TARGET_NORMALIZATION_KEYS = frozenset({"targets", "action", "actions"})

# Shipped decoders (ActionHead) are tanh-bounded to (-1, 1) in physical units: per-step
# targets at or beyond 1.0 are unreachable and would silently floor the training loss.
_TANH_TARGET_BOUND = 1.0


def _check_identity_normalization(manifest: Any) -> None:
    """Reject episodes whose manifest declares non-identity normalization for targets."""
    specs = manifest.normalization_specs() or {}
    for key, spec in specs.items():
        if key not in _TARGET_NORMALIZATION_KEYS:
            continue
        if any(m != 0.0 for m in spec.mean) or any(s != 1.0 for s in spec.std):
            raise ValueError(
                f"episode {manifest.episode_id}: manifest declares a non-identity "
                f"NormalizationSpec for {key!r}, but the training pipeline applies no "
                "normalization (identity end-to-end, see wam.interfaces.NormalizationSpec) "
                "— refusing to train on data whose declared stats would be ignored"
            )


class EpisodeDataset(Dataset):
    """Windowed (observation, action-chunk) samples over one or more episode directories."""

    def __init__(
        self,
        episodes: str | Path | Sequence[str | Path],
        *,
        camera: str = "front",
        num_frames: int = 4,
        chunk_steps: int | None = None,
        verify_checksums: bool = True,
    ) -> None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if chunk_steps is not None and chunk_steps < 1:
            raise ValueError(f"chunk_steps must be >= 1 or None, got {chunk_steps}")
        if isinstance(episodes, (str, Path)):
            self._episodes: tuple[Path, ...] = (Path(episodes),)
            self._maybe_root = True  # a single path may be a root dir of many episodes
        else:
            self._episodes = tuple(Path(p) for p in episodes)
            self._maybe_root = False
        self.camera = camera
        self.num_frames = int(num_frames)
        self.chunk_steps = chunk_steps
        self.verify_checksums = verify_checksums
        self._samples: list[dict[str, Any]] | None = None

    # -- lazy loading ------------------------------------------------------------------

    def _resolve_dirs(self) -> list[Path]:
        from wam.data.episode import MANIFEST_FILENAME, list_episodes

        if self._maybe_root and not (self._episodes[0] / MANIFEST_FILENAME).is_file():
            dirs = list_episodes(self._episodes[0])
            if not dirs:
                raise FileNotFoundError(f"no episodes under {self._episodes[0]}")
            return dirs
        return list(self._episodes)

    def _ensure_loaded(self) -> list[dict[str, Any]]:
        if self._samples is not None:
            return self._samples
        from wam.data.episode import EpisodeReader

        samples: list[dict[str, Any]] = []
        for episode_dir in self._resolve_dirs():
            reader = EpisodeReader(episode_dir, verify_checksums=self.verify_checksums)
            samples.extend(self._window_episode(reader))
        if not samples:
            raise ValueError(
                f"no usable samples (camera={self.camera!r}, chunk_steps={self.chunk_steps})"
            )
        self._samples = samples
        return samples

    def _window_episode(self, reader: Any) -> list[dict[str, Any]]:
        from wam.data.episode import frame_window_indices

        _check_identity_normalization(reader.manifest)
        frames = reader.read_frames(self.camera)  # uint8 [n, H, W, 3]
        frame_ts = reader.frame_timestamps(self.camera)  # int64 [n]
        if frames.shape[0] == 0:
            raise ValueError(
                f"camera {self.camera!r} has no frames in {reader.manifest.episode_id}"
            )
        states = reader.read_states()
        if not states:
            raise ValueError(f"episode {reader.manifest.episode_id} has no states")
        state_ts = np.asarray([s.timestamp_ns for s in states], dtype=np.int64)
        instruction = reader.manifest.instruction

        samples: list[dict[str, Any]] = []
        for chunk, _executed_prefix, ts in reader.read_actions():
            targets = np.asarray(chunk.targets, dtype=np.float32)
            gripper_target = np.asarray(chunk.gripper_target, dtype=np.float32)
            if self.chunk_steps is not None:
                if targets.shape[0] < self.chunk_steps:
                    continue  # shorter chunks are skipped (documented contract)
                targets = targets[: self.chunk_steps]
                gripper_target = gripper_target[: self.chunk_steps]
            if targets.size > 0 and float(np.abs(targets).max()) >= _TANH_TARGET_BOUND:
                raise ValueError(
                    f"episode {reader.manifest.episode_id}: per-step |targets| reaches "
                    f"{float(np.abs(targets).max()):.4f} >= {_TANH_TARGET_BOUND} — outside "
                    "the tanh-reachable output range of the shipped decoders (physical "
                    "units, identity normalization); such data is silently unlearnable"
                )

            frame_idx = max(int(np.searchsorted(frame_ts, ts, side="right")) - 1, 0)
            # Shared with build_eval_pairs so training and evaluation cannot select different
            # windows — which is precisely what happened before T-29.
            indices = frame_window_indices(frame_idx, self.num_frames, frames.shape[0])
            state = states[max(int(np.searchsorted(state_ts, ts, side="right")) - 1, 0)]

            imu = np.concatenate(
                [
                    np.asarray(state.imu.orientation_wxyz, dtype=np.float32).reshape(-1),
                    np.asarray(state.imu.angular_velocity, dtype=np.float32).reshape(-1),
                    np.asarray(state.imu.linear_acceleration, dtype=np.float32).reshape(-1),
                ]
            )
            validity = state.validity.as_dict()
            samples.append(
                {
                    "frames": np.ascontiguousarray(frames[indices]),
                    "q": np.asarray(state.q, dtype=np.float32),
                    "dq": np.asarray(state.dq, dtype=np.float32),
                    "imu": imu,
                    "gripper": np.asarray(state.gripper_state, dtype=np.float32),
                    "validity": np.asarray(
                        [validity["q"], validity["dq"], validity["imu"], validity["gripper"]],
                        dtype=bool,
                    ),
                    "targets": targets,
                    "gripper_target": gripper_target,
                    "instruction": instruction,
                }
            )
        return samples

    # -- Dataset protocol --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._ensure_loaded())

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._ensure_loaded()[index]
        out: dict[str, Any] = {
            key: torch.from_numpy(np.ascontiguousarray(sample[key])) for key in _STACK_KEYS
        }
        out["instruction"] = sample["instruction"]
        return out


def collate_episode_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensor fields along a new batch dim; ``instruction`` becomes ``list[str]``."""
    if not batch:
        raise ValueError("empty batch")
    out: dict[str, Any] = {key: torch.stack([item[key] for item in batch]) for key in _STACK_KEYS}
    out["instruction"] = [item["instruction"] for item in batch]
    return out
