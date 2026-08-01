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
- ``gripper_state`` = per-hand grasp synergy: mean of the 7 Dex3-class hand joints, mapped to
  [0, 1]. Two mappings, selected by ``--gripper-mapping``:

  * ``legacy`` (default): ``clip((mean+1)/2, 0, 1)`` per hand, and ``gripper_target`` = the mean
    of BOTH hands. This is what ``datasets/gr00t-apple-full`` was built with. It is retained as
    the default so re-running the converter reproduces that dataset exactly —
    ``runs/t16-lora-seed0``'s ``dataset_snapshot_ref`` is pinned to its manifest hashes.
    It is also WRONG on GR00T-N1.7-AppleToPlate and the wrongness is measurable: the (x+1)/2
    mapping assumes a [-1, 1] joint range the Dex3 hand never uses, attenuating a 0.69 rad grasp
    into 0.16 of synergy, and averaging in a right hand that never moves halves it again to
    0.08 centred on the 0.5 binarization threshold. ``scripts/audit_gripper.py`` FAILS the
    result on all four clauses while PASSING the source snapshot it came from (T-31).
    Its clip is now gated on the same terms as the pinned affine's: ``legacy_clipped_frac`` is
    measured over the whole conversion set before anything is written, and a source that rails is
    REFUSED with a pointer to ``active-hand``. Until 2026-08-01 it was not, which left the one
    mapping that ASSUMES a scale as the only one allowed to be silently wrong about it.
  * ``active-hand``: one DATASET-level affine ``(x - lo) / (hi - lo)`` over the raw synergy of
    the hand that actually moves, applied to both hands, with ``gripper_target`` taken from the
    active hand alone. Dataset-level, not per-episode, on purpose: a per-episode min-max would
    make the same physical aperture mean a different number in every episode, which is
    unlearnable. The affine is recorded in the manifest's ``normalization`` provenance slot so
    the mapping travels with the data.

  Neither is calibrated to physical aperture (OD-01) — both are monotone in closure only.

  DATASET-LEVEL IS NOT DATASET-INDEPENDENT. ``active-hand`` fits its affine over the episodes of
  THIS invocation, so the same physical aperture is a different number in two conversions of
  different size or of different data. Measured on ``data/raw/gr00t_apple`` (parquet only, no
  decode): 30 episodes give offset −0.39980 span 0.41004, 120 give −0.41043/0.43853, and the
  full 402 give −0.43865/0.46675 — a raw synergy of −0.40 maps to 0.000 under the first fit and
  0.083 under the last. Two conversions are therefore comparable only if they share an affine,
  which is what ``--gripper-affine OFFSET SPAN`` is for: pass the pair printed (and recorded in
  the manifest) by the first conversion. A pinned affine is REFUSED when it would clip any
  sample of the new set, because clipping is silent in the output and inflates every
  admissibility clause of ``scripts/audit_gripper.py`` in the passing direction; the error prints
  the affine that would have been fitted so the choice is explicit.
- Actions are relabeled from executed states (standard BC on demonstrations): per-step
  ``targets[t] = q[t+1] - q[t]`` (JOINT_DELTA), ``gripper_target[t]`` = the gripper synergy at
  t+1 (see the mapping note above). Steps are grouped into non-overlapping chunks of
  ``--chunk-steps``; ``executed_prefix`` = chunk length (demonstrations are fully executed).
  The 43-dim ``action`` column (absolute position targets for the vendor controller) is not used.
- Frames: ego mp4 decoded, resized to ``--resize H W`` (even, backbone-patch friendly),
  written as camera ``ego`` with the parquet timestamps (mp4 pts are not sync-grade).

Usage:
  .venv/bin/python scripts/convert_lerobot_g1.py --source <snapshot_dir> \
      --out datasets/gr00t-apple --episodes 10

  # the fixed gripper mapping — ALWAYS into a new root, never over an existing dataset
  .venv/bin/python scripts/convert_lerobot_g1.py --source data/raw/gr00t_apple \
      --out datasets/gr00t-apple-grip --episodes 402 --gripper-mapping active-hand

  # a second conversion that has to stay comparable with the first: reuse its affine
  .venv/bin/python scripts/convert_lerobot_g1.py --source data/raw/gr00t_apple2 \
      --out datasets/gr00t-apple2-grip --episodes 200 --gripper-mapping active-hand \
      --gripper-affine -0.438654 0.466748 --source-dataset <second-dataset-id>
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from wam.data import MANIFEST_FILENAME
from wam.data.episode import EpisodeWriter
from wam.interfaces import load_config
from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    NormalizationSpec,
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

GRIPPER_MAPPINGS = ("legacy", "active-hand")

DEFAULT_SOURCE_DATASET = "nvidia/GR00T-N1.7-AppleToPlate"
"""Fallback dataset id, used only when ``--source-dataset`` is not given.

LeRobot's ``meta/info.json`` carries ``robot_type``/``codebase_version``/``fps`` but no repo id,
so the id genuinely cannot be derived. It used to be hardcoded into every manifest regardless of
``--source``, which silently mislabelled the provenance of any second dataset; now the fallback
warns and the manifest records which of the two it was.
"""


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


LEGACY_MAX_CLIPPED_FRAC = 0.0
"""How much of the source the legacy mapping may rail before the conversion is refused.

Zero, and for the same reason ``pinned_hand_affine`` refuses at zero: a mapping that clips
nothing is achievable by construction (``active-hand`` fits the scale), so zero is what the
alternative already delivers, and any clipped sample is a measurement replaced by a rail in a
direction that makes every downstream gate easier to pass. It is a named constant rather than a
literal so the refusal can be relaxed deliberately, in writing, if a source ever needs it.
"""


def legacy_clipped_frac(states: list[np.ndarray]) -> float:
    """Fraction of samples ``hand_synergy`` would push outside [0, 1] before clipping.

    Computed on the RAW synergy over both hands, because that is the signal the legacy mapping
    rails; asking the already-clipped output how much it clipped is the question that cannot
    answer itself.
    """
    total = 0
    outside = 0
    for state in states:
        arr = np.asarray(state, dtype=np.float32)
        for span in (LEFT_HAND, RIGHT_HAND):
            mapped = (raw_synergy(arr[..., span]) + 1.0) / 2.0
            total += mapped.size
            outside += int(np.count_nonzero((mapped < 0.0) | (mapped > 1.0)))
    return 0.0 if total == 0 else outside / total


def raw_synergy(hand: np.ndarray) -> np.ndarray:
    """[..., 7] hand joints -> the un-mapped mean, in source joint units.

    Split out of ``hand_synergy`` because the dataset-level affine has to be fitted on the raw
    signal: fitting it on the already-clipped [0, 1] output would inherit exactly the attenuation
    it exists to undo.
    """
    return np.asarray(hand, dtype=np.float32).mean(axis=-1)


class HandAffine(NamedTuple):
    """The dataset-level gripper mapping ``clip((x - offset) / span, 0, 1)``.

    A ``NamedTuple`` rather than a ``@dataclass`` on purpose: this file is loaded by
    ``importlib.util.module_from_spec`` in the tests, which never registers it in ``sys.modules``,
    and ``dataclass`` resolves its (stringified, ``from __future__ import annotations``) field
    types through exactly that registry.

    ``active`` names the hand the affine was fitted on. It is applied to BOTH hands, deliberately:
    they are the same hardware in the same units, so a shared affine keeps them comparable and
    leaves a hand that never moves reading as the near-constant it is. Giving each hand its own
    affine would stretch a frozen hand's micrometres of drift across the full [0, 1] range and
    manufacture a gripper signal out of sensor noise.

    ``fitted_episodes`` / ``pinned`` are provenance: the affine is a property of the conversion
    set (see the module docstring), so the number that produced it has to travel with it or a
    later reader cannot tell whether two datasets are on the same scale.
    """

    active: str
    offset: float
    span: float
    p2p_left: float
    p2p_right: float
    fitted_episodes: int = 0
    pinned: bool = False

    def apply(self, values: np.ndarray) -> np.ndarray:
        """``clip((x - offset) / span, 0, 1)`` — the clip is load-bearing for a PINNED affine.

        For a fitted affine the clip only touches the two extremal samples it was fitted on. For
        a pinned one it silently folds everything outside the borrowed range onto a rail, which
        is why :func:`pinned_hand_affine` refuses that case up front rather than letting the
        conversion record clipped values as measurements.
        """
        if self.span <= 0.0:
            return np.zeros_like(np.asarray(values, dtype=np.float32))
        return np.clip((np.asarray(values, dtype=np.float32) - self.offset) / self.span, 0.0, 1.0)

    def clipped_frac(self, values: np.ndarray) -> float:
        """Fraction of ``values`` this affine would push outside [0, 1] before clipping."""
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.span <= 0.0 or v.size == 0:
            return 0.0
        z = (v - self.offset) / self.span
        return float(((z < 0.0) | (z > 1.0)).mean())

    def to_spec(self, dim: int) -> NormalizationSpec:
        """The PRE-CLIP affine as a ``NormalizationSpec`` (z = (x - mean) / std).

        ``NormalizationSpec`` has exactly ``mean``/``std`` and no clip bounds, so what is recorded
        here inverts to the raw synergy only for values that were inside [offset, offset + span];
        at the endpoints the stored channel is clipped and the inverse of the spec is not the
        original. The clip is therefore recorded separately and machine-readably by
        :func:`convert_episode` as ``mapping.gripper_clip``; this method deliberately does not
        pretend the spec alone is invertible.
        """
        return NormalizationSpec(mean=(self.offset,) * dim, std=(self.span,) * dim)


def _hand_series(states: list[np.ndarray]) -> tuple[dict[str, list[np.ndarray]], dict[str, float]]:
    """Per-hand raw synergy per episode, and each hand's MEAN PER-EPISODE peak-to-peak."""
    per_hand: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for state in states:
        per_hand["left"].append(raw_synergy(state[..., LEFT_HAND]).astype(np.float64))
        per_hand["right"].append(raw_synergy(state[..., RIGHT_HAND]).astype(np.float64))
    p2p = {
        name: float(np.mean([float(s.max() - s.min()) for s in series]))
        for name, series in per_hand.items()
    }
    return per_hand, p2p


def fit_hand_affine(states: list[np.ndarray]) -> HandAffine:
    """Fit the dataset-level gripper affine over every episode that will be converted.

    The active hand is chosen on the MEAN PER-EPISODE peak-to-peak, not on the global one: a
    frozen hand that rests at slightly different angles across a session has a large global range
    and no motion inside any episode, and picking it would produce a channel that never opens
    while looking like it has full range.

    The fit is over THIS set of episodes and nothing else — see the module docstring on why that
    makes two conversions incomparable unless one of them pins the other's affine.
    """
    if not states:
        raise ValueError("fit_hand_affine: no episodes to fit on")
    per_hand, p2p = _hand_series(states)
    active = "left" if p2p["left"] >= p2p["right"] else "right"
    flat = np.concatenate(per_hand[active])
    lo, hi = float(flat.min()), float(flat.max())
    if hi <= lo:
        raise ValueError(
            f"fit_hand_affine: the {active} hand is constant across the whole conversion set "
            "(span 0) — there is no gripper signal to map; audit the source with "
            "scripts/audit_gripper.py --lerobot before converting"
        )
    return HandAffine(
        active=active,
        offset=lo,
        span=hi - lo,
        p2p_left=p2p["left"],
        p2p_right=p2p["right"],
        fitted_episodes=len(states),
    )


def pinned_hand_affine(states: list[np.ndarray], offset: float, span: float) -> HandAffine:
    """Reuse an affine measured elsewhere, and refuse it if it clips this conversion set.

    The active hand is still determined from this set (same rule as :func:`fit_hand_affine`) —
    only the [0, 1] mapping is borrowed, which is the part that has to match for two datasets to
    mean the same thing by the same number.

    The refusal threshold is not a tuned tolerance: an affine fitted on the set it is applied to
    clips exactly zero samples by construction, so zero is the value the alternative achieves and
    anything above it is the borrowed range being too narrow for this data. Clipping cannot be
    recovered from the written dataset and it moves every clause of ``scripts/audit_gripper.py``
    in the passing direction, so it has to be refused at the point where the unclipped values
    still exist.
    """
    if not states:
        raise ValueError("pinned_hand_affine: no episodes to map")
    if not (span > 0.0):
        raise ValueError(f"pinned_hand_affine: span must be > 0, got {span}")
    per_hand, p2p = _hand_series(states)
    active = "left" if p2p["left"] >= p2p["right"] else "right"
    affine = HandAffine(
        active=active,
        offset=offset,
        span=span,
        p2p_left=p2p["left"],
        p2p_right=p2p["right"],
        fitted_episodes=len(states),
        pinned=True,
    )
    flat = np.concatenate(per_hand[active])
    clipped = affine.clipped_frac(flat)
    if clipped > 0.0:
        lo, hi = float(flat.min()), float(flat.max())
        raise ValueError(
            f"pinned_hand_affine: --gripper-affine {offset:.6g} {span:.6g} clips "
            f"{clipped:.4%} of the {active} hand's samples over these {len(states)} episodes "
            f"(raw range [{lo:.6g}, {hi:.6g}], pinned range "
            f"[{offset:.6g}, {offset + span:.6g}]). Clipped samples are indistinguishable from "
            "measurements in the written dataset. Either widen the pinned affine to cover this "
            f"set (fitted here: offset {lo:.6g} span {hi - lo:.6g}) and RE-CONVERT the dataset "
            "it came from with the same pair, or drop --gripper-affine and accept that the two "
            "datasets are on different scales"
        )
    return affine


def gripper_state(state: np.ndarray, affine: HandAffine | None = None) -> np.ndarray:
    """[..., 43] source state -> [..., 2] canonical gripper [left, right] synergies."""
    state = np.asarray(state, dtype=np.float32)
    if affine is None:
        return np.stack(
            [hand_synergy(state[..., LEFT_HAND]), hand_synergy(state[..., RIGHT_HAND])], axis=-1
        )
    return np.stack(
        [
            affine.apply(raw_synergy(state[..., LEFT_HAND])),
            affine.apply(raw_synergy(state[..., RIGHT_HAND])),
        ],
        axis=-1,
    ).astype(np.float32)


def relabel_chunks(
    q: np.ndarray,
    grip: np.ndarray,
    *,
    chunk_steps: int,
    dt_s: float,
    gripper_column: int | None = None,
) -> list[tuple[ActionChunk, int]]:
    """Executed states -> non-overlapping JOINT_DELTA chunks; returns (chunk, start_index).

    ``targets[t] = q[start+t+1] - q[start+t]``. ``gripper_target[t]`` is the mean over both hands
    at ``start+t+1``, or — when ``gripper_column`` is given — that single hand's channel. The
    both-hand mean is the second half of the T-31 mapping bug: averaging a live hand against a
    frozen one halves whatever amplitude survived the synergy mapping and drags the result onto
    the binarization threshold. The trailing remainder shorter than ``chunk_steps`` is dropped.
    """
    n = q.shape[0]
    chunks: list[tuple[ActionChunk, int]] = []
    for start in range(0, n - chunk_steps, chunk_steps):
        deltas = np.diff(q[start : start + chunk_steps + 1], axis=0).astype(np.float32)
        window = grip[start + 1 : start + chunk_steps + 1]
        grip_next = (
            window.mean(axis=-1) if gripper_column is None else window[..., gripper_column]
        ).astype(np.float32)
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


def source_info(source: Path) -> dict[str, Any]:
    """``meta/info.json`` fields worth recording as provenance (empty dict if absent).

    Deliberately does NOT invent a dataset id: LeRobot's info.json does not carry one, and a
    guessed id in a manifest is worse than an absent one because it looks verified.
    """
    path = source / "meta" / "info.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: raw[k] for k in ("robot_type", "codebase_version", "fps") if k in raw}


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
    gripper_affine: HandAffine | None = None,
    source_dataset: str = DEFAULT_SOURCE_DATASET,
    source_dataset_verified: bool = False,
    record_provenance: bool = False,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one source episode into ``out_dir``; returns summary stats.

    ``gripper_affine`` is the dataset-level mapping fitted by :func:`fit_hand_affine` over the
    whole conversion set. ``None`` keeps the legacy mapping, which is the default so that
    re-running the converter reproduces ``datasets/gr00t-apple-full`` byte for byte.

    ``record_provenance`` adds the extra ``source`` fields (``dataset_id_source`` plus
    ``meta/info.json``). It is off by default and must stay keyed on something that actually
    differs from the shipped conversion, because those fields change the manifest BYTES and
    ``datasets/gr00t-apple-full``'s bytes are what ``runs/t16-lora-seed0``'s
    ``dataset_snapshot_ref`` is pinned to.
    """
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
    grip = gripper_state(state, gripper_affine)
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
    gripper_column = (
        None if gripper_affine is None else (0 if gripper_affine.active == "left" else 1)
    )
    chunks = relabel_chunks(
        q, grip, chunk_steps=chunk_steps, dt_s=dt_s, gripper_column=gripper_column
    )

    source_meta: dict[str, Any] = {
        "dataset": source_dataset,
        "license": "CC-BY-4.0",
        "robot_type": "unitree_g1",
        "episode_index": episode_index,
    }
    if record_provenance or gripper_affine is not None:
        # The richer provenance is added only once something already differs from the shipped
        # default conversion. datasets/gr00t-apple-full's manifest BYTES are hashed into
        # runs/t16-lora-seed0's dataset_snapshot_ref (PolicyContract.from_dataset), so a default
        # re-conversion has to keep producing exactly the bytes it produced in July or T-28's
        # split proof stops verifying. `record_provenance` is therefore keyed on the recorded id
        # actually DIFFERING from the default (or on an explicit opt-in) and not on
        # --source-dataset merely being present: naming the dataset the default already assumes
        # states nothing new, and it must not move a byte.
        source_meta["dataset_id_source"] = (
            "--source-dataset" if source_dataset_verified else "converter default (unverified)"
        )
        source_meta.update(info or {})
    mapping: dict[str, Any] = {
        "converter": Path(__file__).name,
        "action_relabel": "executed-state deltas (BC)",
        "dq": "finite-difference" if dq_from_diff else "invalid",
        "gripper_synergy": "clip((mean(hand_7dof)+1)/2, 0, 1)",
        "dropped": ["legs", "waist_roll", "waist_pitch", "per-finger hand joints"],
    }
    normalization = None
    if gripper_affine is not None:
        fitted_on = (
            f"pinned via --gripper-affine, active hand re-derived from these "
            f"{gripper_affine.fitted_episodes} episodes"
            if gripper_affine.pinned
            else f"fitted on these {gripper_affine.fitted_episodes} episodes"
        )
        mapping["gripper_mapping"] = "active-hand"
        mapping["gripper_synergy"] = (
            f"clip((mean(hand_7dof) - {gripper_affine.offset:.6g}) / "
            f"{gripper_affine.span:.6g}, 0, 1), dataset-level affine "
            f"{'pinned onto' if gripper_affine.pinned else 'fitted on'} the "
            f"{gripper_affine.active} hand"
        )
        # The affine is a property of the conversion SET, so a dataset converted from a different
        # number of episodes carries a different scale for the same physical aperture. Recording
        # which it was is what lets a later reader tell "comparable" from "looks comparable".
        mapping["gripper_affine_source"] = fitted_on
        # NormalizationSpec has no clip field (wam.interfaces.schema), so the clip that makes the
        # recorded affine non-invertible at the endpoints is recorded here instead of being left
        # implicit in the spec — see HandAffine.to_spec.
        mapping["gripper_clip"] = [0.0, 1.0]
        mapping["gripper_target"] = f"{gripper_affine.active} hand only (both-hand mean is T-31)"
        mapping["gripper_hand_p2p"] = {
            "left": gripper_affine.p2p_left,
            "right": gripper_affine.p2p_right,
        }
        # Provenance only — nothing in the MVP pipeline APPLIES a NormalizationSpec (see
        # wam.interfaces.NormalizationSpec). Recording it here is what makes the mapping travel
        # with the data instead of living in a converter flag nobody re-reads. The keys are
        # deliberately not 'targets'/'action' so EpisodeDataset's identity check still passes:
        # the joint deltas remain unnormalized physical units.
        normalization = {
            "gripper_state": gripper_affine.to_spec(2),
            "gripper_target": gripper_affine.to_spec(1),
        }

    with EpisodeWriter(
        out_dir,
        episode_id,
        spec,
        fps=1.0 / dt_s,
        instruction=instruction,
        normalization=normalization,
        extra={"d_phase": "D1-real", "source": source_meta, "mapping": mapping},
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
    parser.add_argument(
        "--gripper-mapping",
        choices=GRIPPER_MAPPINGS,
        default="legacy",
        help="how hand joints become the [0, 1] gripper channel (default: legacy, which is what "
        "datasets/gr00t-apple-full was built with — see the module docstring)",
    )
    parser.add_argument(
        "--gripper-affine",
        type=float,
        nargs=2,
        default=None,
        metavar=("OFFSET", "SPAN"),
        help="reuse a mapping instead of fitting one (active-hand only), so a second conversion "
        "or a second dataset means the same aperture by the same number; refused if it would "
        "clip any sample of this set",
    )
    parser.add_argument(
        "--source-dataset",
        type=str,
        default=None,
        help="dataset id recorded as provenance; required for any snapshot that is not "
        f"{DEFAULT_SOURCE_DATASET}, which the converter otherwise assumes",
    )
    parser.add_argument(
        "--record-provenance",
        action="store_true",
        help="also record dataset_id_source + meta/info.json in every manifest. Implied when "
        "--source-dataset differs from the converter default; off otherwise because these "
        "fields change the manifest bytes that datasets/gr00t-apple-full is pinned by",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    robot_cfg = load_config(args.robot_config)
    spec = CanonicalSpaceSpec(**robot_cfg["robot"]["canonical_space"])
    instructions = load_instructions(args.source)
    info = source_info(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    indices = list(range(args.start, args.start + args.episodes))

    source_dataset = args.source_dataset or DEFAULT_SOURCE_DATASET
    if args.source_dataset is None:
        print(
            f"WARNING: no --source-dataset given; recording provenance as {DEFAULT_SOURCE_DATASET}"
            f" for {args.source}. Pass --source-dataset when converting anything else, or every "
            "manifest will name the wrong dataset."
        )
    # Naming the dataset the default already assumes must not change a single manifest byte —
    # otherwise following the warning above silently breaks the byte-reproduction of
    # datasets/gr00t-apple-full that this converter's default invocation promises.
    record_provenance = args.record_provenance or source_dataset != DEFAULT_SOURCE_DATASET

    if args.gripper_affine is not None and args.gripper_mapping != "active-hand":
        print(
            "ERROR: --gripper-affine only applies to --gripper-mapping active-hand; the legacy "
            "mapping is a fixed formula with no affine to pin.",
            file=sys.stderr,
        )
        return 2

    affine = None
    if args.gripper_mapping == "legacy":
        # The legacy mapping clips too, and until now nothing checked it. `pinned_hand_affine`
        # refuses an affine that clips because clipping is silent in the output and moves every
        # downstream admissibility clause in the PASSING direction — a flattened channel reads as
        # a wide, decisive, two-state gripper. That argument is not about which mapping is in use.
        # Legacy is `clip((mean + 1) / 2, 0, 1)`, i.e. it assumes the source hand joints live in
        # [-1, 1]; a source on any other scale is silently railed. Measured on gr00t_apple, the
        # left hand spans 0.826 rad and legacy squashes it to 0.157, so this is not hypothetical.
        states = [
            read_source_episode(
                args.source / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
            )["state"]
            for i in indices
        ]
        clipped = legacy_clipped_frac(states)
        if clipped > LEGACY_MAX_CLIPPED_FRAC:
            print(
                f"ERROR: the legacy gripper mapping clips {clipped:.4f} of samples "
                f"(> {LEGACY_MAX_CLIPPED_FRAC}). clip((mean + 1) / 2, 0, 1) assumes source hand "
                "joints in [-1, 1]; this source is on another scale, so the written "
                "gripper_target would be railed rather than measured — and a railed channel "
                "PASSES the audit's range and transition clauses for the wrong reason. Convert "
                "with --gripper-mapping active-hand, which fits the scale instead of assuming "
                "it.",
                file=sys.stderr,
            )
            return 2
        if clipped > 0.0:
            print(
                f"WARNING: the legacy gripper mapping clips {clipped:.4f} of samples. Under the "
                f"{LEGACY_MAX_CLIPPED_FRAC} refusal threshold, but every clipped sample is a "
                "measurement replaced by a rail."
            )
    if args.gripper_mapping == "active-hand":
        # One pass over the state column of every episode BEFORE writing any of them: the affine
        # has to be a property of the whole conversion set, and an affine that only existed once
        # the first half was already written would not be one.
        states = [
            read_source_episode(
                args.source / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
            )["state"]
            for i in indices
        ]
        if args.gripper_affine is None:
            affine = fit_hand_affine(states)
        else:
            affine = pinned_hand_affine(states, *args.gripper_affine)
        origin = "PINNED via --gripper-affine" if affine.pinned else "fitted"
        print(
            f"gripper affine ({origin}, dataset-level over {len(indices)} episodes): "
            f"active={affine.active} offset={affine.offset:.6g} span={affine.span:.6g} "
            f"(mean per-episode p2p left={affine.p2p_left:.4f} right={affine.p2p_right:.4f}). "
            "Pass --gripper-affine with exactly this offset/span to keep a later conversion on "
            "the same scale; a conversion over a different episode count fits a different one."
        )

    for i in indices:
        out_dir = args.out / f"gr00t-apple-{i:06d}"
        stats = convert_episode(
            args.source,
            i,
            out_dir,
            spec,
            instruction=instructions.get(i, "move the apple to the plate"),
            resize_hw=tuple(args.resize),
            chunk_steps=args.chunk_steps,
            gripper_affine=affine,
            source_dataset=source_dataset,
            source_dataset_verified=args.source_dataset is not None,
            record_provenance=record_provenance,
            info=info,
        )
        print(
            f"converted episode {i}: steps={stats['steps']} chunks={stats['chunks']} "
            f"dt_s={stats['dt_s']:.4f} max|delta|={stats['max_abs_delta']:.4f} -> {out_dir}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
