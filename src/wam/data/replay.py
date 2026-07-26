"""Episode replay + report generation (T-09, FR-08).

Contracts:
- ``replay_episode`` merges the recorded streams (states, per-camera frames, commanded
  chunks) of one episode into a single time-ordered iterator of ``ReplayStep``. Events
  sharing a timestamp are merged into ONE step; a same-timestamp collision on an already
  filled slot (second state, second frame of the same camera, second chunk) starts a new
  step at the same ``t_ns`` so no event is ever dropped.
- ``episode_report`` computes an ``EpisodeReport`` from stored data only (parquet tables +
  manifest; the video streams are NOT decoded). All statistics are float64 aggregates over
  the bit-exact float32 rows, so they are exactly reproducible offline (FR-08).
- Flags are intervention-relevant findings for dataset gates/operators (T-11 consumers),
  not hard errors: an empty or broken episode still yields a report.
- Torch-free; numpy + pydantic only.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wam.data.episode import EpisodeReader
from wam.interfaces.schema import ActionChunk, CanonicalSpaceSpec, RobotState

__all__ = [
    "GRIPPER_ACTIVITY_EPS",
    "ActionReport",
    "CameraReport",
    "EpisodeReport",
    "JointReport",
    "ReplayStep",
    "episode_report",
    "replay_episode",
]

#: |gripper delta| above this counts as gripper activity (gripper unit is [0, 1]).
GRIPPER_ACTIVITY_EPS = 1e-3

_KIND_STATE = 0
_KIND_FRAME = 1
_KIND_ACTION = 2

_NS_PER_S = 1_000_000_000


# -- replay ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayStep:
    """All episode events sharing one timestamp.

    Any subset of the slots may be filled; ``executed_prefix`` is ``None`` exactly when
    ``commanded`` is ``None``.
    """

    t_ns: int
    state: RobotState | None = None
    frames: dict[str, np.ndarray] | None = None  # camera -> uint8 [H, W, 3] RGB
    commanded: ActionChunk | None = None
    executed_prefix: int | None = None


def replay_episode(reader: EpisodeReader) -> Iterator[ReplayStep]:
    """Replay one episode as a time-ordered stream of ``ReplayStep`` (FR-08).

    Ordering: ascending ``t_ns``; ties are merged into one step (slot collisions split
    into consecutive steps at the same ``t_ns``, preserving recording order). Within one
    timestamp the merge order is state, then frames (camera name order), then action.

    Frames are decoded once per camera up-front via ``reader.read_frames`` — fine for
    MVP-length episodes; streaming decode is a dataset-scale concern.
    """
    events: list[tuple[int, int, dict[str, Any]]] = []
    for state in reader.read_states():
        events.append((int(state.timestamp_ns), _KIND_STATE, {"state": state}))
    for camera in sorted(reader.manifest.cameras):
        frames = reader.read_frames(camera)
        stamps = reader.frame_timestamps(camera)
        for idx in range(frames.shape[0]):
            events.append(
                (int(stamps[idx]), _KIND_FRAME, {"camera": camera, "frame": frames[idx]})
            )
    for chunk, prefix, ts in reader.read_actions():
        events.append((int(ts), _KIND_ACTION, {"chunk": chunk, "prefix": int(prefix)}))
    # Stable sort: within (t_ns, kind) the per-stream recording order is preserved.
    events.sort(key=lambda e: (e[0], e[1]))

    steps: list[ReplayStep] = []
    cur_t: int | None = None
    cur_state: RobotState | None = None
    cur_frames: dict[str, np.ndarray] = {}
    cur_chunk: ActionChunk | None = None
    cur_prefix: int | None = None

    def flush() -> None:
        if cur_t is None:
            return
        steps.append(
            ReplayStep(
                t_ns=cur_t,
                state=cur_state,
                frames=dict(cur_frames) if cur_frames else None,
                commanded=cur_chunk,
                executed_prefix=cur_prefix,
            )
        )

    for t_ns, kind, payload in events:
        collision = (
            (kind == _KIND_STATE and cur_state is not None)
            or (kind == _KIND_FRAME and payload["camera"] in cur_frames)
            or (kind == _KIND_ACTION and cur_chunk is not None)
        )
        if cur_t is not None and (t_ns != cur_t or collision):
            flush()
            cur_state, cur_frames, cur_chunk, cur_prefix = None, {}, None, None
        cur_t = t_ns
        if kind == _KIND_STATE:
            cur_state = payload["state"]
        elif kind == _KIND_FRAME:
            cur_frames[payload["camera"]] = payload["frame"]
        else:
            cur_chunk = payload["chunk"]
            cur_prefix = payload["prefix"]
    flush()
    return iter(steps)


# -- report models ---------------------------------------------------------------------------


class CameraReport(BaseModel):
    """Per-camera frame statistics (from manifest + timestamp sidecar, no video decode)."""

    model_config = ConfigDict(frozen=True)

    num_frames: int = Field(ge=0)
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    fps_nominal: float = Field(gt=0)
    duration_s: float = 0.0  # last_ts - first_ts; 0.0 with fewer than 2 frames
    mean_dt_ns: float | None = None  # (last - first) / (n - 1); None with < 2 frames
    fps_actual: float | None = None  # 1e9 / mean_dt_ns; None with < 2 frames


class JointReport(BaseModel):
    """Range coverage + velocity statistics of one canonical joint over all states."""

    model_config = ConfigDict(frozen=True)

    name: str
    q_min: float
    q_max: float
    q_mean: float
    dq_abs_max: float
    dq_abs_mean: float


class ActionReport(BaseModel):
    """Aggregate statistics over all commanded chunks."""

    model_config = ConfigDict(frozen=True)

    num_chunks: int = Field(ge=0)
    num_steps: int = Field(ge=0)
    executed_steps: int = Field(ge=0)  # sum of executed prefixes
    executed_ratio: float = 0.0  # executed_steps / num_steps (0.0 when no steps)
    modes: dict[str, int] = Field(default_factory=dict)  # ActionMode value -> chunk count
    step_norm_mean: float = 0.0  # mean L2 norm of per-step target vectors (delta magnitude)
    step_norm_max: float = 0.0
    gripper_min: float = 0.0
    gripper_max: float = 0.0
    gripper_mean_abs_delta: float = 0.0  # mean |g[t+1]-g[t]| within chunks
    gripper_active_fraction: float = 0.0  # fraction of within-chunk deltas > eps


class EpisodeReport(BaseModel):
    """Episode-level report (FR-08): everything a dataset gate / operator review needs."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    instruction: str
    format_version: str
    schema_version: str
    t0_ns: int
    t1_ns: int
    duration_s: float
    num_states: int = Field(ge=0)
    max_sync_error_ns: int = Field(ge=0)
    sync_tolerance_ns: int | None = None  # resolved tolerance; None when no cameras
    cameras: dict[str, CameraReport] = Field(default_factory=dict)
    joints: tuple[JointReport, ...] = ()
    actions: ActionReport
    flags: tuple[str, ...] = ()

    def render_markdown(self) -> str:
        """Human-readable markdown rendering of the full report."""
        a = self.actions
        lines = [
            f"# Episode report — {self.episode_id}",
            "",
            f"- instruction: {self.instruction or '(none)'}",
            f"- duration: {self.duration_s:.3f} s (t0={self.t0_ns} ns, t1={self.t1_ns} ns)",
            f"- states: {self.num_states} | chunks: {a.num_chunks} | steps: {a.num_steps}",
            f"- versions: format {self.format_version}, schema {self.schema_version}",
            "- max sync error: "
            + f"{self.max_sync_error_ns} ns"
            + (
                f" (tolerance {self.sync_tolerance_ns} ns)"
                if self.sync_tolerance_ns is not None
                else ""
            ),
            "",
            "## Cameras",
            "",
        ]
        if self.cameras:
            lines += [
                "| camera | frames | size | fps nominal | fps actual | mean dt [ms] |",
                "|---|---|---|---|---|---|",
            ]
            for name in sorted(self.cameras):
                c = self.cameras[name]
                actual = _fmt(c.fps_actual) if c.fps_actual is not None else "-"
                dt_ms = _fmt(c.mean_dt_ns / 1e6) if c.mean_dt_ns is not None else "-"
                lines.append(
                    f"| {name} | {c.num_frames} | {c.width}x{c.height} "
                    f"| {_fmt(c.fps_nominal)} | {actual} | {dt_ms} |"
                )
        else:
            lines.append("(no cameras)")
        lines += ["", "## Joint coverage", ""]
        if self.joints:
            lines += [
                "| joint | q min | q max | q mean | \\|dq\\| max | \\|dq\\| mean |",
                "|---|---|---|---|---|---|",
            ]
            for j in self.joints:
                lines.append(
                    f"| {j.name} | {_fmt(j.q_min)} | {_fmt(j.q_max)} | {_fmt(j.q_mean)} "
                    f"| {_fmt(j.dq_abs_max)} | {_fmt(j.dq_abs_mean)} |"
                )
        else:
            lines.append("(no states)")
        lines += [
            "",
            "## Actions",
            "",
            f"- chunks: {a.num_chunks} ({_fmt_modes(a.modes)})",
            (
                f"- steps: {a.num_steps}, executed: {a.executed_steps} "
                f"(ratio {_fmt(a.executed_ratio)})"
            ),
            f"- step target norm: mean {_fmt(a.step_norm_mean)}, max {_fmt(a.step_norm_max)}",
            (
                f"- gripper: range [{_fmt(a.gripper_min)}, {_fmt(a.gripper_max)}], "
                f"mean |delta| {_fmt(a.gripper_mean_abs_delta)}, "
                f"active fraction {_fmt(a.gripper_active_fraction)}"
            ),
            "",
            "## Flags",
            "",
        ]
        if self.flags:
            lines += [f"- {flag}" for flag in self.flags]
        else:
            lines.append("- none")
        return "\n".join(lines) + "\n"


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _fmt_modes(modes: dict[str, int]) -> str:
    if not modes:
        return "no modes"
    return ", ".join(f"{k}: {modes[k]}" for k in sorted(modes))


# -- report computation ----------------------------------------------------------------------


def episode_report(
    reader: EpisodeReader, *, sync_tolerance_ns: int | None = None
) -> EpisodeReport:
    """Compute an ``EpisodeReport`` from stored tables + manifest (video is not decoded).

    ``sync_tolerance_ns``: threshold for the ``sync_error_exceeds_tolerance`` flag.
    Default: half the nominal frame period of the fastest camera; ``None`` stays ``None``
    when the episode has no cameras (no sync check possible).
    """
    manifest = reader.manifest
    states = reader.read_states()
    chunks = reader.read_actions()
    spec = manifest.spec

    cameras = {
        name: _camera_report(info.fps, info.width, info.height, reader.frame_timestamps(name))
        for name, info in manifest.cameras.items()
    }
    if sync_tolerance_ns is None and manifest.cameras:
        fastest = max(info.fps for info in manifest.cameras.values())
        sync_tolerance_ns = int(0.5 * _NS_PER_S / fastest)

    return EpisodeReport(
        episode_id=manifest.episode_id,
        instruction=manifest.instruction,
        format_version=manifest.format_version,
        schema_version=manifest.schema_version,
        t0_ns=manifest.t0_ns,
        t1_ns=manifest.t1_ns,
        duration_s=(manifest.t1_ns - manifest.t0_ns) / _NS_PER_S,
        num_states=len(states),
        max_sync_error_ns=manifest.max_sync_error_ns,
        sync_tolerance_ns=sync_tolerance_ns,
        cameras=cameras,
        joints=_joint_reports(states, spec),
        actions=_action_report(chunks),
        flags=_flags(manifest, states, chunks, spec, sync_tolerance_ns),
    )


def _camera_report(fps: float, width: int, height: int, stamps: np.ndarray) -> CameraReport:
    n = int(stamps.shape[0])
    duration_s = 0.0
    mean_dt: float | None = None
    fps_actual: float | None = None
    if n >= 2:
        span = int(stamps[-1]) - int(stamps[0])
        duration_s = span / _NS_PER_S
        mean_dt = span / (n - 1)
        fps_actual = _NS_PER_S / mean_dt if mean_dt > 0 else None
    return CameraReport(
        num_frames=n,
        width=width,
        height=height,
        fps_nominal=fps,
        duration_s=duration_s,
        mean_dt_ns=mean_dt,
        fps_actual=fps_actual,
    )


def _joint_reports(
    states: list[RobotState], spec: CanonicalSpaceSpec
) -> tuple[JointReport, ...]:
    """Per-joint q min/max/mean and |dq| max/mean; empty if shapes are unusable."""
    n = spec.num_joints
    usable = states and all(
        isinstance(s.q, np.ndarray)
        and s.q.shape == (n,)
        and isinstance(s.dq, np.ndarray)
        and s.dq.shape == (n,)
        for s in states
    )
    if not usable:
        return ()
    q = np.stack([np.asarray(s.q, dtype=np.float64) for s in states])  # [S, N]
    dq_abs = np.abs(np.stack([np.asarray(s.dq, dtype=np.float64) for s in states]))
    return tuple(
        JointReport(
            name=spec.joint_names[j],
            q_min=float(q[:, j].min()),
            q_max=float(q[:, j].max()),
            q_mean=float(q[:, j].mean()),
            dq_abs_max=float(dq_abs[:, j].max()),
            dq_abs_mean=float(dq_abs[:, j].mean()),
        )
        for j in range(n)
    )


def _action_report(chunks: list[tuple[ActionChunk, int, int]]) -> ActionReport:
    num_steps = 0
    executed = 0
    modes: dict[str, int] = {}
    norms: list[float] = []
    grip: list[float] = []
    deltas: list[float] = []
    for chunk, prefix, _ts in chunks:
        steps = chunk.num_steps
        num_steps += steps
        executed += min(max(int(prefix), 0), steps)
        key = chunk.mode.value if hasattr(chunk.mode, "value") else str(chunk.mode)
        modes[key] = modes.get(key, 0) + 1
        targets = np.asarray(chunk.targets, dtype=np.float64)
        if targets.ndim == 2:
            norms.extend(float(v) for v in np.linalg.norm(targets, axis=1))
        g = np.asarray(chunk.gripper_target, dtype=np.float64).reshape(-1)
        grip.extend(float(v) for v in g)
        if g.size >= 2:
            deltas.extend(float(v) for v in np.abs(np.diff(g)))
    return ActionReport(
        num_chunks=len(chunks),
        num_steps=num_steps,
        executed_steps=executed,
        executed_ratio=executed / num_steps if num_steps else 0.0,
        modes=modes,
        step_norm_mean=float(np.mean(norms)) if norms else 0.0,
        step_norm_max=float(np.max(norms)) if norms else 0.0,
        gripper_min=float(np.min(grip)) if grip else 0.0,
        gripper_max=float(np.max(grip)) if grip else 0.0,
        gripper_mean_abs_delta=float(np.mean(deltas)) if deltas else 0.0,
        gripper_active_fraction=(
            float(np.mean([d > GRIPPER_ACTIVITY_EPS for d in deltas])) if deltas else 0.0
        ),
    )


def _flags(
    manifest: Any,
    states: list[RobotState],
    chunks: list[tuple[ActionChunk, int, int]],
    spec: CanonicalSpaceSpec,
    sync_tolerance_ns: int | None,
) -> tuple[str, ...]:
    """Intervention-relevant findings, in a fixed deterministic order."""
    flags: list[str] = []
    if not states:
        flags.append("no_states")
    if not chunks:
        flags.append("no_actions")
    if not manifest.cameras:
        flags.append("no_cameras")
    if states:
        if any(s.validate(spec) for s in states):
            flags.append("invalid_states")
        ts = np.asarray([s.timestamp_ns for s in states], dtype=np.int64)
        if ts.size >= 2 and bool(np.any(np.diff(ts) < 0)):
            flags.append("state_timestamps_not_monotonic")
        for group in ("q", "dq", "imu", "gripper"):
            if any(not getattr(s.validity, group) for s in states):
                flags.append(f"validity_gap:{group}")
    if chunks:
        if any(chunk.validate(spec) for chunk, _p, _t in chunks):
            flags.append("invalid_action_chunks")
        if any(prefix == 0 for _c, prefix, _t in chunks):
            flags.append("discarded_chunks")
    if sync_tolerance_ns is not None and manifest.max_sync_error_ns > sync_tolerance_ns:
        flags.append("sync_error_exceeds_tolerance")
    return tuple(flags)
