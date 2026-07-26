"""Automatic dataset validation gates (T-11, PRD §10/§15 R-04): release gates, not hints.

Contracts:
- ``validate_episode`` accepts a directory (preferred — checksum failures are then a FAILED
  GATE instead of an exception) or an already-open ``EpisodeReader``.
- Gates never raise: a crashing gate is a failed gate with the exception in ``detail``.
  If the checksum gate fails, the remaining gates are NOT run (their results would be
  meaningless on tampered data) — the report contains exactly the failed checksum gate.
- All thresholds live in ``ValidationThresholds`` (frozen pydantic); defaults suit the mock
  D0 setup and MUST be tuned per real deployment.
- Reports are JSON-serializable (``to_json``) so they can be archived next to the dataset
  snapshot (AC-04 traceability).
- Torch-free; numpy + pyarrow via the episode reader.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from wam.data.episode import (
    ACTIONS_TABLE,
    STATES_TABLE,
    EpisodeFormatError,
    EpisodeManifest,
    EpisodeReader,
    list_episodes,
)
from wam.interfaces import ActionChunk, RobotState

_NS_PER_S = 1_000_000_000

# Episode-level gate names (stable identifiers, used in reports and tests).
GATE_READABLE = "readable"
GATE_CHECKSUMS = "checksums"
GATE_MONOTONIC = "monotonic_timestamps"
GATE_SYNC = "sync_error"
GATE_FINITE = "finite_values"
GATE_COVERAGE = "state_coverage"
GATE_COUNTS = "counts"
GATE_DURATION = "duration"
GATE_FRAMES = "frame_integrity"

EPISODE_GATES: tuple[str, ...] = (
    GATE_CHECKSUMS,
    GATE_MONOTONIC,
    GATE_SYNC,
    GATE_FINITE,
    GATE_COVERAGE,
    GATE_COUNTS,
    GATE_DURATION,
    GATE_FRAMES,
)

# Dataset-level gate names.
GATE_EPISODE_COUNT = "episode_count"
GATE_TOTAL_DURATION = "total_duration"
GATE_EPISODES_VALID = "episodes_valid"
GATE_UNIQUE_IDS = "unique_episode_ids"

DATASET_GATES: tuple[str, ...] = (
    GATE_EPISODE_COUNT,
    GATE_TOTAL_DURATION,
    GATE_EPISODES_VALID,
    GATE_UNIQUE_IDS,
)


class ValidationThresholds(BaseModel):
    """All tunable gate thresholds; defaults match the mock D0 recording setup."""

    model_config = ConfigDict(frozen=True)

    sync_tolerance_ns: int = Field(default=20_000_000, ge=0)
    min_duration_s: float = Field(default=0.5, ge=0)
    max_duration_s: float = Field(default=3600.0, gt=0)
    min_states: int = Field(default=2, ge=1)
    min_state_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    min_episodes: int = Field(default=1, ge=1)
    min_total_duration_s: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> ValidationThresholds:
        if self.max_duration_s < self.min_duration_s:
            raise ValueError("max_duration_s must be >= min_duration_s")
        return self


class GateResult(BaseModel):
    """Outcome of one validation gate."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """All gate results for one episode."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    path: str
    duration_s: float = 0.0
    gates: list[GateResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    def failed_gates(self) -> list[str]:
        return [gate.name for gate in self.gates if not gate.passed]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


class DatasetReport(BaseModel):
    """Aggregated per-episode reports plus dataset-level gates."""

    model_config = ConfigDict(frozen=True)

    root: str
    episodes: list[ValidationReport] = Field(default_factory=list)
    gates: list[GateResult] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return (
            bool(self.gates)
            and all(gate.passed for gate in self.gates)
            and all(episode.passed for episode in self.episodes)
        )

    def failed_gates(self) -> list[str]:
        return [gate.name for gate in self.gates if not gate.passed]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)


# -- internals ------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _EpisodeContext:
    """Lazily loaded, cached views of one episode for the gate functions."""

    def __init__(self, reader: EpisodeReader, dir: Path, thresholds: ValidationThresholds):
        self.reader = reader
        self.dir = dir
        self.thresholds = thresholds
        self.manifest: EpisodeManifest = reader.manifest
        self._states: list[RobotState] | None = None
        self._actions: list[tuple[ActionChunk, int, int]] | None = None
        self._frame_ts: dict[str, np.ndarray] = {}

    @property
    def duration_s(self) -> float:
        return (self.manifest.t1_ns - self.manifest.t0_ns) / _NS_PER_S

    def states(self) -> list[RobotState]:
        if self._states is None:
            self._states = self.reader.read_states()
        return self._states

    def actions(self) -> list[tuple[ActionChunk, int, int]]:
        if self._actions is None:
            self._actions = self.reader.read_actions()
        return self._actions

    def frame_ts(self, camera: str) -> np.ndarray:
        if camera not in self._frame_ts:
            self._frame_ts[camera] = self.reader.frame_timestamps(camera)
        return self._frame_ts[camera]


def _result(name: str, problems: list[str], metrics: dict[str, Any]) -> GateResult:
    return GateResult(name=name, passed=not problems, detail="; ".join(problems), metrics=metrics)


# -- episode gates --------------------------------------------------------------------------


def _gate_checksums(ctx: _EpisodeContext) -> GateResult:
    problems: list[str] = []
    for fname, expected in sorted(ctx.manifest.checksums.items()):
        path = ctx.dir / fname
        if not path.is_file():
            problems.append(f"missing file {fname!r}")
        elif _sha256(path) != expected:
            problems.append(f"sha256 mismatch for {fname!r}")
    return _result(
        GATE_CHECKSUMS,
        problems,
        {"files": len(ctx.manifest.checksums), "problems": len(problems)},
    )


def _gate_monotonic(ctx: _EpisodeContext) -> GateResult:
    problems: list[str] = []
    state_ts = np.asarray([s.timestamp_ns for s in ctx.states()], dtype=np.int64)
    if state_ts.size >= 2 and not (np.diff(state_ts) > 0).all():
        problems.append("state timestamps not strictly increasing")
    action_ts = np.asarray([ts for _, _, ts in ctx.actions()], dtype=np.int64)
    if action_ts.size >= 2 and (np.diff(action_ts) < 0).any():
        problems.append("action timestamps decreasing")
    for camera in sorted(ctx.manifest.cameras):
        ts = ctx.frame_ts(camera)
        if ts.size >= 2 and (np.diff(ts) < 0).any():
            problems.append(f"{camera}: frame timestamps decreasing")
    return _result(
        GATE_MONOTONIC,
        problems,
        {"num_states": int(state_ts.size), "num_action_chunks": int(action_ts.size)},
    )


def _gate_sync(ctx: _EpisodeContext) -> GateResult:
    """Camera-camera spread per frame index + worst state<->frame alignment vs tolerance."""
    tolerance = ctx.thresholds.sync_tolerance_ns
    cameras = sorted(ctx.manifest.cameras)
    series = [ctx.frame_ts(camera) for camera in cameras]

    camera_spread = 0
    if len(series) >= 2:
        n = min(s.size for s in series)
        if n:
            stacked = np.stack([s[:n] for s in series])
            camera_spread = int((stacked.max(axis=0) - stacked.min(axis=0)).max())

    state_frame = 0
    state_ts = np.sort(np.asarray([s.timestamp_ns for s in ctx.states()], dtype=np.int64))
    if state_ts.size:
        for s in series:
            if not s.size:
                continue
            idx = np.searchsorted(state_ts, s)
            hi = np.abs(s - state_ts[np.clip(idx, 0, state_ts.size - 1)])
            lo = np.abs(s - state_ts[np.clip(idx - 1, 0, state_ts.size - 1)])
            state_frame = max(state_frame, int(np.minimum(hi, lo).max()))

    observed = max(camera_spread, state_frame, ctx.manifest.max_sync_error_ns)
    problems: list[str] = []
    if observed > tolerance:
        problems.append(f"max sync error {observed} ns > tolerance {tolerance} ns")
    return _result(
        GATE_SYNC,
        problems,
        {
            "camera_spread_ns": camera_spread,
            "state_frame_ns": state_frame,
            "manifest_ns": ctx.manifest.max_sync_error_ns,
            "tolerance_ns": tolerance,
        },
    )


def _gate_finite(ctx: _EpisodeContext) -> GateResult:
    problems: list[str] = []
    for i, state in enumerate(ctx.states()):
        arrays = (
            state.q,
            state.dq,
            state.gripper_state,
            state.imu.orientation_wxyz,
            state.imu.angular_velocity,
            state.imu.linear_acceleration,
        )
        if any(not np.isfinite(np.asarray(a, dtype=np.float64)).all() for a in arrays):
            problems.append(f"state[{i}] contains NaN/Inf")
    for k, (chunk, _, _) in enumerate(ctx.actions()):
        if (
            not np.isfinite(chunk.targets).all()
            or not np.isfinite(chunk.gripper_target).all()
            or not math.isfinite(chunk.dt_s)
        ):
            problems.append(f"chunk[{k}] contains NaN/Inf")
    return _result(GATE_FINITE, problems[:8], {"problems": len(problems)})


def _gate_coverage(ctx: _EpisodeContext) -> GateResult:
    """Fields marked valid must have canonical shapes; q/dq coverage must meet the minimum."""
    spec = ctx.manifest.spec
    n, g = spec.num_joints, spec.gripper_dims
    problems: list[str] = []
    states = ctx.states()
    total = len(states)
    q_ok = dq_ok = 0
    for i, state in enumerate(states):
        if state.validity.q:
            if state.q.shape == (n,):
                q_ok += 1
            else:
                problems.append(f"state[{i}].q shape {state.q.shape} != ({n},)")
        if state.validity.dq:
            if state.dq.shape == (n,):
                dq_ok += 1
            else:
                problems.append(f"state[{i}].dq shape {state.dq.shape} != ({n},)")
        if state.validity.gripper and state.gripper_state.shape != (g,):
            problems.append(f"state[{i}].gripper_state shape {state.gripper_state.shape} != ({g},)")
    coverage_q = q_ok / total if total else 0.0
    coverage_dq = dq_ok / total if total else 0.0
    minimum = ctx.thresholds.min_state_coverage
    if coverage_q < minimum:
        problems.append(f"q coverage {coverage_q:.3f} < {minimum:.3f}")
    if coverage_dq < minimum:
        problems.append(f"dq coverage {coverage_dq:.3f} < {minimum:.3f}")
    return _result(
        GATE_COVERAGE,
        problems[:8],
        {"coverage_q": coverage_q, "coverage_dq": coverage_dq, "num_states": total},
    )


def _gate_counts(ctx: _EpisodeContext) -> GateResult:
    """Action/state/frame counts consistent with the manifest and the minimum thresholds."""
    manifest = ctx.manifest
    problems: list[str] = []
    states = ctx.states()
    chunks = ctx.actions()

    states_info = manifest.tables.get(STATES_TABLE)
    if states_info is None or states_info.num_rows != len(states):
        rows = None if states_info is None else states_info.num_rows
        problems.append(f"states rows {len(states)} != manifest {rows}")
    steps_total = sum(chunk.num_steps for chunk, _, _ in chunks)
    actions_info = manifest.tables.get(ACTIONS_TABLE)
    if actions_info is None or actions_info.num_rows != steps_total:
        problems.append(
            f"action rows {steps_total} != manifest {getattr(actions_info, 'num_rows', None)}"
        )
    if len(states) < ctx.thresholds.min_states:
        problems.append(f"{len(states)} states < min_states {ctx.thresholds.min_states}")
    if not chunks:
        problems.append("no action chunks")
    for k, (chunk, prefix, _) in enumerate(chunks):
        expected_dim = manifest.spec.target_dim(chunk.mode)
        if chunk.targets.shape[1] != expected_dim:
            problems.append(f"chunk[{k}] target dim {chunk.targets.shape[1]} != {expected_dim}")
        if not 0 <= prefix <= chunk.num_steps:
            problems.append(f"chunk[{k}] executed_prefix {prefix} outside [0, {chunk.num_steps}]")
    for camera, info in sorted(manifest.cameras.items()):
        sidecar = manifest.tables.get(f"{camera}_timestamps")
        if sidecar is None or sidecar.num_rows != info.num_frames:
            problems.append(f"{camera}: timestamp rows != num_frames {info.num_frames}")
    return _result(
        GATE_COUNTS,
        problems[:8],
        {"num_states": len(states), "num_chunks": len(chunks), "action_rows": steps_total},
    )


def _gate_duration(ctx: _EpisodeContext) -> GateResult:
    duration = ctx.duration_s
    lo, hi = ctx.thresholds.min_duration_s, ctx.thresholds.max_duration_s
    problems: list[str] = []
    if not lo <= duration <= hi:
        problems.append(f"duration {duration:.3f} s outside [{lo}, {hi}] s")
    return _result(GATE_DURATION, problems, {"duration_s": duration})


def _gate_frames(ctx: _EpisodeContext) -> GateResult:
    """Decode every stream; count/shape vs manifest is enforced by the reader."""
    problems: list[str] = []
    counts: dict[str, int] = {}
    for camera in sorted(ctx.manifest.cameras):
        try:
            frames = ctx.reader.read_frames(camera)
        except Exception as exc:  # noqa: BLE001 — any decode failure must fail the gate
            problems.append(f"{camera}: {type(exc).__name__}: {exc}")
            continue
        counts[camera] = int(frames.shape[0])
        if frames.shape[0] == 0:
            problems.append(f"{camera}: zero frames")
    return _result(GATE_FRAMES, problems, {"frames": counts})


_EPISODE_GATE_FUNCS: tuple[tuple[str, Callable[[_EpisodeContext], GateResult]], ...] = (
    (GATE_MONOTONIC, _gate_monotonic),
    (GATE_SYNC, _gate_sync),
    (GATE_FINITE, _gate_finite),
    (GATE_COVERAGE, _gate_coverage),
    (GATE_COUNTS, _gate_counts),
    (GATE_DURATION, _gate_duration),
    (GATE_FRAMES, _gate_frames),
)


# -- public API -----------------------------------------------------------------------------


def validate_episode(
    episode: EpisodeReader | str | Path,
    thresholds: ValidationThresholds | None = None,
) -> ValidationReport:
    """Run all episode gates; never raises for bad data (bad data == failed gates)."""
    t = thresholds if thresholds is not None else ValidationThresholds()
    if isinstance(episode, EpisodeReader):
        reader = episode
        dir = Path(getattr(reader, "_dir"))  # noqa: B009 — reader does not expose its dir
    else:
        dir = Path(episode)
        try:
            reader = EpisodeReader(dir, verify_checksums=False)
        except (FileNotFoundError, EpisodeFormatError, ValueError) as exc:
            return ValidationReport(
                episode_id=dir.name,
                path=str(dir),
                gates=[
                    GateResult(
                        name=GATE_READABLE,
                        passed=False,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                ],
            )

    ctx = _EpisodeContext(reader, dir, t)
    gates: list[GateResult] = [_gate_checksums(ctx)]
    if gates[0].passed:
        for name, gate_fn in _EPISODE_GATE_FUNCS:
            try:
                gates.append(gate_fn(ctx))
            except Exception as exc:  # noqa: BLE001 — a crashing gate is a failed gate
                gates.append(
                    GateResult(
                        name=name, passed=False, detail=f"gate crashed: {type(exc).__name__}: {exc}"
                    )
                )
    return ValidationReport(
        episode_id=ctx.manifest.episode_id,
        path=str(dir),
        duration_s=ctx.duration_s,
        gates=gates,
    )


def validate_dataset(
    root: str | Path,
    thresholds: ValidationThresholds | None = None,
) -> DatasetReport:
    """Validate every episode under ``root`` and apply the dataset-level gates."""
    t = thresholds if thresholds is not None else ValidationThresholds()
    root = Path(root)
    episodes = [validate_episode(dir, t) for dir in list_episodes(root)]

    count = len(episodes)
    total_duration = float(sum(report.duration_s for report in episodes))
    failing = [report.episode_id for report in episodes if not report.passed]
    ids = [report.episode_id for report in episodes]
    duplicates = sorted({episode_id for episode_id in ids if ids.count(episode_id) > 1})

    gates = [
        GateResult(
            name=GATE_EPISODE_COUNT,
            passed=count >= t.min_episodes,
            detail="" if count >= t.min_episodes else f"{count} episodes < min {t.min_episodes}",
            metrics={"episodes": count, "min_episodes": t.min_episodes},
        ),
        GateResult(
            name=GATE_TOTAL_DURATION,
            passed=total_duration >= t.min_total_duration_s,
            detail=""
            if total_duration >= t.min_total_duration_s
            else f"total {total_duration:.2f} s < min {t.min_total_duration_s} s",
            metrics={
                "total_duration_s": total_duration,
                "min_total_duration_s": t.min_total_duration_s,
            },
        ),
        GateResult(
            name=GATE_EPISODES_VALID,
            passed=not failing,
            detail="" if not failing else f"failing episodes: {', '.join(failing)}",
            metrics={"failing": len(failing)},
        ),
        GateResult(
            name=GATE_UNIQUE_IDS,
            passed=not duplicates,
            detail="" if not duplicates else f"duplicate episode_ids: {', '.join(duplicates)}",
            metrics={"duplicates": len(duplicates)},
        ),
    ]
    return DatasetReport(root=str(root), episodes=episodes, gates=gates)
