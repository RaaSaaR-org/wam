"""Synchronized capture (T-08, FR-01): timestamped pull sources -> aligned episode recording.

Contracts:
- Caller-driven stepping: no threads, no wall-clock sleeps. One ``SyncRecorder.step()`` pulls
  ONE state and ONE frame per camera, checks the timestamp spread across all of them against
  ``sync_tolerance_ns`` and only then writes to the ``EpisodeWriter``. This makes capture
  deterministic and lets it run on MockRobot's simulated clock.
- A violating sample is never partially written: with ``on_violation="raise"`` the step raises
  BEFORE anything reaches the writer; with ``"flag"`` the sample is recorded, marked
  ``within_tolerance=False`` and counted in ``num_violations``.
- ``max_sync_error_ns`` tracks the worst observed spread (state + all cameras), including
  samples that were raised/flagged.
- ``MockCaptureSession`` records the SAFE chunk (post safety filter — the action that was
  actually commanded, FR-07) together with the actually executed prefix (FR-05 semantics).
- Torch-free; numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from wam.data.episode import EpisodeManifest, EpisodeWriter
from wam.interfaces import (
    ActionChunk,
    NormalizationSpec,
    Observation,
    Policy,
    RobotAdapter,
    RobotState,
    SafetyFilter,
)
from wam.robot import MockRobot

DEFAULT_SYNC_TOLERANCE_NS = 20_000_000  # 20 ms
_DEFAULT_INSTRUCTION = "Greife die rote Tasse."


class SyncToleranceError(RuntimeError):
    """Timestamp spread of one capture step exceeded the configured tolerance (FR-01)."""


# -- source protocols -----------------------------------------------------------------------


@runtime_checkable
class FrameSource(Protocol):
    """Timestamped pull interface for one camera stream."""

    @property
    def name(self) -> str:
        """Camera name; becomes the episode stream name (``<name>.mp4``)."""
        ...

    def capture(self) -> tuple[np.ndarray, int]:
        """Return ``(rgb uint8 [H, W, 3], timestamp_ns)`` of the freshest frame."""
        ...


@runtime_checkable
class StateSource(Protocol):
    """Timestamped pull interface for the canonical robot state."""

    def capture(self) -> RobotState:
        """Return the freshest state; ``state.timestamp_ns`` is its capture time."""
        ...


# -- concrete sources -----------------------------------------------------------------------


class MockCameraSource:
    """``FrameSource`` over MockRobot's synthetic camera.

    Timestamps come from the robot's simulated clock plus a fixed ``offset_ns`` that models
    per-sensor clock skew (used to exercise the sync tolerance path in tests).
    """

    def __init__(self, robot: MockRobot, camera: str, *, offset_ns: int = 0) -> None:
        self._robot = robot
        self._camera = str(camera)
        self._offset_ns = int(offset_ns)

    @property
    def name(self) -> str:
        return self._camera

    def capture(self) -> tuple[np.ndarray, int]:
        frames = self._robot.render_frames(1)
        if self._camera not in frames:
            raise KeyError(f"unknown camera {self._camera!r}; robot has {sorted(frames)}")
        return frames[self._camera][0], self._robot.sim_time_ns + self._offset_ns


class RobotStateSource:
    """``StateSource`` over any ``RobotAdapter.read_state()``."""

    def __init__(self, robot: RobotAdapter) -> None:
        self._robot = robot

    def capture(self) -> RobotState:
        return self._robot.read_state()


# -- recorder -------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncSample:
    """One aligned capture step: state + one frame per camera + sync diagnostics."""

    state: RobotState
    images: dict[str, np.ndarray]  # camera -> uint8 [H, W, 3]
    frame_timestamps_ns: dict[str, int]  # camera -> capture timestamp
    sync_error_ns: int  # spread over state + all frame timestamps
    within_tolerance: bool


class SyncRecorder:
    """Polls sources, aligns by timestamp and records to an ``EpisodeWriter`` (FR-01).

    Caller-driven: each ``step()`` records exactly one sample (or raises/flags). Actions are
    recorded via ``add_action`` so a whole episode goes through one recording facade.
    """

    def __init__(
        self,
        writer: EpisodeWriter,
        sources: list[FrameSource] | tuple[FrameSource, ...],
        state_source: StateSource,
        sync_tolerance_ns: int,
        *,
        on_violation: Literal["raise", "flag"] = "raise",
    ) -> None:
        if sync_tolerance_ns < 0:
            raise ValueError(f"sync_tolerance_ns must be >= 0, got {sync_tolerance_ns}")
        if on_violation not in ("raise", "flag"):
            raise ValueError(f"on_violation must be 'raise' or 'flag', got {on_violation!r}")
        names = [source.name for source in sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate camera names in sources: {names}")
        self._writer = writer
        self._sources = tuple(sources)
        self._state_source = state_source
        self._tolerance_ns = int(sync_tolerance_ns)
        self._on_violation = on_violation
        self._max_sync_error_ns = 0
        self._num_samples = 0
        self._num_violations = 0
        self._num_chunks = 0

    # -- introspection -----------------------------------------------------------------

    @property
    def max_sync_error_ns(self) -> int:
        """Worst observed spread across all steps, including raised/flagged ones."""
        return self._max_sync_error_ns

    @property
    def num_samples(self) -> int:
        """Samples actually written (violating samples in 'raise' mode are not)."""
        return self._num_samples

    @property
    def num_violations(self) -> int:
        return self._num_violations

    @property
    def num_chunks(self) -> int:
        return self._num_chunks

    # -- recording ---------------------------------------------------------------------

    def step(self) -> SyncSample:
        """Pull one state + one frame per camera; check tolerance; write; return the sample."""
        state = self._state_source.capture()
        images: dict[str, np.ndarray] = {}
        frame_ts: dict[str, int] = {}
        for source in self._sources:
            img, ts = source.capture()
            images[source.name] = img
            frame_ts[source.name] = int(ts)

        values = [int(state.timestamp_ns), *frame_ts.values()]
        sync_error_ns = max(values) - min(values)
        self._max_sync_error_ns = max(self._max_sync_error_ns, sync_error_ns)
        within = sync_error_ns <= self._tolerance_ns
        if not within:
            self._num_violations += 1
            if self._on_violation == "raise":
                raise SyncToleranceError(
                    f"sync error {sync_error_ns} ns exceeds tolerance {self._tolerance_ns} ns "
                    f"(state at {state.timestamp_ns}, frames at {frame_ts})"
                )

        self._writer.add_state(state)
        for name, img in images.items():
            self._writer.add_frame(name, img, frame_ts[name])
        self._num_samples += 1
        return SyncSample(
            state=state,
            images=images,
            frame_timestamps_ns=frame_ts,
            sync_error_ns=sync_error_ns,
            within_tolerance=within,
        )

    def add_action(self, commanded: ActionChunk, executed_prefix: int, timestamp_ns: int) -> None:
        """Record one commanded chunk with its actually executed prefix."""
        self._writer.add_action(commanded, executed_prefix, timestamp_ns)
        self._num_chunks += 1


# -- mock capture session -------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of one recorded episode (manifest + sync/safety statistics)."""

    manifest: EpisodeManifest
    iterations: int
    max_sync_error_ns: int
    sync_violations: int
    interventions_total: int
    intervention_kinds: dict[str, int]


class MockCaptureSession:
    """Wires MockRobot + Policy + SafetyFilter into a ``SyncRecorder`` (T-08, no hardware).

    Per iteration: synchronized sample (state + all cameras) -> policy -> safety filter ->
    record the SAFE chunk with the actually executed prefix -> execute on the mock robot
    (which advances its simulated clock). Deterministic for fixed robot/policy seeds.
    """

    def __init__(
        self,
        robot: MockRobot,
        policy: Policy,
        safety: SafetyFilter,
        *,
        fps: float = 20.0,
        sync_tolerance_ns: int = DEFAULT_SYNC_TOLERANCE_NS,
        camera_offsets_ns: dict[str, int] | None = None,
        instruction: str = _DEFAULT_INSTRUCTION,
        on_violation: Literal["raise", "flag"] = "raise",
    ) -> None:
        self._robot = robot
        self._policy = policy
        self._safety = safety
        self._fps = float(fps)
        self._sync_tolerance_ns = int(sync_tolerance_ns)
        self._instruction = str(instruction)
        self._on_violation: Literal["raise", "flag"] = on_violation
        cameras = tuple(robot.render_frames(1))
        offsets = dict(camera_offsets_ns or {})
        unknown = set(offsets) - set(cameras)
        if unknown:
            raise ValueError(f"camera_offsets_ns for unknown cameras {sorted(unknown)}")
        self._sources: tuple[MockCameraSource, ...] = tuple(
            MockCameraSource(robot, cam, offset_ns=offsets.get(cam, 0)) for cam in cameras
        )

    def record_episode(
        self,
        dir: str | Path,
        episode_id: str,
        *,
        iterations: int,
        prefix_steps: int,
        normalization: dict[str, NormalizationSpec | dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CaptureResult:
        """Record one complete episode (frames + states + commanded/executed actions).

        ``normalization`` is stored in the manifest as provenance only and is NOT applied
        anywhere in the MVP pipeline (identity normalization — see ``NormalizationSpec``).
        """
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if prefix_steps < 1:
            raise ValueError(f"prefix_steps must be >= 1, got {prefix_steps}")
        kinds: dict[str, int] = {}
        with EpisodeWriter(
            dir,
            episode_id,
            self._robot.spec,
            self._fps,
            self._instruction,
            normalization=normalization,
            extra=extra,
        ) as writer:
            recorder = SyncRecorder(
                writer,
                self._sources,
                RobotStateSource(self._robot),
                self._sync_tolerance_ns,
                on_violation=self._on_violation,
            )
            for _ in range(iterations):
                sample = recorder.step()
                observation = Observation(
                    images=sample.images, state=sample.state, instruction=self._instruction
                )
                chunk = self._policy.predict(observation)
                safe_chunk, interventions = self._safety.filter(sample.state, chunk)
                for intervention in interventions:
                    kinds[intervention.kind] = kinds.get(intervention.kind, 0) + 1
                executed = (
                    0
                    if getattr(self._robot, "is_estopped", False)
                    else min(prefix_steps, safe_chunk.num_steps)
                )
                recorder.add_action(safe_chunk, executed, int(sample.state.timestamp_ns))
                self._robot.execute(safe_chunk, executed)
            manifest = writer.close()
        return CaptureResult(
            manifest=manifest,
            iterations=iterations,
            max_sync_error_ns=recorder.max_sync_error_ns,
            sync_violations=recorder.num_violations,
            interventions_total=sum(kinds.values()),
            intervention_kinds=kinds,
        )
