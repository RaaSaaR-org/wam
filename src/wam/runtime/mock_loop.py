"""M0 end-to-end mock loop (T-03): read -> predict -> safety filter -> execute -> log.

Contracts:
- No hardware and no wall-clock control flow: robot time is the adapter's simulated clock,
  watchdog time is injected from it (plus an optional simulated policy stall), so runs are
  deterministic and testable.
- Policy output ALWAYS passes the safety filter before the robot adapter sees it (FR-07).
- Every iteration writes exactly one JSONL record: state digest, chunk digests, ALL safety
  interventions, watchdog decision and timings. Interventions are never swallowed.
- On watchdog expiry the stale chunk is DISCARDED, never executed: HOLD -> ``robot.hold()``,
  STOP -> ``robot.estop()`` (PRD §11.1 Recovery). The watchdog is re-armed only after the
  safe state has been commanded.
- Torch-free; numpy only.
"""

from __future__ import annotations

import math
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    JsonlRunLogger,
    Observation,
    Policy,
    RobotAdapter,
    RobotState,
    SafetyFilter,
    SafetyIntervention,
)
from wam.safety import Watchdog, WatchdogAction

_NS_PER_S = 1_000_000_000

DEFAULT_INSTRUCTION = "Greife die rote Tasse."


class DummyPolicy:
    """Deterministic sinusoid joint-delta policy (T-03 dummy). Implements ``Policy``.

    Contracts:
    - Phase derives ONLY from ``observation.state.timestamp_ns``: identical observation
      timestamps yield bit-identical chunks (stateless, replayable).
    - Per joint j (1-based): q_j(t) = A * (j/N) * (1 - cos(2*pi*t/period)) — zero velocity at
      t=0, bounded velocity A*(j/N)*omega and acceleration A*(j/N)*omega^2. Defaults stay
      inside ``configs/safety/default.yaml`` limits, so a clean run needs zero interventions.
    - Gripper follows a slow cosine in [0, 1]; chunk targets are raw joint deltas in rad
      (identity normalization, matching MockRobot's interpretation).
    """

    def __init__(
        self,
        spec: CanonicalSpaceSpec,
        *,
        steps: int = 8,
        dt_s: float = 0.05,
        amplitude_rad: float = 0.2,
        period_s: float = 2.0,
        gripper_period_s: float = 4.0,
    ) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if not dt_s > 0:
            raise ValueError(f"dt_s must be > 0, got {dt_s}")
        if amplitude_rad < 0:
            raise ValueError(f"amplitude_rad must be >= 0, got {amplitude_rad}")
        if not period_s > 0 or not gripper_period_s > 0:
            raise ValueError("period_s and gripper_period_s must be > 0")
        self._spec = spec
        self._steps = int(steps)
        self._dt_s = float(dt_s)
        self._amplitude_rad = float(amplitude_rad)
        self._period_s = float(period_s)
        self._gripper_period_s = float(gripper_period_s)
        n = spec.num_joints
        self._joint_scale = np.arange(1, n + 1, dtype=np.float64) / n

    def predict(self, observation: Observation) -> ActionChunk:
        """Fresh observation -> next JOINT_DELTA chunk; phase anchored at the state timestamp."""
        t0 = observation.state.timestamp_ns / _NS_PER_S
        omega = 2.0 * math.pi / self._period_s
        times = t0 + np.arange(self._steps + 1, dtype=np.float64) * self._dt_s
        envelope = 1.0 - np.cos(omega * times)  # [steps + 1]
        q = self._amplitude_rad * envelope[:, None] * self._joint_scale[None, :]
        targets = np.diff(q, axis=0)  # [steps, num_joints] joint deltas in rad
        g_omega = 2.0 * math.pi / self._gripper_period_s
        gripper = 0.5 - 0.5 * np.cos(g_omega * times[1:])
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=targets.astype(np.float32),
            gripper_target=np.clip(gripper, 0.0, 1.0).astype(np.float32),
            dt_s=self._dt_s,
        )


@dataclass
class LoopResult:
    """Aggregate outcome of one mock-loop run; per-iteration detail lives in the JSONL log."""

    iterations: int
    executed_iterations: int
    watchdog_timeouts: int
    interventions_total: int
    intervention_kinds: dict[str, int] = field(default_factory=dict)


def _state_digest(state: RobotState) -> dict[str, Any]:
    """Compact JSON-safe summary of a RobotState (full arrays stay out of the log)."""
    q = np.asarray(state.q, dtype=np.float64)
    dq = np.asarray(state.dq, dtype=np.float64)
    gripper = np.asarray(state.gripper_state, dtype=np.float64).ravel()
    return {
        "timestamp_ns": int(state.timestamp_ns),
        "q_mean": float(q.mean()) if q.size else None,
        "q_abs_max": float(np.abs(q).max()) if q.size else None,
        "dq_abs_max": float(np.abs(dq).max()) if dq.size else None,
        "gripper": [float(g) for g in gripper],
        "validity": state.validity.as_dict(),
    }


def _chunk_digest(chunk: ActionChunk) -> dict[str, Any]:
    """Compact JSON-safe summary of an ActionChunk (never emits NaN into the log)."""
    targets = np.asarray(chunk.targets, dtype=np.float64)
    finite = np.isfinite(targets)
    return {
        "mode": chunk.mode.value if isinstance(chunk.mode, ActionMode) else str(chunk.mode),
        "num_steps": int(chunk.num_steps),
        "dt_s": float(chunk.dt_s),
        "targets_abs_max": float(np.abs(targets[finite]).max()) if finite.any() else None,
        "targets_finite": bool(finite.all()),
    }


def _render_images(robot: RobotAdapter) -> dict[str, np.ndarray]:
    """One [H, W, 3] frame per camera if the adapter exposes ``render_frames`` (mock does)."""
    render = getattr(robot, "render_frames", None)
    if not callable(render):
        return {}
    return {name: frames[0] for name, frames in render(1).items()}


def run_mock_loop(
    robot: RobotAdapter,
    policy: Policy,
    safety: SafetyFilter,
    logger: JsonlRunLogger,
    iterations: int,
    prefix_steps: int,
    *,
    watchdog: Watchdog | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
    stall_at: Collection[int] = (),
    stall_s: float | None = None,
) -> LoopResult:
    """Run the M0 closed loop for ``iterations`` cycles; one JSONL record per iteration.

    Contracts:
    - ``logger`` must already be open; records are stamped with run_id + config_hash (AC-04).
    - Receding horizon: only ``prefix_steps`` of each safe chunk are executed (FR-05).
    - ``stall_at`` iteration indices simulate a late policy: watchdog time jumps by
      ``stall_s`` (default 2x the watchdog timeout) AFTER prediction, so the chunk arrives
      expired, is discarded, and the configured HOLD/STOP action is commanded instead.
    - A never-fed watchdog is armed at the first iteration's state timestamp; without that
      the fail-safe contract (never fed == expired) would trip immediately.
    """
    if iterations < 0:
        raise ValueError(f"iterations must be >= 0, got {iterations}")
    if prefix_steps < 1:
        raise ValueError(f"prefix_steps must be >= 1, got {prefix_steps}")
    if stall_s is not None and stall_s < 0:
        raise ValueError(f"stall_s must be >= 0, got {stall_s}")
    stall_set = frozenset(int(i) for i in stall_at)
    if stall_s is None:
        stall_s = 2.0 * watchdog.timeout_ns / _NS_PER_S if watchdog is not None else 0.0

    kinds: dict[str, int] = {}
    executed_count = 0
    timeouts = 0

    for i in range(iterations):
        t_iter = time.perf_counter()
        state = robot.read_state()
        now_ns = int(state.timestamp_ns)
        if watchdog is not None and watchdog.last_feed_ns is None:
            watchdog.feed(now_ns)

        observation = Observation(
            images=_render_images(robot), state=state, instruction=instruction
        )
        t0 = time.perf_counter()
        chunk = policy.predict(observation)
        predict_ms = (time.perf_counter() - t0) * 1e3

        stalled = i in stall_set
        if stalled:
            now_ns += round(stall_s * _NS_PER_S)

        decision: WatchdogAction | None = None
        interventions: list[SafetyIntervention] = []
        safe_digest: dict[str, Any] | None = None
        filter_ms = 0.0
        execute_ms = 0.0
        executed = False

        timed_out = watchdog is not None and watchdog.expired(now_ns)
        if timed_out and watchdog is not None:
            decision = watchdog.decide(now_ns)
            wd_intervention = watchdog.intervention(now_ns)
            if wd_intervention is not None:
                interventions.append(wd_intervention)
            if decision is WatchdogAction.STOP:
                robot.estop()
            else:
                robot.hold()
            watchdog.feed(now_ns)  # re-arm only after the safe state was commanded
            timeouts += 1
        else:
            t0 = time.perf_counter()
            safe_chunk, interventions = safety.filter(state, chunk)
            filter_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            robot.execute(safe_chunk, prefix_steps)
            execute_ms = (time.perf_counter() - t0) * 1e3
            if watchdog is not None:
                watchdog.feed(now_ns)
            safe_digest = _chunk_digest(safe_chunk)
            executed = True
            executed_count += 1

        for intervention in interventions:
            kinds[intervention.kind] = kinds.get(intervention.kind, 0) + 1

        logger.log(
            {
                "kind": "loop_iteration",
                "iteration": i,
                "now_ns": now_ns,
                "stalled": stalled,
                "instruction": instruction,
                "state": _state_digest(state),
                "chunk": _chunk_digest(chunk),
                "safe_chunk": safe_digest,
                "executed": executed,
                "prefix_steps": prefix_steps,
                "watchdog": {
                    "enabled": watchdog is not None,
                    "expired": timed_out,
                    "action": decision.value if decision is not None else None,
                },
                "interventions": [
                    {"kind": iv.kind, "detail": iv.detail, "timestamp_ns": iv.timestamp_ns}
                    for iv in interventions
                ],
                "timings_ms": {
                    "predict": predict_ms,
                    "filter": filter_ms,
                    "execute": execute_ms,
                    "total": (time.perf_counter() - t_iter) * 1e3,
                },
            }
        )

    return LoopResult(
        iterations=iterations,
        executed_iterations=executed_count,
        watchdog_timeouts=timeouts,
        interventions_total=sum(kinds.values()),
        intervention_kinds=kinds,
    )
