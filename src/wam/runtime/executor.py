"""Closed-loop runtime executor (T-19, FR-05): receding horizon with replanning.

REPLANNING SEMANTIC (FR-05): every cycle produces a FRESH action chunk from a fresh
observation; only the first ``prefix_steps`` of the (safety-filtered) chunk are executed
and the unexecuted remainder is DISCARDED — the next prediction replaces it. The robot
therefore only ever sees prefixes; no stale chunk tail survives a replan.

Contracts:
- Torch-free. Model inference is behind the ``Policy`` protocol
  (:class:`wam.runtime.policies.CheckpointPolicy` for trained checkpoints).
- Every policy output passes the deterministic safety filter before the robot adapter
  sees it (FR-07). The learned model never commands the robot directly.
- Deadline (PRD §11.1): a prediction arriving later than ``policy_deadline_ms`` is
  DISCARDED — never executed late — and ``robot.hold()`` is commanded instead. The
  watchdog is intentionally NOT fed on that path, so a persistently late policy trips it.
- Rejected states are NOT watchdog food: when the safety filter rejects the cycle
  (``nan_reject``/``schema_reject``/``state_reject`` -> HOLD chunk), the watchdog is not
  fed — a robot that keeps serving stale/unusable states must not keep the watchdog armed.
  Because a stalled robot also freezes ``state.timestamp_ns`` (the watchdog's clock), the
  executor additionally measures an uninterrupted reject streak on the HOST clock and
  escalates per the watchdog's HOLD/STOP action once it exceeds the watchdog timeout.
- Watchdog expiry: stale loop -> HOLD or STOP (e-stop) per its configured action; the
  watchdog is re-armed only after the safe state was commanded (PRD §11.2 recovery).
- Time: policy latency is measured with ``time.monotonic`` by default; a ``clock``
  callable (seconds) can be injected so tests enforce deadlines deterministically.
  ``now_ns`` and the watchdog's normal feed/expiry run on ROBOT time
  (``state.timestamp_ns`` — simulated for the mock, wall-clock-ish on hardware); the
  stale-state reject-streak escalation above runs on the HOST clock because a stalled
  robot freezes its own timestamps.
- Logging: writes exactly the SHARED ROLLOUT LOG CONTRACT records — one
  ``kind="control_cycle"`` line per cycle and one ``kind="rollout_summary"`` line per
  rollout, stamped run_id + config_hash by :class:`JsonlRunLogger` (AC-04). The
  below-min-rate flag lives ONLY in :class:`RolloutResult` (``below_min_policy_rate``);
  the summary record keeps the fixed contract keys and consumers derive the flag from
  ``policy_rate_hz``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wam.interfaces import (
    JsonlRunLogger,
    Observation,
    Policy,
    RobotAdapter,
    RobotState,
    SafetyFilter,
    SafetyIntervention,
)
from wam.runtime.mock_loop import DEFAULT_INSTRUCTION
from wam.safety import Watchdog, WatchdogAction

__all__ = ["ClosedLoopExecutor", "ExecutorConfig", "RolloutResult", "run_rollouts"]

SuccessFn = Callable[[RobotState], bool]

# Safety-filter rejection kinds: cycles carrying one of these executed only a HOLD chunk
# built from an unusable prediction/state and therefore never feed the watchdog.
_REJECT_KINDS = frozenset({"nan_reject", "schema_reject", "state_reject"})

_NS_PER_S = 1_000_000_000


class ExecutorConfig(BaseModel):
    """Closed-loop executor parameters (FR-05, PRD §11.1).

    - ``prefix_steps``: steps of each safe chunk to execute before replanning.
    - ``max_cycles``: hard rollout length bound (cycles == policy predictions).
    - ``policy_deadline_ms``: predictions later than this are discarded -> hold.
    - ``min_policy_rate_hz``: MVP floor (>= 2 Hz, PRD §11.1); rollouts below it are
      flagged in :attr:`RolloutResult.below_min_policy_rate`.
    - ``stop_on_estop``: end the rollout as soon as the robot reports an e-stop.
    """

    model_config = ConfigDict(frozen=True)

    prefix_steps: int = Field(ge=1)
    max_cycles: int = Field(ge=1)
    policy_deadline_ms: float = Field(default=500.0, gt=0)
    min_policy_rate_hz: float = Field(default=2.0, gt=0)
    instruction: str = DEFAULT_INSTRUCTION
    task: str = "pick_and_place"
    stop_on_estop: bool = True


@dataclass
class RolloutResult:
    """Outcome of one rollout; mirrors the ``rollout_summary`` log-record contract.

    ``below_min_policy_rate`` is a derived convenience flag (``policy_rate_hz <
    ExecutorConfig.min_policy_rate_hz``) and is deliberately NOT part of the shared
    summary record.
    """

    rollout_id: str
    success: bool
    task: str
    duration_s: float
    cycles: int
    executed_cycles: int
    interventions_total: int
    intervention_kinds: dict[str, int]
    watchdog_timeouts: int
    deadline_misses: int
    estopped: bool
    policy_rate_hz: float
    below_min_policy_rate: bool = field(default=False, compare=False)

    def summary_record(self) -> dict[str, Any]:
        """Exactly the SHARED ROLLOUT LOG CONTRACT ``rollout_summary`` payload."""
        return {
            "kind": "rollout_summary",
            "rollout_id": self.rollout_id,
            "success": self.success,
            "task": self.task,
            "duration_s": self.duration_s,
            "cycles": self.cycles,
            "executed_cycles": self.executed_cycles,
            "interventions_total": self.interventions_total,
            "intervention_kinds": dict(self.intervention_kinds),
            "watchdog_timeouts": self.watchdog_timeouts,
            "deadline_misses": self.deadline_misses,
            "estopped": self.estopped,
            "policy_rate_hz": self.policy_rate_hz,
        }


def _render_images(robot: RobotAdapter) -> dict[str, np.ndarray]:
    """One [H, W, 3] frame per camera if the adapter exposes ``render_frames``."""
    render = getattr(robot, "render_frames", None)
    if not callable(render):
        return {}
    return {name: frames[0] for name, frames in render(1).items()}


def _is_estopped(robot: RobotAdapter) -> bool:
    return bool(getattr(robot, "is_estopped", False))


class ClosedLoopExecutor:
    """Receding-horizon closed loop: observe -> predict -> filter -> execute prefix.

    Per cycle: ``read_state`` -> render frames (if the adapter can) -> ``Observation``
    -> timed ``policy.predict`` -> watchdog/deadline gates -> ``safety.filter`` ->
    ``robot.execute(safe_chunk, prefix_steps)`` -> feed watchdog -> success check.
    Only the prefix executes; the remainder is replaced by the next prediction (FR-05).
    """

    def __init__(
        self,
        robot: RobotAdapter,
        policy: Policy,
        safety: SafetyFilter,
        watchdog: Watchdog | None,
        logger: JsonlRunLogger,
        config: ExecutorConfig,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._robot = robot
        self._policy = policy
        self._safety = safety
        self._watchdog = watchdog
        self._logger = logger
        self._config = config
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def run_rollout(self, rollout_id: str, success_fn: SuccessFn | None = None) -> RolloutResult:
        """Run one rollout of up to ``max_cycles`` cycles; one JSONL record per cycle.

        ``success_fn(state)`` is evaluated on the freshly read post-action state each
        cycle; returning True ends the rollout early with ``success=True``. An e-stop
        ends it with ``success=False`` when ``stop_on_estop`` is set.
        """
        cfg = self._config
        robot, policy, safety, watchdog = self._robot, self._policy, self._safety, self._watchdog
        t_start = self._clock()

        kinds: dict[str, int] = {}
        cycles = 0
        executed_cycles = 0
        watchdog_timeouts = 0
        deadline_misses = 0
        success = False
        estopped = False
        # Host-clock start of the current uninterrupted safety-reject streak (None when
        # the last cycle was clean). The watchdog itself runs on robot time, which FREEZES
        # exactly when the robot stalls — so stale-state escalation is timed on the host.
        reject_streak_start_s: float | None = None

        for cycle in range(cfg.max_cycles):
            cycles = cycle + 1
            state = robot.read_state()
            now_ns = int(state.timestamp_ns)
            if watchdog is not None and watchdog.last_feed_ns is None:
                watchdog.feed(now_ns)  # arm; never-fed == expired would trip immediately

            observation = Observation(
                images=_render_images(robot), state=state, instruction=cfg.instruction
            )
            t0 = self._clock()
            timeout_detail: str | None = None
            try:
                chunk = policy.predict(observation)
            except TimeoutError as exc:
                # RemotePolicy contract (T-20): a client-side timeout surfaces as a
                # deadline miss — no chunk arrived, so there is nothing to execute.
                chunk = None
                timeout_detail = f"policy timed out: {exc}"
            policy_latency_ms = (self._clock() - t0) * 1e3
            deadline_missed = chunk is None or policy_latency_ms > cfg.policy_deadline_ms
            if deadline_missed:
                deadline_misses += 1

            interventions: list[SafetyIntervention] = []
            decision: WatchdogAction | None = None
            executed = False
            prefix_executed = 0
            expired = watchdog is not None and watchdog.expired(now_ns)

            if expired and watchdog is not None:
                # Stale loop: the fresh chunk is discarded, HOLD/STOP is commanded.
                decision = watchdog.decide(now_ns)
                wd_intervention = watchdog.intervention(now_ns)
                if wd_intervention is not None:
                    interventions.append(wd_intervention)
                if decision is WatchdogAction.STOP:
                    robot.estop()
                else:
                    robot.hold()
                watchdog.feed(now_ns)  # re-arm only after the safe state was commanded
                watchdog_timeouts += 1
            elif deadline_missed:
                # Late prediction: discard, hold. Watchdog NOT fed -> chronic lateness
                # eventually trips it (PRD §11.1: never execute a stale action).
                robot.hold()
                interventions.append(
                    SafetyIntervention(
                        kind="deadline_miss",
                        detail=timeout_detail
                        or (
                            f"policy latency {policy_latency_ms:.3f} ms > deadline "
                            f"{cfg.policy_deadline_ms:.3f} ms; chunk discarded, hold commanded"
                        ),
                        timestamp_ns=now_ns,
                    )
                )
            else:
                assert chunk is not None  # deadline_missed covers the timeout path
                safe_chunk, filter_interventions = safety.filter(state, chunk)
                interventions.extend(filter_interventions)
                rejected = any(iv.kind in _REJECT_KINDS for iv in filter_interventions)
                host_now_s = self._clock()
                if rejected and reject_streak_start_s is None:
                    reject_streak_start_s = host_now_s
                elif not rejected:
                    reject_streak_start_s = None
                if (
                    rejected
                    and watchdog is not None
                    and reject_streak_start_s is not None
                    and (host_now_s - reject_streak_start_s) * _NS_PER_S > watchdog.timeout_ns
                ):
                    # Frozen/unusable robot state persisting past the watchdog timeout: the
                    # robot clock (the watchdog's time source) is frozen too, so escalate
                    # here on host time — HOLD or STOP per the watchdog's configured action.
                    decision = watchdog.action
                    interventions.append(
                        SafetyIntervention(
                            kind="watchdog_timeout",
                            detail=(
                                f"safety filter rejected the state for "
                                f"{host_now_s - reject_streak_start_s:.3f} s (host clock) > "
                                f"timeout {watchdog.timeout_ns} ns; robot time frozen; "
                                f"decision={decision.value}"
                            ),
                            timestamp_ns=now_ns,
                        )
                    )
                    if decision is WatchdogAction.STOP:
                        robot.estop()
                    else:
                        robot.hold()
                    # Re-arm the streak only after the safe state was commanded.
                    reject_streak_start_s = host_now_s
                    watchdog_timeouts += 1
                    expired = True
                else:
                    prefix_executed = min(cfg.prefix_steps, safe_chunk.num_steps)
                    # Receding horizon: only the prefix runs; the remainder is discarded
                    # and replaced by the next cycle's prediction (FR-05). A rejected
                    # cycle executes the safety layer's HOLD chunk but does NOT feed the
                    # watchdog: stale robot data must not keep it armed.
                    robot.execute(safe_chunk, cfg.prefix_steps)
                    if watchdog is not None and not rejected:
                        watchdog.feed(now_ns)
                    executed = True
                    executed_cycles += 1

            for intervention in interventions:
                kinds[intervention.kind] = kinds.get(intervention.kind, 0) + 1
            if _is_estopped(robot):
                estopped = True

            self._logger.log(
                {
                    "kind": "control_cycle",
                    "rollout_id": rollout_id,
                    "cycle": cycle,
                    "now_ns": now_ns,
                    "policy_latency_ms": float(policy_latency_ms),
                    "deadline_missed": bool(deadline_missed),
                    "executed": bool(executed),
                    "prefix_steps": int(prefix_executed),
                    "chunk_steps": int(chunk.num_steps) if chunk is not None else 0,
                    "interventions": [
                        {"kind": iv.kind, "detail": iv.detail, "timestamp_ns": int(iv.timestamp_ns)}
                        for iv in interventions
                    ],
                    "watchdog": {
                        "expired": bool(expired),
                        "action": decision.value if decision is not None else None,
                    },
                }
            )

            if estopped and cfg.stop_on_estop:
                break
            if success_fn is not None and success_fn(robot.read_state()):
                success = True
                break

        duration_s = max(self._clock() - t_start, 0.0)
        policy_rate_hz = cycles / duration_s if duration_s > 0 else 0.0
        result = RolloutResult(
            rollout_id=rollout_id,
            success=success,
            task=cfg.task,
            duration_s=duration_s,
            cycles=cycles,
            executed_cycles=executed_cycles,
            interventions_total=sum(kinds.values()),
            intervention_kinds=kinds,
            watchdog_timeouts=watchdog_timeouts,
            deadline_misses=deadline_misses,
            estopped=estopped,
            policy_rate_hz=policy_rate_hz,
            below_min_policy_rate=policy_rate_hz < cfg.min_policy_rate_hz,
        )
        self._logger.log(result.summary_record())
        return result


def run_rollouts(
    executor: ClosedLoopExecutor,
    n: int,
    rollout_id_prefix: str = "rollout",
    success_fn: SuccessFn | None = None,
) -> list[RolloutResult]:
    """Run ``n`` sequential rollouts with ids ``<prefix>-0000`` ... ``<prefix>-<n-1>``."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return [
        executor.run_rollout(f"{rollout_id_prefix}-{i:04d}", success_fn=success_fn)
        for i in range(n)
    ]
