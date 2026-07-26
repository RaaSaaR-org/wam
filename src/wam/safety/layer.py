"""Deterministic safety layer v0 (FR-07, PRD §11.2). Implements ``wam.interfaces.SafetyFilter``.

Contracts:
- NO ML, no torch, no randomness: identical inputs produce identical outputs.
- Operates on PHYSICAL canonical units (rad / m; gripper in [0, 1]) and compares against
  physical limits directly. The MVP pipeline is identity-normalized end-to-end: decoder
  output IS physical units (no denormalization step exists anywhere; NormalizationSpec is
  parked, see its docstring).
- The returned chunk is safe-by-construction: every step satisfies position, velocity,
  acceleration, workspace and gripper-rate limits (given the declared start state). Unusable
  input (NaN/Inf, wrong shape, invalid state) is replaced by a zero-delta HOLD chunk.
- A start state OUTSIDE the position limits / workspace AABB (overtravel, miscalibration,
  manual repositioning) is never snapped back in one step: recovery toward the limits is
  re-bounded to the velocity limits per step (``joint_limit_recovery`` /
  ``workspace_recovery`` interventions), so re-entry happens at legal speed over multiple
  steps.
- Projection is step-wise (per step, per joint), never truncation of the whole chunk.
- Every modification is reported as a ``SafetyIntervention`` (FR-07: log every event).
- ``filter`` is pure w.r.t. its inputs (never mutates state/chunk); the ONLY internal
  mutation is a monotonic intervention counter (``intervention_count``).
- EE rotation deltas (quaternion part) pass through unmodified in v0; only translation is
  bounded. Documented limitation, not an oversight.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    RobotState,
    SafetyIntervention,
)
from wam.safety.config import SafetyConfig

_EE_LIN_DIMS = 3
_EE_QUAT_DIMS = 4
_EE_TARGET_DIM = _EE_LIN_DIMS + _EE_QUAT_DIMS
# Threshold below which a numerical change is not counted as an intervention (float noise).
_ATOL = 1e-9

# Identity rotation delta as quat wxyz — the "no rotation" EE step.
_QUAT_IDENTITY_WXYZ = (1.0, 0.0, 0.0, 0.0)

FkCallable = Callable[[RobotState], np.ndarray]
"""Forward kinematics: RobotState -> EE position [3] float in the workspace AABB frame."""


class SafetyLayer:
    """Deterministic projection/rejection gate between policy output and robot adapter.

    ``fk``: optional forward-kinematics callable for EE-mode workspace checking. Without it,
    EE translation deltas are only magnitude-bounded (``ee_max_step_m``) and every filtered
    EE chunk carries a ``workspace_skipped`` intervention.
    """

    def __init__(
        self,
        config: SafetyConfig,
        spec: CanonicalSpaceSpec | None = None,
        fk: FkCallable | None = None,
    ) -> None:
        if spec is not None and spec.num_joints != config.num_joints:
            raise ValueError(
                f"spec.num_joints {spec.num_joints} != config.num_joints {config.num_joints}"
            )
        self._config = config
        self._spec = spec
        self._fk = fk
        self._intervention_count = 0

    @property
    def config(self) -> SafetyConfig:
        return self._config

    @property
    def intervention_count(self) -> int:
        """Total interventions emitted over the lifetime of this layer (monotonic)."""
        return self._intervention_count

    # ------------------------------------------------------------------ filter

    def filter(
        self, state: RobotState, chunk: ActionChunk
    ) -> tuple[ActionChunk, list[SafetyIntervention]]:
        """Return a safe chunk plus all interventions applied (empty list == passed unchanged)."""
        ts = int(state.timestamp_ns)
        interventions: list[SafetyIntervention] = []

        reject = self._reject_reason(state, chunk)
        if reject is not None:
            kind, detail = reject
            interventions.append(SafetyIntervention(kind=kind, detail=detail, timestamp_ns=ts))
            hold = self._hold_chunk(state, chunk)
            self._intervention_count += len(interventions)
            return hold, interventions

        if chunk.mode is ActionMode.JOINT_DELTA:
            targets = self._filter_joint(state, chunk, interventions, ts)
        else:
            targets = self._filter_ee(state, chunk, interventions, ts)
        gripper = self._filter_gripper(state, chunk, interventions, ts)

        safe = ActionChunk(
            mode=chunk.mode,
            targets=targets.astype(np.float32),
            gripper_target=gripper.astype(np.float32),
            dt_s=float(chunk.dt_s),
            schema_version=chunk.schema_version,
        )
        self._intervention_count += len(interventions)
        return safe, interventions

    # ---------------------------------------------------------------- reject

    def _reject_reason(self, state: RobotState, chunk: ActionChunk) -> tuple[str, str] | None:
        """Whole-chunk rejection: NaN/Inf, structural problems, unusable state."""
        # 1. NaN/Inf anywhere in the chunk -> nan_reject (PRD §11.2: stop on NaN/Inf output).
        for name, arr in (("targets", chunk.targets), ("gripper_target", chunk.gripper_target)):
            if isinstance(arr, np.ndarray) and arr.size > 0 and not np.isfinite(arr).all():
                return "nan_reject", f"{name} contains NaN/Inf"
        if not np.isfinite(chunk.dt_s):
            return "nan_reject", f"dt_s is not finite: {chunk.dt_s}"

        # 2. Structural problems (shape, dtype, mode, dt<=0, version) -> schema_reject.
        # Out-of-range gripper values are NOT rejected: they are projectable and handled by
        # ``_filter_gripper`` (clamp + ``gripper_range`` intervention).
        issues = [
            i
            for i in chunk.validate(self._spec)
            if not i.startswith("gripper_target: values outside")
        ]
        if issues:
            return "schema_reject", "; ".join(issues)
        expected_d = (
            self._config.num_joints if chunk.mode is ActionMode.JOINT_DELTA else _EE_TARGET_DIM
        )
        if chunk.targets.shape[1] != expected_d:
            return (
                "schema_reject",
                (
                    f"targets: expected D={expected_d} for {chunk.mode.value}, "
                    f"got {chunk.targets.shape[1]}"
                ),
            )

        # 3. State must be usable for integration -> state_reject.
        state_issues = state.validate(self._spec)
        if state_issues:
            return "state_reject", "; ".join(state_issues)
        if chunk.mode is ActionMode.JOINT_DELTA:
            if not state.validity.q:
                return "state_reject", "state.q flagged invalid; cannot integrate joint deltas"
            if state.q.shape != (self._config.num_joints,):
                return (
                    "state_reject",
                    f"state.q shape {state.q.shape} != ({self._config.num_joints},)",
                )
        return None

    # ------------------------------------------------------------------ hold

    def _hold_chunk(self, state: RobotState, chunk: ActionChunk) -> ActionChunk:
        """Zero-delta single-step chunk: hold position, hold gripper (0.5 if unknown)."""
        mode = chunk.mode if isinstance(chunk.mode, ActionMode) else ActionMode.JOINT_DELTA
        if mode is ActionMode.JOINT_DELTA:
            step = np.zeros(self._config.num_joints, dtype=np.float32)
        else:
            step = np.array((0.0, 0.0, 0.0, *_QUAT_IDENTITY_WXYZ), dtype=np.float32)
        gripper = 0.5
        gs = state.gripper_state
        if (
            state.validity.gripper
            and isinstance(gs, np.ndarray)
            and gs.size > 0
            and np.isfinite(gs.flat[0])
        ):
            gripper = float(np.clip(gs.flat[0], 0.0, 1.0))
        dt = (
            float(chunk.dt_s)
            if np.isfinite(chunk.dt_s) and chunk.dt_s > 0
            else (self._config.hold_dt_s)
        )
        return ActionChunk(
            mode=mode,
            targets=step[None, :],
            gripper_target=np.array([gripper], dtype=np.float32),
            dt_s=dt,
        )

    # ----------------------------------------------------------------- joint

    def _filter_joint(
        self,
        state: RobotState,
        chunk: ActionChunk,
        interventions: list[SafetyIntervention],
        ts: int,
    ) -> np.ndarray:
        """Step-wise accel -> velocity -> position projection of joint deltas.

        Order guarantees the limits hold simultaneously: the velocity clamp can only
        shrink |dv| (both endpoints inside [-dq_max, dq_max]), and the position clamp can only
        shrink the step when the running position is inside [q_min, q_max]. When the running
        position is OUTSIDE [q_min, q_max] the position clamp would snap it back in a single
        (arbitrarily fast) step, so the resulting step is re-clipped to the velocity limit
        (``joint_limit_recovery``): the position ramps back at legal speed over multiple steps.
        """
        cfg = self._config
        dt = float(chunk.dt_s)
        deltas = chunk.targets.astype(np.float64)
        q = state.q.astype(np.float64).copy()
        if state.validity.dq and state.dq.shape == q.shape and np.isfinite(state.dq).all():
            v_prev = state.dq.astype(np.float64).copy()
        else:
            v_prev = np.zeros_like(q)
        dq_max = cfg.dq_max_arr()
        dv_max = cfg.ddq_max_arr() * dt
        q_min, q_max = cfg.q_min_arr(), cfg.q_max_arr()

        out = np.empty_like(deltas)
        for t in range(deltas.shape[0]):
            v = deltas[t] / dt
            # Acceleration limit: |v - v_prev| <= ddq_max * dt, per joint.
            dv = np.clip(v - v_prev, -dv_max, dv_max)
            v_acc = v_prev + dv
            self._record_per_joint(
                interventions, ts, "accel_limit", t, v, v_acc, "rad/s (accel-limited)"
            )
            # Velocity limit: |v| <= dq_max, per joint.
            v_lim = np.clip(v_acc, -dq_max, dq_max)
            self._record_per_joint(interventions, ts, "velocity_limit", t, v_acc, v_lim, "rad/s")
            # Position limit: q + d inside [q_min, q_max], per joint (exact projection).
            q_next = np.clip(q + v_lim * dt, q_min, q_max)
            self._record_per_joint(
                interventions, ts, "joint_limit", t, q + v_lim * dt, q_next, "rad"
            )
            step = q_next - q
            # Out-of-limits start recovery: if q was outside [q_min, q_max], the position
            # clamp above produces a step that jumps to the boundary in one dt — re-clip it
            # to the velocity limit so recovery ramps back at legal speed (no-op whenever
            # q was inside the limits, where the clamp can only shrink the step).
            step_limited = np.clip(step, -dq_max * dt, dq_max * dt)
            self._record_per_joint(
                interventions, ts, "joint_limit_recovery", t, step, step_limited, "rad"
            )
            out[t] = step_limited
            v_prev = out[t] / dt
            q = q + out[t]
        return out

    # -------------------------------------------------------------------- ee

    def _filter_ee(
        self,
        state: RobotState,
        chunk: ActionChunk,
        interventions: list[SafetyIntervention],
        ts: int,
    ) -> np.ndarray:
        """Bound EE translation deltas; quaternion part passes through (v0 limitation)."""
        cfg = self._config
        dt = float(chunk.dt_s)
        out = chunk.targets.astype(np.float64)

        # Linear velocity limit: ||dxyz|| / dt <= ee_max_lin_vel_m_s (uniform per-step scaling).
        max_step_vel = cfg.ee_max_lin_vel_m_s * dt
        for t in range(out.shape[0]):
            norm = float(np.linalg.norm(out[t, :_EE_LIN_DIMS]))
            if norm > max_step_vel + _ATOL:
                out[t, :_EE_LIN_DIMS] *= max_step_vel / norm
                interventions.append(
                    SafetyIntervention(
                        kind="velocity_limit",
                        detail=(
                            f"step {t}: EE speed {norm / dt:.6f} m/s > "
                            f"{cfg.ee_max_lin_vel_m_s} m/s; scaled"
                        ),
                        timestamp_ns=ts,
                    )
                )

        position = self._ee_position(state)
        if position is None:
            # No usable fk: bound per-step delta magnitude only, flag the skipped check.
            for t in range(out.shape[0]):
                norm = float(np.linalg.norm(out[t, :_EE_LIN_DIMS]))
                if norm > cfg.ee_max_step_m + _ATOL:
                    out[t, :_EE_LIN_DIMS] *= cfg.ee_max_step_m / norm
                    interventions.append(
                        SafetyIntervention(
                            kind="ee_step_limit",
                            detail=(
                                f"step {t}: |EE delta| {norm:.6f} m > {cfg.ee_max_step_m} m; scaled"
                            ),
                            timestamp_ns=ts,
                        )
                    )
            interventions.append(
                SafetyIntervention(
                    kind="workspace_skipped",
                    detail="no fk available; workspace AABB not checked, "
                    "per-step delta magnitude bounded only",
                    timestamp_ns=ts,
                )
            )
            return out

        # fk available: integrate positions, project into the workspace AABB step-wise.
        # A start position OUTSIDE the AABB would make the projection emit one huge
        # re-entry step, so each projected step is re-clipped to the linear velocity
        # limit (``workspace_recovery``) — re-entry happens at legal speed; a no-op
        # whenever the running position is inside the AABB (the projection can only
        # shrink steps there).
        ws_min, ws_max = cfg.workspace_min_arr(), cfg.workspace_max_arr()
        p = position
        for t in range(out.shape[0]):
            p_next = p + out[t, :_EE_LIN_DIMS]
            p_clamped = np.clip(p_next, ws_min, ws_max)
            if np.any(np.abs(p_next - p_clamped) > _ATOL):
                axes = np.flatnonzero(np.abs(p_next - p_clamped) > _ATOL).tolist()
                interventions.append(
                    SafetyIntervention(
                        kind="workspace",
                        detail=f"step {t}: EE position clamped to AABB on axes {axes}",
                        timestamp_ns=ts,
                    )
                )
            step = p_clamped - p
            norm = float(np.linalg.norm(step))
            if norm > max_step_vel + _ATOL:
                step = step * (max_step_vel / norm)
                interventions.append(
                    SafetyIntervention(
                        kind="workspace_recovery",
                        detail=(
                            f"step {t}: workspace re-entry step {norm:.6f} m re-clipped to "
                            f"{max_step_vel:.6f} m (EE velocity limit)"
                        ),
                        timestamp_ns=ts,
                    )
                )
            out[t, :_EE_LIN_DIMS] = step
            p = p + step
        return out

    def _ee_position(self, state: RobotState) -> np.ndarray | None:
        """Current EE position via fk, or None (fk absent, raised, or returned garbage)."""
        if self._fk is None:
            return None
        try:
            p = np.asarray(self._fk(state), dtype=np.float64).reshape(-1)
        except Exception:  # noqa: BLE001 — fk is caller-supplied; any failure degrades to skip
            return None
        if p.shape != (_EE_LIN_DIMS,) or not np.isfinite(p).all():
            return None
        return p

    # --------------------------------------------------------------- gripper

    def _filter_gripper(
        self,
        state: RobotState,
        chunk: ActionChunk,
        interventions: list[SafetyIntervention],
        ts: int,
    ) -> np.ndarray:
        """Clamp gripper targets to [0, 1] and rate-limit against the current gripper state."""
        cfg = self._config
        dt = float(chunk.dt_s)
        max_step = cfg.gripper_rate_max * dt
        out = chunk.gripper_target.astype(np.float64).copy()

        g_prev: float | None = None
        gs = state.gripper_state
        if (
            state.validity.gripper
            and isinstance(gs, np.ndarray)
            and gs.size > 0
            and np.isfinite(gs.flat[0])
        ):
            g_prev = float(np.clip(gs.flat[0], 0.0, 1.0))

        for t in range(out.shape[0]):
            g = float(np.clip(out[t], 0.0, 1.0))
            if abs(g - out[t]) > _ATOL:
                interventions.append(
                    SafetyIntervention(
                        kind="gripper_range",
                        detail=f"step {t}: gripper_target {out[t]:.6f} clamped to [0, 1]",
                        timestamp_ns=ts,
                    )
                )
            if g_prev is not None and abs(g - g_prev) > max_step + _ATOL:
                g = g_prev + float(np.sign(g - g_prev)) * max_step
                interventions.append(
                    SafetyIntervention(
                        kind="gripper_rate",
                        detail=(
                            f"step {t}: gripper rate > {cfg.gripper_rate_max}/s; limited to {g:.6f}"
                        ),
                        timestamp_ns=ts,
                    )
                )
            out[t] = g
            g_prev = g
        return out

    # ----------------------------------------------------------------- utils

    @staticmethod
    def _record_per_joint(
        interventions: list[SafetyIntervention],
        ts: int,
        kind: str,
        step: int,
        before: np.ndarray,
        after: np.ndarray,
        unit: str,
    ) -> None:
        """Emit one intervention per step in which ``after`` differs from ``before``."""
        changed = np.flatnonzero(np.abs(after - before) > _ATOL)
        if changed.size > 0:
            interventions.append(
                SafetyIntervention(
                    kind=kind,
                    detail=f"step {step}: joints {changed.tolist()} clamped ({unit})",
                    timestamp_ns=ts,
                )
            )
