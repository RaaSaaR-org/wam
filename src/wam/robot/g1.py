"""Unitree G1 robot adapter (T-21, FR-06, OD-08).

Contracts:
- This module imports WITHOUT ``unitree_sdk2py``. All hardware I/O goes through the
  swappable :class:`~wam.robot.g1_transport.G1Transport` seam; the default transport
  (:class:`~wam.robot.g1_transport.DdsG1Transport`, built lazily in ``connect()``) raises
  ``RuntimeError('G1 hardware support requires unitree_sdk2py')`` without the SDK. With an
  injected transport (e.g. ``FakeG1Transport``) the full adapter runs SDK-free.
- ``read_state``/``execute``/``hold`` require a prior ``connect()``. ``estop()`` is safe to
  call at any time (latches; damps via the transport when one is attached).
- Joint mapping lives HERE as explicit data (``G1_JOINT_MAP``) — the ONLY place canonical
  order meets G1 motor indices. Pure mapping/unit helpers are implemented and testable
  without hardware.
- Ordering assumption: canonical order is waist first, then left arm proximal->distal, then
  right arm proximal->distal. G1 motor indices follow the unitree_sdk2py 29-DoF
  ``G1JointIndex`` convention (legs 0-11, waist yaw/roll/pitch 12-14, left arm 15-21,
  right arm 22-28). Only WaistYaw is used, so the subset is valid for the 23-DoF variant
  (which locks WaistRoll/WaistPitch) as well.
- Units: joint positions/velocities are rad and rad/s on BOTH sides (identity conversion,
  kept explicit in helpers). Gripper is canonical-normalized [0, 1] <-> vendor range from
  config.
- Safety: ``execute()`` clips per-step deltas to ``dq_max * dt`` and positions to
  ``[q_min, q_max]`` BEFORE sending (defense in depth — the upstream ``SafetyLayer`` is
  authoritative, FR-07). Unmapped motors (legs, waist roll/pitch) are commanded to HOLD
  their current position with zero gains, so the vendor controller keeps authority there.
- Timing: ``execute()`` paces the per-step command stream on the wall clock so successive
  targets are sent ``chunk.dt_s`` apart — the ``dq_max * dt`` clip encodes a velocity limit
  that is only valid with that temporal spacing. ``clock``/``sleep`` are injectable for
  deterministic tests.
- Torch-free; numpy + pydantic only.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
    ValidityMask,
)
from wam.robot.g1_transport import (
    G1_NUM_MOTORS,
    SDK_MISSING_MSG,
    DdsG1Transport,
    G1Transport,
)

_SDK_MISSING_MSG = SDK_MISSING_MSG

# Canonical name -> G1 motor index (29-DoF G1JointIndex convention). Upper-body subset:
# waist + 2x7 arm joints. Tuple order IS the canonical joint order.
G1_JOINT_MAP: tuple[tuple[str, int], ...] = (
    ("waist_yaw", 12),
    ("left_shoulder_pitch", 15),
    ("left_shoulder_roll", 16),
    ("left_shoulder_yaw", 17),
    ("left_elbow", 18),
    ("left_wrist_roll", 19),
    ("left_wrist_pitch", 20),
    ("left_wrist_yaw", 21),
    ("right_shoulder_pitch", 22),
    ("right_shoulder_roll", 23),
    ("right_shoulder_yaw", 24),
    ("right_elbow", 25),
    ("right_wrist_roll", 26),
    ("right_wrist_pitch", 27),
    ("right_wrist_yaw", 28),
)

G1_NUM_CANONICAL_JOINTS = len(G1_JOINT_MAP)

# Canonical space for the G1 upper-body subset. gripper_dims=2: [left, right], each in [0, 1].
G1_SPEC = CanonicalSpaceSpec(
    joint_names=tuple(name for name, _ in G1_JOINT_MAP),
    gripper_dims=2,
)

_G1_MOTOR_INDICES = np.array([idx for _, idx in G1_JOINT_MAP], dtype=np.int64)

# True for motor indices NOT covered by the canonical subset (legs, waist roll/pitch).
_G1_UNMAPPED_MASK = np.ones(G1_NUM_MOTORS, dtype=bool)
_G1_UNMAPPED_MASK[_G1_MOTOR_INDICES] = False


def _uniform(value: float) -> tuple[float, ...]:
    return (value,) * G1_NUM_CANONICAL_JOINTS


class G1Config(BaseModel):
    """G1 adapter configuration. Limit defaults are CONSERVATIVE PLACEHOLDERS pending OD-08
    (vendor datasheet / controller verification) — override from a versioned config file
    before any real-robot use."""

    model_config = ConfigDict(frozen=True)

    network_interface: str = "eth0"
    control_dt_s: float = Field(default=0.02, gt=0.0)
    q_min: tuple[float, ...] = Field(default_factory=lambda: _uniform(-1.5708))
    q_max: tuple[float, ...] = Field(default_factory=lambda: _uniform(1.5708))
    dq_max: tuple[float, ...] = Field(default_factory=lambda: _uniform(2.0))
    # Position-control gains per canonical joint, CONSERVATIVE defaults (low stiffness,
    # light damping) — tune upward only after E2/E3 verification (OD-08).
    kp: tuple[float, ...] = Field(default_factory=lambda: _uniform(20.0))
    kd: tuple[float, ...] = Field(default_factory=lambda: _uniform(0.5))
    gripper_vendor_min: float = 0.0
    gripper_vendor_max: float = 1.0

    @model_validator(mode="after")
    def _check(self) -> G1Config:
        for name in ("q_min", "q_max", "dq_max", "kp", "kd"):
            v = getattr(self, name)
            if len(v) != G1_NUM_CANONICAL_JOINTS:
                raise ValueError(f"{name}: expected {G1_NUM_CANONICAL_JOINTS} entries, got {len(v)}")
        if any(lo >= hi for lo, hi in zip(self.q_min, self.q_max)):
            raise ValueError("q_min must be < q_max per joint")
        if any(d <= 0 for d in self.dq_max):
            raise ValueError("dq_max entries must be > 0")
        if any(g < 0 for g in self.kp) or any(g < 0 for g in self.kd):
            raise ValueError("kp/kd entries must be >= 0")
        if self.gripper_vendor_min >= self.gripper_vendor_max:
            raise ValueError("gripper_vendor_min must be < gripper_vendor_max")
        return self


class G1Adapter:
    """``RobotAdapter`` for the Unitree G1 upper body (waist + both arms + grippers).

    All hardware I/O goes through the injected :class:`G1Transport`; without one, a
    :class:`DdsG1Transport` is built lazily in ``connect()`` (which needs the vendor SDK).
    Importable and configurable without the SDK.
    """

    def __init__(
        self,
        config: G1Config | Mapping[str, Any] | None = None,
        transport: G1Transport | None = None,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if config is None:
            config = G1Config()
        elif isinstance(config, Mapping):
            config = G1Config(**config)
        self._config = config
        self._transport: G1Transport | None = transport
        self._connected = False
        self._estopped = False
        self._last_tick_ns: int | None = None
        # Wall-clock pacing seam for execute(); injectable for deterministic tests.
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._sleep: Callable[[float], None] = sleep if sleep is not None else time.sleep

    @property
    def spec(self) -> CanonicalSpaceSpec:
        return G1_SPEC

    @property
    def config(self) -> G1Config:
        return self._config

    @property
    def transport(self) -> G1Transport | None:
        return self._transport

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_estopped(self) -> bool:
        return self._estopped

    # -- connection guard ----------------------------------------------------------------

    def connect(self) -> None:
        """Attach the transport. With an injected transport this succeeds without the SDK;
        otherwise a :class:`DdsG1Transport` is built and opened, which raises
        ``RuntimeError('G1 hardware support requires unitree_sdk2py')`` when the vendor SDK
        is missing."""
        if self._transport is None:
            transport = DdsG1Transport(self._config)
            transport.open()  # RuntimeError without unitree_sdk2py
            self._transport = transport
        self._connected = True

    def _require_connected(self, op: str) -> G1Transport:
        if not self._connected or self._transport is None:
            raise RuntimeError(f"{op}: not connected — call connect() first ({_SDK_MISSING_MSG})")
        return self._transport

    # -- RobotAdapter protocol -----------------------------------------------------------

    @property
    def limits(self) -> dict[str, np.ndarray]:
        """Canonical-order limit arrays from config (float32)."""
        g = G1_SPEC.gripper_dims
        return {
            "q_min": np.asarray(self._config.q_min, dtype=np.float32),
            "q_max": np.asarray(self._config.q_max, dtype=np.float32),
            "dq_max": np.asarray(self._config.dq_max, dtype=np.float32),
            "gripper_min": np.zeros(g, dtype=np.float32),
            "gripper_max": np.ones(g, dtype=np.float32),
        }

    def read_state(self) -> RobotState:
        """Read one low-state sample and map it into the canonical schema.

        - q/dq: motor arrays [29] gathered into canonical order [15] (rad, rad/s).
        - imu: passthrough (quat wxyz, gyro, acc).
        - gripper: vendor units -> canonical [0, 1]; missing "gripper" key degrades
          gripper validity.
        - timestamp: the transport tick. A tick UNCHANGED since the previous read means
          the sample is stale — ALL validity flags are degraded so the upstream safety
          layer rejects the state (state_reject) and the watchdog is not fed fresh data.

        ``validity`` describes THE SENSOR, not any policy's training distribution. A G1 has
        an IMU, so a fresh sample reports ``imu=True`` and the payload is real (the DDS
        passthrough; even ``FakeG1Transport`` supplies ``acc=(0, 0, 9.81)``). That is the
        honest answer and it must stay the honest answer: the T-16 checkpoints were trained
        on converted gr00t episodes whose every state carries ``imu=False``, so their
        encoder only ever saw the learned ``missing['imu']`` vector — but the fix for that
        is NOT to lie here about what the robot has. Matching a checkpoint's expectations to
        a robot's sensors belongs to :class:`wam.runtime.executor.PolicyContract`, which
        masks the observation down for the policy while the safety layer keeps seeing what
        the robot actually reported. Flipping this flag would instead break every adapter
        consumer that is not that one checkpoint.
        """
        transport = self._require_connected("read_state")
        low = transport.read_low_state()
        tick_ns = int(low["tick_ns"])
        stale = self._last_tick_ns is not None and tick_ns == self._last_tick_ns
        self._last_tick_ns = tick_ns

        imu_raw = low["imu"]
        imu = IMUState(
            orientation_wxyz=np.asarray(imu_raw["quat_wxyz"], dtype=np.float32),
            angular_velocity=np.asarray(imu_raw["gyro"], dtype=np.float32),
            linear_acceleration=np.asarray(imu_raw["acc"], dtype=np.float32),
        )
        gripper_raw = low.get("gripper")
        gripper_missing = gripper_raw is None
        if gripper_missing:
            gripper_state = np.zeros(G1_SPEC.gripper_dims, dtype=np.float32)
        else:
            gripper_state = self.vendor_to_gripper(np.asarray(gripper_raw, dtype=np.float32))
        fresh = not stale
        return RobotState(
            timestamp_ns=tick_ns,
            q=self.motor_to_canonical(low["q"]),
            dq=self.motor_to_canonical(low["dq"]),
            imu=imu,
            gripper_state=gripper_state,
            validity=ValidityMask(
                q=fresh, dq=fresh, imu=fresh, gripper=fresh and not gripper_missing
            ),
        )

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        """Stream the first ``prefix_steps`` steps as per-step position targets.

        JOINT_DELTA only (EE_DELTA is post-MVP, OD-02: joint deltas were chosen for the
        MVP; an EE variant needs an IK layer that does not exist yet). Deltas are
        integrated onto the CURRENT canonical q (read from the transport), clipped to
        ``dq_max * dt`` per step and to ``[q_min, q_max]`` before sending — defense in
        depth; the upstream SafetyLayer remains authoritative (FR-07). Successive step
        commands are PACED ``chunk.dt_s`` apart on the wall clock (step i is sent no
        earlier than ``t0 + i * dt_s``): the per-step ``dq_max * dt`` clip is a velocity
        limit only if the targets actually arrive ``dt_s`` apart — streaming them
        back-to-back would collapse the chunk into one large position step at the motors.
        Unmapped motors are commanded to hold their current position with zero gains. A
        latched e-stop makes this a no-op. The chunk MUST already have passed the
        SafetyFilter.
        """
        if prefix_steps < 0:
            raise ValueError(f"prefix_steps must be >= 0, got {prefix_steps}")
        if self._estopped:
            return
        transport = self._require_connected("execute")
        if chunk.mode is not ActionMode.JOINT_DELTA:
            raise NotImplementedError(
                "G1Adapter executes JOINT_DELTA only for the MVP; EE_DELTA needs an IK "
                "layer (OD-02)"
            )
        targets = np.asarray(chunk.targets, dtype=np.float64)
        if targets.ndim != 2 or targets.shape[1] != G1_NUM_CANONICAL_JOINTS:
            raise ValueError(
                f"expected targets [T, {G1_NUM_CANONICAL_JOINTS}], got {targets.shape}"
            )
        steps = min(int(prefix_steps), targets.shape[0])
        if steps == 0:
            return

        cfg = self._config
        low = transport.read_low_state()
        motor_q = np.asarray(low["q"], dtype=np.float64)
        q_can = motor_q[_G1_MOTOR_INDICES].copy()
        q_min = np.asarray(cfg.q_min, dtype=np.float64)
        q_max = np.asarray(cfg.q_max, dtype=np.float64)
        max_step = np.asarray(cfg.dq_max, dtype=np.float64) * float(chunk.dt_s)
        kp_motor = self.canonical_to_motor(np.asarray(cfg.kp, dtype=np.float32))
        kd_motor = self.canonical_to_motor(np.asarray(cfg.kd, dtype=np.float32))
        dq_target = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
        gripper = np.asarray(chunk.gripper_target, dtype=np.float64)

        dt_s = float(chunk.dt_s)
        t0 = self._clock()
        for i in range(steps):
            if i > 0:
                # Pace on the wall clock: never send step i before t0 + i * dt_s (FR-07:
                # the dq_max * dt clip assumes commands are issued dt_s apart).
                delay = t0 + i * dt_s - self._clock()
                if delay > 0.0:
                    self._sleep(delay)
            delta = np.clip(targets[i], -max_step, max_step)
            q_can = np.clip(q_can + delta, q_min, q_max)
            q_target = np.where(
                _G1_UNMAPPED_MASK, motor_q, self.canonical_to_motor(q_can.astype(np.float32))
            ).astype(np.float32)
            transport.write_motor_cmd(q_target, dq_target, kp_motor, kd_motor)
            # Scalar canonical gripper command drives both hands (MVP simplification).
            vendor = self.gripper_to_vendor(np.array([gripper[i], gripper[i]]))
            transport.write_gripper_cmd(float(vendor[0]), float(vendor[1]))

    def hold(self) -> None:
        """Re-send the current position as target with ``dq_target = 0`` (damped position
        hold). No-op while e-stopped (damping is already active)."""
        transport = self._require_connected("hold")
        if self._estopped:
            return
        low = transport.read_low_state()
        q_target = np.asarray(low["q"], dtype=np.float32)
        zeros = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
        kp_motor = self.canonical_to_motor(np.asarray(self._config.kp, dtype=np.float32))
        kd_motor = self.canonical_to_motor(np.asarray(self._config.kd, dtype=np.float32))
        transport.write_motor_cmd(q_target, zeros, kp_motor, kd_motor)

    def estop(self) -> None:
        """Emergency stop: vendor damping mode via the transport + latch. Safe to call at
        any time — latches even without a transport attached; subsequent ``execute()``
        calls are no-ops until ``clear_estop()``.

        The latch is set even if the transport raises while damping (the exception still
        propagates): a failed damp must never leave the adapter willing to keep commanding
        motion via ``execute()``.
        """
        try:
            if self._transport is not None:
                self._transport.emergency_damp()
        finally:
            self._estopped = True

    def clear_estop(self) -> None:
        """Release the e-stop latch (deliberate operator action)."""
        self._estopped = False

    def forget_tick(self) -> None:
        """Drop the cached previous transport tick, so the NEXT ``read_state()`` cannot be
        judged stale.

        ``read_state()`` marks a sample stale (and clears every validity flag) when the
        transport tick is unchanged since the previous read. That is the runtime's only
        liveness signal, so it has exactly one owner — this method — rather than callers
        poking the cache.

        Call it ONLY after the transport's clock has been deliberately rewound or restarted
        (a simulator episode reset, a reconnect), where "the tick did not change" no longer
        means "no new sample". Calling it in a running loop would mask a stalled controller.
        """
        self._last_tick_ns = None

    # -- pure mapping / unit helpers (no hardware, fully testable) ------------------------

    @staticmethod
    def canonical_to_motor(values: np.ndarray) -> np.ndarray:
        """Scatter canonical-order joint values [15] (rad or rad/s) into a G1 motor array
        [29] (float32); unmapped motors (legs, waist roll/pitch) are zero."""
        v = np.asarray(values, dtype=np.float32)
        if v.shape != (G1_NUM_CANONICAL_JOINTS,):
            raise ValueError(f"expected shape ({G1_NUM_CANONICAL_JOINTS},), got {v.shape}")
        motor = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
        motor[_G1_MOTOR_INDICES] = v
        return motor

    @staticmethod
    def motor_to_canonical(motor_values: np.ndarray) -> np.ndarray:
        """Gather a G1 motor array [29] into canonical order [15] (float32). Inverse of
        ``canonical_to_motor`` on the mapped subset."""
        m = np.asarray(motor_values, dtype=np.float32)
        if m.shape != (G1_NUM_MOTORS,):
            raise ValueError(f"expected shape ({G1_NUM_MOTORS},), got {m.shape}")
        return m[_G1_MOTOR_INDICES].copy()

    def gripper_to_vendor(self, g: np.ndarray) -> np.ndarray:
        """Canonical gripper command in [0, 1] -> vendor units (affine, from config)."""
        cfg = self._config
        g64 = np.clip(np.asarray(g, dtype=np.float64), 0.0, 1.0)
        out = cfg.gripper_vendor_min + g64 * (cfg.gripper_vendor_max - cfg.gripper_vendor_min)
        return out.astype(np.float32)

    def vendor_to_gripper(self, v: np.ndarray) -> np.ndarray:
        """Vendor gripper units -> canonical [0, 1] (affine inverse, clipped)."""
        cfg = self._config
        span = cfg.gripper_vendor_max - cfg.gripper_vendor_min
        out = (np.asarray(v, dtype=np.float64) - cfg.gripper_vendor_min) / span
        return np.clip(out, 0.0, 1.0).astype(np.float32)
