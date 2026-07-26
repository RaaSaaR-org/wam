"""Mock robot adapter (T-03): deterministic kinematic integrator behind ``RobotAdapter``.

Contracts:
- JOINT_DELTA only. ``chunk.targets`` are interpreted DIRECTLY as joint deltas in rad
  (identity normalization — sufficient for pipeline tests); EE_DELTA raises NotImplementedError.
- ``q`` integrates deltas with hard clipping to ``q_min``/``q_max``; ``dq`` is the finite
  difference of consecutive ``q`` over ``chunk.dt_s`` (clipping is reflected in ``dq``).
- No real time passes: per-step latency is SIMULATED on an internal monotonic clock
  (``timestamp_ns`` advances by ``dt_s + step_latency_s`` per executed step). Never sleeps.
- Deterministic under a fixed ``seed``. Optional gaussian noise perturbs READS only; the
  internal kinematic state is never noisy.
- ``estop()`` latches: motion commands are ignored until ``clear_estop()``. ``hold()`` zeroes
  ``dq`` and is released by the next ``execute()``.
- Torch-free; numpy only.
"""

from __future__ import annotations

import math

import numpy as np

from wam.interfaces.schema import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    RobotState,
)

_DOT_HALF = 1  # rendered dot is (2*_DOT_HALF+1)^2 pixels

_CAMERA_BG: dict[str, tuple[int, int, int]] = {
    "front": (30, 30, 40),
    "wrist": (40, 30, 30),
}
_DEFAULT_BG = (32, 32, 32)


def _as_limit(value: float | np.ndarray, n: int, name: str) -> np.ndarray:
    arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (n,)).copy()
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: limits must be finite")
    return arr


class MockRobot:
    """In-memory robot implementing the ``RobotAdapter`` protocol without hardware.

    Also exposes a synthetic camera (``render_frames``) so vision pipelines can be tested:
    flat background plus a bright dot whose COLUMN position encodes ``q[0]`` within
    ``[q_min[0], q_max[0]]``.
    """

    def __init__(
        self,
        spec: CanonicalSpaceSpec | None = None,
        num_joints: int = 7,
        gripper_dims: int = 1,
        initial_q: np.ndarray | None = None,
        q_min: float | np.ndarray = -math.pi,
        q_max: float | np.ndarray = math.pi,
        dq_max: float | np.ndarray = 2.0 * math.pi,
        step_latency_s: float = 0.0,
        noise_std: float = 0.0,
        seed: int = 0,
        image_hw: tuple[int, int] = (64, 64),
        cameras: tuple[str, ...] = ("front", "wrist"),
    ) -> None:
        if spec is None:
            spec = CanonicalSpaceSpec(
                joint_names=tuple(f"joint_{i}" for i in range(num_joints)),
                gripper_dims=gripper_dims,
            )
        self._spec = spec
        n = spec.num_joints
        self._q_min = _as_limit(q_min, n, "q_min")
        self._q_max = _as_limit(q_max, n, "q_max")
        if (self._q_min >= self._q_max).any():
            raise ValueError("q_min must be < q_max per joint")
        self._dq_max = _as_limit(dq_max, n, "dq_max")
        if step_latency_s < 0:
            raise ValueError("step_latency_s must be >= 0")
        if noise_std < 0:
            raise ValueError("noise_std must be >= 0")
        self._step_latency_s = float(step_latency_s)
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)
        self._image_hw = image_hw
        self._cameras = cameras

        q0 = np.zeros(n) if initial_q is None else np.asarray(initial_q, dtype=np.float64).copy()
        if q0.shape != (n,):
            raise ValueError(f"initial_q: expected shape ({n},), got {q0.shape}")
        self._q = np.clip(q0, self._q_min, self._q_max)
        self._dq = np.zeros(n)
        self._gripper = np.zeros(max(spec.gripper_dims, 0))
        self._clock_ns = 0
        self._estopped = False
        self._holding = False

    # -- introspection -------------------------------------------------------------------

    @property
    def spec(self) -> CanonicalSpaceSpec:
        return self._spec

    @property
    def is_estopped(self) -> bool:
        return self._estopped

    @property
    def is_holding(self) -> bool:
        return self._holding

    @property
    def sim_time_ns(self) -> int:
        """Simulated monotonic clock; advanced only by ``execute`` (never wall time)."""
        return self._clock_ns

    # -- RobotAdapter protocol -----------------------------------------------------------

    @property
    def limits(self) -> dict[str, np.ndarray]:
        g = self._spec.gripper_dims
        return {
            "q_min": self._q_min.astype(np.float32),
            "q_max": self._q_max.astype(np.float32),
            "dq_max": self._dq_max.astype(np.float32),
            "gripper_min": np.zeros(g, dtype=np.float32),
            "gripper_max": np.ones(g, dtype=np.float32),
        }

    def read_state(self) -> RobotState:
        """Current canonical state. Noise (if configured) is applied per read; internal
        state is untouched. After ``estop()``/``hold()``, ``dq`` reads zero."""
        q = self._q.copy()
        dq = self._dq.copy()
        if self._noise_std > 0.0:
            q += self._rng.normal(0.0, self._noise_std, q.shape)
            dq += self._rng.normal(0.0, self._noise_std, dq.shape)
        return RobotState(
            timestamp_ns=self._clock_ns,
            q=q.astype(np.float32),
            dq=dq.astype(np.float32),
            imu=IMUState(
                orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                angular_velocity=np.zeros(3, dtype=np.float32),
                linear_acceleration=np.zeros(3, dtype=np.float32),
            ),
            gripper_state=self._gripper.astype(np.float32),
        )

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        """Integrate only the first ``prefix_steps`` steps (receding horizon, FR-05).

        No-op while e-stopped. Clears a previous ``hold()``. Raises for negative
        ``prefix_steps``, non-JOINT_DELTA chunks, or a target width mismatch.
        """
        if prefix_steps < 0:
            raise ValueError(f"prefix_steps must be >= 0, got {prefix_steps}")
        if self._estopped:
            return
        if chunk.mode is not ActionMode.JOINT_DELTA:
            raise NotImplementedError("MockRobot supports ActionMode.JOINT_DELTA only")
        n = self._spec.num_joints
        if chunk.targets.ndim != 2 or chunk.targets.shape[1] != n:
            raise ValueError(f"targets: expected shape [T, {n}], got {chunk.targets.shape}")
        self._holding = False
        dt = float(chunk.dt_s)
        steps = min(prefix_steps, chunk.num_steps)
        for i in range(steps):
            delta = np.asarray(chunk.targets[i], dtype=np.float64)
            q_new = np.clip(self._q + delta, self._q_min, self._q_max)
            self._dq = (q_new - self._q) / dt
            self._q = q_new
            if self._gripper.size > 0:
                self._gripper[:] = float(np.clip(chunk.gripper_target[i], 0.0, 1.0))
            self._clock_ns += round((dt + self._step_latency_s) * 1e9)

    def hold(self) -> None:
        """Freeze position: zero ``dq``, keep ``q``. Released by the next ``execute``."""
        self._dq[:] = 0.0
        self._holding = True

    def estop(self) -> None:
        """Latched emergency stop: zero ``dq`` and ignore all motion until ``clear_estop``.
        Idempotent and safe to call at any time."""
        self._dq[:] = 0.0
        self._estopped = True

    def clear_estop(self) -> None:
        """Release a latched e-stop (mock-only convenience for tests)."""
        self._estopped = False

    # -- synthetic camera ----------------------------------------------------------------

    def render_frames(self, n: int) -> dict[str, np.ndarray]:
        """Render ``n`` frames per camera: ``{name: uint8 array [n, H, W, 3]}``.

        Flat per-camera background + bright dot; the dot's column encodes the CURRENT
        ``q[0]`` normalized within ``[q_min[0], q_max[0]]`` (frames within one call are
        identical — the dot moves between calls as ``q`` changes).
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        h, w = self._image_hw
        span = self._q_max[0] - self._q_min[0]
        frac = float(np.clip((self._q[0] - self._q_min[0]) / span, 0.0, 1.0))
        col = int(np.clip(round(frac * (w - 1)), _DOT_HALF, w - 1 - _DOT_HALF))
        row = h // 2
        out: dict[str, np.ndarray] = {}
        for cam in self._cameras:
            frame = np.empty((h, w, 3), dtype=np.uint8)
            frame[:, :] = _CAMERA_BG.get(cam, _DEFAULT_BG)
            frame[
                row - _DOT_HALF : row + _DOT_HALF + 1,
                col - _DOT_HALF : col + _DOT_HALF + 1,
            ] = 255
            out[cam] = np.repeat(frame[None, ...], n, axis=0)
        return out
