"""Transport seam for the Unitree G1 adapter (T-21, FR-06, OD-08).

``G1Transport`` is the narrow hardware boundary: everything above it (joint mapping, unit
conversion, limit clipping, e-stop latching in :mod:`wam.robot.g1`) is pure logic and fully
testable against :class:`FakeG1Transport`. :class:`DdsG1Transport` is the real DDS-backed
implementation; it imports cleanly without ``unitree_sdk2py`` and every hardware method is
gated behind a lazy SDK import.

Low-state dict contract (returned by ``read_low_state``):

- ``"q"``: ``np.ndarray`` [29] float32, motor positions in rad (G1JointIndex order)
- ``"dq"``: ``np.ndarray`` [29] float32, motor velocities in rad/s
- ``"imu"``: ``{"quat_wxyz": [4] f32, "gyro": [3] f32 rad/s, "acc": [3] f32 m/s^2}``
- ``"gripper"``: ``np.ndarray`` [2] float32 vendor units (left, right); OPTIONAL — adapters
  must degrade gripper validity when absent
- ``"tick_ns"``: int, vendor controller tick. A tick that does not advance between reads
  means the sample is stale (watchdog food).

Torch-free; numpy only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import with wam.robot.g1
    from wam.robot.g1 import G1Config

G1_NUM_MOTORS = 29
SDK_MISSING_MSG = "G1 hardware support requires unitree_sdk2py"

_IMU_DEFAULT: dict[str, tuple[float, ...]] = {
    "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    "gyro": (0.0, 0.0, 0.0),
    "acc": (0.0, 0.0, 9.81),
}


@runtime_checkable
class G1Transport(Protocol):
    """Minimal hardware I/O surface the G1 adapter depends on."""

    def read_low_state(self) -> dict[str, Any]:
        """Return one low-state sample (see module docstring for the dict contract)."""
        ...

    def write_motor_cmd(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        """Send one position-control command for all 29 motors (arrays [29], rad / rad/s)."""
        ...

    def write_gripper_cmd(self, left: float, right: float) -> None:
        """Send one gripper command pair in VENDOR units (left, right)."""
        ...

    def emergency_damp(self) -> None:
        """Vendor damping mode = safe stop. Must be safe to call at any time."""
        ...


def _as_motor_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (G1_NUM_MOTORS,):
        raise ValueError(f"{name}: expected shape ({G1_NUM_MOTORS},), got {arr.shape}")
    return arr


class FakeG1Transport:
    """Deterministic in-memory :class:`G1Transport` for tests (no hardware, no threads).

    Physics model: first-order lag toward ``q_target`` per motor command
    (``q += lag * (q_target - q)``); ``dq`` is the finite difference of the last write over
    one tick period. The tick advances by ``tick_step_ns`` per write (motor or gripper)
    unless ``freeze_tick`` is set — freezing simulates a stalled vendor controller for
    stale-tick / watchdog tests. Optional per-motor Gaussian read noise (seeded, applied on
    read only; the internal state stays noise-free). Every command is recorded for
    assertions.
    """

    def __init__(
        self,
        initial_q: np.ndarray | None = None,
        *,
        lag: float = 0.5,
        tick_step_ns: int = 2_000_000,
        imu: dict[str, Any] | None = None,
        initial_gripper: tuple[float, float] = (0.0, 0.0),
        noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        if not 0.0 < lag <= 1.0:
            raise ValueError(f"lag must be in (0, 1], got {lag}")
        if tick_step_ns <= 0:
            raise ValueError(f"tick_step_ns must be > 0, got {tick_step_ns}")
        self._q = (
            np.zeros(G1_NUM_MOTORS, dtype=np.float64)
            if initial_q is None
            else _as_motor_array("initial_q", initial_q).copy()
        )
        self._dq = np.zeros(G1_NUM_MOTORS, dtype=np.float64)
        self._gripper = np.asarray(initial_gripper, dtype=np.float64)
        self._lag = float(lag)
        self._tick_step_ns = int(tick_step_ns)
        self._tick_ns = 0
        imu_in = imu or _IMU_DEFAULT
        self._imu = {
            key: np.asarray(imu_in.get(key, _IMU_DEFAULT[key]), dtype=np.float32)
            for key in _IMU_DEFAULT
        }
        self._noise_std = float(noise_std)
        self._rng = np.random.default_rng(seed)

        #: When True, writes no longer advance the tick (stale-sample injection).
        self.freeze_tick: bool = False
        #: Every write_motor_cmd call: dicts with q_target/dq_target/kp/kd copies + tick_ns.
        self.motor_commands: list[dict[str, Any]] = []
        #: Every write_gripper_cmd call: (left, right) vendor units.
        self.gripper_commands: list[tuple[float, float]] = []
        #: Number of emergency_damp calls.
        self.damp_count: int = 0

    # -- test hooks ------------------------------------------------------------------------

    def set_imu(
        self,
        quat_wxyz: tuple[float, ...] | np.ndarray,
        gyro: tuple[float, ...] | np.ndarray,
        acc: tuple[float, ...] | np.ndarray,
    ) -> None:
        """Inject the IMU sample returned by subsequent reads."""
        self._imu = {
            "quat_wxyz": np.asarray(quat_wxyz, dtype=np.float32),
            "gyro": np.asarray(gyro, dtype=np.float32),
            "acc": np.asarray(acc, dtype=np.float32),
        }

    @property
    def q(self) -> np.ndarray:
        """Current (noise-free) motor positions, copy."""
        return self._q.copy()

    @property
    def tick_ns(self) -> int:
        return self._tick_ns

    # -- G1Transport -----------------------------------------------------------------------

    def read_low_state(self) -> dict[str, Any]:
        q = self._q.copy()
        dq = self._dq.copy()
        if self._noise_std > 0.0:
            q += self._rng.normal(0.0, self._noise_std, G1_NUM_MOTORS)
            dq += self._rng.normal(0.0, self._noise_std, G1_NUM_MOTORS)
        return {
            "q": q.astype(np.float32),
            "dq": dq.astype(np.float32),
            "imu": {key: value.copy() for key, value in self._imu.items()},
            "gripper": self._gripper.astype(np.float32),
            "tick_ns": self._tick_ns,
        }

    def write_motor_cmd(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        q_t = _as_motor_array("q_target", q_target)
        dq_t = _as_motor_array("dq_target", dq_target)
        kp_a = _as_motor_array("kp", kp)
        kd_a = _as_motor_array("kd", kd)
        self.motor_commands.append(
            {
                "q_target": q_t.copy(),
                "dq_target": dq_t.copy(),
                "kp": kp_a.copy(),
                "kd": kd_a.copy(),
                "tick_ns": self._tick_ns,
            }
        )
        q_new = self._q + self._lag * (q_t - self._q)
        dt_s = self._tick_step_ns * 1e-9
        self._dq = (q_new - self._q) / dt_s
        self._q = q_new
        self._advance_tick()

    def write_gripper_cmd(self, left: float, right: float) -> None:
        self.gripper_commands.append((float(left), float(right)))
        target = np.array([left, right], dtype=np.float64)
        self._gripper = self._gripper + self._lag * (target - self._gripper)
        self._advance_tick()

    def emergency_damp(self) -> None:
        self.damp_count += 1
        self._dq = np.zeros(G1_NUM_MOTORS, dtype=np.float64)

    def _advance_tick(self) -> None:
        if not self.freeze_tick:
            self._tick_ns += self._tick_step_ns


class DdsG1Transport:
    """Real DDS transport for the Unitree G1 via ``unitree_sdk2py``.

    The constructor ONLY stores config — it never touches the SDK, so this module imports
    and constructs cleanly on machines without the vendor stack. Every hardware method is
    gated behind a lazy ``import unitree_sdk2py`` and raises
    ``RuntimeError('G1 hardware support requires unitree_sdk2py')`` when it is missing.

    Intended SDK mapping (unitree_sdk2py conventions, G1 = ``unitree_hg`` IDL family):

    - ``open()``:
      ``ChannelFactoryInitialize(0, network_interface)`` from
      ``unitree_sdk2py.core.channel``; then
      ``ChannelSubscriber("rt/lowstate", LowState_).Init()`` and
      ``ChannelPublisher("rt/lowcmd", LowCmd_).Init()`` with
      ``LowState_``/``LowCmd_`` from ``unitree_sdk2py.idl.unitree_hg.msg.dds_``;
      ``CRC`` instance from ``unitree_sdk2py.utils.crc``.
    - ``read_low_state()``: ``subscriber.Read()`` -> ``LowState_``; map
      ``motor_state[i].q/.dq`` (i in 0..28) into the q/dq arrays,
      ``imu_state.quaternion`` (w,x,y,z) / ``.gyroscope`` / ``.accelerometer`` into the imu
      dict, and ``tick`` (ms) * 1_000_000 into ``tick_ns``. Gripper state comes from the
      Dex3 hand state topics (``rt/dex3/left/state``, ``rt/dex3/right/state``) when hands
      are fitted.
    - ``write_motor_cmd()``: fill ``unitree_hg_msg_dds__LowCmd_()`` (from
      ``unitree_sdk2py.idl.default``): ``motor_cmd[i].q/.dq/.kp/.kd`` from the arrays,
      ``tau = 0``, set ``mode_machine`` from the reported low state; stamp
      ``cmd.crc = CRC().Crc(cmd)``; ``publisher.Write(cmd)``.
    - ``write_gripper_cmd()``: publish Dex3 ``HandCmd_`` on ``rt/dex3/left/cmd`` /
      ``rt/dex3/right/cmd`` (vendor units passed through unchanged).
    - ``emergency_damp()``: vendor damping service —
      ``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient().Damp()`` (all joints to damping
      mode; the deterministic safe stop per OD-08).
    """

    def __init__(self, config: G1Config | None = None) -> None:
        self._config = config
        self._channel: Any = None  # set by open() when the SDK is present

    @property
    def config(self) -> G1Config | None:
        return self._config

    def _require_sdk(self) -> Any:
        try:
            import unitree_sdk2py
        except ImportError as exc:
            raise RuntimeError(SDK_MISSING_MSG) from exc
        return unitree_sdk2py

    def open(self) -> None:
        """Initialize the DDS channel factory, low-state subscriber and low-cmd publisher."""
        self._require_sdk()
        raise NotImplementedError("G1 DDS channel setup pending first hardware access (OD-08)")

    def read_low_state(self) -> dict[str, Any]:
        self._require_sdk()
        raise NotImplementedError("G1 LowState_ read pending first hardware access (OD-08)")

    def write_motor_cmd(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        self._require_sdk()
        raise NotImplementedError("G1 LowCmd_ write pending first hardware access (OD-08)")

    def write_gripper_cmd(self, left: float, right: float) -> None:
        self._require_sdk()
        raise NotImplementedError("G1 HandCmd_ write pending first hardware access (OD-08)")

    def emergency_damp(self) -> None:
        self._require_sdk()
        raise NotImplementedError("G1 damping service pending first hardware access (OD-08)")
