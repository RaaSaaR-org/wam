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

import threading
import time
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import with wam.robot.g1
    from wam.robot.g1 import G1Config

G1_NUM_MOTORS = 29
SDK_MISSING_MSG = "G1 hardware support requires unitree_sdk2py"

# -- DDS wire constants (unitree_hg IDL family) -------------------------------------------
# Verified against unitree_sdk2py @ 65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5 (v1.0.1):
# LowCmd_/LowState_ carry 35 motor slots, of which the 29-DoF G1 uses 0..28.
G1_LOWSTATE_TOPIC = "rt/lowstate"
G1_LOWCMD_TOPIC = "rt/lowcmd"
#: Upper-body-only alternative: the vendor loco controller keeps the legs, the SDK blends
#: in arm targets weighted by ``motor_cmd[29].q``. See ``DdsG1Transport(cmd_topic=...)``.
G1_ARM_SDK_TOPIC = "rt/arm_sdk"
G1_ARM_SDK_WEIGHT_MOTOR = 29
#: Dex3-1 hands. Topic names are vendor documentation, NOT verified against hardware (OD-08).
DEX3_CMD_TOPICS = ("rt/dex3/left/cmd", "rt/dex3/right/cmd")
DEX3_STATE_TOPICS = ("rt/dex3/left/state", "rt/dex3/right/state")
DEX3_NUM_MOTORS = 7
#: MotorCmd_.mode: 1 = enable, 0 = disable (vendor low-level convention).
MOTOR_MODE_ENABLE = 1
#: LowCmd_.mode_pr: 0 = series (pitch/roll) control of the ankle/waist parallel joints.
MODE_PR_SERIES = 0


def dex3_mode_byte(motor_id: int, status: int = 1, timeout: int = 0) -> int:
    """Pack a Dex3-1 ``MotorCmd_.mode`` byte: ``id | status << 4 | timeout << 7``.

    Vendor bit layout (RIS mode): bits 0-3 motor id, bits 4-6 status (1 = enable),
    bit 7 timeout flag. NOT verified against hardware (OD-08).
    """
    return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)


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
    """Real DDS transport for the Unitree G1 via ``unitree_sdk2py`` (unitree_hg IDL family).

    Contracts:

    - **Import- and construct-safe without the vendor stack.** The constructor only stores
      config; it never touches the SDK. Every I/O method calls ``_require_sdk()`` FIRST and
      raises ``RuntimeError('G1 hardware support requires unitree_sdk2py')`` when the SDK is
      absent, so this module imports on any machine.
    - **``open()`` before I/O.** ``read_low_state``/``write_*``/``emergency_damp`` raise
      ``RuntimeError`` when called before ``open()``. ``open()`` is idempotent.
    - **Latest-sample semantics.** A DDS listener stores the most recent ``LowState_``;
      ``read_low_state()`` never blocks after the first sample and returns that cached
      sample. If the robot stops publishing, ``tick_ns`` stops advancing and the adapter
      above degrades every validity flag — the stale-sample path is fed by exactly this.
      ``open()`` waits up to ``state_timeout_s`` for the FIRST sample; if none arrives it
      raises (a silent no-state connect would look healthy while being blind).
    - **Wire mapping** (verified against unitree_sdk2py @ 65691c8, v1.0.1):
      ``LowState_.motor_state[i].q/.dq`` for ``i in 0..28`` -> ``q``/``dq`` [29];
      ``imu_state.quaternion`` (w,x,y,z) / ``.gyroscope`` / ``.accelerometer`` -> ``imu``;
      ``LowState_.tick`` (uint32 ms) * 1e6 -> ``tick_ns``. The tick WRAPS every ~49.7 days;
      only "did it advance" is contractual, never the absolute value.
      ``write_motor_cmd`` fills ``LowCmd_.motor_cmd[0..28]`` (``mode=1`` enable, ``tau=0``),
      copies ``mode_machine`` from the last received ``LowState_``, stamps
      ``crc = CRC().Crc(cmd)`` and publishes on ``cmd_topic``. Motor slots 29..34 stay at
      their defaults (``mode=0``, disabled) — the 29-DoF G1 does not use them.
    - **``emergency_damp()`` is a wire command, not a service call.** It publishes a LowCmd
      with ``kp=0, kd=damp_kd, q=0, dq=0, tau=0`` on ALL 29 motors (pure viscous damping),
      ``damp_repeats`` times. This is deliberate: it works even when the vendor loco service
      is dead or busy, which is exactly the situation an e-stop has to survive (OD-08).
      ``unitree_sdk2py.g1.loco.g1_loco_client.LocoClient().Damp()`` remains the vendor
      high-level escalation and should be added ON TOP during hardware bring-up — it is not
      used here because it needs a live service peer.
    - **Gripper mapping is a PLACEHOLDER pending OD-08.** The Dex3-1 has 7 joints per hand
      and no single "aperture" axis. ``write_gripper_cmd`` commands all 7 finger motors of a
      hand to the same target and ``read_low_state`` reports the MEAN of the 7 measured
      joint angles, so the pair round-trips consistently — but it is an open/close proxy,
      not a grasp policy. Dex3 topic names and the RIS mode-byte layout come from vendor
      documentation and are NOT verified against hardware.
    - Torch-free; numpy only. Thread-safe for one reader and one writer: the DDS listener
      threads only ever rebind ``_low_state`` / ``_hand_states`` under ``self._lock``.

    Bring-up note: ``cmd_topic=G1_ARM_SDK_TOPIC`` with ``arm_sdk_weight=1.0`` is the SAFER
    first-contact path on a standing robot — the vendor loco controller keeps the legs and
    blends in the SDK's arm targets — whereas ``rt/lowcmd`` takes over all 29 motors at once.
    """

    #: Dex3 finger gains for ``write_gripper_cmd`` — conservative PLACEHOLDERS pending OD-08.
    HAND_KP = 1.5
    HAND_KD = 0.1

    def __init__(
        self,
        config: G1Config | None = None,
        *,
        domain_id: int = 0,
        cmd_topic: str = G1_LOWCMD_TOPIC,
        arm_sdk_weight: float | None = None,
        state_timeout_s: float = 5.0,
        damp_kd: float = 2.0,
        damp_repeats: int = 3,
        enable_hands: bool = True,
    ) -> None:
        """Store config only — no SDK import, no I/O.

        ``domain_id``: DDS domain (vendor default 0). ``cmd_topic``: ``rt/lowcmd`` (full
        low-level authority) or ``G1_ARM_SDK_TOPIC`` (upper-body blend, see class docstring).
        ``arm_sdk_weight``: written to ``motor_cmd[29].q`` when set; REQUIRED (1.0) for the
        arm_sdk topic, where a missing weight makes the robot ignore every command.
        ``state_timeout_s``: how long ``open()`` waits for the first ``LowState_``.
        ``damp_kd``: viscous damping gain used by ``emergency_damp()`` — a conservative
        vendor-typical value, CONFIRM against the datasheet before real use (OD-08).
        ``enable_hands``: subscribe/publish the Dex3 hand topics.
        """
        if state_timeout_s <= 0.0:
            raise ValueError(f"state_timeout_s must be > 0, got {state_timeout_s}")
        if damp_kd < 0.0:
            raise ValueError(f"damp_kd must be >= 0, got {damp_kd}")
        if damp_repeats < 1:
            raise ValueError(f"damp_repeats must be >= 1, got {damp_repeats}")
        if cmd_topic == G1_ARM_SDK_TOPIC and arm_sdk_weight is None:
            raise ValueError(
                f"{G1_ARM_SDK_TOPIC} ignores commands without a weight — pass "
                "arm_sdk_weight=1.0 (see DdsG1Transport docstring)"
            )
        self._config = config
        self._domain_id = int(domain_id)
        self._cmd_topic = str(cmd_topic)
        self._arm_sdk_weight = None if arm_sdk_weight is None else float(arm_sdk_weight)
        self._state_timeout_s = float(state_timeout_s)
        self._damp_kd = float(damp_kd)
        self._damp_repeats = int(damp_repeats)
        self._enable_hands = bool(enable_hands)

        self._opened = False
        self._lock = threading.Lock()
        self._state_event = threading.Event()
        self._low_state: Any = None  # latest LowState_, rebound by the DDS listener thread
        self._hand_states: list[Any] = [None, None]  # latest HandState_ (left, right)
        self._low_cmd: Any = None  # reused LowCmd_ instance
        self._hand_cmds: list[Any] = [None, None]
        self._crc: Any = None
        self._cmd_pub: Any = None
        self._state_sub: Any = None
        self._hand_pubs: list[Any] = [None, None]
        self._hand_subs: list[Any] = [None, None]

    @property
    def config(self) -> G1Config | None:
        return self._config

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def cmd_topic(self) -> str:
        return self._cmd_topic

    @property
    def damp_kd(self) -> float:
        """Viscous damping gain published by :meth:`emergency_damp`."""
        return self._damp_kd

    def _require_sdk(self) -> Any:
        try:
            import unitree_sdk2py
        except ImportError as exc:
            raise RuntimeError(SDK_MISSING_MSG) from exc
        return unitree_sdk2py

    def _require_open(self, op: str) -> None:
        if not self._opened:
            raise RuntimeError(f"{op}: DDS transport not open — call open() first")

    # -- lifecycle -------------------------------------------------------------------------

    def open(self) -> None:
        """Initialize the DDS channel factory, the low-state subscriber and the command
        publisher, then wait for the first ``LowState_``. Idempotent."""
        self._require_sdk()
        if self._opened:
            return

        from unitree_sdk2py.core.channel import (
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (
            unitree_hg_msg_dds__HandCmd_,
            unitree_hg_msg_dds__LowCmd_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_, LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        interface = self._config.network_interface if self._config is not None else None
        # ChannelFactory is a process-wide singleton: a second Init() with different
        # arguments is a silent no-op, which is why domain/interface live in the config.
        ChannelFactoryInitialize(self._domain_id, interface)

        self._crc = CRC()
        self._low_cmd = unitree_hg_msg_dds__LowCmd_()
        self._hand_cmds = [unitree_hg_msg_dds__HandCmd_(), unitree_hg_msg_dds__HandCmd_()]

        self._state_sub = ChannelSubscriber(G1_LOWSTATE_TOPIC, LowState_)
        self._state_sub.Init(self._on_low_state, 0)
        self._cmd_pub = ChannelPublisher(self._cmd_topic, LowCmd_)
        self._cmd_pub.Init()

        if self._enable_hands:
            for side in (0, 1):
                sub = ChannelSubscriber(DEX3_STATE_TOPICS[side], HandState_)
                sub.Init(self._make_hand_handler(side), 0)
                self._hand_subs[side] = sub
                pub = ChannelPublisher(DEX3_CMD_TOPICS[side], HandCmd_)
                pub.Init()
                self._hand_pubs[side] = pub

        self._opened = True
        if not self._state_event.wait(self._state_timeout_s):
            self.close()
            raise RuntimeError(
                f"no {G1_LOWSTATE_TOPIC} sample within {self._state_timeout_s:.1f}s on "
                f"interface {interface!r} (domain {self._domain_id}) — is the robot "
                "publishing and on the same DDS domain?"
            )

    def close(self) -> None:
        """Tear down publishers/subscribers and drop the cached state. Safe to call more than
        once and on a never-opened transport (no-op, no SDK needed)."""
        for holder in (self._hand_subs, self._hand_pubs):
            for i, channel in enumerate(holder):
                if channel is not None:
                    channel.Close()
                    holder[i] = None
        if self._state_sub is not None:
            self._state_sub.Close()
            self._state_sub = None
        if self._cmd_pub is not None:
            self._cmd_pub.Close()
            self._cmd_pub = None
        with self._lock:
            self._low_state = None
            self._hand_states = [None, None]
        self._state_event.clear()
        self._opened = False

    # -- DDS listener callbacks (run on the Cyclone listener thread) ------------------------

    def _on_low_state(self, msg: Any) -> None:
        with self._lock:
            self._low_state = msg
        self._state_event.set()

    def _make_hand_handler(self, side: int) -> Any:
        def _handler(msg: Any) -> None:
            with self._lock:
                self._hand_states[side] = msg

        return _handler

    # -- G1Transport ------------------------------------------------------------------------

    def read_low_state(self) -> dict[str, Any]:
        """Return the latest ``LowState_`` mapped to the low-state dict contract.

        Never blocks (``open()`` already waited for the first sample). A robot that stopped
        publishing yields the SAME ``tick_ns`` again — that is the stale-sample signal the
        adapter turns into cleared validity flags.
        """
        self._require_sdk()
        self._require_open("read_low_state")
        with self._lock:
            state = self._low_state
            hand_states = list(self._hand_states)
        if state is None:  # pragma: no cover - open() guarantees a first sample
            raise RuntimeError(f"read_low_state: no {G1_LOWSTATE_TOPIC} sample received yet")

        q = np.empty(G1_NUM_MOTORS, dtype=np.float32)
        dq = np.empty(G1_NUM_MOTORS, dtype=np.float32)
        motors = state.motor_state
        for i in range(G1_NUM_MOTORS):
            q[i] = motors[i].q
            dq[i] = motors[i].dq

        imu_msg = state.imu_state
        low: dict[str, Any] = {
            "q": q,
            "dq": dq,
            "imu": {
                "quat_wxyz": np.asarray(imu_msg.quaternion, dtype=np.float32),
                "gyro": np.asarray(imu_msg.gyroscope, dtype=np.float32),
                "acc": np.asarray(imu_msg.accelerometer, dtype=np.float32),
            },
            "tick_ns": int(state.tick) * 1_000_000,
        }
        gripper = self._gripper_from_hand_states(hand_states)
        if gripper is not None:
            # Key ABSENT (not zeroed) while the hands are silent: the adapter degrades
            # gripper validity on the missing key, which is the honest signal.
            low["gripper"] = gripper
        return low

    @staticmethod
    def _gripper_from_hand_states(hand_states: list[Any]) -> np.ndarray | None:
        """Mean finger-joint angle per hand, or None when either hand has not reported."""
        out = np.zeros(2, dtype=np.float32)
        for side, hand_state in enumerate(hand_states):
            if hand_state is None:
                return None
            joints = [float(motor.q) for motor in hand_state.motor_state[:DEX3_NUM_MOTORS]]
            if not joints:
                return None
            out[side] = float(np.mean(joints))
        return out

    def write_motor_cmd(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        """Publish one 29-motor position command with a valid CRC on ``cmd_topic``."""
        self._require_sdk()
        self._require_open("write_motor_cmd")
        q = _as_motor_array("q_target", q_target)
        dq = _as_motor_array("dq_target", dq_target)
        kp_a = _as_motor_array("kp", kp)
        kd_a = _as_motor_array("kd", kd)
        self._publish_low_cmd(q, dq, kp_a, kd_a)

    def write_gripper_cmd(self, left: float, right: float) -> None:
        """Publish one Dex3-1 ``HandCmd_`` per hand (PLACEHOLDER mapping, class docstring).

        ``left``/``right`` are vendor units (finger joint angle in rad); every finger motor of
        a hand receives the same target.
        """
        self._require_sdk()
        self._require_open("write_gripper_cmd")
        if not self._enable_hands:
            raise RuntimeError("write_gripper_cmd: transport was opened with enable_hands=False")
        for side, value in enumerate((float(left), float(right))):
            cmd = self._hand_cmds[side]
            for motor_id in range(DEX3_NUM_MOTORS):
                motor = cmd.motor_cmd[motor_id]
                motor.mode = dex3_mode_byte(motor_id)
                motor.q = value
                motor.dq = 0.0
                motor.tau = 0.0
                motor.kp = float(self.HAND_KP)
                motor.kd = float(self.HAND_KD)
            self._hand_pubs[side].Write(cmd)

    def emergency_damp(self) -> None:
        """Publish a pure-damping LowCmd (kp=0, kd=``damp_kd``, q=dq=tau=0) on all 29 motors,
        ``damp_repeats`` times. Deterministic safe stop — no vendor service involved."""
        self._require_sdk()
        self._require_open("emergency_damp")
        zeros = np.zeros(G1_NUM_MOTORS, dtype=np.float64)
        kd = np.full(G1_NUM_MOTORS, self._damp_kd, dtype=np.float64)
        for i in range(self._damp_repeats):
            if i > 0:
                time.sleep(0.002)  # one vendor control tick between repeats
            self._publish_low_cmd(zeros, zeros, zeros, kd)

    # -- LowCmd_ assembly --------------------------------------------------------------------

    def _publish_low_cmd(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        cmd = self._low_cmd
        with self._lock:
            state = self._low_state
        cmd.mode_pr = MODE_PR_SERIES
        cmd.mode_machine = int(state.mode_machine) if state is not None else 0
        for i in range(G1_NUM_MOTORS):
            motor = cmd.motor_cmd[i]
            motor.mode = MOTOR_MODE_ENABLE
            motor.q = float(q[i])
            motor.dq = float(dq[i])
            motor.tau = 0.0
            motor.kp = float(kp[i])
            motor.kd = float(kd[i])
        if self._arm_sdk_weight is not None:
            cmd.motor_cmd[G1_ARM_SDK_WEIGHT_MOTOR].q = self._arm_sdk_weight
        cmd.crc = self._crc.Crc(cmd)
        self._cmd_pub.Write(cmd)
