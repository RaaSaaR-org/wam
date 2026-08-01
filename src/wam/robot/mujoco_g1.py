"""MuJoCo-backed G1 robot for the WAM runtime (T-21, FR-06, E2).

Thin wrapper that composes the REAL :class:`~wam.robot.g1.G1Adapter` with a
:class:`~wam.robot.mujoco_transport.MujocoG1Transport`, so the full hardware code path —
joint mapping, unit conversion, defense-in-depth clipping, wall-/sim-clock pacing and the
latched e-stop — runs on MuJoCo contact physics and real rendered pixels instead of a
kinematic mock. Everything a rollout harness needs beyond the ``RobotAdapter`` protocol
(cameras, episode reset, sim clock) lives here; nothing robot-specific is added.

COMPOSITION, NOT SUBCLASSING (deliberate)
  ``MujocoG1Robot`` HOLDS a ``G1Adapter`` and forwards the four protocol methods verbatim.
  Subclassing was rejected: a subclass would inherit ``connect()``'s
  ``DdsG1Transport`` fallback (a sim robot that can silently try to open a DDS socket is a
  hazard, not a convenience), would be free to override the very safety code this class
  exists to exercise unchanged, and would tie the sim wrapper to the adapter's constructor
  signature. Delegation makes the boundary explicit: the adapter is the robot, this is the
  simulator around it. ``RobotAdapter`` is structural (``@runtime_checkable``), so the
  wrapper satisfies it without inheriting anything.

Contracts:

- **``mujoco`` is an OPTIONAL dependency** (``uv pip install wam[sim]``). This module imports
  without it; ``mujoco`` is imported lazily in ``__init__`` and a missing install raises
  ``RuntimeError`` naming the fix. ``wam.robot.registry`` imports THIS module lazily and
  lists ``"mujoco_g1"`` under ``optional_robots()`` rather than ``available_robots()``, so
  everything else in the registry keeps working without the dependency.
- **Constructed connected.** ``__init__`` builds the transport and calls
  ``G1Adapter.connect()`` with it injected — no SDK, no socket, no hardware. ``get_robot
  ("mujoco_g1")`` therefore returns a ready-to-read robot, like ``MockRobot``.
- **Sim gains by default.** With no ``config``, gains are :data:`SIM_KP` / :data:`SIM_KD`
  (the vendor model's own stiffness with per-joint CRITICAL damping), ``dq_max`` is
  :data:`SIM_DQ_MAX` and ``q_min``/``q_max`` are read from the SCENE'S OWN joint ranges —
  i.e. the no-config path matches ``configs/robot/mujoco_g1.yaml`` field for field.
  ``G1Config``'s own defaults are conservative HARDWARE placeholders (kp=20: the arm visibly
  sags, 0.17 rad steady-state error), its +-1.5708 rad range is narrower than the G1's real
  one and its dq_max=2.0 is looser than the shipped config — none is right for a sim.
- **``gripper_vendor_min/max`` must stay (0.0, 1.0).** The transport's gripper channel IS the
  Dex3 synergy fraction, so the canonical unit and the vendor unit coincide; any other range
  would make ``G1Adapter.gripper_to_vendor`` command a fully-closed hand for every input above
  ~0.01 with no error anywhere. ``__init__`` rejects it.
- **Every commanded joint delta is UNDER-EXECUTED, by a prefix-dependent factor.**
  ``G1Adapter.execute()`` re-bases its target on the MEASURED ``q`` at each call, so the
  position loop's lag is discarded rather than caught up. With the sim gains a joint executes
  ~0.39 of a one-control-period step within that period, rising to ~0.9 over a 25-step chunk.
  This is a property of the control architecture (no feed-forward, no integral action) on a
  plant with a finite servo bandwidth, NOT a MuJoCo artifact and NOT tunable away at any
  physically defensible stiffness — see the PER-CONTROL-PERIOD EXECUTION section of the
  ``mujoco_transport`` module docstring for the measurements, and ``docs/sim.md`` for what it
  invalidates (recorded action labels, safety-intervention rates, velocity-envelope claims).
- **Pacing runs on SIM TIME, not wall time.** ``G1Adapter.execute()`` paces its per-step
  command stream so step ``i`` is sent no earlier than ``t0 + i * dt_s``; that pacing is
  what makes its ``dq_max * dt`` clip a genuine velocity limit. The injected clock is the
  transport's sim tick (which advances by exactly ``control_dt_s`` per motor command), and
  the injected sleep advances PHYSICS rather than blocking a thread. Consequence: the
  velocity-limit reasoning holds exactly as on hardware, while a rollout runs as fast as
  the machine can step (~28x realtime, physics only). Pass ``clock=time.monotonic`` and
  ``sleep=time.sleep`` for a realtime-paced sim.
  Keep ``chunk.dt_s == config.control_dt_s``: a LARGER chunk ``dt_s`` makes the executor
  step extra hold-physics between commands (correct, just slower), a SMALLER one advances
  more sim time per step than the chunk claims (the trajectory plays back slow-motion).
- **Rendering never steps physics.** ``render_frames(n)`` renders the CURRENT state once
  per camera and repeats it ``n`` times — the ``n`` frames are IDENTICAL (same contract as
  ``MockRobot.render_frames``). Frames differ between calls, i.e. after ``execute()``.
  One ``mujoco.Renderer`` is created lazily on the first call and reused for every camera
  and every frame; ``close()`` releases it.
- **``reset()`` is an EPISODE reset, not an e-stop release.** It restores the keyframe pose
  and pristine actuator gains, and it clears the adapter's stale-tick memory. A latched
  e-stop SURVIVES it and is released only by the deliberate ``clear_estop()`` (mirroring
  ``MockRobot.clear_estop``).
- **Determinism.** No Python-level RNG anywhere in this class. Same scene + same keyframe +
  same command sequence => bit-identical states.
- Torch-free; numpy only.

Typical use::

    robot = get_robot("mujoco_g1")            # or MujocoG1Robot()
    state = robot.read_state()                 # canonical 15-joint state, validity flags
    images = robot.render_frames(1)            # {"head": [1, H, W, 3], "wrist_left": ...}
    robot.execute(safe_chunk, prefix_steps=5)  # steps 5 x control_dt_s of physics
    robot.reset()                              # next episode
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from wam.interfaces.schema import ActionChunk, CanonicalSpaceSpec, RobotState
from wam.robot.g1 import G1_JOINT_MAP, G1_NUM_CANONICAL_JOINTS, G1_SPEC, G1Adapter, G1Config
from wam.robot.mujoco_transport import (
    DEFAULT_KEYFRAME,
    MJCF_JOINT_SUFFIX,
    MUJOCO_MISSING_MSG,
    VENDOR_MODEL,
    VENDOR_MODEL_MISSING_MSG,
    MujocoG1Transport,
)

__all__ = [
    "DEFAULT_CAMERAS",
    "DEFAULT_IMAGE_HW",
    "SIM_DQ_MAX",
    "SIM_KD",
    "SIM_KP",
    "VENDOR_MODEL",
    "VENDOR_MODEL_MISSING_MSG",
    "MujocoG1Robot",
    "scene_critical_damping",
    "scene_joint_limits",
]

#: Sim position gain per canonical joint: the vendor Menagerie ``g1`` actuator class stiffness,
#: i.e. the value the model itself was authored for. SIM gains for a model with NO gravity
#: compensation — NOT the hardware placeholders in ``configs/robot/g1.yaml`` (OD-08).
SIM_KP = 500.0

#: Sim damping per canonical joint — CRITICAL damping against each joint's own effective
#: inertia, ``kd = 2 * sqrt(SIM_KP * m_eff)``. These are exactly the values MuJoCo compiles the
#: vendor model's ``dampratio="1"`` into (see :func:`scene_critical_damping`, and the test that
#: re-derives them from the scene). A FLAT kd is the wrong shape here: the wrists' effective
#: inertia is ~40x the waist's, so one number is either 3x overdamped at the wrist or 3x
#: underdamped at the waist. Measured against the criterion ``G1Adapter`` actually imposes
#: (fraction of a per-control-period position step executed within that period), the previous
#: flat kp=300/kd=15 executed 0.14 of a step on average; this set executes 0.39 — see the
#: PER-CONTROL-PERIOD EXECUTION section of the ``mujoco_transport`` module docstring, which
#: also states why no physically defensible gain reaches 1.0.
SIM_KD: tuple[float, ...] = (
    28.20,  # waist_yaw
    20.68,  # left_shoulder_pitch
    14.94,  # left_shoulder_roll
    14.41,  # left_shoulder_yaw
    13.57,  # left_elbow
    4.70,  # left_wrist_roll
    7.65,  # left_wrist_pitch
    6.30,  # left_wrist_yaw
    20.68,  # right_shoulder_pitch
    14.94,  # right_shoulder_roll
    14.41,  # right_shoulder_yaw
    13.57,  # right_elbow
    4.70,  # right_wrist_roll
    7.65,  # right_wrist_pitch
    6.30,  # right_wrist_yaw
)

#: Per-joint velocity cap for the no-config path, kept IDENTICAL to
#: ``configs/robot/mujoco_g1.yaml`` (and to ``configs/robot/g1.yaml``): these are conservative
#: MVP policy caps (PRD §11.2), not hardware facts. ``G1Config``'s own 2.0 rad/s default is a
#: placeholder and is 33% looser, which would make ``get_robot("mujoco_g1")`` enforce a WIDER
#: defense-in-depth velocity clip than the versioned config claims.
SIM_DQ_MAX: tuple[float, ...] = (
    1.5,  # waist_yaw
    1.5,  # left_shoulder_pitch
    1.5,  # left_shoulder_roll
    1.5,  # left_shoulder_yaw
    1.5,  # left_elbow
    2.0,  # left_wrist_roll
    2.0,  # left_wrist_pitch
    2.0,  # left_wrist_yaw
    1.5,  # right_shoulder_pitch
    1.5,  # right_shoulder_roll
    1.5,  # right_shoulder_yaw
    1.5,  # right_elbow
    2.0,  # right_wrist_roll
    2.0,  # right_wrist_pitch
    2.0,  # right_wrist_yaw
)

#: Cameras defined by ``configs/sim/g1_scene.xml``; these names become the ``Observation.images``
#: keys the policy sees.
DEFAULT_CAMERAS: tuple[str, str] = ("head", "wrist_left")

#: Default render size (H, W). 256x256 costs ~10x more than physics (~20 control steps/s with
#: both cameras) — still ~10x the 2 Hz closed-loop floor.
DEFAULT_IMAGE_HW: tuple[int, int] = (256, 256)


def _require_mujoco() -> Any:
    """Import ``mujoco`` lazily; raise a RuntimeError naming the fix when it is absent."""
    try:
        import mujoco as _mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise RuntimeError(MUJOCO_MISSING_MSG) from exc
    return _mujoco


def _build_transport(
    scene_path: str | Path | None,
    control_dt_s: float,
    keyframe: str | None,
    cameras: tuple[str, ...],
    image_hw: tuple[int, int],
) -> MujocoG1Transport:
    """Construct the transport. The un-fetched-vendor-model hint lives in
    ``MujocoG1Transport.__init__`` so BOTH entry points produce it."""
    return MujocoG1Transport(
        scene_path,
        control_dt_s=control_dt_s,
        keyframe=keyframe,
        camera_names=cameras,
        render_hw=image_hw,
    )


def scene_critical_damping(mj: Any, model: Any, kp: float = SIM_KP) -> tuple[float, ...]:
    """Critically-damped kd per CANONICAL joint for stiffness ``kp``, read off the scene.

    The vendor actuators declare ``dampratio="1"``, which MuJoCo compiles into
    ``biasprm[2] = -2 * sqrt(kp_vendor * m_eff)`` for each joint's own effective inertia. This
    inverts that to recover ``m_eff`` and re-derives kd for the requested ``kp``, so a gain set
    is never a flat guess. :data:`SIM_KD` is this function's output at :data:`SIM_KP`, rounded
    to 2 decimals and frozen so the shipped configs and the code agree.
    """
    out: list[float] = []
    for name, _ in G1_JOINT_MAP:
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name + MJCF_JOINT_SUFFIX)
        if aid < 0:  # pragma: no cover - MujocoG1Transport already fails on a missing actuator
            raise ValueError(f"scene has no actuator {name + MJCF_JOINT_SUFFIX!r}")
        kp_vendor = float(model.actuator_gainprm[aid][0])
        kd_vendor = -float(model.actuator_biasprm[aid][2])
        m_eff = (kd_vendor / 2.0) ** 2 / kp_vendor
        out.append(2.0 * float(np.sqrt(kp * m_eff)))
    return tuple(out)


def scene_joint_limits(mj: Any, model: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(q_min, q_max) in CANONICAL order, read off the MJCF ``jnt_range`` of the 15 mapped
    joints. The scene's ranges are the ground truth for what the sim can physically reach;
    MuJoCo clamps every actuator target to them anyway, so a wider config limit would only
    make the adapter's clipping lie."""
    lo: list[float] = []
    hi: list[float] = []
    for name, _ in G1_JOINT_MAP:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name + MJCF_JOINT_SUFFIX)
        if jid < 0:  # pragma: no cover - MujocoG1Transport already fails on a missing joint
            raise ValueError(f"scene has no joint {name + MJCF_JOINT_SUFFIX!r}")
        lo.append(float(model.jnt_range[jid][0]))
        hi.append(float(model.jnt_range[jid][1]))
    return tuple(lo), tuple(hi)


class MujocoG1Robot:
    """``RobotAdapter`` backed by MuJoCo: ``G1Adapter`` + ``MujocoG1Transport`` + cameras.

    See the module docstring for the contracts (composition rationale, sim-time pacing,
    render semantics, reset vs e-stop).
    """

    def __init__(
        self,
        config: G1Config | Mapping[str, Any] | None = None,
        *,
        scene_path: str | Path | None = None,
        keyframe: str | None = DEFAULT_KEYFRAME,
        cameras: tuple[str, ...] = DEFAULT_CAMERAS,
        image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
        transport: MujocoG1Transport | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """Load the scene, wire a connected ``G1Adapter`` to it and validate the cameras.

        ``config``: ``G1Config`` or a mapping of its fields. ``None`` => sim gains
        (:data:`SIM_KP`/:data:`SIM_KD`) plus the scene's own joint ranges (see the module
        docstring). ``scene_path``: MJCF file, relative paths resolve against the repo root
        (default ``configs/sim/g1_scene.xml``). ``keyframe``: pose restored by ``reset()``.
        ``cameras``/``image_hw``: what ``render_frames`` returns; camera names are validated
        against the scene at construction. ``transport``: pre-built transport (its
        ``control_dt_s`` must match the config, else the sim-time pacing would silently
        desync). ``clock``/``sleep``: pacing seams handed to ``G1Adapter`` — default to sim
        time (see the module docstring); pass ``time.monotonic``/``time.sleep`` for realtime.

        Raises ``RuntimeError`` without ``mujoco``, and ``ValueError`` for a missing scene,
        an unknown keyframe/camera or a transport whose control period disagrees with the
        config.
        """
        self._mj = _require_mujoco()

        if config is None:
            # The limits come from the scene, so the transport must exist first.
            if transport is None:
                transport = _build_transport(
                    scene_path, G1Config().control_dt_s, keyframe, tuple(cameras), image_hw
                )
            q_min, q_max = scene_joint_limits(self._mj, transport.model)
            cfg = G1Config(
                q_min=q_min,
                q_max=q_max,
                dq_max=SIM_DQ_MAX,
                kp=(SIM_KP,) * G1_NUM_CANONICAL_JOINTS,
                kd=SIM_KD,
                control_dt_s=transport.control_dt_s,
            )
        else:
            cfg = config if isinstance(config, G1Config) else G1Config(**config)

        # The transport's gripper channel IS vendor units [0, 1] (the Dex3 synergy fraction) and
        # clips to that range. G1Adapter.gripper_to_vendor maps the canonical [0, 1] onto
        # config.gripper_vendor_min/max, so a non-unit vendor range would silently produce a
        # fully-closed hand for every command above ~0.01 AND a readback that under-reports
        # closure by the span factor — with no error at any layer. Reject it here.
        if (cfg.gripper_vendor_min, cfg.gripper_vendor_max) != (0.0, 1.0):
            raise ValueError(
                f"gripper_vendor_min/max must be (0.0, 1.0) for the MuJoCo transport, got "
                f"({cfg.gripper_vendor_min}, {cfg.gripper_vendor_max}) — the Dex3 synergy "
                f"fraction IS the vendor unit here"
            )

        if transport is None:
            transport = _build_transport(
                scene_path, cfg.control_dt_s, keyframe, tuple(cameras), image_hw
            )
        elif abs(transport.control_dt_s - cfg.control_dt_s) > 1e-12:
            raise ValueError(
                f"transport.control_dt_s={transport.control_dt_s} != "
                f"config.control_dt_s={cfg.control_dt_s} — sim-time pacing needs them equal"
            )
        for cam in cameras:
            if cam not in transport.camera_names:
                raise ValueError(
                    f"camera {cam!r} is not among the transport's cameras "
                    f"{list(transport.camera_names)}"
                )

        self._config = cfg
        self._transport = transport
        self._cameras = tuple(cameras)
        self._image_hw = (int(image_hw[0]), int(image_hw[1]))
        self._renderer: Any | None = None
        self._adapter = G1Adapter(
            cfg,
            transport,
            clock=clock if clock is not None else self._sim_clock,
            sleep=sleep if sleep is not None else self._sim_sleep,
        )
        self._adapter.connect()  # injected transport: no SDK, no socket, no hardware

    # -- introspection ---------------------------------------------------------------------

    @property
    def spec(self) -> CanonicalSpaceSpec:
        """Canonical space — identical to ``G1_SPEC`` (same 15 joints, ``gripper_dims=2``)."""
        return G1_SPEC

    @property
    def config(self) -> G1Config:
        return self._config

    @property
    def adapter(self) -> G1Adapter:
        """The wrapped adapter — the code path that also runs on hardware."""
        return self._adapter

    @property
    def transport(self) -> MujocoG1Transport:
        """The MuJoCo transport (``.model`` / ``.data`` for scene-level inspection)."""
        return self._transport

    @property
    def scene_path(self) -> Path:
        return self._transport.scene_path

    @property
    def cameras(self) -> tuple[str, ...]:
        """Camera names ``render_frames`` returns (the ``Observation.images`` keys)."""
        return self._cameras

    @property
    def image_hw(self) -> tuple[int, int]:
        """(H, W) of the rendered frames."""
        return self._image_hw

    @property
    def is_estopped(self) -> bool:
        """Latched e-stop state (the executor duck-types this for its rollout summary)."""
        return self._adapter.is_estopped

    @property
    def sim_time_ns(self) -> int:
        """Simulated monotonic clock in ns — the transport tick, advanced ONLY by physics
        steps (``execute``/``hold``), never by wall time. Mirrors ``MockRobot.sim_time_ns``
        so recording/replay can timestamp on sim time."""
        return self._transport.tick_ns

    # -- RobotAdapter protocol (delegated verbatim to the real G1Adapter) --------------------

    @property
    def limits(self) -> dict[str, np.ndarray]:
        """Canonical-order limit arrays from the config (float32)."""
        return self._adapter.limits

    def read_state(self) -> RobotState:
        """Canonical state from MuJoCo. Two consecutive reads with no physics step in
        between are STALE (the tick did not advance) and come back with cleared validity
        flags — exactly as a stalled vendor controller would look.

        The IMU is genuine here too (the scene carries gyro + accelerometer sensors and the
        torso body's world orientation), so a fresh sample reports ``imu=True``. A T-16
        checkpoint trained on gr00t episodes never saw a valid IMU group; reconciling that
        is :class:`wam.runtime.executor.PolicyContract`'s job, not this adapter's — see
        ``G1Adapter.read_state`` for why the flag must keep describing the sensor."""
        return self._adapter.read_state()

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        """Execute the first ``prefix_steps`` steps (FR-05), stepping ``control_dt_s`` of
        physics per step. JOINT_DELTA only; no-op while e-stopped. The chunk MUST already
        have passed the SafetyFilter."""
        self._adapter.execute(chunk, prefix_steps)

    def hold(self) -> None:
        """Re-command the current position with ``dq_target = 0``. Unlike ``MockRobot``
        this ADVANCES sim time by one ``control_dt_s`` — holding a real arm is physics."""
        self._adapter.hold()

    def estop(self) -> None:
        """Latched emergency stop: viscous damping in the sim + the adapter's latch.

        Safe to call at any time AND from any thread (``RobotAdapter.estop()``'s hard
        guarantee): the transport serialises every ``MjData`` access behind its lock, so an
        e-stop from a watchdog thread blocks behind the in-flight command instead of racing it
        into a segfault. The latch is always set, even if damping fails — but a failed damp
        RAISES rather than reporting success (see ``MujocoG1Transport.emergency_damp``).
        Released only by ``clear_estop()``. Advances sim time by ``damp_duration_s``.
        """
        self._adapter.estop()

    # -- rollout-harness extras ---------------------------------------------------------------

    def clear_estop(self) -> None:
        """Release the e-stop latch (deliberate operator action; mirrors
        ``MockRobot.clear_estop``). The transport's damping gains stay in force until the
        next motor command restores the configured gains."""
        self._adapter.clear_estop()

    def reset(self) -> None:
        """Reset the EPISODE: keyframe pose, pristine actuator gains, sim clock back to 0.

        A latched e-stop deliberately SURVIVES (use ``clear_estop``). Deterministic: the
        same scene + keyframe always yields bit-identical ``qpos``/``qvel``.
        """
        self._transport.reset()
        # G1Adapter caches the previous tick to detect stale samples, and reset() rewinds the
        # sim clock to 0. Without dropping that cache, a reset() taken before any motion
        # (tick 0 -> tick 0) would make the first post-reset read look stale. forget_tick() is
        # the adapter's public, single-owner hook for exactly this deliberate clock rewind.
        self._adapter.forget_tick()

    # -- cameras --------------------------------------------------------------------------------

    def render_frames(self, n: int) -> dict[str, np.ndarray]:
        """Render ``n`` frames per camera: ``{name: uint8 array [n, H, W, 3]}``.

        The frames within one call are IDENTICAL copies of the CURRENT sim state — rendering
        never steps physics (that would advance the tick behind the adapter's back and break
        the staleness contract). Call ``render_frames(1)`` between ``execute()`` calls to get
        distinct frames; that is exactly what ``ClosedLoopExecutor`` does.

        One ``mujoco.Renderer`` is built on first use and reused for every camera and call.
        Holds the transport's lock: ``update_scene`` reads ``MjData`` and must not run while
        another thread steps it.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        renderer = self._require_renderer()
        out: dict[str, np.ndarray] = {}
        with self._transport.lock:
            data = self._transport.data
            for cam in self._cameras:
                renderer.update_scene(data, camera=cam)
                frame = np.asarray(renderer.render(), dtype=np.uint8)
                out[cam] = np.repeat(frame[None, ...], n, axis=0)
        return out

    def close(self) -> None:
        """Release the offscreen renderer (idempotent). Physics needs no teardown."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _require_renderer(self) -> Any:
        """The single reused ``mujoco.Renderer``, built on first use (so constructing the
        robot never needs a GL context)."""
        if self._renderer is None:
            h, w = self._image_hw
            self._renderer = self._mj.Renderer(self._transport.model, height=h, width=w)
        return self._renderer

    # -- sim-time pacing seams handed to G1Adapter ------------------------------------------------

    def _sim_clock(self) -> float:
        """Sim seconds (the transport tick). Advances only when physics steps."""
        return self._transport.tick_ns * 1e-9

    def _sim_sleep(self, seconds: float) -> None:
        """Advance SIM time by ``seconds`` holding the last command, instead of blocking.

        Only reached when a chunk's ``dt_s`` exceeds ``control_dt_s``; the arm keeps tracking
        its last target (ctrl, gains and the feed-forward torque are untouched), which is what
        a real arm does while the next command is late. Holds the transport's lock — this is
        the one place outside the transport that steps physics.
        """
        if seconds <= 0.0:
            return
        with self._transport.lock:
            model = self._transport.model
            data = self._transport.data
            steps = round(seconds / float(model.opt.timestep))
            for _ in range(steps):
                self._mj.mj_step(model, data)
