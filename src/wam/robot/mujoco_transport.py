"""MuJoCo-backed :class:`~wam.robot.g1_transport.G1Transport` (T-21, FR-06, E2).

This is the THIRD implementation of the ``G1Transport`` protocol, next to ``DdsG1Transport``
(real hardware) and ``FakeG1Transport`` (unit tests). It is deliberately NOT a second robot
adapter: joint mapping, unit conversion, defense-in-depth clipping and the latched e-stop all
stay in :class:`~wam.robot.g1.G1Adapter`, which drives this transport unchanged. Swapping
``FakeG1Transport`` for ``MujocoG1Transport`` turns the same closed loop from a kinematic
integrator into real contact physics and real rendered pixels (E2 on the evaluation ladder).

Contracts:

- **``mujoco`` is an OPTIONAL dependency.** This module imports without it; the package is
  imported lazily inside ``__init__`` and a missing install raises ``RuntimeError`` naming the
  ``uv pip install mujoco`` fix. ``wam.robot`` / ``wam.robot.registry`` never import this module
  at package import time.
- **Scene.** ``configs/sim/g1_scene.xml`` by default (resolved against the repo root, so the
  transport works from any cwd). Fixed base, locked legs + waist roll/pitch, one table, one
  cube, two cameras. The scene file documents its own simplifications.
- **Motor ordering is resolved BY NAME, once, at construction** into the 29-slot
  ``G1JointIndex`` convention (legs 0-11, waist yaw/roll/pitch 12-14, left arm 15-21, right arm
  22-28). Construction fails loudly if any of the 29 joints or 43 actuators is missing, and
  cross-checks the result against :data:`wam.robot.g1.G1_JOINT_MAP` — the scene's qpos layout
  is never assumed, only verified. The Dex3 ACTUATOR order differs between the hands (the right
  hand lists index before middle) while the JOINT order does not; every finger is therefore
  resolved by name too.
- **``write_motor_cmd`` steps physics** by exactly ``control_dt_s``
  (``round(control_dt_s / timestep)`` substeps, default 10 x 2 ms). ``write_gripper_cmd`` does
  NOT step (see below), so sim time advances 1:1 with motor commands — which is what the
  adapter's wall-clock pacing assumes.
- **Gains are honoured, never overridden.** The scene's 43 actuators are MuJoCo ``position``
  actuators (``ctrl`` is a desired POSITION). ``write_motor_cmd`` writes the caller's per-motor
  gains straight into the actuator parameters of the 29 body motors
  (``gainprm[0] = kp``, ``biasprm[1] = -kp``, ``biasprm[2] = -kd``), giving
  ``tau = kp * (ctrl - q) - kd * dq``. ``kp = kd = 0`` (what ``G1Adapter`` sends for unmapped
  motors) yields exactly zero actuator force — no NaN, no explosion. The 14 Dex3 actuators keep
  the vendor gains (kp=500, dampratio=1); the canonical gripper channel is a scalar, not a gain
  surface.
- **``dq_target`` is applied as a feed-forward torque** ``kd * dq_target`` on ``qfrc_applied``,
  because a position actuator's affine bias can only express the ``dq = 0`` target. In practice
  ``G1Adapter`` always sends ``dq_target = 0``, so this term is normally zero; it is implemented
  so the PD contract is honest rather than silently truncated. ``qfrc_applied`` is rewritten on
  every command and cleared by ``reset()`` / ``emergency_damp()``.
- **``tick_ns`` is derived from ``data.time``**, rounded onto the exact substep grid. It
  advances iff physics stepped, i.e. by exactly ``control_dt_s`` per ``write_motor_cmd``, by
  ``damp_duration_s`` (10 control periods at the defaults) per ``emergency_damp()``, and not at
  all for a bare ``read_low_state`` or ``write_gripper_cmd``. That is precisely the staleness
  signal ``G1Adapter`` turns into cleared validity flags. An e-stop therefore injects a
  ``damp_duration_s`` discontinuity into any log that timestamps on sim time, and the arm
  creeps ~0.04 rad under gravity DURING that interval — ``emergency_damp()`` brings the arm to
  rest, it does not freeze it.
- **Non-finite input is rejected at the seam.** ``write_motor_cmd`` AND ``write_gripper_cmd``
  raise ``ValueError`` on NaN/Inf rather than letting it reach ``data.ctrl``: MuJoCo zeroes ALL
  43 ``ctrl`` entries on a step with any non-finite control, which silently slams the whole
  robot to its zero pose while ``read_low_state`` keeps returning finite values and an advancing
  tick. That failure is undetectable downstream, so it must not be reachable.
- **THREAD SAFETY.** ``MjModel``/``MjData`` are not thread-safe and stepping them from two
  threads segfaults the process. Every method that touches them — ``read_low_state``,
  ``write_motor_cmd``, ``write_gripper_cmd``, ``emergency_damp``, ``reset`` — holds
  :attr:`lock` (a re-entrant lock). ``RobotAdapter.estop()`` promises to be safe "at any time,
  from any thread", so a watchdog/e-stop thread calling ``emergency_damp()`` BLOCKS behind the
  in-flight command instead of racing it. Anything outside this class that touches ``.model`` /
  ``.data`` (rendering, ad-hoc stepping) must hold :attr:`lock` too —
  :class:`~wam.robot.mujoco_g1.MujocoG1Robot` does.
- **``imu``**: ``quat_wxyz`` is the TORSO BODY's world orientation (``data.xquat``) — the scene
  has gyro and accelerometer sensors but NO orientation sensor. The ``imu_in_torso`` site is
  aligned with ``torso_link`` (identity relative quat), so body orientation and sensor frame
  agree. ``gyro`` / ``acc`` come from the named torso sensors.
- **``gripper``**: MEASURED closure per hand in vendor units [0, 1], not the last command — the
  inverse of the grasp synergy, obtained by projecting the 7 measured finger angles onto the
  open->closed line. A blocked finger therefore reads back below its command, which is the
  honest signal.
- **Determinism.** No Python-level RNG. Two transports built from the same scene, reset to the
  same keyframe and driven with the same command sequence produce bit-identical ``qpos``
  (verified: max abs diff 0.0 over 200 control steps).

GRIPPER SYNERGY (OD-01: canonical gripper stays scalar, per-finger control is post-MVP)

  The Dex3-1 has 7 joints per hand and no aperture axis, so ``write_gripper_cmd`` interpolates
  one scalar along a fixed open->closed line derived FROM THE JOINT RANGES:
  open = 0 rad for all seven (which is also the scene's ``ready`` pose), closed = the range
  endpoint farther from zero for the six curling joints (``thumb_1``, ``thumb_2``, ``middle_0``,
  ``middle_1``, ``index_0``, ``index_1``). The signs are mirrored between the hands and this
  rule picks up the mirroring automatically. ``thumb_0`` is the thumb's opposition ROLL and is
  held at 0 in both poses: measured at full curl, the thumb tip sits 37.2 mm from BOTH the
  middle and index tips at ``thumb_0 = 0`` and moves away in either direction (44.6-69.1 mm at
  +-1.047 rad), so zero is the opposing pose. It contributes no travel and therefore does not
  enter the measured-closure projection either.

  ``write_gripper_cmd`` only writes the finger ``ctrl`` targets; it does NOT step physics. The
  fingers move on the next ``write_motor_cmd``. This keeps sim time advancing 1:1 with motor
  commands — ``G1Adapter.execute()`` issues one motor command and one gripper command per step,
  and stepping in both would double the simulated period behind the adapter's back. A gripper
  command with no following motor command has no effect.

SIM GAINS — MEASURE THEM AGAINST THE PROTOCOL THE ADAPTER ACTUALLY USES

  ``G1Config``'s defaults (kp=20, kd=0.5 per canonical joint) are conservative HARDWARE
  placeholders pending OD-08. They are far too soft for this model, which carries the arm's real
  inertia with no gravity compensation, and the arm visibly sags.

  There are TWO protocols and they give very different-looking numbers. Both are reported here,
  because quoting only the first one flatters the gains:

  (A) FIXED ABSOLUTE TARGET, held. Steady-state |error| in rad after a 1.5 s hold of one fixed
      target, all 15 canonical joints commanded, a -0.5 rad step on ``left_elbow`` (free space —
      see the contact caveat below); "mean"/"max" are over the 15 canonical joints:

          kp=20,  kd=0.5           elbow 0.1711  mean 0.0468  max 0.1874  (34% of the step)
          kp=100, kd=5             elbow 0.0323  mean 0.0144  max 0.0436
          kp=200, kd=10            elbow 0.0160  mean 0.0073  max 0.0223
          kp=300, kd=15            elbow 0.0107  mean 0.0049  max 0.0150
          kp=500, critical damping elbow 0.0064  mean 0.0030  max 0.0091   <- SIM_KP / SIM_KD

      Holding ``ready`` against gravity for 5 s with a FIXED target: kp=300/kd=15 -> max droop
      0.01502 rad; kp=500/critdamp -> 0.00909 rad. BOUNDED, because the target never moves.

  (B) THROUGH ``G1Adapter.execute()``, which is the only protocol the runtime ever uses. The
      adapter re-bases its target on the MEASURED ``q`` at every call, so whatever the position
      loop has not caught up with by the end of a chunk is DISCARDED, not carried forward.
      Metric: the fraction of one control period's commanded position step that is actually
      executed within that period (``prefix_steps=1``, one joint at a time, free space,
      0.004 rad/step, 60 steps):

          kp=300, kd=15 (flat)         min -0.20  mean 0.14  max 0.30
          kp=500, critical damping     min -0.11  mean 0.39  max 0.97   <- SIM_KP / SIM_KD

      and the same total travel executed with different ``prefix_steps`` (``left_shoulder_yaw``,
      0.400 rad commanded as 100 x 0.004, identical 2 s of sim in every row):

          prefix_steps     1      2      5     10     25    100
          kp=300/kd=15   0.162  0.269  0.477  0.675  0.895  0.975
          SIM_KP/SIM_KD  0.309  0.439  0.666  0.851  0.947  0.987

  WHY (B) NEVER REACHES 1.0, AT ANY GAIN. Executing ~all of a 20 ms position step inside 20 ms
  needs a closed-loop bandwidth of several hundred rad/s. The scene's own effective inertias
  (recovered from the vendor ``dampratio="1"``, see ``mujoco_g1.scene_critical_damping``) are
  0.011 kg m^2 at the wrist roll and 0.398 kg m^2 at the waist, so wn = sqrt(kp/m) puts the
  waist at 35 rad/s for the vendor kp=500 and would need kp ~ 1.6e4 Nm/rad to reach 200 rad/s.
  That is not a robot arm. Measured: even flat kp=4000 with critical damping only reaches
  mean 0.86 / min 0.56. The under-execution is therefore a property of the CONTROL ARCHITECTURE
  (re-base on measurement, no feed-forward, no integral action) on a plant with finite servo
  bandwidth — not a MuJoCo artifact and not a gain to tune away.

  FIXED BY T-25c, ARCHITECTURALLY, since that is the only level it could be fixed at. The
  adapter now carries the previous commanded target into the next call, clamped to
  ``G1Config.q_track_window`` of the measured ``q`` (``mujoco_g1.SIM_Q_TRACK_WINDOW`` = 0.05 rad
  here). Every row of the ``prefix_steps`` table above becomes 0.987. The tables are kept as
  measured because they are still exactly what happens at ``q_track_window = 0``, which is what
  ``configs/robot/g1.yaml`` ships pending OD-08 — so on the hardware config the sentence below
  still stands: an achieved (state, action) pair recorded there is NOT the commanded action, the
  safety-intervention rate is not calibrated, and both move when ``prefix_steps`` moves.

  THE RATCHET (same root cause, hold edition; same T-25c fix). At ``q_track_window = 0`` a
  zero-delta chunk is NOT a position hold: the adapter re-reads ``q``, so each cycle forgives the
  gravity droop accumulated in the previous one and the arm creeps monotonically. Max
  |q - keyframe| over the 15 joints, through ``execute()``: 0.080 rad @ 2 s, 0.329 @ 10 s,
  0.730 @ 30 s, 0.971 @ 60 s (SIM_KP/SIM_KD; kp=300/kd=15 gives 0.087 / 0.268 / 0.636 / 0.957 —
  the same order, still growing at 60 s). Protocol (A)'s bounded 0.009 rad droop does NOT
  describe that, and must never be quoted as the hold accuracy of a zero-window config.

  With the window at SIM_Q_TRACK_WINDOW the carried target holds the pose and the same
  measurement is 0.0091 rad, FLAT at 2 / 4 / 6 / 8 / 10 s — i.e. it converges onto protocol
  (A)'s bounded droop, which is the point: the runtime protocol now achieves what the fixed
  target always could.

  RECOMMENDED SIM GAINS: ``mujoco_g1.SIM_KP`` (500, the vendor Menagerie ``g1`` class stiffness
  — the value the model was authored for) with ``mujoco_g1.SIM_KD``, the per-joint CRITICAL
  damping ``2 * sqrt(kp * m_eff)`` that the vendor's own ``dampratio="1"`` compiles to. A FLAT
  kd is the wrong shape: the wrist roll's effective inertia is 1/36 of the waist's, so one
  number is either heavily overdamped at the wrist or underdamped at the waist. They live in
  ``configs/robot/mujoco_g1.yaml`` — NOT in ``configs/robot/g1.yaml``, whose gains must stay the
  conservative hardware placeholders. This transport deliberately does not clamp or substitute
  gains: honouring the caller's gains is the contract, and a hidden override would make a sim
  result unreproducible on hardware.

  CONTACT CAVEAT — a stiffer arm presses harder. ``left/right_wrist_pitch_joint`` carries a
  vendor ``actuatorfrcrange +-5 Nm`` while its position actuator inherits kp=500, so it
  saturates as soon as the hand touches anything. The mirrored ``left_elbow +0.5`` step drives
  the open hand into ``table_top``; the wrist then sits 0.14 rad off command at kp=20/kd=0.5 and
  0.54 rad off at kp=300/kd=15 (the demanded torque is 163 Nm, clipped to 5). That is real
  physics plus a vendor-model quirk, not a gain to tune away — treat commanded and achieved
  wrist pitch as different quantities whenever the hand is in contact.

Torch-free; numpy only (``mujoco`` itself is numpy-based).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from wam.robot.g1 import G1_JOINT_MAP
from wam.robot.g1_transport import _as_motor_array

MUJOCO_MISSING_MSG = (
    "MuJoCo simulation support requires the 'mujoco' package — install it with "
    "`uv pip install mujoco` from the repo root"
)

#: Repo-root-relative default scene (see ``configs/sim/g1_scene.xml``).
DEFAULT_SCENE_PATH = Path("configs/sim/g1_scene.xml")

#: The vendor model the default scene ``<include>``s. Fetched, never committed — its absence is
#: checked ONLY to turn MuJoCo's raw "XML Error: Error opening file" into an actionable message.
VENDOR_MODEL = Path("assets/mujoco/unitree_g1/g1_with_hands.xml")
VENDOR_MODEL_MISSING_MSG = (
    "the vendor G1 model is missing — fetch it with `.venv/bin/python scripts/fetch_g1_model.py`"
)

#: Keyframe used by ``reset()`` unless overridden — the scene's manipulation-ready pose.
DEFAULT_KEYFRAME = "ready"

#: MJCF joint/actuator names are the canonical joint name plus this suffix.
MJCF_JOINT_SUFFIX = "_joint"

#: Canonical joint name per G1 motor slot, in the 29-DoF ``G1JointIndex`` convention:
#: legs 0-11, waist yaw/roll/pitch 12-14, left arm 15-21, right arm 22-28.
G1_MOTOR_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)

#: Dex3-1 finger joints per hand, in the model's JOINT order (the ACTUATOR order differs
#: between the hands — always resolve by name).
DEX3_FINGER_JOINTS: tuple[str, ...] = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)

#: The six finger joints that curl during a Dex3 grasp. ``thumb_0`` (opposition roll) is held at
#: 0 in both the open and the closed pose — see the module docstring for the measurement.
DEX3_CLOSING_JOINTS: frozenset[str] = frozenset(
    {"thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"}
)

_HAND_SIDES: tuple[str, str] = ("left", "right")

#: Body whose world orientation stands in for the missing IMU orientation sensor.
IMU_BODY = "torso_link"
IMU_GYRO_SENSOR = "imu-torso-angular-velocity"
IMU_ACC_SENSOR = "imu-torso-linear-acceleration"


def _repo_root() -> Path:
    """Repo root: ``src/wam/robot/mujoco_transport.py`` -> up four levels."""
    return Path(__file__).resolve().parents[3]


def _require_mujoco() -> Any:
    """Import ``mujoco`` lazily; raise a RuntimeError naming the fix when it is absent."""
    try:
        import mujoco as _mujoco
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency
        raise RuntimeError(MUJOCO_MISSING_MSG) from exc
    return _mujoco


class MujocoG1Transport:
    """MuJoCo physics behind the ``G1Transport`` seam (see the module docstring for contracts).

    Typical use::

        transport = MujocoG1Transport()
        adapter = G1Adapter(G1Config(kp=(300.0,) * 15, kd=(15.0,) * 15), transport)
        adapter.connect()
        adapter.execute(chunk, prefix_steps=5)   # steps physics 5 x control_dt_s
        frames = renderer.render(transport.model, transport.data)   # rendering lives upstream
    """

    def __init__(
        self,
        scene_path: str | Path | None = None,
        *,
        control_dt_s: float = 0.02,
        keyframe: str | None = DEFAULT_KEYFRAME,
        camera_names: tuple[str, ...] = ("head", "wrist_left"),
        render_hw: tuple[int, int] = (256, 256),
        damp_kd: float = 20.0,
        damp_duration_s: float = 0.2,
        grasp_closure: float = 1.0,
    ) -> None:
        """Load the scene and resolve every name-based index once.

        ``scene_path``: MJCF file; relative paths resolve against the repo root
        (default ``configs/sim/g1_scene.xml``). ``control_dt_s``: simulated time advanced per
        ``write_motor_cmd``; must be an integer multiple of the model timestep.
        ``keyframe``: keyframe NAME restored by ``reset()`` (``None`` = the model's qpos0).
        ``camera_names``: cameras the wrapper is going to render; validated here so a renamed
        camera fails at construction, not mid-rollout. ``render_hw``: (H, W) the wrapper should
        render at — stored, not used (this class owns no GL context).
        ``damp_kd`` / ``damp_duration_s``: viscous gain and simulated settle time used by
        ``emergency_damp()``; the default 20.0 is SIM-TUNED (measured below), deliberately
        higher than ``DdsG1Transport``'s vendor-typical 2.0. ``grasp_closure``: fraction of each
        curling finger joint's travel that ``write_gripper_cmd(1.0)`` commands (1.0 = the joint
        limit).

        Raises ``RuntimeError`` without ``mujoco``; ``ValueError`` for a bad control period,
        an unknown keyframe/camera, or a scene missing any of the 29 body joints, 43 actuators,
        14 finger joints or the torso IMU sensors.
        """
        mj = _require_mujoco()
        self._mj = mj

        path = Path(scene_path) if scene_path is not None else DEFAULT_SCENE_PATH
        if not path.is_absolute():
            path = _repo_root() / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"scene_path: no such MJCF file: {path}")
        self._scene_path = path
        #: Re-entrant lock guarding every MjModel/MjData access (see the module docstring).
        #: Public so a renderer or an ad-hoc stepper outside this class can hold it too.
        self.lock = threading.RLock()
        try:
            self._model = mj.MjModel.from_xml_path(str(path))
        except ValueError as exc:
            # The scene <include>s the fetched vendor model; MuJoCo reports only a bare
            # "XML Error: Error opening file" for a missing include.
            if not (_repo_root() / VENDOR_MODEL).is_file():
                raise ValueError(f"{exc}\n{VENDOR_MODEL_MISSING_MSG}") from exc
            raise
        self._data = mj.MjData(self._model)

        # -- control period -> substeps ----------------------------------------------------
        timestep = float(self._model.opt.timestep)
        if control_dt_s <= 0.0:
            raise ValueError(f"control_dt_s must be > 0, got {control_dt_s}")
        substeps = round(control_dt_s / timestep)
        if substeps < 1 or abs(substeps * timestep - control_dt_s) > 1e-12:
            raise ValueError(
                f"control_dt_s={control_dt_s} is not an integer multiple of the model "
                f"timestep {timestep}"
            )
        self._control_dt_s = float(control_dt_s)
        self._substeps = substeps
        self._timestep = timestep
        self._timestep_ns = round(timestep * 1e9)

        # -- 29 body motors, resolved BY NAME ----------------------------------------------
        self._motor_joint_ids = self._resolve_ids(
            mj.mjtObj.mjOBJ_JOINT,
            [name + MJCF_JOINT_SUFFIX for name in G1_MOTOR_JOINT_NAMES],
            "joint",
        )
        self._motor_act_ids = self._resolve_ids(
            mj.mjtObj.mjOBJ_ACTUATOR,
            [name + MJCF_JOINT_SUFFIX for name in G1_MOTOR_JOINT_NAMES],
            "actuator",
        )
        self._motor_qpos_adr = self._model.jnt_qposadr[self._motor_joint_ids].astype(np.int64)
        self._motor_dof_adr = self._model.jnt_dofadr[self._motor_joint_ids].astype(np.int64)
        self._verify_motor_convention()

        # -- Dex3 fingers, resolved BY NAME (actuator order differs between hands) ----------
        if not 0.0 < grasp_closure <= 1.0:
            raise ValueError(f"grasp_closure must be in (0, 1], got {grasp_closure}")
        self._grasp_closure = float(grasp_closure)
        self._finger_joint_ids: list[np.ndarray] = []
        self._finger_act_ids: list[np.ndarray] = []
        self._finger_qpos_adr: list[np.ndarray] = []
        self._finger_open: list[np.ndarray] = []
        self._finger_closed: list[np.ndarray] = []
        for side in _HAND_SIDES:
            names = [f"{side}_hand_{joint}{MJCF_JOINT_SUFFIX}" for joint in DEX3_FINGER_JOINTS]
            jnt_ids = self._resolve_ids(mj.mjtObj.mjOBJ_JOINT, names, "joint")
            act_ids = self._resolve_ids(mj.mjtObj.mjOBJ_ACTUATOR, names, "actuator")
            self._finger_joint_ids.append(jnt_ids)
            self._finger_act_ids.append(act_ids)
            self._finger_qpos_adr.append(self._model.jnt_qposadr[jnt_ids].astype(np.int64))
            open_pose, closed_pose = self._finger_synergy_poses(jnt_ids)
            self._finger_open.append(open_pose)
            self._finger_closed.append(closed_pose)
        # Projection direction per hand: <q - open, d> / <d, d> inverts the synergy.
        self._finger_dir = [c - o for o, c in zip(self._finger_open, self._finger_closed)]
        self._finger_dir_sq = [float(d @ d) for d in self._finger_dir]

        # -- IMU stand-ins ------------------------------------------------------------------
        self._imu_body_id = self._resolve_one(mj.mjtObj.mjOBJ_BODY, IMU_BODY, "body")
        self._gyro_slice = self._sensor_slice(IMU_GYRO_SENSOR)
        self._acc_slice = self._sensor_slice(IMU_ACC_SENSOR)

        # -- keyframe / cameras --------------------------------------------------------------
        self._keyframe_name = keyframe
        if keyframe is None:
            self._keyframe_id = -1
        else:
            self._keyframe_id = int(self._resolve_one(mj.mjtObj.mjOBJ_KEY, keyframe, "keyframe"))
        for cam in camera_names:
            self._resolve_one(mj.mjtObj.mjOBJ_CAMERA, cam, "camera")
        self._camera_names = tuple(camera_names)
        if len(render_hw) != 2 or any(int(v) < 1 for v in render_hw):
            raise ValueError(f"render_hw must be two positive ints, got {render_hw!r}")
        self._render_hw = (int(render_hw[0]), int(render_hw[1]))

        # -- emergency damping ----------------------------------------------------------------
        if damp_kd < 0.0:
            raise ValueError(f"damp_kd must be >= 0, got {damp_kd}")
        if damp_duration_s < 0.0:
            raise ValueError(f"damp_duration_s must be >= 0, got {damp_duration_s}")
        self._damp_kd = float(damp_kd)
        self._damp_steps = round(damp_duration_s / timestep)
        #: Number of ``emergency_damp()`` calls (diagnostics; mirrors FakeG1Transport).
        self.damp_count: int = 0
        #: Exception raised by the last ``emergency_damp()``, if any. Recorded for diagnostics
        #: AND re-raised — a swallowed e-stop failure is invisible to every layer above.
        self.last_damp_error: Exception | None = None

        # Pristine actuator parameters, restored by reset() so a fresh episode never inherits
        # the previous episode's (or a damping) gain override.
        self._gainprm0 = self._model.actuator_gainprm[:, 0].copy()
        self._biasprm12 = self._model.actuator_biasprm[:, 1:3].copy()

        self.reset()

    # -- name resolution helpers -------------------------------------------------------------

    def _resolve_one(self, obj_type: Any, name: str, kind: str) -> int:
        idx = self._mj.mj_name2id(self._model, obj_type, name)
        if idx < 0:
            raise ValueError(f"scene {self._scene_path.name}: no {kind} named {name!r}")
        return int(idx)

    def _resolve_ids(self, obj_type: Any, names: list[str], kind: str) -> np.ndarray:
        missing = [n for n in names if self._mj.mj_name2id(self._model, obj_type, n) < 0]
        if missing:
            raise ValueError(
                f"scene {self._scene_path.name}: missing {len(missing)} {kind}(s): "
                f"{missing} — this scene does not model the 29-DoF G1 with Dex3-1 hands"
            )
        return np.array(
            [self._mj.mj_name2id(self._model, obj_type, n) for n in names], dtype=np.int64
        )

    def _sensor_slice(self, name: str) -> slice:
        sid = self._resolve_one(self._mj.mjtObj.mjOBJ_SENSOR, name, "sensor")
        adr = int(self._model.sensor_adr[sid])
        return slice(adr, adr + int(self._model.sensor_dim[sid]))

    def _verify_motor_convention(self) -> None:
        """Fail loudly unless the resolved motor slots match ``G1_JOINT_MAP``.

        ``G1Adapter`` gathers canonical joints out of the 29-slot array by hard-coded index;
        if the scene ever renames or reorders a joint, that gather would silently address the
        wrong motor. Checking the resolved names here turns that into a construction error.
        """
        resolved = [
            self._mj.mj_id2name(self._model, self._mj.mjtObj.mjOBJ_JOINT, int(jid))
            for jid in self._motor_joint_ids
        ]
        expected = [name + MJCF_JOINT_SUFFIX for name in G1_MOTOR_JOINT_NAMES]
        if resolved != expected:  # pragma: no cover - _resolve_ids already guarantees this
            raise ValueError(f"motor name resolution mismatch: {resolved} != {expected}")
        for canonical_name, motor_idx in G1_JOINT_MAP:
            if G1_MOTOR_JOINT_NAMES[motor_idx] != canonical_name:
                raise ValueError(
                    f"G1_JOINT_MAP disagrees with the motor convention: slot {motor_idx} is "
                    f"{G1_MOTOR_JOINT_NAMES[motor_idx]!r}, adapter expects {canonical_name!r}"
                )

    def _finger_synergy_poses(self, jnt_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Derive (open, closed) finger poses for one hand from the model's joint ranges.

        open = 0 rad everywhere (the scene's ``ready`` pose). closed = ``grasp_closure`` x the
        range endpoint farther from zero, for the six curling joints; ``thumb_0`` stays at 0.
        The left/right sign mirroring falls out of the ranges — nothing is hard-coded per hand.
        """
        open_pose = np.zeros(len(DEX3_FINGER_JOINTS), dtype=np.float64)
        closed_pose = np.zeros(len(DEX3_FINGER_JOINTS), dtype=np.float64)
        for i, (joint, jid) in enumerate(zip(DEX3_FINGER_JOINTS, jnt_ids)):
            if joint not in DEX3_CLOSING_JOINTS:
                continue
            lo, hi = (float(v) for v in self._model.jnt_range[int(jid)])
            closed_pose[i] = (hi if abs(hi) >= abs(lo) else lo) * self._grasp_closure
        return open_pose, closed_pose

    # -- introspection --------------------------------------------------------------------

    @property
    def model(self) -> Any:
        """Live ``mujoco.MjModel`` handle (for rendering upstream). Read-only property; the
        object itself is mutable — ``write_motor_cmd`` rewrites the 29 body actuators' gains.
        Hold :attr:`lock` while touching it from anywhere but the owning thread."""
        return self._model

    @property
    def data(self) -> Any:
        """Live ``mujoco.MjData`` handle (for rendering upstream). Read-only property.
        Hold :attr:`lock` while touching it from anywhere but the owning thread."""
        return self._data

    @property
    def control_dt_s(self) -> float:
        """Simulated time advanced by one ``write_motor_cmd``."""
        return self._control_dt_s

    @property
    def substeps(self) -> int:
        """Physics substeps per ``write_motor_cmd`` (``control_dt_s / timestep``)."""
        return self._substeps

    @property
    def scene_path(self) -> Path:
        return self._scene_path

    @property
    def keyframe(self) -> str | None:
        """Keyframe name restored by ``reset()`` (``None`` = the model's qpos0)."""
        return self._keyframe_name

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Cameras validated at construction, for the wrapper that owns the renderer."""
        return self._camera_names

    @property
    def render_hw(self) -> tuple[int, int]:
        """(H, W) the wrapper should render at. Stored only — this class owns no GL context."""
        return self._render_hw

    @property
    def motor_joint_names(self) -> tuple[str, ...]:
        """MJCF joint name per motor slot 0..28 (``G1JointIndex`` order)."""
        return tuple(name + MJCF_JOINT_SUFFIX for name in G1_MOTOR_JOINT_NAMES)

    @property
    def tick_ns(self) -> int:
        """Sim time on the exact substep grid, in ns. Advances iff physics stepped."""
        return round(self._data.time / self._timestep) * self._timestep_ns

    def finger_synergy(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """(open, closed) finger poses [7] in rad for ``side`` in {'left', 'right'}, in
        :data:`DEX3_FINGER_JOINTS` order. Copies."""
        idx = _HAND_SIDES.index(side)
        return self._finger_open[idx].copy(), self._finger_closed[idx].copy()

    # -- lifecycle ---------------------------------------------------------------------------

    def reset(self) -> None:
        """Restore the keyframe pose, the pristine actuator gains and a zero tick.

        Deterministic: the same scene + keyframe always yields bit-identical ``qpos``/``qvel``.
        Also clears the ``dq_target`` feed-forward and any damping gain override left by
        ``emergency_damp()``. Holds :attr:`lock`.
        """
        with self.lock:
            if self._keyframe_id >= 0:
                self._mj.mj_resetDataKeyframe(self._model, self._data, self._keyframe_id)
            else:
                self._mj.mj_resetData(self._model, self._data)
            self._model.actuator_gainprm[:, 0] = self._gainprm0
            self._model.actuator_biasprm[:, 1:3] = self._biasprm12
            self._data.qfrc_applied[:] = 0.0
            self._mj.mj_forward(self._model, self._data)

    # -- G1Transport --------------------------------------------------------------------------

    def read_low_state(self) -> dict[str, Any]:
        """One low-state sample in the documented dict contract.

        Pure read — never steps, so ``tick_ns`` is unchanged since the last ``write_motor_cmd``.
        ``gripper`` is the MEASURED closure per hand in [0, 1] (the inverse of the grasp
        synergy), so a blocked finger reads back below its command. Holds :attr:`lock`, so the
        sample is a coherent snapshot rather than a tear across a concurrent step.
        """
        with self.lock:
            d = self._data
            return {
                "q": d.qpos[self._motor_qpos_adr].astype(np.float32),
                "dq": d.qvel[self._motor_dof_adr].astype(np.float32),
                "imu": {
                    "quat_wxyz": np.asarray(d.xquat[self._imu_body_id], dtype=np.float32).copy(),
                    "gyro": np.asarray(d.sensordata[self._gyro_slice], dtype=np.float32).copy(),
                    "acc": np.asarray(d.sensordata[self._acc_slice], dtype=np.float32).copy(),
                },
                "gripper": self._measure_gripper(),
                "tick_ns": self.tick_ns,
            }

    def write_motor_cmd(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        """Apply one 29-motor PD command and step physics by ``control_dt_s``.

        The caller's gains are written into the position actuators verbatim (see the module
        docstring — the default ``G1Config`` gains are too soft for this model and the arm will
        sag; that is a config problem, not something this transport hides). ``kp = kd = 0``
        gives exactly zero actuator force. ``q_target`` is clamped by MuJoCo to each actuator's
        ``ctrlrange`` (= the joint range). Locked/welded joints simply will not move.
        Holds :attr:`lock` for the whole command + step.
        """
        q = _as_motor_array("q_target", q_target)
        dq = _as_motor_array("dq_target", dq_target)
        kp_a = _as_motor_array("kp", kp)
        kd_a = _as_motor_array("kd", kd)
        for name, arr in (("q_target", q), ("dq_target", dq), ("kp", kp_a), ("kd", kd_a)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}: non-finite values would destroy the simulation")
        if (kp_a < 0.0).any() or (kd_a < 0.0).any():
            raise ValueError("kp/kd entries must be >= 0 (negative gains are unstable)")

        with self.lock:
            model, data, act = self._model, self._data, self._motor_act_ids
            # position actuator: tau = gainprm[0]*ctrl + biasprm[1]*q + biasprm[2]*dq
            model.actuator_gainprm[act, 0] = kp_a
            model.actuator_biasprm[act, 1] = -kp_a
            model.actuator_biasprm[act, 2] = -kd_a
            data.ctrl[act] = q
            # Feed-forward for the dq target the affine bias cannot express (normally zero).
            data.qfrc_applied[self._motor_dof_adr] = kd_a * dq
            for _ in range(self._substeps):
                self._mj.mj_step(model, data)

    def write_gripper_cmd(self, left: float, right: float) -> None:
        """Set both hands' 7 finger targets from one scalar each (0 = open, 1 = closed).

        Vendor units are [0, 1] (matching ``G1Config``'s default gripper range) and are clipped.
        NON-FINITE INPUT IS REJECTED, exactly as in ``write_motor_cmd``: ``np.clip(nan, 0, 1)``
        is ``nan``, and one NaN anywhere in ``data.ctrl`` makes MuJoCo zero ALL 43 control
        entries on every subsequent step — the whole robot slams to its zero pose in 0.2 s with
        no exception, finite ``read_low_state`` output and a normally advancing tick, i.e. a
        failure nothing downstream can observe.

        Does NOT step physics — the fingers move on the next ``write_motor_cmd``; see the module
        docstring for why. Holds :attr:`lock`.
        """
        for name, value in (("left", left), ("right", right)):
            if not np.isfinite(value):
                raise ValueError(f"{name}: non-finite values would destroy the simulation")
        with self.lock:
            for idx, value in enumerate((left, right)):
                g = float(np.clip(value, 0.0, 1.0))
                target = self._finger_open[idx] + g * self._finger_dir[idx]
                self._data.ctrl[self._finger_act_ids[idx]] = target

    def emergency_damp(self) -> None:
        """Safe stop: drop the 29 body motors to pure viscous damping and let them settle.

        Sets ``kp = 0`` and ``kd = damp_kd`` on the body actuators (``tau = -damp_kd * dq``),
        clears the feed-forward torque and steps ``damp_duration_s`` of physics so the arm
        actually comes to rest instead of holding a pose. The damping gains STAY in force until
        the next ``write_motor_cmd`` — which, with ``G1Adapter``'s latched e-stop, means until
        an operator clears the latch. The 14 Dex3 actuators keep their vendor gains, mirroring
        ``DdsG1Transport.emergency_damp()`` (which also commands only the 29 body motors), so a
        held object is not dropped by the stop itself.

        Pure damping cannot reach exactly zero velocity — with kp=0 the arm keeps creeping down
        under gravity at the terminal rate ``tau_gravity / damp_kd``, exactly like a real vendor
        damp mode. Measured from a moving arm (peak |dq| 2.55 rad/s), after one 0.2 s call:
        damp_kd=2 -> 1.53, 5 -> 0.61, **20 -> 0.198 (default)**, 50 -> 0.082, 100 -> 0.041
        rad/s; further calls plateau (20 -> 0.145 rad/s), which IS the gravity creep.

        Advances ``tick_ns`` by ``damp_duration_s`` (the damping IS physics), and the arm creeps
        ~0.04 rad under gravity during that interval.

        A FAILED DAMP PROPAGATES. The exception is recorded on ``last_damp_error`` for
        diagnostics and then re-raised, matching ``FakeG1Transport``/``DdsG1Transport`` and the
        contract in ``G1Adapter.estop()`` ("the latch is set even if the transport raises while
        damping — the exception still propagates"). ``G1Adapter.estop()`` latches in a
        ``finally``, so propagating can never leave the adapter willing to command motion; a
        silently swallowed failure, by contrast, is indistinguishable from a successful stop at
        every layer above this seam — which is the opposite of what an e-stop needs. Only
        ``Exception`` is caught, never ``BaseException``: ``KeyboardInterrupt``/``SystemExit``
        must not be intercepted here. Holds :attr:`lock`, so an e-stop raised from a watchdog
        thread blocks behind an in-flight ``write_motor_cmd`` instead of racing it.
        """
        with self.lock:
            self.damp_count += 1
            self.last_damp_error = None
            try:
                model, data, act = self._model, self._data, self._motor_act_ids
                model.actuator_gainprm[act, 0] = 0.0
                model.actuator_biasprm[act, 1] = 0.0
                model.actuator_biasprm[act, 2] = -self._damp_kd
                data.qfrc_applied[self._motor_dof_adr] = 0.0
                for _ in range(self._damp_steps):
                    self._mj.mj_step(model, data)
            except Exception as exc:
                self.last_damp_error = exc
                raise

    # -- internals ------------------------------------------------------------------------------

    def _measure_gripper(self) -> np.ndarray:
        """Measured closure per hand in [0, 1]: the 7 finger angles projected onto the
        open->closed line, ``<q - open, d> / <d, d>``, clipped. Exactly inverts
        ``write_gripper_cmd`` when the fingers track their targets."""
        out = np.zeros(2, dtype=np.float32)
        for idx in range(2):
            q = self._data.qpos[self._finger_qpos_adr[idx]]
            denom = self._finger_dir_sq[idx]
            if denom <= 0.0:  # pragma: no cover - only if a scene gives the fingers no travel
                continue
            projection = float((q - self._finger_open[idx]) @ self._finger_dir[idx]) / denom
            out[idx] = np.clip(projection, 0.0, 1.0)
        return out
