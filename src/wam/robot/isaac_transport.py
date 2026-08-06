"""Isaac Sim behind the :class:`~wam.robot.g1_transport.G1Transport` seam (FR-06, E2).

    NOTHING IN THIS FILE HAS EVER RUN AGAINST ISAAC SIM.

    It was written on a Mac with no Isaac Sim and no GPU, against
    :class:`~wam.robot.isaac_binding.IsaacBinding` — which is itself written against NVIDIA's
    *documentation* for Isaac Sim 6.0.1. What IS executed, on that Mac, is every line below
    driven by :class:`~wam.robot.isaac_binding.FakeIsaacBinding` (``tests/test_isaac_g1.py``).
    So the CONTRACT is tested and the VENDOR HALF is not: ``scripts/preflight_isaac.py`` is
    the thing that proves the vendor half on the box, and it must pass before a single number
    out of an Isaac rollout is trusted.

This is the FOURTH implementation of the four-method ``G1Transport`` protocol, next to
``FakeG1Transport`` (unit tests), ``DdsG1Transport`` (real hardware) and ``MujocoG1Transport``
(the other simulator). Like those, it is NOT a second robot adapter: joint mapping, unit
conversion, defense-in-depth clipping, wall-/sim-clock pacing and the latched e-stop all stay
in :class:`~wam.robot.g1.G1Adapter`, which drives this transport unchanged.

WHY THIS IS TORCH-FREE, AND WHY THAT IS NOT NEGOTIABLE

  ``isaacsim-core==6.0.1.0`` hard-pins ``torch==2.11.0``; this repo's ``uv.lock`` resolves
  2.13.0. Isaac Sim and the WAM backbone therefore CANNOT share a virtualenv. The topology on
  the box is two interpreters: the Isaac venv runs
  ``rollout.py --robot isaac_g1 --policy remote --server-uri ws://...`` and the WAM venv runs
  ``scripts/serve_policy.py``. Everything on the Isaac side of that split — this module,
  ``isaac_binding.py`` and ``isaac_g1.py`` — imports numpy and nothing heavier.
  ``tests/test_isaac_g1.py`` proves it in a subprocess rather than trusting the import list.

CONTRACTS

- **The binding is INJECTED, never constructed here.** All ``isaacsim``/``omni``/``pxr``/
  ``warp`` imports live in ``isaac_binding.py`` and happen inside
  ``IsaacSimBinding.__init__``; this module never imports the vendor stack at all. Passing
  ``FakeIsaacBinding`` is what makes the tick contract, the e-stop latch, the gain round-trip
  and the 43-DoF name resolution testable on a laptop.
- **43 DoFs, resolved BY NAME, once, at construction.** ``binding.dof_indices`` (from
  :func:`~wam.robot.isaac_binding.resolve_g1_dof_indices`) maps the 29 body motors and the
  2 x 7 Dex3-1 finger joints onto the articulation's own PhysX ordering. This module does NOT
  re-implement that resolution and never indexes positionally: PhysX walks the articulation
  breadth-first from the base link, so its order is neither the URDF's nor
  :data:`~wam.robot.g1_transport.G1_MOTOR_JOINT_NAMES`', and ``G1Adapter`` gathers canonical
  joints out of the 29-slot motor array by HARD-CODED index — one permuted entry moves a
  physical arm silently. Construction additionally cross-checks the resolved slots against
  :data:`~wam.robot.g1.G1_JOINT_MAP`, exactly as ``MujocoG1Transport`` does.
- **``write_motor_cmd`` steps physics** by ``round(control_dt_s / binding.physics_dt)`` steps
  (default 10 x 2 ms at 500 Hz), validated to be an exact integer multiple.
  ``write_gripper_cmd`` does NOT step — the fingers move on the next motor command — so sim
  time advances 1:1 with motor commands, which is what the adapter's pacing assumes.
- **``tick_ns`` is the PHYSICS STEP COUNT, not a clock.**
  ``binding.get_physics_step_count()`` is an exact ``int`` that increments once per physics
  step and is not moved by a read, a gain write or a render;
  ``tick_ns = step_count * round(physics_dt * 1e9)``. ``G1Adapter.read_state`` decides
  staleness by EQUALITY against the previous tick, so a float clock — or one derived from sim
  time — would make that comparison meaningless. A non-integral tick is a ``TypeError`` here,
  not a cast.
- **The caller owns the gains.** The 29 body slots get the caller's ``kp``/``kd`` verbatim,
  including zeros. The 14 finger slots keep whatever gains the ASSET shipped, read back once
  at construction and never invented (see HAND GAINS below). This is also why the backend is
  raw Isaac Sim and NOT Isaac Lab: Isaac Lab's explicit actuator models (``DCMotorCfg``, used
  for the G1 legs in its shipped cfg) compute torque in Python and neutralise the sim's PD
  gains, and its G1 cfg is a legacy 23-DoF model besides.
- **Non-finite input is rejected at the seam**, before any transport state is mutated. A NaN
  reaching PhysX corrupts the articulation for the rest of the episode while the readbacks may
  stay finite — a failure nothing downstream can observe.
- **Reads are cheap and coherent.** ``read_low_state`` makes exactly two articulation reads
  (positions, velocities) and slices both; it never steps.

SIX DELIBERATE DIFFERENCES FROM THE OTHER TRANSPORTS — read these, they are not details

  1. **The e-stop is not at parity with hardware.** See E-STOP below. The short version: the
     Omniverse API is main-thread-only, so an e-stop arriving on a watchdog thread can only
     latch a flag in pure Python; a ``PHYSICS_PRE_STEP`` callback drains it on the main
     thread. ``DdsG1Transport.emergency_damp()`` puts damping on the DDS wire immediately and
     synchronously, independent of the control loop's health. This one cannot.
  2. **The IMU is a CONSTANT STAND-IN, not a measurement.** ``IsaacBinding`` exposes joint
     state only — no root pose, no sensors — and ``scripts/preflight_isaac.py`` checks no such
     symbol, so adding one here would be an unverified assumption smuggled past the preflight.
     ``read_low_state`` therefore reports identity orientation, zero angular velocity and
     ``(0, 0, 9.81)`` acceleration, and :attr:`imu_is_measured` is ``False``.
     ``MujocoG1Transport`` reports the torso body's real world orientation plus real gyro and
     accelerometer sensors; ``DdsG1Transport`` passes the robot's real IMU through.
     CONSEQUENCE, stated plainly: ``G1Adapter.read_state`` sets ``validity.imu`` from
     freshness alone, so an Isaac rollout reports ``imu=True`` over a payload that is not a
     measurement. Do not train on, evaluate against, or safety-gate on the IMU group from this
     backend. The fix is a root-pose getter on ``IsaacBinding`` plus a preflight check for the
     symbol it uses — in that order.
  3. **``dq_target`` is REFUSED, not silently dropped.** ``IsaacBinding`` has no velocity-target
     and no effort-write channel, so a non-zero velocity feed-forward cannot be honoured.
     ``MujocoG1Transport`` applies it as a ``kd * dq_target`` feed-forward torque on
     ``qfrc_applied``; ``DdsG1Transport`` puts it on the wire. Here a non-zero ``dq_target``
     raises ``ValueError``. ``G1Adapter`` always sends zeros (``execute`` and ``hold`` both
     do), so this never fires in the runtime — it exists so that a future caller who starts
     using the channel finds out at the seam instead of wondering why it does nothing.
  4. **After the e-stop latches, the GRIPPER is refused too.** ``DdsG1Transport`` damps only
     the 29 body motors on separate topics, so its hands stay commandable;
     ``MujocoG1Transport`` likewise leaves the 14 Dex3 actuators alone. Isaac's articulation
     has ONE write path — a full 43-vector — so letting a gripper command through would
     re-send the body position targets as a side effect. The fingers keep their gains and
     their last target (a grasped object is not dropped by the stop itself), but a NEW grip
     command is refused until :meth:`clear_estop`.
  5. **The Dex3 open/closed poses are CONSTANTS here, not read from the model.**
     ``MujocoG1Transport`` derives them from the scene's own ``jnt_range``; ``IsaacBinding``
     exposes no joint-limit getter and the preflight checks none, so :data:`DEX3_CLOSED_POSE`
     is a table (provenance in its own docstring). If the Isaac USD's finger limits differ
     from the Menagerie model's, a full-close command is clipped by PhysX and the measured
     closure reads back below 1.0 — visible, not silent, but it IS a discrepancy to check.
  6. **``emergency_damp()`` NEVER BLOCKS ON AN IN-FLIGHT COMMAND.** (It does block on the main
     thread, where it steps ``damp_settle_s`` of physics before returning — that is the
     synchronous settle, not lock contention.) ``MujocoG1Transport`` serialises everything behind
     one re-entrant lock, so an e-stop from a watchdog thread waits out the in-flight command
     instead of racing it. That trade is wrong here and buys nothing: the damp cannot be
     applied off the main thread ANYWAY, so blocking the watchdog for a control period would
     only delay the latch — and if the main thread wedged inside a command, the e-stop call
     itself would hang, which is the last thing a watchdog may do. The latch therefore lives
     under a plain :class:`threading.Lock` that is never held across a binding call, and
     ``emergency_damp()`` returns promptly from any thread. THE RESIDUAL WINDOW, stated
     exactly: a ``write_motor_cmd`` that had already passed the latch check when the e-stop
     arrived completes — one command, already in flight before ``estop()`` was called, at most
     one ``control_dt_s`` of motion. No command STARTED after the latch reaches the simulator.
     Nothing is serialised, because nothing needs to be: every array this class mutates is
     touched only on the main thread, and the two latch booleans are the only cross-thread
     state.

E-STOP: WHAT IT GUARANTEES AND WHAT IT DOES NOT

  ``emergency_damp()`` does two things, in this order, and the order is the whole design:

  (a) It LATCHES, in pure Python, under a plain :class:`threading.Lock`, touching nothing that
      belongs to Isaac. From that moment no ``write_motor_cmd`` and no ``write_gripper_cmd``
      reaches the simulator, whatever thread the call came from and whether or not the damp
      itself ever gets applied. This is the property ``G1Adapter.estop()`` depends on — it
      latches its own flag in a ``finally`` so ``execute()`` becomes a no-op — and this
      transport closes the remaining race: an e-stop landing between the adapter's per-step
      ``self._estopped`` check and the transport call would otherwise still write one command.
  (b) It DRAINS, i.e. writes ``kp = 0, kd = damp_kd`` on the 29 body slots, but only where
      that is legal:

      - **on the main thread**, synchronously, right away, and then steps ``damp_settle_s`` of
        physics so the arm comes to rest instead of hanging in its last pose. Errors PROPAGATE
        on this path, matching ``MujocoG1Transport``/``DdsG1Transport`` — a swallowed e-stop
        failure is indistinguishable from a successful stop at every layer above.
      - **from any other thread**, not at all: the flag stays pending and the
        ``PHYSICS_PRE_STEP`` callback registered at construction applies it on the main thread
        at the next physics step. Errors CANNOT propagate on this path (there is no caller to
        propagate to, and a Python exception escaping into Isaac's C++ event dispatcher is
        undocumented behaviour), so they are recorded on :attr:`last_damp_error` and the drain
        is retried on the next step. **A rollout harness that never inspects
        :attr:`last_damp_error` cannot tell a failed asynchronous damp from a successful one.**

  Two real costs, neither papered over:

  1. **A latency floor of one physics step** on the asynchronous path — up to one
     ``physics_dt`` (2 ms at 500 Hz), plus however long the current ``step(steps=N)`` batch
     has left to run, since the callback fires per step but the Python caller does not regain
     control until the batch ends.
  2. **No damp at all if the main loop is wedged.** If the main thread is blocked, deadlocked
     or simply not calling ``step()``, the flag is never drained and NOTHING happens in the
     simulator. On hardware the DDS write is independent of the control loop's health, which
     is the entire point of an e-stop. ``FakeIsaacBinding.wedge_main_thread()`` reproduces
     this exactly and ``tests/test_isaac_g1.py`` asserts the DOCUMENTED behaviour (the latch
     holds, the damp never lands), not the desired one.

  A third consequence, and the one that actually bites: after a latched e-stop nothing calls
  ``step()`` any more (``G1Adapter.execute``/``hold`` are no-ops while latched). So on the
  asynchronous path the arm settles only for whatever remained of the batch that was in
  flight — and if the request arrived BETWEEN batches, which is the common case, there is no
  remainder and the damping is never applied at all. ``damp_count == 1``,
  ``damp_applied_count == 0``, ``is_damping`` False, body kp untouched. ``clear_estop()``
  then discards the pending flag, so an operator resume closes the window for good. A frozen
  clock is a safe state in simulation. It is emphatically not one on a robot, and that
  asymmetry is the reason this file refuses to call itself an e-stop equivalent.

HAND GAINS ARE READ, NOT INVENTED

  ``set_dof_gains`` writes all 43 DoFs at once, so this transport has to have a number for the
  14 finger slots. It does not make one up: ``__init__`` reads the articulation's own gains
  back once and keeps the finger entries untouched forever after. That mirrors
  ``MujocoG1Transport`` ("the 14 Dex3 actuators keep the vendor gains"). If the shipped USD
  ships zero finger gains the hands will not move — visible immediately, and
  :attr:`hand_gains` reports it; ``hand_gains=(kp, kd)`` overrides.

TUNING CONSTANTS THAT ARE NOT MEASURED ON THIS BACKEND

  ``damp_kd=20.0`` and ``damp_settle_s=0.2`` are carried over from ``MujocoG1Transport``,
  where they WERE measured (see its ``emergency_damp`` docstring for the settle table). PhysX
  is a different solver with different effective inertias, so on Isaac they are STARTING
  POINTS, not measurements. Re-run that measurement on the box — drive a joint to a known
  ``|dq|``, call ``emergency_damp()``, record the residual velocity — before quoting a stop
  distance anywhere.

Torch-free; numpy only.
"""

from __future__ import annotations

import numbers
import threading
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from wam.robot.g1 import G1_JOINT_MAP
from wam.robot.g1_transport import (
    DEX3_FINGER_JOINTS,
    G1_MOTOR_JOINT_NAMES,
    G1_NUM_MOTORS,
    _as_motor_array,
)
from wam.robot.isaac_binding import EXPECTED_NUM_DOFS, IsaacBinding

__all__ = [
    "DEX3_CLOSED_POSE",
    "DEX3_OPEN_POSE",
    "ISAAC_IMU_STANDIN",
    "IsaacG1Transport",
]

_HAND_SIDES: tuple[str, str] = ("left", "right")

#: Fully-open Dex3-1 pose, in :data:`~wam.robot.g1_transport.DEX3_FINGER_JOINTS` order. Zero
#: for all seven joints — the same open pose ``configs/sim/g1_scene.xml`` uses for its
#: ``ready`` keyframe.
DEX3_OPEN_POSE: tuple[float, ...] = (0.0,) * len(DEX3_FINGER_JOINTS)

#: Fully-closed Dex3-1 pose per hand, in :data:`DEX3_FINGER_JOINTS` order.
#:
#: PROVENANCE, because these are numbers and numbers need one: they are the joint-range
#: endpoints farther from zero, read off ``assets/mujoco/unitree_g1/g1_with_hands.xml``
#: (MuJoCo Menagerie ``unitree_g1``, revision ``g1_29dof_with_hand_rev_1_0``) — the same
#: kinematic model the Isaac asset ``{assets_root}/Isaac/Robots/Unitree/G1/g1.usd`` is a
#: conversion of, and the same rule ``MujocoG1Transport._finger_synergy_poses`` applies at
#: runtime. ``thumb_0`` is the thumb's opposition ROLL and is held at 0 in BOTH poses: measured
#: on that model at full curl, the thumb tip sits 37.2 mm from both the middle and the index
#: tip at ``thumb_0 = 0`` and moves away in either direction. The left/right sign mirroring is
#: the model's, not a convention imposed here.
#:
#: UNVERIFIED AGAINST THE ISAAC USD. ``IsaacBinding`` exposes no joint-limit getter and
#: ``scripts/preflight_isaac.py`` checks none, so if the converted asset's finger limits
#: differ, PhysX clips the target and the measured closure reads back below 1.0 — detectable
#: from a rollout, but nothing here will say so.
DEX3_CLOSED_POSE: Mapping[str, tuple[float, ...]] = {
    "left": (0.0, 1.0472, 1.74533, -1.5708, -1.74533, -1.5708, -1.74533),
    "right": (0.0, -1.0472, -1.74533, 1.5708, 1.74533, 1.5708, 1.74533),
}

#: The IMU sample ``read_low_state`` reports. A CONSTANT STAND-IN, not a measurement — see
#: difference (2) in the module docstring. Identical to ``FakeG1Transport``'s default, on
#: purpose: a test that passes against the fake transport and against this one is testing the
#: adapter, not the sensor, which is exactly what it should be doing.
ISAAC_IMU_STANDIN: Mapping[str, tuple[float, ...]] = {
    "quat_wxyz": (1.0, 0.0, 0.0, 0.0),
    "gyro": (0.0, 0.0, 0.0),
    "acc": (0.0, 0.0, 9.81),
}


class IsaacG1Transport:
    """``G1Transport`` on top of an injected :class:`IsaacBinding`.

    See the module docstring for every contract, and especially for the six documented
    differences from ``MujocoG1Transport``/``DdsG1Transport``. Typical use::

        binding = IsaacSimBinding()            # or FakeIsaacBinding() in a test
        transport = IsaacG1Transport(binding)
        adapter = G1Adapter(config, transport)
        adapter.connect()
        adapter.execute(chunk, prefix_steps=5)  # steps 5 x control_dt_s of physics
    """

    def __init__(
        self,
        binding: IsaacBinding,
        *,
        control_dt_s: float = 0.02,
        damp_kd: float = 20.0,
        damp_settle_s: float = 0.2,
        grasp_closure: float = 1.0,
        hand_gains: tuple[float, float] | None = None,
        finger_closed: Mapping[str, Sequence[float]] | None = None,
        imu: Mapping[str, Sequence[float]] | None = None,
        expected_num_dofs: int = EXPECTED_NUM_DOFS,
    ) -> None:
        """Resolve every DoF index once, read the asset's gains back and arm the e-stop drain.

        ``binding``: any :class:`IsaacBinding` — :class:`IsaacSimBinding` on the box,
        :class:`FakeIsaacBinding` in tests. Constructed by the CALLER, never here.
        ``control_dt_s``: simulated time advanced per ``write_motor_cmd``; must be an exact
        integer multiple of ``binding.physics_dt``. ``damp_kd`` / ``damp_settle_s``: viscous
        gain and settle time used by :meth:`emergency_damp` — carried over from the MuJoCo
        transport and NOT measured on PhysX (module docstring). ``grasp_closure``: fraction of
        each curling finger joint's travel that ``write_gripper_cmd(1.0)`` commands.
        ``hand_gains``: ``(kp, kd)`` for all 14 finger slots; ``None`` keeps the ASSET's own
        gains, which is the honest default. ``finger_closed``: per-side closed pose override,
        each 7 values in :data:`DEX3_FINGER_JOINTS` order; ``None`` uses
        :data:`DEX3_CLOSED_POSE`. ``imu``: the constant stand-in sample (see
        :data:`ISAAC_IMU_STANDIN`); this backend cannot measure one. ``expected_num_dofs``:
        asserted against the articulation, 43 for the G1 with Dex3-1 hands.

        Raises ``ValueError`` for a bad control period, a bad closure/gain argument, a DoF
        count that is not ``expected_num_dofs``, or an articulation whose joint names cannot be
        resolved (the message names the joint and dumps the articulation's own names);
        ``RuntimeError`` when the resolved slots disagree with ``G1_JOINT_MAP``.
        """
        self._binding = binding

        num_dofs = int(binding.num_dofs)
        if num_dofs != int(expected_num_dofs):
            raise ValueError(
                f"articulation has {num_dofs} DOFs, expected {int(expected_num_dofs)} "
                f"(29 body + 2 x 7 Dex3 fingers). This is not g1_29dof_with_hand_rev_1_0 — "
                f"Isaac Lab's G1 cfg is a legacy 23-DoF model, do not point at "
                f"ISAACLAB_NUCLEUS_DIR.\nDOF names: {list(binding.dof_names)}"
            )
        self._num_dofs = num_dofs

        # ONE place turns names into indices, and it is not this one: resolve_g1_dof_indices
        # lives in the binding and is shared with the real backend. Re-implementing it here
        # would be a second copy of a 43-entry ordering, i.e. a second chance to permute it.
        self._dof = binding.dof_indices
        self._body_idx = self._dof.body_array()
        self._finger_idx = [self._dof.hand_array(side) for side in _HAND_SIDES]
        self._verify_motor_convention()

        # -- control period -> physics steps -------------------------------------------------
        physics_dt = float(binding.physics_dt)
        if physics_dt <= 0.0:
            raise ValueError(f"binding.physics_dt must be > 0, got {physics_dt}")
        if control_dt_s <= 0.0:
            raise ValueError(f"control_dt_s must be > 0, got {control_dt_s}")
        substeps = round(control_dt_s / physics_dt)
        if substeps < 1 or abs(substeps * physics_dt - control_dt_s) > 1e-12:
            raise ValueError(
                f"control_dt_s={control_dt_s} is not an integer multiple of the binding's "
                f"physics_dt {physics_dt}"
            )
        self._physics_dt = physics_dt
        # Integer ns per physics step. tick_ns is a MULTIPLE of the step count, never a float
        # clock: G1Adapter compares ticks for equality and a float would make that meaningless.
        self._physics_dt_ns = round(physics_dt * 1e9)
        if self._physics_dt_ns <= 0:
            raise ValueError(
                f"physics_dt {physics_dt} rounds to 0 ns per step — tick_ns would never advance"
            )
        self._control_dt_s = float(control_dt_s)
        self._substeps = int(substeps)

        # -- grasp synergy --------------------------------------------------------------------
        if not 0.0 < grasp_closure <= 1.0:
            raise ValueError(f"grasp_closure must be in (0, 1], got {grasp_closure}")
        self._grasp_closure = float(grasp_closure)
        closed_in = DEX3_CLOSED_POSE if finger_closed is None else finger_closed
        self._finger_open: list[np.ndarray] = []
        self._finger_closed: list[np.ndarray] = []
        for side in _HAND_SIDES:
            if side not in closed_in:
                raise ValueError(f"finger_closed: missing the {side!r} hand")
            closed = np.asarray(closed_in[side], dtype=np.float64)
            if closed.shape != (len(DEX3_FINGER_JOINTS),):
                raise ValueError(
                    f"finger_closed[{side!r}]: expected {len(DEX3_FINGER_JOINTS)} values in "
                    f"DEX3_FINGER_JOINTS order, got shape {closed.shape}"
                )
            if not np.isfinite(closed).all():
                raise ValueError(f"finger_closed[{side!r}]: non-finite values")
            self._finger_open.append(np.asarray(DEX3_OPEN_POSE, dtype=np.float64))
            self._finger_closed.append(closed * self._grasp_closure)
        # Projection direction per hand: <q - open, d> / <d, d> inverts the synergy.
        self._finger_dir = [c - o for o, c in zip(self._finger_open, self._finger_closed)]
        self._finger_dir_sq = [float(d @ d) for d in self._finger_dir]

        # -- damping ----------------------------------------------------------------------------
        if damp_kd < 0.0:
            raise ValueError(f"damp_kd must be >= 0, got {damp_kd}")
        if damp_settle_s < 0.0:
            raise ValueError(f"damp_settle_s must be >= 0, got {damp_settle_s}")
        self._damp_kd = float(damp_kd)
        self._settle_steps = round(damp_settle_s / physics_dt)

        # -- IMU stand-in ------------------------------------------------------------------------
        imu_in = ISAAC_IMU_STANDIN if imu is None else imu
        self._imu = {
            key: np.asarray(imu_in.get(key, ISAAC_IMU_STANDIN[key]), dtype=np.float32)
            for key in ISAAC_IMU_STANDIN
        }

        # -- gains: the asset's own, read back once ------------------------------------------------
        kp0, kd0 = binding.get_dof_gains()
        self._kp0 = np.asarray(kp0, dtype=np.float64).copy()
        self._kd0 = np.asarray(kd0, dtype=np.float64).copy()
        if self._kp0.shape != (num_dofs,) or self._kd0.shape != (num_dofs,):
            raise RuntimeError(
                f"binding.get_dof_gains() returned {self._kp0.shape}/{self._kd0.shape}, "
                f"expected ({num_dofs},) each"
            )
        if hand_gains is not None:
            hand_kp, hand_kd = (float(v) for v in hand_gains)
            if hand_kp < 0.0 or hand_kd < 0.0:
                raise ValueError(f"hand_gains must be >= 0, got {hand_gains!r}")
            for idx in self._finger_idx:
                self._kp0[idx] = hand_kp
                self._kd0[idx] = hand_kd
        self._kp_all = self._kp0.copy()
        self._kd_all = self._kd0.copy()
        self._targets_all = np.asarray(binding.get_dof_positions(), dtype=np.float64).copy()
        self._damping = False

        # -- e-stop latch -------------------------------------------------------------------------
        #: Plain lock, plain bools: emergency_damp() may arrive on ANY thread and must not
        #: touch a single Isaac object there (the Omniverse API is main-thread-only).
        self._latch_lock = threading.Lock()
        self._estop_latched = False
        self._pending_damp = False
        #: Number of emergency_damp() REQUESTS (mirrors FakeG1Transport/MujocoG1Transport).
        self.damp_count: int = 0
        #: Number of times the damping gains actually reached the simulator. Lags damp_count
        #: on the asynchronous path, and commonly stays at 0 there: a latched e-stop is what
        #: stops the control loop calling step(), so unless the request lands inside a batch
        #: already in flight there is no physics step left to drain it on.
        self.damp_applied_count: int = 0
        #: Exception from the last drain attempt, if any. On the synchronous (main-thread)
        #: path it is ALSO re-raised; on the callback path it cannot be, so this attribute is
        #: the only signal an asynchronous damp failed. Harnesses must check it.
        self.last_damp_error: Exception | None = None
        #: Commands refused because the latch was set (diagnostics; both must stay 0 in a
        #: rollout that never e-stopped).
        self.blocked_motor_writes: int = 0
        self.blocked_gripper_writes: int = 0

        # The drain point. Registered LAST so a construction failure above never leaves a
        # callback pointing at a half-built transport.
        binding.register_pre_physics_callback(self._on_pre_physics)

    # -- construction-time verification ----------------------------------------------------------

    def _verify_motor_convention(self) -> None:
        """Fail loudly unless the resolved body slots match ``G1_JOINT_MAP``.

        ``G1Adapter`` gathers canonical joints out of the 29-slot motor array by hard-coded
        index. If ``G1_MOTOR_JOINT_NAMES`` and ``G1_JOINT_MAP`` ever drift apart, that gather
        addresses the wrong motor and the only symptom is a physical arm in the wrong place.
        Same check ``MujocoG1Transport`` runs, for the same reason.
        """
        if len(self._body_idx) != G1_NUM_MOTORS:
            raise RuntimeError(
                f"resolved {len(self._body_idx)} body DOFs, expected {G1_NUM_MOTORS}"
            )
        for canonical_name, motor_idx in G1_JOINT_MAP:
            if G1_MOTOR_JOINT_NAMES[motor_idx] != canonical_name:
                raise RuntimeError(
                    f"G1_JOINT_MAP disagrees with the motor convention: slot {motor_idx} is "
                    f"{G1_MOTOR_JOINT_NAMES[motor_idx]!r}, adapter expects {canonical_name!r}"
                )

    # -- introspection ---------------------------------------------------------------------------

    @property
    def binding(self) -> IsaacBinding:
        """The injected binding (rendering and lifecycle live upstream, in ``isaac_g1``)."""
        return self._binding

    @property
    def control_dt_s(self) -> float:
        """Simulated time advanced by one ``write_motor_cmd``."""
        return self._control_dt_s

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def substeps(self) -> int:
        """Physics steps per ``write_motor_cmd`` (``control_dt_s / physics_dt``)."""
        return self._substeps

    @property
    def step_count(self) -> int:
        """The binding's raw physics-step counter, validated to be an exact integer."""
        raw = self._binding.get_physics_step_count()
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
            raise TypeError(
                f"get_physics_step_count() returned {type(raw).__name__} ({raw!r}); the tick "
                "must be an integer counter — G1Adapter decides staleness by EQUALITY against "
                "the previous one, and a float clock makes that comparison meaningless"
            )
        return int(raw)

    @property
    def tick_ns(self) -> int:
        """Physics step count in ns. Advances iff physics stepped — never on a read, a gain
        write or a render. This IS the staleness signal ``G1Adapter`` consumes."""
        return self.step_count * self._physics_dt_ns

    @property
    def is_estopped(self) -> bool:
        """True once :meth:`emergency_damp` has latched, until :meth:`clear_estop`."""
        with self._latch_lock:
            return self._estop_latched

    @property
    def pending_damp(self) -> bool:
        """True while a latched damp has not yet reached the simulator (asynchronous path)."""
        with self._latch_lock:
            return self._pending_damp

    @property
    def is_damping(self) -> bool:
        """True while the body slots carry ``kp = 0, kd = damp_kd``."""
        return self._damping

    @property
    def imu_is_measured(self) -> bool:
        """Always ``False``: this backend reports a CONSTANT IMU stand-in (module docstring)."""
        return False

    @property
    def hand_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """``(kp, kd)`` per finger slot, 14 entries each (left hand then right), as currently
        written to the articulation. These come from the ASSET unless ``hand_gains=`` was
        passed; all-zero means the USD ships no finger gains and the hands will not move."""
        idx = np.concatenate(self._finger_idx)
        return self._kp_all[idx].astype(np.float32), self._kd_all[idx].astype(np.float32)

    def finger_synergy(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """``(open, closed)`` finger poses [7] in rad for ``side``, in
        :data:`DEX3_FINGER_JOINTS` order. Copies."""
        if side not in _HAND_SIDES:
            raise ValueError(f"side must be one of {_HAND_SIDES}, got {side!r}")
        i = _HAND_SIDES.index(side)
        return self._finger_open[i].copy(), self._finger_closed[i].copy()

    # -- lifecycle ---------------------------------------------------------------------------------

    def reset(self) -> None:
        """Episode reset: restore the articulation's default state and the asset's own gains.

        Mirrors ``MujocoG1Transport.reset()``: the pose goes back, the pristine gains go back
        (so a new episode never inherits a damping override), and the position targets are
        re-based on the MEASURED post-reset pose so the first command afterwards does not yank.

        Does NOT rewind the tick — ``get_physics_step_count`` is a raw counter on both bindings
        — and does NOT clear a latched e-stop. The caller owes ``G1Adapter.forget_tick()`` and
        ``forget_command()`` after this; :class:`~wam.robot.isaac_g1.IsaacG1Robot.reset` does
        both.
        """
        self._binding.reset()
        self._targets_all = np.asarray(self._binding.get_dof_positions(), dtype=np.float64).copy()
        self._kp_all = self._kp0.copy()
        self._kd_all = self._kd0.copy()
        self._damping = False
        self._binding.set_dof_gains(
            self._kp_all.astype(np.float32), self._kd_all.astype(np.float32)
        )
        self._binding.set_dof_position_targets(self._targets_all.astype(np.float32))

    def advance(self, seconds: float) -> None:
        """Step physics by ``seconds`` WITHOUT issuing a command (sim-time sleep).

        The arm keeps tracking its last target, which is what a real arm does while the next
        command is late. Used by :class:`~wam.robot.isaac_g1.IsaacG1Robot` as the injected
        ``sleep`` for ``G1Adapter``'s pacing. Steps even while e-stopped — it commands nothing,
        and stepping is exactly how a pending asynchronous damp gets drained.
        """
        if seconds <= 0.0:
            return
        steps = round(seconds / self._physics_dt)
        if steps > 0:
            self._binding.step(steps)

    def clear_estop(self) -> None:
        """Release the transport's latch (deliberate operator action).

        The damping gains STAY in force until the next ``write_motor_cmd`` restores the
        caller's — same as ``MujocoG1Transport``. This does NOT release ``G1Adapter``'s own
        latch; :meth:`~wam.robot.isaac_g1.IsaacG1Robot.clear_estop` releases both, and both
        have to be released or the arm stays silently uncommandable.
        """
        with self._latch_lock:
            self._estop_latched = False
            self._pending_damp = False

    # -- G1Transport ---------------------------------------------------------------------------------

    def read_low_state(self) -> dict[str, Any]:
        """One low-state sample in the documented dict contract.

        Pure read — never steps, so ``tick_ns`` is unchanged since the last ``write_motor_cmd``
        and two consecutive reads make the second one STALE, which is what clears the adapter's
        validity flags. Allowed while e-stopped: reading does not move anything.

        ``gripper`` is the MEASURED closure per hand in [0, 1] (the inverse of the grasp
        synergy), so a blocked finger reads back below its command. ``imu`` is a CONSTANT
        STAND-IN — this backend cannot measure one; see difference (2) in the module docstring
        before using it for anything.
        """
        q_all = np.asarray(self._binding.get_dof_positions(), dtype=np.float32)
        dq_all = np.asarray(self._binding.get_dof_velocities(), dtype=np.float32)
        for name, arr in (("positions", q_all), ("velocities", dq_all)):
            if arr.shape != (self._num_dofs,):
                raise RuntimeError(
                    f"binding returned {arr.shape} for dof {name}, expected "
                    f"({self._num_dofs},)"
                )
        return {
            "q": q_all[self._body_idx].copy(),
            "dq": dq_all[self._body_idx].copy(),
            "imu": {key: value.copy() for key, value in self._imu.items()},
            "gripper": self._measure_gripper(q_all),
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

        The caller's gains go into the 29 body slots verbatim (``kp = kd = 0`` gives exactly
        zero actuator force, which is what ``G1Adapter`` sends for unmapped motors); the 14
        finger slots are untouched. ``q_target`` is clamped by PhysX to each joint's limits.

        Validation happens BEFORE any state is mutated, so a rejected command leaves the
        transport exactly as it found it. ``dq_target`` must be all zeros — see difference (3)
        in the module docstring; Isaac has no velocity-target channel in this binding and
        silently dropping a feed-forward would be worse than refusing it.

        NO-OP AFTER THE E-STOP LATCH, counted on :attr:`blocked_motor_writes`. That is the
        guarantee ``G1Adapter.estop()`` rests on, held here rather than only in the adapter,
        because an e-stop from a watchdog thread can land between the adapter's per-step check
        and this call. The latch is read under the lock and released again immediately: this
        method must never make ``emergency_damp()`` wait (difference (6) in the module
        docstring), so a command already past this check does finish.
        """
        q = _as_motor_array("q_target", q_target)
        dq = _as_motor_array("dq_target", dq_target)
        kp_a = _as_motor_array("kp", kp)
        kd_a = _as_motor_array("kd", kd)
        for name, arr in (("q_target", q), ("dq_target", dq), ("kp", kp_a), ("kd", kd_a)):
            if not np.isfinite(arr).all():
                raise ValueError(f"{name}: non-finite values would corrupt the simulation")
        if (kp_a < 0.0).any() or (kd_a < 0.0).any():
            # Prefixed with the method name on purpose: IsaacBinding rejects negative gains
            # too, one layer lower and AFTER this transport has already mutated its shadow
            # state. The prefix is how a caller (and a test) can tell which guard fired.
            raise ValueError(
                "write_motor_cmd: kp/kd entries must be >= 0 (negative gains are unstable)"
            )
        if dq.any():
            raise ValueError(
                "dq_target must be all zeros for the Isaac backend: IsaacBinding exposes no "
                "velocity-target and no effort-write channel, so a velocity feed-forward "
                "cannot be honoured and will not be silently dropped (see the module "
                "docstring). G1Adapter always sends zeros."
            )

        with self._latch_lock:
            if self._estop_latched:
                self.blocked_motor_writes += 1
                return

        self._kp_all[self._body_idx] = kp_a
        self._kd_all[self._body_idx] = kd_a
        self._targets_all[self._body_idx] = q
        self._damping = False
        self._binding.set_dof_gains(
            self._kp_all.astype(np.float32), self._kd_all.astype(np.float32)
        )
        self._binding.set_dof_position_targets(self._targets_all.astype(np.float32))
        self._binding.step(self._substeps)

    def write_gripper_cmd(self, left: float, right: float) -> None:
        """Set both hands' 7 finger targets from one scalar each (0 = open, 1 = closed).

        Vendor units are [0, 1] (matching ``G1Config``'s default gripper range) and are
        clipped; non-finite input is refused. Does NOT step physics — the fingers move on the
        next ``write_motor_cmd``, which keeps sim time advancing 1:1 with motor commands.

        NO-OP AFTER THE E-STOP LATCH, counted on :attr:`blocked_gripper_writes` — see
        difference (4) in the module docstring. The hands keep their gains and their last
        target, so a grasped object is not dropped by the stop itself.
        """
        for name, value in (("left", left), ("right", right)):
            if not np.isfinite(value):
                raise ValueError(f"{name}: non-finite values would corrupt the simulation")
        with self._latch_lock:
            if self._estop_latched:
                self.blocked_gripper_writes += 1
                return
        for i, value in enumerate((left, right)):
            g = float(np.clip(value, 0.0, 1.0))
            self._targets_all[self._finger_idx[i]] = (
                self._finger_open[i] + g * self._finger_dir[i]
            )
        self._binding.set_dof_position_targets(self._targets_all.astype(np.float32))

    def emergency_damp(self) -> None:
        """Latch the e-stop, then apply damping where and when that is legal.

        SAFE FROM ANY THREAD, and that is the whole reason this method is shaped the way it is:
        the latch is pure Python under a plain lock and touches nothing belonging to Isaac. The
        Omniverse API is main-thread-only, so

        - called ON the main thread, this drains synchronously (``kp = 0, kd = damp_kd`` on the
          29 body slots) and then steps ``damp_settle_s`` of physics so the arm comes to rest.
          A failure PROPAGATES, matching ``MujocoG1Transport``/``DdsG1Transport``;
          ``G1Adapter.estop()`` latches in a ``finally``, so propagating can never leave the
          adapter willing to command motion.
        - called from any other thread, this ONLY latches and returns. The registered
          ``PHYSICS_PRE_STEP`` callback applies the damping on the main thread at the next
          physics step. A failure there cannot propagate to any caller; it is recorded on
          :attr:`last_damp_error` and retried on the following step.

        Either way, from the instant the latch is set, no ``write_motor_cmd`` and no
        ``write_gripper_cmd`` STARTED afterwards reaches the simulator until
        :meth:`clear_estop`. This never blocks on an in-flight command — see difference (6) in
        the module docstring for why, and for the one-command window that leaves.

        NOT AT PARITY WITH HARDWARE. ``DdsG1Transport.emergency_damp()`` puts damping on the
        DDS wire immediately, synchronously and independently of the control loop's health.
        This one has a latency floor of one physics step on the asynchronous path, and does
        NOTHING AT ALL if the main loop is wedged — the flag simply stays pending. See the
        module docstring; ``tests/test_isaac_g1.py`` asserts that documented behaviour rather
        than the behaviour anyone would prefer.
        """
        with self._latch_lock:
            self.damp_count += 1
            self._estop_latched = True
            self._pending_damp = True
        if threading.current_thread() is threading.main_thread():
            self._drain(synchronous=True)

    # -- internals ------------------------------------------------------------------------------------

    def _on_pre_physics(self) -> None:
        """``PHYSICS_PRE_STEP`` subscriber: drain a pending damp on the main thread.

        MUST NOT RAISE. On the real binding this runs inside Isaac's C++ event dispatcher,
        where the behaviour of a propagating Python exception is undocumented — so the belt
        (``_drain(synchronous=False)`` swallows) gets braces here.
        """
        try:
            self._drain(synchronous=False)
        except Exception as exc:  # noqa: BLE001 - a callback that raises into C++ is worse
            self.last_damp_error = exc

    def _drain(self, *, synchronous: bool) -> None:
        """Apply the latched damping. Main thread only; one implementation, two call sites."""
        with self._latch_lock:
            if not self._pending_damp:
                return
            # Cleared BEFORE the settle steps below, which fire this very callback: a flag
            # still marked pending would re-enter the drain and step recursively.
            self._pending_damp = False
        # Cleared per ATTEMPT, matching MujocoG1Transport.emergency_damp(). This attribute is
        # documented (module docstring, and emergency_damp's) as the ONLY way a harness can
        # tell a failed asynchronous damp from a successful one — so it has to describe the
        # last attempt, not the worst one ever seen. Leaving it sticky meant one transient
        # failure poisoned it for the rest of the process: the retry on the next physics step
        # would succeed, damp_applied_count would climb, and a harness following the
        # instruction to check last_damp_error would still read a stale exception forever.
        self.last_damp_error = None
        try:
            self._kp_all[self._body_idx] = 0.0
            self._kd_all[self._body_idx] = self._damp_kd
            self._binding.set_dof_gains(
                self._kp_all.astype(np.float32), self._kd_all.astype(np.float32)
            )
            # Set only after the write landed, so is_damping never claims a damping the
            # simulator never received.
            self._damping = True
            self.damp_applied_count += 1
            if synchronous and self._settle_steps:
                self._binding.step(self._settle_steps)
        except Exception as exc:
            # Recorded AND retried: a damp is idempotent, so the next physics step is a free
            # second chance. Only Exception, never BaseException — KeyboardInterrupt and
            # SystemExit must not be intercepted by an e-stop.
            self.last_damp_error = exc
            with self._latch_lock:
                self._pending_damp = True
            if synchronous:
                raise

    def _measure_gripper(self, q_all: np.ndarray) -> np.ndarray:
        """Measured closure per hand in [0, 1]: the 7 finger angles projected onto the
        open->closed line, ``<q - open, d> / <d, d>``, clipped. Exactly inverts
        ``write_gripper_cmd`` when the fingers track their targets."""
        out = np.zeros(2, dtype=np.float32)
        for i in range(2):
            denom = self._finger_dir_sq[i]
            if denom <= 0.0:  # pragma: no cover - only for a synergy with no travel at all
                continue
            q = np.asarray(q_all[self._finger_idx[i]], dtype=np.float64)
            projection = float((q - self._finger_open[i]) @ self._finger_dir[i]) / denom
            out[i] = np.clip(projection, 0.0, 1.0)
        return out
