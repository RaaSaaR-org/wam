"""Isaac-Sim-backed G1 robot for the WAM runtime (FR-06, E2).

Thin wrapper that composes the REAL :class:`~wam.robot.g1.G1Adapter` with an
:class:`~wam.robot.isaac_transport.IsaacG1Transport`, so the full hardware code path — joint
mapping, unit conversion, defense-in-depth clipping, sim-clock pacing and the latched e-stop —
runs on PhysX contact physics and RTX-rendered pixels instead of a kinematic mock. Everything
a rollout harness needs beyond the ``RobotAdapter`` protocol (cameras, episode reset, sim
clock) lives here; nothing robot-specific is added.

    NOTHING IN THIS FILE HAS EVER RUN AGAINST ISAAC SIM. It was written on a Mac with no
    Isaac Sim and no GPU. What IS executed here, on that Mac, is every line below driven by
    :class:`~wam.robot.isaac_binding.FakeIsaacBinding` (``tests/test_isaac_g1.py``).
    ``scripts/preflight_isaac.py`` is what proves the vendor half on the box, and it must pass
    FIRST.

COMPOSITION, NOT SUBCLASSING (deliberate, and the same call ``MujocoG1Robot`` made)
  ``IsaacG1Robot`` HOLDS a ``G1Adapter`` and forwards the four protocol methods verbatim.
  Subclassing was rejected: a subclass would inherit ``connect()``'s ``DdsG1Transport``
  fallback (a sim robot that can silently try to open a DDS socket is a hazard, not a
  convenience), would be free to override the very safety code this class exists to exercise
  unchanged, and would tie the sim wrapper to the adapter's constructor signature.
  ``RobotAdapter`` is structural (``@runtime_checkable``), so the wrapper satisfies it without
  inheriting anything.

Contracts:

- **Isaac Sim is not importable from this repo's venv, and that is by design.**
  ``isaacsim-core==6.0.1.0`` pins ``torch==2.11.0`` while ``uv.lock`` resolves 2.13.0, so the
  box runs TWO interpreters: the Isaac venv runs
  ``rollout.py --robot isaac_g1 --policy remote --server-uri ws://...`` and the WAM venv runs
  ``scripts/serve_policy.py``. This module, ``isaac_transport.py`` and ``isaac_binding.py``
  are torch-free so they can live in the Isaac venv; ``tests/test_isaac_g1.py`` proves that in
  a subprocess. Constructing this class in the WAM venv raises ``RuntimeError`` naming that
  topology (:data:`~wam.robot.isaac_binding.ISAAC_MISSING_MSG`), not a bare ImportError.
- **Constructed connected.** ``__init__`` boots the binding, builds the transport and calls
  ``G1Adapter.connect()`` with it injected — no SDK, no socket, no hardware. Pass
  ``binding=FakeIsaacBinding()`` to get the whole path on CPU.
- **NO SIM GAINS ARE SUBSTITUTED, unlike ``MujocoG1Robot``.** With no ``config`` this uses
  ``G1Config``'s defaults, which are conservative HARDWARE placeholders (kp=20, kd=0.5,
  ``q_track_window=0``) pending OD-08 — under them the arm sags and per-chunk deltas are
  under-executed by a prefix-dependent factor (measured on MuJoCo: ~0.39 of a one-step delta).
  ``MujocoG1Robot`` substitutes gains it MEASURED on its own scene. Nobody has measured
  anything on PhysX, so nothing is substituted here: inventing an Isaac gain table would be a
  number no one computed. **Pass a config explicitly** — ``configs/robot/isaac_g1.yaml`` is
  the one ``scripts/rollout.py --robot isaac_g1`` loads, and it ships Isaac Lab's published
  magnitudes for this exact USD (waist 5000/5, arms 3000/10). Two candidate starting points
  therefore exist and they DISAGREE by an order of magnitude: those, and the gains in
  ``configs/robot/mujoco_g1.yaml``, which were measured on the same robot revision but in a
  different engine (its ``kd`` is per-joint critical damping derived from MuJoCo's effective
  inertia, so it does not transfer). NEITHER has been measured on PhysX; ``isaac_g1.yaml``'s
  ``gains`` block records why it picked the one it did. Re-measure on the box with the
  protocol in the ``mujoco_transport`` module docstring (fraction of a per-control-period
  position step executed within that period) before quoting any tracking number.
- **``gripper_vendor_min/max`` must stay (0.0, 1.0).** The transport's gripper channel IS the
  Dex3 synergy fraction, so canonical and vendor units coincide; any other range would make
  ``G1Adapter.gripper_to_vendor`` command a fully-closed hand for every input above ~0.01 with
  no error anywhere. ``__init__`` rejects it, exactly as ``MujocoG1Robot`` does.
- **Pacing runs on SIM TIME, not wall time.** ``G1Adapter.execute()`` paces its per-step
  command stream so step ``i`` is sent no earlier than ``t0 + i * dt_s``; that pacing is what
  makes its ``dq_max * dt`` clip a genuine velocity limit. The injected clock is the
  transport's physics-step tick (which advances by exactly ``control_dt_s`` per motor command)
  and the injected sleep advances PHYSICS rather than blocking a thread. Pass
  ``clock=time.monotonic`` / ``sleep=time.sleep`` for a realtime-paced sim.
- **Rendering never steps physics.** ``RenderingManager.render()`` is documented to render
  without advancing the simulation (preflight check I is the proof on the box), so
  ``render_frames(n)`` renders the CURRENT state once per camera and repeats it ``n`` times —
  the ``n`` frames are IDENTICAL, the same contract ``MockRobot`` and ``MujocoG1Robot`` give.
- **A warmup frame is NEVER recorded.** Isaac's rgb annotator returns ``None`` for the first
  frames — up to 20 of them in NVIDIA's own test. ``render_frames`` retries up to
  :data:`DEFAULT_RENDER_WARMUP_TICKS` times and then RAISES, because a black frame passes the
  T-11 data-quality gates and poisons training silently, which is strictly worse than a crash.
- **``reset()`` is an EPISODE reset, not an e-stop release.** It restores the articulation's
  default state and the asset's own gains, and clears the adapter's stale-tick memory and
  carried feed-forward target. A latched e-stop SURVIVES it and is released only by the
  deliberate ``clear_estop()`` — which, here, releases BOTH latches (the adapter's and the
  transport's); releasing only one leaves the arm silently uncommandable.
- **The IMU is a CONSTANT STAND-IN.** ``MujocoG1Robot`` reports a genuine IMU (the scene
  carries gyro and accelerometer sensors); this backend cannot. See difference (2) in the
  ``isaac_transport`` module docstring — and do not train on, evaluate against or safety-gate
  on the IMU group from an Isaac rollout.
- **The e-stop is NOT at parity with hardware.** It has a latency floor of one physics step
  when it arrives from a watchdog thread, and it does nothing at all if the main loop is
  wedged. Read the E-STOP section of the ``isaac_transport`` module docstring before relying
  on it; ``tests/test_isaac_g1.py`` asserts that documented behaviour, not the desired one.
- Torch-free; numpy only.

WHAT THIS CLASS DOES NOT SET UP, AND YOU PROBABLY NEED

  :class:`~wam.robot.isaac_binding.IsaacSimBinding` adds the G1 reference to an otherwise
  EMPTY stage: no ground plane, no fixed base, no table, no cube, no dome light, and no
  head/wrist cameras (its default camera is the viewport's ``/OmniverseKit_Persp``). A G1 on
  an empty stage falls forever, and an unlit stage renders a frame that is uniform — which
  ``render_frames`` will happily hand you, and only preflight check K ("camera_frame_is_not_
  blank", pixel std > 1.0) is currently looking for it. Point ``asset=`` at a prepared USD
  that contains the scene, and pass the camera prims through ``cameras=``. That USD is a data
  deliverable and does not exist in this repo yet.

Typical use::

    robot = IsaacG1Robot(config=g1_sim_config)     # boots Isaac Sim, loads the G1
    state = robot.read_state()                     # canonical 15-joint state, validity flags
    images = robot.render_frames(1)                # {"persp": [1, H, W, 3]}
    robot.execute(safe_chunk, prefix_steps=5)      # steps 5 x control_dt_s of physics
    robot.reset()                                  # next episode
    robot.close()                                  # shuts the SimulationApp down
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from wam.interfaces.schema import ActionChunk, CanonicalSpaceSpec, RobotState
from wam.robot.g1 import G1_SPEC, G1Adapter, G1Config
from wam.robot.isaac_binding import (
    DEFAULT_ASSET_SUBPATH,
    ISAAC_MISSING_MSG,
    IsaacBinding,
)
from wam.robot.isaac_transport import IsaacG1Transport

__all__ = [
    "DEFAULT_ASSET_SUBPATH",
    "DEFAULT_CAMERAS",
    "DEFAULT_IMAGE_HW",
    "DEFAULT_PHYSICS_HZ",
    "DEFAULT_RENDER_WARMUP_TICKS",
    "ISAAC_MISSING_MSG",
    "IsaacG1Robot",
]

#: Cameras rendered by default. ``persp`` is :class:`IsaacSimBinding`'s own default, the
#: viewport camera ``/OmniverseKit_Persp`` that exists on any stage — which is exactly what
#: makes it a safe default and a poor observation. A real scene supplies head/wrist prims.
DEFAULT_CAMERAS: tuple[str, ...] = ("persp",)

#: Default render size (H, W), matching ``MujocoG1Robot`` and the preflight's default so the
#: two simulators' image pipelines are comparable without a resize.
DEFAULT_IMAGE_HW: tuple[int, int] = (256, 256)

#: Default physics rate. 500 Hz survives PhysX's ``int(1.0 / dt)`` truncation exactly (the
#: binding refuses rates that do not) and gives 10 physics steps per 20 ms control period.
DEFAULT_PHYSICS_HZ = 500

#: How many times ``render_frames`` re-renders while the annotator returns ``None``. 20 is
#: NVIDIA's own number in the warmup test the preflight mirrors (check K).
DEFAULT_RENDER_WARMUP_TICKS = 20


class IsaacG1Robot:
    """``RobotAdapter`` backed by Isaac Sim: ``G1Adapter`` + ``IsaacG1Transport`` + cameras.

    See the module docstring for the contracts (composition rationale, why no sim gains are
    substituted, sim-time pacing, render semantics, reset vs e-stop, and the several ways this
    backend is NOT at parity with hardware).
    """

    def __init__(
        self,
        config: G1Config | Mapping[str, Any] | None = None,
        *,
        asset: str | Path | None = None,
        scene_path: str | Path | None = None,
        cameras: tuple[str, ...] = DEFAULT_CAMERAS,
        image_hw: tuple[int, int] = DEFAULT_IMAGE_HW,
        physics_hz: int = DEFAULT_PHYSICS_HZ,
        device: str = "cuda:0",
        headless: bool = True,
        camera_prims: Mapping[str, str] | None = None,
        render_warmup_ticks: int = DEFAULT_RENDER_WARMUP_TICKS,
        binding: IsaacBinding | None = None,
        transport: IsaacG1Transport | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        **binding_kwargs: Any,
    ) -> None:
        """Boot Isaac Sim (or accept an injected binding), wire a connected ``G1Adapter``.

        ``config``: ``G1Config`` or a mapping of its fields. ``None`` uses ``G1Config``'s
        HARDWARE PLACEHOLDER defaults — nothing sim-specific is substituted, see the module
        docstring. ``asset``: USD to load; ``None`` resolves the Isaac asset root plus
        :data:`~wam.robot.isaac_binding.DEFAULT_ASSET_SUBPATH`. ``scene_path`` is an alias for
        ``asset`` (it is the key ``scripts/rollout.py`` maps a config's ``sim.scene`` onto);
        passing both is an error. ``cameras``: names ``render_frames`` returns, validated
        against the binding at construction. ``camera_prims``: name -> USD prim path handed to
        the binding when this class builds one; defaults to the binding's own mapping.
        ``image_hw``, ``physics_hz``, ``device``, ``headless`` and any extra keyword go to
        :class:`~wam.robot.isaac_binding.IsaacSimBinding` and are IGNORED when ``binding`` is
        injected (that binding already owns them). ``binding`` / ``transport``: pre-built
        objects — this is how the whole path runs on CPU against
        :class:`~wam.robot.isaac_binding.FakeIsaacBinding`. ``clock``/``sleep``: pacing seams
        handed to ``G1Adapter``; default to SIM time (module docstring).

        Raises ``RuntimeError`` without Isaac Sim (message names the two-venv topology), and
        ``ValueError`` for a non-unit gripper range, an unknown camera, both ``asset`` and
        ``scene_path``, or a transport whose control period disagrees with the config.
        """
        if asset is not None and scene_path is not None:
            raise ValueError("pass either asset= or scene_path= (they are the same knob)")
        if asset is None:
            asset = scene_path

        cfg = self._resolve_config(config)
        # The transport's gripper channel IS vendor units [0, 1] (the Dex3 synergy fraction)
        # and clips to that range. G1Adapter.gripper_to_vendor maps the canonical [0, 1] onto
        # config.gripper_vendor_min/max, so a non-unit vendor range would silently produce a
        # fully-closed hand for every command above ~0.01 AND a readback that under-reports
        # closure by the span factor — with no error at any layer. Reject it here.
        if (cfg.gripper_vendor_min, cfg.gripper_vendor_max) != (0.0, 1.0):
            raise ValueError(
                f"gripper_vendor_min/max must be (0.0, 1.0) for the Isaac transport, got "
                f"({cfg.gripper_vendor_min}, {cfg.gripper_vendor_max}) — the Dex3 synergy "
                f"fraction IS the vendor unit here"
            )
        if render_warmup_ticks < 1:
            raise ValueError(f"render_warmup_ticks must be >= 1, got {render_warmup_ticks}")

        if transport is not None:
            if binding is not None and transport.binding is not binding:
                raise ValueError(
                    "transport and binding disagree: the injected transport is bound to a "
                    "different IsaacBinding"
                )
            binding = transport.binding
        elif binding is None:
            binding = self._build_binding(
                asset=asset,
                physics_hz=physics_hz,
                device=device,
                headless=headless,
                cameras=camera_prims,
                render_hw=image_hw,
                **binding_kwargs,
            )
        self._binding = binding

        if transport is None:
            transport = IsaacG1Transport(binding, control_dt_s=cfg.control_dt_s)
        elif abs(transport.control_dt_s - cfg.control_dt_s) > 1e-12:
            raise ValueError(
                f"transport.control_dt_s={transport.control_dt_s} != "
                f"config.control_dt_s={cfg.control_dt_s} — sim-time pacing needs them equal"
            )
        self._transport = transport

        for cam in cameras:
            if cam not in binding.camera_names:
                raise ValueError(
                    f"camera {cam!r} is not among the binding's cameras "
                    f"{list(binding.camera_names)}"
                )
        self._cameras = tuple(cameras)
        self._image_hw = (int(image_hw[0]), int(image_hw[1]))
        self._render_warmup_ticks = int(render_warmup_ticks)
        self._config = cfg

        self._adapter = G1Adapter(
            cfg,
            transport,
            clock=clock if clock is not None else self._sim_clock,
            sleep=sleep if sleep is not None else self._sim_sleep,
        )
        self._adapter.connect()  # injected transport: no SDK, no socket, no hardware

    # -- construction helpers ---------------------------------------------------------------

    @staticmethod
    def _resolve_config(config: G1Config | Mapping[str, Any] | None) -> G1Config:
        if config is None:
            return G1Config()
        return config if isinstance(config, G1Config) else G1Config(**config)

    @staticmethod
    def _build_binding(*, asset: str | Path | None, **kwargs: Any) -> IsaacBinding:
        """Construct the real :class:`IsaacSimBinding`.

        Imported INSIDE the function purely for symmetry with the rest of the lazy-vendor
        pattern; the class itself is importable anywhere (it only reaches for ``isaacsim``
        inside its own ``__init__``, which is what raises :data:`ISAAC_MISSING_MSG`).
        """
        from wam.robot.isaac_binding import IsaacSimBinding

        return IsaacSimBinding(asset=None if asset is None else str(asset), **kwargs)

    # -- introspection ----------------------------------------------------------------------

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
    def transport(self) -> IsaacG1Transport:
        return self._transport

    @property
    def binding(self) -> IsaacBinding:
        """The Isaac binding (DOF names, the physics tick, the render surface)."""
        return self._binding

    @property
    def cameras(self) -> tuple[str, ...]:
        """Camera names ``render_frames`` returns (the ``Observation.images`` keys)."""
        return self._cameras

    @property
    def image_hw(self) -> tuple[int, int]:
        """(H, W) requested at construction. When a binding is INJECTED it owns the real
        render size and this is only what the caller asked for — ``render_frames`` returns
        whatever the binding produces and never resizes."""
        return self._image_hw

    @property
    def is_estopped(self) -> bool:
        """Latched e-stop state (the executor duck-types this for its rollout summary)."""
        return self._adapter.is_estopped

    @property
    def sim_time_ns(self) -> int:
        """Simulated monotonic clock in ns — the physics-step tick, advanced ONLY by physics
        steps (``execute``/``hold``), never by wall time. Mirrors ``MockRobot.sim_time_ns`` so
        recording/replay can timestamp on sim time."""
        return self._transport.tick_ns

    # -- RobotAdapter protocol (delegated verbatim to the real G1Adapter) --------------------

    @property
    def limits(self) -> dict[str, np.ndarray]:
        """Canonical-order limit arrays from the config (float32)."""
        return self._adapter.limits

    def read_state(self) -> RobotState:
        """Canonical state from Isaac. Two consecutive reads with no physics step in between
        are STALE (the tick did not advance) and come back with cleared validity flags —
        exactly as a stalled vendor controller would look.

        THE IMU IS NOT MEASURED HERE. ``G1Adapter.read_state`` sets ``validity.imu`` from
        freshness alone, so this reports ``imu=True`` over the constant stand-in
        :data:`~wam.robot.isaac_transport.ISAAC_IMU_STANDIN`. That is a real gap, not a
        convention: ``MujocoG1Robot`` and the DDS transport both report a genuine IMU. See
        difference (2) in the ``isaac_transport`` module docstring.
        """
        return self._adapter.read_state()

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        """Execute the first ``prefix_steps`` steps (FR-05), stepping ``control_dt_s`` of
        physics per step. JOINT_DELTA only; no-op while e-stopped. The chunk MUST already have
        passed the SafetyFilter."""
        self._adapter.execute(chunk, prefix_steps)

    def hold(self) -> None:
        """Re-command the current position with ``dq_target = 0``. Unlike ``MockRobot`` this
        ADVANCES sim time by one ``control_dt_s`` — holding a real arm is physics."""
        self._adapter.hold()

    def estop(self) -> None:
        """Latched emergency stop: viscous damping in the sim + the adapter's latch.

        Safe to call at any time and from any thread. **It is not at parity with hardware.**

        Called on the MAIN thread it damps synchronously, steps ``damp_settle_s`` of physics
        so the arm comes to rest — so it DOES block, for 100 physics steps at the shipped
        settings, and advances the sim clock by that much — and a failed damp RAISES rather
        than reporting success.

        Called from ANY OTHER thread (a watchdog, say) it only LATCHES, and the damping is
        applied by the ``PHYSICS_PRE_STEP`` callback at the next physics step. **In the
        ordinary case that step never comes:** the latch is exactly what stops ``execute()``
        and ``hold()`` from stepping, so unless the e-stop lands inside a batch that was
        already in flight, ``damp_applied_count`` stays 0 and the gains are never lowered. A
        wedged main loop is the same story for a more exotic reason. Safe in sim, because a
        clock that stops means an arm that stops; on a robot it would be the opposite.

        ``clear_estop()`` discards a still-pending damp, so watchdog-estop → operator-resume
        can complete with ``damp_count == 1`` and ``damp_applied_count == 0``.

        Released only by :meth:`clear_estop`. Read the E-STOP section of the
        ``isaac_transport`` module docstring before relying on any of this.
        """
        self._adapter.estop()

    # -- rollout-harness extras --------------------------------------------------------------

    def clear_estop(self) -> None:
        """Release BOTH latches — the adapter's and the transport's (deliberate operator
        action; mirrors ``MockRobot.clear_estop``).

        Both, because they are independent: the adapter's makes ``execute()`` a no-op, the
        transport's makes every write a no-op, and clearing only one leaves a robot that
        accepts commands and silently drops them. The transport's damping gains stay in force
        until the next motor command restores the configured ones.
        """
        self._adapter.clear_estop()
        self._transport.clear_estop()

    def reset(self) -> None:
        """Reset the EPISODE: the articulation's default state and the asset's own gains.

        A latched e-stop deliberately SURVIVES (use :meth:`clear_estop`). The physics-step
        counter is NOT rewound — it is a raw counter on both bindings — but the pose changes
        without a step, so the adapter's stale-tick memory and its carried feed-forward target
        are both dropped: "the tick did not advance" no longer means "no new sample", which is
        the one case ``forget_tick()`` exists for.
        """
        self._transport.reset()
        self._adapter.forget_tick()
        self._adapter.forget_command()

    # -- cameras -------------------------------------------------------------------------------

    def render_frames(self, n: int) -> dict[str, np.ndarray]:
        """Render ``n`` frames per camera: ``{name: uint8 array [n, H, W, 3]}``.

        The frames within one call are IDENTICAL copies of the CURRENT sim state — rendering
        never steps physics (that would advance the tick behind the adapter's back and break
        the staleness contract). Call ``render_frames(1)`` between ``execute()`` calls to get
        distinct frames; that is exactly what ``ClosedLoopExecutor`` does.

        A camera still warming up returns ``None`` from the binding; this retries up to
        ``render_warmup_ticks`` times and then raises ``RuntimeError``. It NEVER substitutes a
        black frame: black frames pass the T-11 data-quality gates and poison training
        silently, which is strictly worse than a crash.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        out: dict[str, np.ndarray] = {}
        for cam in self._cameras:
            frame = None
            for _ in range(self._render_warmup_ticks):
                frame = self._binding.render_frame(cam)
                if frame is not None:
                    break
            if frame is None:
                raise RuntimeError(
                    f"camera {cam!r} returned no frame after {self._render_warmup_ticks} "
                    "render ticks — the renderer is still warming up, or the stage has no "
                    "lighting. Raise render_warmup_ticks or add a dome light; a black frame "
                    "is NOT substituted because it would pass the T-11 data-quality gates"
                )
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            out[cam] = np.repeat(frame[None, ...], n, axis=0)
        return out

    def close(self) -> None:
        """Shut the binding down (idempotent — ``IsaacBinding.close`` is).

        On the real backend this closes the ``SimulationApp``. A leaked ``SimulationApp``
        wedges the interpreter, so this is not optional housekeeping.
        """
        self._binding.close()

    # -- sim-time pacing seams handed to G1Adapter ---------------------------------------------

    def _sim_clock(self) -> float:
        """Sim seconds (the physics-step tick). Advances only when physics steps."""
        return self._transport.tick_ns * 1e-9

    def _sim_sleep(self, seconds: float) -> None:
        """Advance SIM time by ``seconds`` holding the last command, instead of blocking.

        Only reached when a chunk's ``dt_s`` exceeds ``control_dt_s``; the arm keeps tracking
        its last target (targets and gains are untouched), which is what a real arm does while
        the next command is late.
        """
        self._transport.advance(seconds)
