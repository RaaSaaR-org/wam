"""Isaac Sim binding behind the G1 transport seam (FR-06, E2).

    THE REAL BINDING IN THIS FILE HAS NEVER BEEN EXECUTED.

    :class:`IsaacSimBinding` was written against NVIDIA's *documentation* for Isaac Sim
    6.0.1, on a machine with no Isaac Sim and no GPU. Every module path, symbol name,
    argument name and return shape below is an ASSUMPTION, not a measurement.
    ``scripts/preflight_isaac.py`` is the thing that proves or refutes each of them on the
    target box; it is the authoritative in-repo record of what this file believes, and it
    must be run — and pass — before any number out of an Isaac rollout is trusted. Anything
    here labelled UNVERIFIED is exactly that.

    :class:`FakeIsaacBinding` is the opposite: pure Python + numpy, no Isaac, no GPU, fully
    executed by ``tests/test_isaac_binding.py`` on a laptop. It exists so that everything
    ABOVE this seam — the transport's tick/staleness contract, the e-stop latch, the gain
    round-trip, the name resolution — is genuinely tested even though the vendor half
    cannot be.

WHY A BINDING AT ALL

  Isaac Sim cannot share a virtualenv with this repo's backbone: ``isaacsim-core==6.0.1.0``
  hard-pins ``torch==2.11.0`` while ``uv.lock`` resolves 2.13.0. The box therefore runs two
  interpreters — the Isaac venv drives ``scripts/rollout.py --robot isaac_g1 --policy remote``
  and the WAM venv serves the policy over the existing websocket seam. **Everything on the
  Isaac side of that split must be torch-free**, so this module, ``isaac_transport.py`` and
  ``isaac_g1.py`` import numpy and nothing heavier. Proved in a subprocess rather than by
  trusting an import list — and by two tests, because they cover different amounts of the
  claim: ``tests/test_isaac_binding.py`` imports THIS module alone, while
  ``tests/test_isaac_g1.py`` imports all three and drives a whole ``IsaacG1Robot``.

  All ``isaacsim``/``omni``/``pxr``/``warp`` imports are confined HERE, and they happen
  inside :meth:`IsaacSimBinding.__init__`, never at module scope: importing
  ``wam.robot.isaac_binding`` on a machine without Isaac Sim raises nothing, which is what
  lets the transport and the adapter be unit-tested on CPU. Same isolation trick as
  ``mujoco_transport.py``, one level stricter (there ``mujoco`` is a normal pip package;
  here the extension system only exists after ``SimulationApp`` has been constructed).

WHAT THE PROTOCOL PROMISES

- **Plain numpy at the boundary.** Isaac's experimental prims API returns ``warp.array``
  objects; they are converted here and nothing downstream ever sees one. No ``warp`` type
  appears in a signature.
- **Single-robot ``(D,)`` views.** ``Articulation`` is a BATCHED view of ``N`` prims and
  returns ``(N, D)``. The squeeze happens in ONE place (:func:`_row`), so no caller has to
  remember whether it is holding a batch.
- **The tick is an exact integer.** ``SimulationManager.get_num_physics_steps()``, coerced
  to a Python ``int``. ``G1Adapter.read_state`` decides staleness by EQUALITY against the
  previous tick, so a float clock (or a tick derived from sim time) would make that
  comparison meaningless. A float here is a hard error, not a cast.
- **Rendering never advances physics.** ``RenderingManager.render()`` is documented to
  toggle ``/app/player/playSimulations`` off, update once and restore it. The adapter owns
  the clock; a render that stepped behind its back would corrupt staleness detection and
  silently widen the ``dq_max * dt`` velocity limit. Preflight check I is the proof.
- **The caller owns the gains.** :meth:`set_dof_gains` writes what it is given, including
  ``kp = 0`` (the e-stop damping mode). This is why the backend is raw Isaac Sim and NOT
  Isaac Lab: Isaac Lab's explicit actuator models (``DCMotorCfg``, used for the G1 legs in
  its shipped cfg) compute torque in Python and neutralise the sim's PD gains, and its G1
  cfg is a legacy 23-DoF model besides.
- **43 DoFs, resolved BY NAME.** The asset is ``{assets_root}/Isaac/Robots/Unitree/G1/g1.usd``
  = ``g1_29dof_with_hand_rev_1_0``: 29 body motors + 2 x 7 Dex3-1 finger joints. PhysX orders
  DOFs breadth-first from the base link, which is neither URDF order nor
  :data:`~wam.robot.g1_transport.G1_MOTOR_JOINT_NAMES` order, and ``G1Adapter`` gathers by
  hard-coded index — so a positional guess would produce a plausible-looking robot moving
  the wrong arm, undetectable without hardware. :func:`resolve_g1_dof_indices` is the only
  place names become indices, it is shared by both bindings, and it fails naming the joint.

E-STOP IS **NOT** AT PARITY WITH HARDWARE — READ THIS BEFORE RELYING ON IT

  The Omniverse API is main-thread-only, so an e-stop arriving on a watchdog thread cannot
  touch Isaac at all. The design (implemented in ``isaac_transport.py``, enabled by
  :meth:`register_pre_physics_callback` here) is: ``emergency_damp()`` latches a pending-damp
  flag in pure Python from any thread, and a ``PHYSICS_PRE_STEP`` callback drains it on the
  main thread. Two real differences from ``DdsG1Transport.emergency_damp()``, which puts
  damping on the DDS wire immediately and synchronously:

  1. **A latency floor of one physics step.** The damp takes effect at the next
     ``PHYSICS_PRE_STEP``, i.e. up to one ``physics_dt`` (2 ms at 500 Hz) later — plus
     however long the current ``step(steps=N)`` batch has left to run, since the callback
     fires per step but the Python caller does not regain control until the batch ends.
  2. **No e-stop at all if the main loop is wedged.** If the main thread is blocked,
     deadlocked or simply not calling ``step()``, the flag is never drained and NOTHING
     happens in the simulator. On hardware the DDS write is independent of the control
     loop's health, which is the entire point of an e-stop.

  Neither is papered over: ``FakeIsaacBinding.wedge_main_thread()`` reproduces (2) exactly
  so the failure is a test, not a footnote. What the transport still owes — and what is
  testable — is the property ``G1Adapter.estop()`` depends on: **no further motor command
  reaches the sim after ``estop()`` returns.** That one is enforced in pure Python by the
  latch, on whatever thread it is called from, with no Isaac involvement.

Torch-free; numpy only.
"""

from __future__ import annotations

import numbers
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from wam.robot.g1_transport import DEX3_FINGER_JOINTS, G1_MOTOR_JOINT_NAMES

__all__ = [
    "BODY_NAME_CANDIDATES",
    "DEFAULT_ASSET_SUBPATH",
    "DEFAULT_CAMERA_PRIMS",
    "EFFORT_GETTER_CANDIDATES",
    "EXPECTED_NUM_DOFS",
    "FINGER_NAME_CANDIDATES",
    "ISAAC_MISSING_MSG",
    "FakeIsaacBinding",
    "G1DofIndices",
    "IsaacBinding",
    "IsaacSimBinding",
    "fake_g1_dof_names",
    "resolve_g1_dof_indices",
]

ISAAC_MISSING_MSG = (
    "Isaac Sim support requires the vendor stack — run this with Isaac Sim's own "
    "interpreter (`isaac-python`), in the Isaac venv, NOT the repo venv. Isaac cannot "
    "share a venv with the WAM backbone (isaacsim-core 6.0.1 pins torch==2.11.0); serve "
    "the policy from the WAM venv and pass --policy remote --server-uri ws://..."
)

#: Candidate joint-naming conventions for the shipped G1 USD, tried in order. The Unitree
#: URDF (g1_29dof_with_hand_rev_1_0) uses ``<canonical>_joint`` for the body and
#: ``<side>_hand_<finger>_joint`` for the fingers; the USD is a conversion of it and the
#: converter may or may not have preserved the suffix. UNVERIFIED until preflight check G
#: runs on the box — a mismatch here is a config/data finding, not a code change.
#:
#: DUPLICATED, deliberately, in ``scripts/preflight_isaac.py``: the preflight must stay
#: runnable without believing anything this module believes (its whole job is to test the
#: vendor API, not our code). ``tests/test_isaac_binding.py`` asserts the two copies are
#: identical, so a divergence is loud instead of silent.
BODY_NAME_CANDIDATES: tuple[str, ...] = ("{name}_joint", "{name}")
FINGER_NAME_CANDIDATES: tuple[str, ...] = (
    "{side}_hand_{finger}_joint",
    "{side}_hand_{finger}",
    "{side}_{finger}_joint",
)

#: Isaac ships the G1 under the asset root: the 29-DoF revision WITH the Dex3-1 hands, i.e.
#: the same kinematic model as the MuJoCo Menagerie file this repo already fetches. The
#: Isaac Lab copy (``ISAACLAB_NUCLEUS_DIR``) is a legacy 23-DoF G1 — do not point at it.
DEFAULT_ASSET_SUBPATH = "/Isaac/Robots/Unitree/G1/g1.usd"

#: 29 body motors + 2 x 7 Dex3-1 finger joints. Asserted at construction, never assumed:
#: the DoF count was cross-checked against the vendor URDF, never read out of the USD.
EXPECTED_NUM_DOFS = len(G1_MOTOR_JOINT_NAMES) + 2 * len(DEX3_FINGER_JOINTS)

#: Default camera name -> prim path. ``/OmniverseKit_Persp`` is the viewport camera that
#: exists on any stage, which is what ``scripts/preflight_isaac.py`` renders from. A G1
#: scene with head/wrist cameras supplies its own mapping; the prims must exist on the stage.
DEFAULT_CAMERA_PRIMS: Mapping[str, str] = {"persp": "/OmniverseKit_Persp"}

#: Candidate names for the Articulation's measured-effort getter, tried in order. UNVERIFIED
#: — unlike positions/velocities/gains this one is NOT in the documentation excerpt the rest
#: of this module was written from, so the name is a guess with fallbacks. Effort readback is
#: DIAGNOSTIC ONLY: it is not part of the ``G1Transport`` low-state dict contract (q, dq, imu,
#: gripper, tick_ns), so a build where none of these resolves still runs a full rollout.
#: ``scripts/preflight_isaac.py`` RECORDS which one exists rather than failing on it.
EFFORT_GETTER_CANDIDATES: tuple[str, ...] = (
    "get_dof_efforts",
    "get_measured_dof_efforts",
    "get_dof_forces",
    "get_measured_joint_efforts",
)

_HAND_SIDES: tuple[str, str] = ("left", "right")


# -- name resolution ---------------------------------------------------------------------


@dataclass(frozen=True)
class G1DofIndices:
    """Where each canonical G1 joint lives in the articulation's own DOF ordering.

    ``body`` is 29 indices in :data:`~wam.robot.g1_transport.G1_MOTOR_JOINT_NAMES` order
    (the ``G1JointIndex`` convention ``G1Adapter`` gathers by), ``left``/``right`` are 7
    each in :data:`~wam.robot.g1_transport.DEX3_FINGER_JOINTS` order. The ``*_pattern``
    fields record WHICH naming convention matched, so a rollout manifest can carry it and an
    asset swap shows up as a diff rather than as a robot moving the wrong arm.
    """

    body: tuple[int, ...]
    left: tuple[int, ...]
    right: tuple[int, ...]
    body_pattern: str
    left_pattern: str
    right_pattern: str

    def body_array(self) -> np.ndarray:
        """The 29 body indices as an ``int64`` array, for gather/scatter."""
        return np.asarray(self.body, dtype=np.int64)

    def hand_array(self, side: str) -> np.ndarray:
        """The 7 finger indices of ``side`` in {'left', 'right'} as an ``int64`` array."""
        if side not in _HAND_SIDES:
            raise ValueError(f"side must be one of {_HAND_SIDES}, got {side!r}")
        return np.asarray(self.left if side == "left" else self.right, dtype=np.int64)


def _match_pattern(
    index: Mapping[str, int],
    canonical: Sequence[str],
    candidates: Sequence[str],
    **fmt: str,
) -> tuple[str, tuple[int, ...]] | tuple[None, list[str]]:
    """First naming convention that covers every canonical name -> (pattern, indices).

    On failure returns ``(None, missing)`` for the candidate that missed the FEWEST names —
    reporting the near-miss is the difference between "the asset is wrong" and "the suffix
    changed", and a caller looking at a 43-name dump needs to be told which.
    """
    best_missing: list[str] = list(canonical)
    for pattern in candidates:
        wanted = [pattern.format(name=n, finger=n, **fmt) for n in canonical]
        missing = [w for w in wanted if w not in index]
        if not missing:
            return pattern, tuple(index[w] for w in wanted)
        if len(missing) < len(best_missing):
            best_missing = missing
    return None, best_missing


def resolve_g1_dof_indices(dof_names: Sequence[str]) -> G1DofIndices:
    """Map the 43 canonical G1 joints onto an articulation's own DOF ordering, BY NAME.

    This is the only place in the Isaac backend where a name becomes an index, and it is
    shared by :class:`IsaacSimBinding` and :class:`FakeIsaacBinding` so the fake exercises
    the real resolution rather than a convenient parallel one. Never index positionally:
    PhysX walks the articulation breadth-first from the base link, so the order is neither
    the URDF's nor ours, and ``G1Adapter`` gathers canonical joints out of the 29-slot motor
    array by HARD-CODED index — one permuted entry moves a physical arm silently.

    Raises ``ValueError`` naming the joints that could not be resolved and dumping the
    articulation's actual DOF names, because that dump is the fix: the naming convention in
    the shipped USD is a discovery (preflight check G), not a fact.
    """
    names = list(dof_names)
    index: dict[str, int] = {}
    duplicates = []
    for i, name in enumerate(names):
        if name in index:
            duplicates.append(name)
            continue
        index[name] = i
    if duplicates:
        # A duplicate makes the name->index map ambiguous, and the wrong branch of the
        # ambiguity is a joint that moves when a different one was commanded.
        raise ValueError(
            f"duplicate DOF names in the articulation: {sorted(set(duplicates))} — the "
            "name -> index map is ambiguous, refusing to guess"
        )

    body_pattern, body = _match_pattern(index, G1_MOTOR_JOINT_NAMES, BODY_NAME_CANDIDATES)
    if body_pattern is None:
        raise ValueError(_resolution_error("body", body, BODY_NAME_CANDIDATES, names))

    hands: dict[str, tuple[int, ...]] = {}
    patterns: dict[str, str] = {}
    for side in _HAND_SIDES:
        pattern, found = _match_pattern(
            index, DEX3_FINGER_JOINTS, FINGER_NAME_CANDIDATES, side=side
        )
        if pattern is None:
            raise ValueError(
                _resolution_error(f"{side} hand", found, FINGER_NAME_CANDIDATES, names)
            )
        patterns[side] = pattern
        hands[side] = found  # type: ignore[assignment]

    # Every canonical slot must own a DISTINCT dof. Each group is internally distinct for free
    # (distinct canonical names -> distinct formatted names -> distinct dict entries), so what
    # this catches is a collision ACROSS groups: a naming convention under which some body joint
    # formats to the same string as a finger joint, or the left hand to the right. Today's
    # candidate patterns cannot produce one. The guard is here because the ones added later
    # might, and because this is the failure with no symptom — resolution succeeds, the count is
    # 43, every readback looks plausible, and two canonical joints quietly share one physical
    # actuator. Assets get renamed; the 3 lines cost nothing.
    resolved = body + hands["left"] + hands["right"]
    if len(set(resolved)) != len(resolved):
        shared = sorted({i for i in resolved if resolved.count(i) > 1})
        raise ValueError(
            f"the naming conventions that matched map two canonical G1 joints onto the same "
            f"DOF index {shared} (body={body_pattern!r}, left={patterns['left']!r}, "
            f"right={patterns['right']!r}). Refusing to guess which one wins.\n"
            f"The articulation reports {len(names)} DOFs: {names}"
        )

    return G1DofIndices(
        body=body,  # type: ignore[arg-type]
        left=hands["left"],
        right=hands["right"],
        body_pattern=body_pattern,
        left_pattern=patterns["left"],
        right_pattern=patterns["right"],
    )


def _resolution_error(
    group: str, missing: Sequence[str], candidates: Sequence[str], dof_names: Sequence[str]
) -> str:
    return (
        f"cannot resolve the G1 {group} joints against this articulation: no naming "
        f"convention in {list(candidates)} covers all of them. Closest candidate is still "
        f"missing {list(missing)}.\n"
        f"The articulation reports {len(dof_names)} DOFs: {list(dof_names)}\n"
        "Fix the asset or add the convention to BODY_NAME_CANDIDATES / "
        "FINGER_NAME_CANDIDATES (and to the copies in scripts/preflight_isaac.py) — do NOT "
        "fall back to positional indexing."
    )


# -- the seam ------------------------------------------------------------------------------


@runtime_checkable
class IsaacBinding(Protocol):
    """Everything the Isaac G1 transport needs from Isaac Sim, and nothing more.

    Kept deliberately small: every member is a thing that can turn out to be wrong on the
    box, and the cost of being wrong is one preflight check each. Shapes are single-robot
    ``(D,)`` numpy — the ``(N, D)`` batch squeeze and the ``warp.array`` conversion both
    happen inside the implementation.
    """

    @property
    def num_dofs(self) -> int:
        """Articulation DOF count (43 for the G1 with Dex3-1 hands)."""
        ...

    @property
    def dof_names(self) -> tuple[str, ...]:
        """The articulation's own DOF names, in ITS order (PhysX breadth-first)."""
        ...

    @property
    def dof_indices(self) -> G1DofIndices:
        """Canonical G1 joints -> DOF indices, resolved by name (never positionally)."""
        ...

    @property
    def physics_dt(self) -> float:
        """Seconds of simulated time per physics step."""
        ...

    @property
    def camera_names(self) -> tuple[str, ...]:
        """Camera names accepted by :meth:`render_frame`."""
        ...

    def get_physics_step_count(self) -> int:
        """Physics steps since the simulation started, as an exact ``int``.

        THE tick. Staleness upstream is an equality test against the previous value, so this
        must never be a float and never be derived from sim time.
        """
        ...

    def step(self, steps: int = 1) -> None:
        """Advance physics by exactly ``steps`` steps, firing pre-physics callbacks."""
        ...

    def reset(self) -> None:
        """Restore the articulation's default state (episode reset).

        Does NOT rewind the tick — ``get_physics_step_count`` is a raw counter — so a caller
        that resets mid-episode must also call ``G1Adapter.forget_tick()``.
        """
        ...

    def get_dof_positions(self) -> np.ndarray:
        """Joint positions ``(D,)`` in rad."""
        ...

    def get_dof_velocities(self) -> np.ndarray:
        """Joint velocities ``(D,)`` in rad/s."""
        ...

    def get_dof_efforts(self) -> np.ndarray:
        """Measured joint efforts ``(D,)`` in Nm. DIAGNOSTIC ONLY — not part of the
        ``G1Transport`` low-state contract, and the vendor getter's name is a guess (see
        :data:`EFFORT_GETTER_CANDIDATES`)."""
        ...

    def set_dof_position_targets(self, targets: np.ndarray) -> None:
        """Write position targets ``(D,)`` in rad for every DOF."""
        ...

    def set_dof_gains(self, stiffnesses: np.ndarray, dampings: np.ndarray) -> None:
        """Write per-DOF PD gains ``(D,)``. ``kp = 0`` is legal — it is the damping mode."""
        ...

    def get_dof_gains(self) -> tuple[np.ndarray, np.ndarray]:
        """Read back ``(kp, kd)``, each ``(D,)``."""
        ...

    def render_frame(self, camera: str) -> np.ndarray | None:
        """Render ONE frame from ``camera`` as ``uint8 (H, W, 3)``, WITHOUT stepping physics.

        Returns ``None`` while the renderer is still warming up (the first frames come back
        empty — up to 20 of them in NVIDIA's own test). The caller must retry rather than
        record a black frame: black frames pass the T-11 data-quality gates and poison
        training silently, which is strictly worse than a crash.
        """
        ...

    def register_pre_physics_callback(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` on the main thread before each physics step.

        The e-stop drain point. Callbacks run in registration order and MUST NOT raise: on
        the real binding they execute inside Isaac's C++ event dispatcher, where the
        behaviour of a propagating Python exception is undocumented.
        """
        ...

    def close(self) -> None:
        """Shut down. Idempotent; every other method raises afterwards."""
        ...


# -- helpers shared by both implementations ------------------------------------------------


def _to_numpy(value: Any) -> np.ndarray:
    """``warp.array`` | ``np.ndarray`` -> ``np.ndarray``, without importing warp or torch.

    Duck-typed on ``.numpy()`` on purpose: naming the type would mean importing ``warp`` at
    module scope, and this module has to import on a machine that has neither warp nor Isaac.
    A CUDA torch tensor would raise out of its own ``.numpy()`` — deliberately loud, since a
    torch object arriving here means the torch-free premise of this venv is already broken.
    """
    if isinstance(value, np.ndarray):
        return value
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        return np.asarray(to_numpy())
    return np.asarray(value)


def _row(value: Any, num_dofs: int, what: str) -> np.ndarray:
    """Squeeze a batched ``(1, D)`` articulation readback to a fresh ``(D,)`` float32 copy.

    ONE place does this. ``Articulation`` is a view over N prims and returns ``(N, D)``; we
    always drive exactly one robot, and a caller that has to remember which of the two
    shapes it is holding will eventually forget.
    """
    arr = np.asarray(_to_numpy(value), dtype=np.float32)
    if arr.shape == (num_dofs,):
        return arr.copy()
    if arr.shape == (1, num_dofs):
        return arr[0].copy()
    raise RuntimeError(
        f"{what}: expected ({num_dofs},) or (1, {num_dofs}) from the articulation, got "
        f"{arr.shape} — a batch of more than one robot is not what this binding drives"
    )


def _batch(values: np.ndarray, num_dofs: int, what: str) -> np.ndarray:
    """Validate a ``(D,)`` command and shape it as the ``(1, D)`` float32 batch Isaac wants."""
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != (num_dofs,):
        raise ValueError(f"{what}: expected shape ({num_dofs},), got {arr.shape}")
    if not np.isfinite(arr).all():
        # A non-finite target reaching PhysX corrupts the articulation for the rest of the
        # episode while readbacks may stay finite — undetectable downstream, so it stops here.
        raise ValueError(f"{what}: non-finite values would corrupt the simulation")
    return np.ascontiguousarray(arr.reshape(1, num_dofs), dtype=np.float32)


def _require_main_thread(op: str) -> None:
    """Refuse an Omniverse call from a non-main thread.

    NVIDIA documents the prohibition ("All methods must be called from the main thread") but
    not the consequence of breaking it. Assume the worst — silently wrong readings — because
    that is the failure mode a safety layer cannot detect. Preflight check M verifies that
    this identity test discriminates at all.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            f"{op}: the Omniverse API is main-thread-only and this call is on "
            f"{threading.current_thread().name!r}. An e-stop from a watchdog thread must "
            "latch a flag and let the PHYSICS_PRE_STEP callback drain it on the main thread "
            "(see the module docstring)."
        )


# -- the real thing (written against docs, never executed) ---------------------------------


class IsaacSimBinding:
    """:class:`IsaacBinding` on real Isaac Sim 6.0.1. **Never executed — see the module
    docstring.** ``scripts/preflight_isaac.py`` is what turns each assumption below into a
    pass or a fail on the box.

    Construction order is load-bearing and mirrors the preflight exactly:

    1. ``SimulationApp`` FIRST — the Omniverse extension system provides every other
       ``omni.*``/``isaacsim.*`` module and is not loaded until its constructor returns.
    2. ``setup_simulation(dt, device)`` and ``RenderingManager.set_dt(dt)`` BEFORE ``play()``
       — changing rates after play is a known fatal crash in 6.0. ``PhysxScene.set_dt`` does
       ``steps_per_second = int(1.0 / dt)``, which silently lands on 499 for some rates, so
       ``physics_hz`` is validated for exact truncation first (Isaac Lab ships a live victim
       of this: dt=0.0167 -> 59, not 60).
    3. ``add_reference_to_stage`` -> ``Articulation`` -> ``play()`` -> ``update_app(steps=2)``:
       the tensor backend only becomes valid after the app has ticked.
    4. Resolve all 43 DOFs BY NAME and assert the count, before anything is commanded.

    Anything that raises after step 1 closes the app before propagating — a leaked
    ``SimulationApp`` wedges the interpreter and the traceback would be the last thing the
    operator ever saw.
    """

    def __init__(
        self,
        *,
        asset: str | None = None,
        prim_path: str = "/World/G1",
        physics_hz: int = 500,
        device: str = "cuda:0",
        cameras: Mapping[str, str] | None = None,
        render_hw: tuple[int, int] = (256, 256),
        headless: bool = True,
        renderer: str = "RaytracedLighting",
        expected_num_dofs: int = EXPECTED_NUM_DOFS,
    ) -> None:
        """Boot Isaac Sim, load the G1 and resolve its DOFs.

        ``asset``: USD path; ``None`` resolves ``{assets_root}`` +
        :data:`DEFAULT_ASSET_SUBPATH`. ``physics_hz``: physics rate; ``int(1/dt)`` must be
        exact. ``cameras``: name -> prim path (default :data:`DEFAULT_CAMERA_PRIMS`); the
        prims must already exist on the stage. ``render_hw``: (H, W) of every render product.
        ``expected_num_dofs``: asserted against the articulation, 43 for the G1 with hands.

        Raises ``RuntimeError`` when Isaac Sim is not importable (:data:`ISAAC_MISSING_MSG`),
        when called off the main thread, or when the DOF count disagrees; ``ValueError`` when
        a joint name cannot be resolved (the message names it and dumps the actual names).
        """
        _require_main_thread("IsaacSimBinding()")
        if physics_hz <= 0:
            raise ValueError(f"physics_hz must be > 0, got {physics_hz}")
        dt = 1.0 / float(physics_hz)
        if int(1.0 / dt) != physics_hz:
            raise ValueError(
                f"physics_hz={physics_hz} does not survive PhysX's int(1.0/dt) truncation "
                f"(int(1/{dt!r}) == {int(1.0 / dt)}) — every velocity limit would silently "
                "be computed against a different rate than the one that runs"
            )
        if len(render_hw) != 2 or any(int(v) < 1 for v in render_hw):
            raise ValueError(f"render_hw must be two positive ints, got {render_hw!r}")

        try:
            # MUST come before every other omni/isaacsim import in this constructor: the
            # extension system that PROVIDES those modules is started by SimulationApp's
            # constructor, so importing them earlier fails (or, worse, half-loads).
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(ISAAC_MISSING_MSG) from exc

        self._app: Any = SimulationApp({"headless": bool(headless), "renderer": str(renderer)})
        self._closed = False
        try:
            import isaacsim.core.experimental.utils.app as app_utils
            import isaacsim.core.experimental.utils.stage as stage_utils
            from isaacsim.core.experimental.prims import Articulation
            from isaacsim.core.rendering_manager import RenderingManager
            from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

            self._app_utils = app_utils
            self._sim = SimulationManager
            self._rendering = RenderingManager

            # (2) every clock set before play(), and only before play().
            SimulationManager.setup_simulation(dt=dt, device=str(device))
            RenderingManager.set_dt(dt)
            self._physics_dt = float(SimulationManager.get_physics_dt())

            if asset is None:
                from isaacsim.storage.native import get_assets_root_path

                root = get_assets_root_path()
                if not root:
                    raise RuntimeError(
                        "get_assets_root_path() returned nothing — the Isaac asset root is "
                        "not reachable; pass asset=<local g1.usd> explicitly"
                    )
                asset = f"{root}{DEFAULT_ASSET_SUBPATH}"
            self._asset = str(asset)

            # (3) SI units, then the robot, then play, then two app ticks.
            stage_utils.set_stage_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
            stage_utils.add_reference_to_stage(usd_path=self._asset, path=str(prim_path))
            self._robot: Any = Articulation(str(prim_path))
            app_utils.play()
            app_utils.update_app(steps=2)  # the tensor backend is invalid before this

            # (4) 43 DOFs, resolved by name. Assert the count separately: an articulation
            # with EXTRA dofs (a floating base, a gripper mount) would still resolve all 43
            # names while every unnamed dof silently keeps whatever the asset left in it.
            self._dof_names = tuple(str(n) for n in self._robot.dof_names)
            self._num_dofs = int(self._robot.num_dofs)
            if self._num_dofs != int(expected_num_dofs):
                raise RuntimeError(
                    f"{self._asset} has {self._num_dofs} DOFs, expected "
                    f"{int(expected_num_dofs)} (29 body + 2 x 7 Dex3 fingers). This is not "
                    f"g1_29dof_with_hand_rev_1_0 — Isaac Lab's G1 cfg is a legacy 23-DoF "
                    f"model, do not point at ISAACLAB_NUCLEUS_DIR.\nDOF names: "
                    f"{list(self._dof_names)}"
                )
            if len(self._dof_names) != self._num_dofs:
                raise RuntimeError(
                    f"articulation reports num_dofs={self._num_dofs} but "
                    f"{len(self._dof_names)} names: {list(self._dof_names)}"
                )
            self._dof_indices = resolve_g1_dof_indices(self._dof_names)

            # Effort getter: discovered, not assumed (see EFFORT_GETTER_CANDIDATES).
            self._effort_getter: str | None = next(
                (n for n in EFFORT_GETTER_CANDIDATES if callable(getattr(self._robot, n, None))),
                None,
            )

            # ONE registration with Isaac, dispatching a plain Python list. Additional
            # callbacks cost nothing and, more to the point, risk nothing: register_callback
            # is called exactly once, at a known moment, on the main thread.
            self._pre_physics: list[Callable[[], None]] = []
            self._callback_handle: Any = SimulationManager.register_callback(
                self._dispatch_pre_physics, event=SimulationEvent.PHYSICS_PRE_STEP
            )

            self._cameras = dict(DEFAULT_CAMERA_PRIMS if cameras is None else cameras)
            self._render_hw = (int(render_hw[0]), int(render_hw[1]))
            self._annotators: dict[str, Any] = {}
            self._render_products: dict[str, Any] = {}
            self._setup_cameras()
        except BaseException:
            self.close()
            raise

    # -- introspection ---------------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    @property
    def dof_indices(self) -> G1DofIndices:
        return self._dof_indices

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(self._cameras)

    @property
    def asset(self) -> str:
        """The USD actually loaded — belongs in the rollout manifest (AC-04)."""
        return self._asset

    @property
    def effort_getter(self) -> str | None:
        """Which of :data:`EFFORT_GETTER_CANDIDATES` this build provides, or ``None``."""
        return self._effort_getter

    # -- clock -----------------------------------------------------------------------------

    def get_physics_step_count(self) -> int:
        """``SimulationManager.get_num_physics_steps()``, coerced to an exact ``int``.

        A float is rejected rather than rounded: staleness upstream is an equality test, and
        a clock that only ALMOST repeats would make the watchdog fire at random.

        ``numbers.Integral``, NOT ``(int, np.integer)`` — the same rule ``preflight_isaac.py``
        and ``IsaacG1Transport.step_count`` apply. When this was the narrower test the three
        disagreed: a pybind counter type that registers as ``Integral`` without subclassing
        either concrete type passed the preflight green and then made every ``read_state()``
        on the box raise. The preflight is only worth running if it believes what the code
        believes.
        """
        self._require_open("get_physics_step_count")
        raw = self._sim.get_num_physics_steps()
        if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
            raise TypeError(
                f"get_num_physics_steps() returned {type(raw).__name__} ({raw!r}); the tick "
                "must be an integer counter — staleness detection compares it for equality"
            )
        return int(raw)

    def step(self, steps: int = 1) -> None:
        self._require_open("step")
        _require_main_thread("step")
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        if steps:
            self._sim.step(steps=int(steps))

    def reset(self) -> None:
        self._require_open("reset")
        _require_main_thread("reset")
        self._robot.reset()

    # -- articulation I/O ------------------------------------------------------------------

    def get_dof_positions(self) -> np.ndarray:
        self._require_open("get_dof_positions")
        _require_main_thread("get_dof_positions")
        return _row(self._robot.get_dof_positions(), self._num_dofs, "dof positions")

    def get_dof_velocities(self) -> np.ndarray:
        self._require_open("get_dof_velocities")
        _require_main_thread("get_dof_velocities")
        return _row(self._robot.get_dof_velocities(), self._num_dofs, "dof velocities")

    def get_dof_efforts(self) -> np.ndarray:
        self._require_open("get_dof_efforts")
        _require_main_thread("get_dof_efforts")
        if self._effort_getter is None:
            raise RuntimeError(
                "this Isaac build exposes none of "
                f"{list(EFFORT_GETTER_CANDIDATES)} on Articulation. Effort readback is "
                "diagnostic only — nothing in the G1Transport low-state contract needs it; "
                "check scripts/preflight_isaac.py's report for the name it observed"
            )
        return _row(
            getattr(self._robot, self._effort_getter)(), self._num_dofs, "dof efforts"
        )

    def set_dof_position_targets(self, targets: np.ndarray) -> None:
        self._require_open("set_dof_position_targets")
        _require_main_thread("set_dof_position_targets")
        self._robot.set_dof_position_targets(_batch(targets, self._num_dofs, "targets"))

    def set_dof_gains(self, stiffnesses: np.ndarray, dampings: np.ndarray) -> None:
        """Write the caller's gains verbatim. ``update_default_gains=False`` on purpose: the
        e-stop's ``kp = 0`` must not become this articulation's new default."""
        self._require_open("set_dof_gains")
        _require_main_thread("set_dof_gains")
        kp = _batch(stiffnesses, self._num_dofs, "stiffnesses")
        kd = _batch(dampings, self._num_dofs, "dampings")
        if (kp < 0.0).any() or (kd < 0.0).any():
            raise ValueError("kp/kd entries must be >= 0 (negative gains are unstable)")
        self._robot.set_dof_gains(stiffnesses=kp, dampings=kd, update_default_gains=False)

    def get_dof_gains(self) -> tuple[np.ndarray, np.ndarray]:
        self._require_open("get_dof_gains")
        _require_main_thread("get_dof_gains")
        kp, kd = self._robot.get_dof_gains()
        return (
            _row(kp, self._num_dofs, "stiffnesses"),
            _row(kd, self._num_dofs, "dampings"),
        )

    # -- rendering -------------------------------------------------------------------------

    def _setup_cameras(self) -> None:
        """One replicator render product + ``rgb`` annotator per camera.

        ``set_capture_on_play(False)`` is what keeps the renderer OUT of the physics loop:
        with capture-on-play the orchestrator drives its own frame cadence and the "render
        never steps physics" guarantee stops being ours to make.
        """
        import omni.replicator.core as rep

        rep.orchestrator.set_capture_on_play(False)
        height, width = self._render_hw
        for name, prim in self._cameras.items():
            product = rep.create.render_product(str(prim), resolution=(width, height))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach(product)
            self._render_products[name] = product
            self._annotators[name] = annotator

    def render_frame(self, camera: str) -> np.ndarray | None:
        """One rendered frame, or ``None`` while the renderer warms up.

        ``RenderingManager.render()`` is documented to render WITHOUT advancing physics (it
        flips ``/app/player/playSimulations`` off, updates once, restores it) — preflight
        check I is the proof on this build. The annotator hands back RGBA; the alpha channel
        is dropped here so no caller has to know.
        """
        self._require_open("render_frame")
        _require_main_thread("render_frame")
        if camera not in self._annotators:
            raise ValueError(f"unknown camera {camera!r}; have {list(self._annotators)}")
        self._rendering.render()
        data = self._annotators[camera].get_data()
        if data is None or getattr(data, "size", 0) == 0:
            return None  # warmup: the caller retries; recording a black frame is worse
        frame = np.asarray(_to_numpy(data))
        if frame.ndim != 3 or frame.shape[2] not in (3, 4):
            raise RuntimeError(
                f"camera {camera!r}: expected (H, W, 3|4) from the rgb annotator, got "
                f"{frame.shape}"
            )
        return np.ascontiguousarray(frame[:, :, :3])

    # -- callbacks -------------------------------------------------------------------------

    def _dispatch_pre_physics(self, *args: Any, **kwargs: Any) -> None:
        """Fan out to the registered callbacks on the main thread.

        ``*args``/``**kwargs`` are swallowed: whether Isaac passes the step size (as
        ``SimulationContext.add_physics_callback`` does) or nothing at all to a
        ``PHYSICS_PRE_STEP`` subscriber is UNVERIFIED, and this dispatcher must survive
        either. ``scripts/preflight_isaac.py`` records what it was actually called with.
        """
        for callback in tuple(self._pre_physics):
            callback()

    def register_pre_physics_callback(self, callback: Callable[[], None]) -> None:
        self._require_open("register_pre_physics_callback")
        self._pre_physics.append(callback)

    # -- lifecycle -------------------------------------------------------------------------

    def _require_open(self, op: str) -> None:
        if self._closed:
            raise RuntimeError(f"{op}: this IsaacSimBinding is closed")

    def close(self) -> None:
        """Detach the annotators, drop the callback and close the app. Idempotent.

        Every teardown step is individually guarded: ``close()`` also runs on the failure
        path of ``__init__``, where half of these attributes may not exist yet, and an
        exception there would replace the real construction error with a noisy AttributeError.
        """
        if getattr(self, "_closed", False):
            return
        self._closed = True
        for name, annotator in getattr(self, "_annotators", {}).items():
            product = self._render_products.get(name)
            try:
                annotator.detach([product] if product is not None else None)
            except Exception:  # noqa: BLE001, S110 - must not mask the original failure
                pass
        handle = getattr(self, "_callback_handle", None)
        if handle is not None:
            try:
                self._sim.deregister_callback(handle)
            except Exception:  # noqa: BLE001, S110 - see above; app.close() is what matters
                pass
        app = getattr(self, "_app", None)
        if app is not None:
            app.close()


# -- the fake (pure python, fully executed) ------------------------------------------------


def fake_g1_dof_names(
    body_pattern: str = BODY_NAME_CANDIDATES[0],
    finger_pattern: str = FINGER_NAME_CANDIDATES[0],
    *, scramble: bool = True,
) -> tuple[str, ...]:
    """43 plausible G1 DOF names, deliberately NOT in canonical order.

    PhysX walks the articulation breadth-first from the base link, so the real order is
    neither the URDF's nor :data:`~wam.robot.g1_transport.G1_MOTOR_JOINT_NAMES`'. The default
    ``scramble`` applies a fixed even-then-odd permutation for exactly that reason: any code
    that indexes positionally instead of resolving by name has to break against this fake, on
    a laptop, instead of on a robot. Deterministic — no RNG anywhere in this module.
    """
    names = [body_pattern.format(name=n) for n in G1_MOTOR_JOINT_NAMES]
    for side in _HAND_SIDES:
        names += [finger_pattern.format(side=side, finger=f) for f in DEX3_FINGER_JOINTS]
    if not scramble:
        return tuple(names)
    order = list(range(0, len(names), 2)) + list(range(1, len(names), 2))
    return tuple(names[i] for i in order)


class FakeIsaacBinding:
    """:class:`IsaacBinding` in pure Python + numpy. No Isaac, no GPU, no threads.

    This is the deliverable that makes the Isaac backend verifiable at all: the transport's
    tick/staleness contract, the e-stop latch and drain, the gain round-trip and the
    name-resolution path all run against this on a laptop. It is a plausible integrator, not
    a physics engine — it exists to be RIGHT ABOUT THE CONTRACT, not about the dynamics.

    Contract fidelity, in the order it matters:

    - **The tick is an ``int`` and advances by exactly 1 per physics step**, never on a
      render, never on a read, never on a gain write. That equality is the whole staleness
      signal upstream.
    - **Pre-physics callbacks fire once per step, before the integration**, in registration
      order — the e-stop drain point.
    - **Gains round-trip verbatim**, including ``kp = 0``.
    - **Positions respond to targets** through ``kp``/``kd``: with the default zero gains
      nothing moves, which is also what a real articulation does.
    - **The main-thread rule is enforced**, like the real binding. This is not pedantry: it
      is what proves the transport's ``emergency_damp()`` never touches the simulator from
      the watchdog thread it is called on. Pass ``enforce_main_thread=False`` only for a test
      that deliberately drives the whole binding from a worker.
    - **Frames are deterministic and non-blank**, and depend on the tick and the joint state,
      so "the frame changed after execute()" is a real assertion.

    Failure modes it can simulate, because they are the ones that bite:

    - ``wedge_main_thread()`` — ``step()`` becomes a no-op: no tick, no callbacks, no
      integration. This is the e-stop's real limitation (module docstring), reproduced.
    - ``warmup_frames=N`` — the first N ``render_frame`` calls return ``None``, like a
      renderer that has not settled.
    - ``dof_names=...`` — hand it a broken or differently-named articulation and watch
      :func:`resolve_g1_dof_indices` refuse.
    """

    def __init__(
        self,
        *,
        dof_names: Sequence[str] | None = None,
        physics_dt: float = 1.0 / 500.0,
        cameras: Sequence[str] = ("persp",),
        render_hw: tuple[int, int] = (64, 64),
        warmup_frames: int = 0,
        enforce_main_thread: bool = True,
        initial_positions: np.ndarray | None = None,
    ) -> None:
        if physics_dt <= 0.0:
            raise ValueError(f"physics_dt must be > 0, got {physics_dt}")
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")
        if len(render_hw) != 2 or any(int(v) < 1 for v in render_hw):
            raise ValueError(f"render_hw must be two positive ints, got {render_hw!r}")
        self._dof_names = tuple(fake_g1_dof_names() if dof_names is None else dof_names)
        self._num_dofs = len(self._dof_names)
        self._physics_dt = float(physics_dt)
        self._cameras = tuple(cameras)
        self._render_hw = (int(render_hw[0]), int(render_hw[1]))
        self._warmup_frames = int(warmup_frames)
        self._enforce_main_thread = bool(enforce_main_thread)
        self._indices: G1DofIndices | None = None

        n = self._num_dofs
        self._q = np.zeros(n, dtype=np.float64)
        if initial_positions is not None:
            q0 = np.asarray(initial_positions, dtype=np.float64)
            if q0.shape != (n,):
                raise ValueError(f"initial_positions: expected ({n},), got {q0.shape}")
            self._q = q0.copy()
        self._q0 = self._q.copy()
        self._dq = np.zeros(n, dtype=np.float64)
        self._targets = self._q.copy()
        self._kp = np.zeros(n, dtype=np.float64)
        self._kd = np.zeros(n, dtype=np.float64)

        self._steps = 0
        self._closed = False
        self._wedged = False
        self._pre_physics: list[Callable[[], None]] = []

        #: Diagnostics, mirroring ``FakeG1Transport``'s recording style.
        self.step_calls: int = 0
        self.render_calls: int = 0
        self.callback_invocations: int = 0
        self.target_writes: list[np.ndarray] = []
        self.gain_writes: list[tuple[np.ndarray, np.ndarray]] = []

    # -- test hooks ------------------------------------------------------------------------

    def wedge_main_thread(self) -> None:
        """Simulate a wedged main loop: ``step()`` stops doing anything at all.

        The tick freezes, callbacks never drain, physics never integrates. This is what an
        e-stop looks like when the main thread is blocked — on hardware
        ``DdsG1Transport.emergency_damp()`` would still have put damping on the wire.
        """
        self._wedged = True

    def release_main_thread(self) -> None:
        """Undo :meth:`wedge_main_thread`."""
        self._wedged = False

    @property
    def is_wedged(self) -> bool:
        return self._wedged

    @property
    def is_closed(self) -> bool:
        return self._closed

    # -- IsaacBinding ----------------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self._dof_names

    @property
    def dof_indices(self) -> G1DofIndices:
        """Resolved lazily, so a fake built with deliberately broken names is still
        constructible — the point of such a fake is to make the RESOLUTION fail, loudly, at
        the moment a caller asks for the mapping."""
        if self._indices is None:
            self._indices = resolve_g1_dof_indices(self._dof_names)
        return self._indices

    @property
    def physics_dt(self) -> float:
        return self._physics_dt

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._cameras

    def get_physics_step_count(self) -> int:
        self._require_open("get_physics_step_count")
        return self._steps

    def step(self, steps: int = 1) -> None:
        """Advance ``steps`` physics steps: callbacks first, then one integration each.

        A wedged fake returns immediately — no tick, no callbacks, no integration.
        """
        self._require_open("step")
        self._require_main_thread("step")
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")
        self.step_calls += 1
        if self._wedged:
            return
        for _ in range(int(steps)):
            for callback in tuple(self._pre_physics):
                self.callback_invocations += 1
                callback()
            self._integrate()
            self._steps += 1

    def _integrate(self) -> None:
        """One semi-implicit PD step with unit inertia.

        ``dq <- (dq + dt * kp * (target - q)) / (1 + dt * kd)`` then ``q <- q + dt * dq``.
        Implicit in the damping term so ``kp = 0, kd = large`` (the e-stop mode) decays
        instead of ringing, whatever gain the caller picks. It is a caricature of an
        articulation, and deliberately so: this fake's job is the CONTRACT.
        """
        dt = self._physics_dt
        self._dq = (self._dq + dt * self._kp * (self._targets - self._q)) / (1.0 + dt * self._kd)
        self._q = self._q + dt * self._dq

    def reset(self) -> None:
        """Restore the initial pose and clear the command state. Does NOT rewind the tick —
        ``get_num_physics_steps`` is a raw counter on the real binding too."""
        self._require_open("reset")
        self._require_main_thread("reset")
        self._q = self._q0.copy()
        self._dq = np.zeros(self._num_dofs, dtype=np.float64)
        self._targets = self._q0.copy()

    def get_dof_positions(self) -> np.ndarray:
        self._require_open("get_dof_positions")
        self._require_main_thread("get_dof_positions")
        return self._q.astype(np.float32)

    def get_dof_velocities(self) -> np.ndarray:
        self._require_open("get_dof_velocities")
        self._require_main_thread("get_dof_velocities")
        return self._dq.astype(np.float32)

    def get_dof_efforts(self) -> np.ndarray:
        self._require_open("get_dof_efforts")
        self._require_main_thread("get_dof_efforts")
        effort = self._kp * (self._targets - self._q) - self._kd * self._dq
        return effort.astype(np.float32)

    def set_dof_position_targets(self, targets: np.ndarray) -> None:
        self._require_open("set_dof_position_targets")
        self._require_main_thread("set_dof_position_targets")
        batch = _batch(targets, self._num_dofs, "targets")
        self._targets = batch[0].astype(np.float64)
        self.target_writes.append(batch[0].copy())

    def set_dof_gains(self, stiffnesses: np.ndarray, dampings: np.ndarray) -> None:
        self._require_open("set_dof_gains")
        self._require_main_thread("set_dof_gains")
        kp = _batch(stiffnesses, self._num_dofs, "stiffnesses")
        kd = _batch(dampings, self._num_dofs, "dampings")
        if (kp < 0.0).any() or (kd < 0.0).any():
            raise ValueError("kp/kd entries must be >= 0 (negative gains are unstable)")
        self._kp = kp[0].astype(np.float64)
        self._kd = kd[0].astype(np.float64)
        self.gain_writes.append((kp[0].copy(), kd[0].copy()))

    def get_dof_gains(self) -> tuple[np.ndarray, np.ndarray]:
        self._require_open("get_dof_gains")
        self._require_main_thread("get_dof_gains")
        return self._kp.astype(np.float32), self._kd.astype(np.float32)

    def render_frame(self, camera: str) -> np.ndarray | None:
        """A deterministic synthetic frame, or ``None`` for the first ``warmup_frames`` calls.

        Never advances the tick — same guarantee ``RenderingManager.render()`` is documented
        to give, which is the one the whole render path rests on.
        """
        self._require_open("render_frame")
        self._require_main_thread("render_frame")
        if camera not in self._cameras:
            raise ValueError(f"unknown camera {camera!r}; have {list(self._cameras)}")
        self.render_calls += 1
        if self.render_calls <= self._warmup_frames:
            return None
        return self._synthetic_frame(self._cameras.index(camera))

    def _synthetic_frame(self, camera_index: int) -> np.ndarray:
        """uint8 (H, W, 3) with real pixel variance, derived from the tick and the pose.

        No RNG (the repo's determinism convention), and it MOVES: a frame that ignored the
        state would make "the image changed after execute()" an assertion that cannot fail.
        """
        height, width = self._render_hw
        total = float(np.sum(np.abs(self._q)))
        # A diverged integrator (someone's kp * dt^2 > 4) must not turn a frame into an
        # OverflowError — it should still render, and the divergence should show up where it
        # belongs, in the joint readback.
        pose = int(total * 1000.0) % 256 if np.isfinite(total) else 0
        phase = (self._steps * 3 + camera_index * 47 + pose) % 256
        yy, xx = np.mgrid[0:height, 0:width]
        red = ((xx * 255) // max(width - 1, 1) + phase) % 256
        green = ((yy * 255) // max(height - 1, 1) + phase) % 256
        blue = red ^ green
        return np.stack([red, green, blue], axis=-1).astype(np.uint8)

    def register_pre_physics_callback(self, callback: Callable[[], None]) -> None:
        self._require_open("register_pre_physics_callback")
        self._pre_physics.append(callback)

    def close(self) -> None:
        self._closed = True

    # -- internals -------------------------------------------------------------------------

    def _require_open(self, op: str) -> None:
        if self._closed:
            raise RuntimeError(f"{op}: this FakeIsaacBinding is closed")

    def _require_main_thread(self, op: str) -> None:
        if self._enforce_main_thread:
            _require_main_thread(op)
