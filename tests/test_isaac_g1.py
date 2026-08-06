"""Tests for the Isaac Sim G1 backend (FR-06, E2).

Every test here runs on a Mac with no Isaac Sim, no GPU and no CUDA, against
:class:`~wam.robot.isaac_binding.FakeIsaacBinding`. That is not a compromise, it is the point:
``IsaacSimBinding`` was written against NVIDIA's documentation and cannot be executed here
(``scripts/preflight_isaac.py`` is what tests it, on the box), so everything ABOVE that seam —
``IsaacG1Transport``, ``IsaacG1Robot`` and the unmodified ``G1Adapter`` driving them — has to
be verifiable without it, or the Isaac backend would be untested code all the way down.

What is under test, in the order it matters:

1. **The tick / staleness contract**, including that ``G1Adapter`` decides staleness by
   EQUALITY and not by "did it go forward" — a tick that jumps BACKWARD is a fresh sample.
2. **The e-stop**, including the two ways it is NOT at parity with hardware: the latch is set
   in pure Python from any thread and the damping only reaches the simulator when the main
   thread next steps — never, if the main loop is wedged. Those tests assert the DOCUMENTED
   behaviour, not the behaviour anyone would prefer.
3. **The 43-DoF resolution**, which is the failure this seam fears most: ``G1Adapter`` gathers
   by hard-coded index, so one permuted entry moves a physical arm silently.
4. **The torch-free promise**, in a subprocess, because the Isaac venv cannot contain this
   repo's torch.

Assertion style: on the CONTRACT, never on ``FakeIsaacBinding``'s integrator constants. The
integrator is a caricature and is allowed to change; "the tick advanced by exactly 10 per
motor command" is not.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wam.interfaces.protocols import RobotAdapter
from wam.interfaces.schema import ActionChunk, ActionMode
from wam.robot.g1 import G1_JOINT_MAP, G1_NUM_MOTORS, G1_SPEC, G1Adapter, G1Config
from wam.robot.g1_transport import DEX3_FINGER_JOINTS, G1_MOTOR_JOINT_NAMES, G1Transport
from wam.robot.isaac_binding import EXPECTED_NUM_DOFS, FakeIsaacBinding, fake_g1_dof_names
from wam.robot.isaac_g1 import IsaacG1Robot
from wam.robot.isaac_transport import (
    DEX3_CLOSED_POSE,
    DEX3_OPEN_POSE,
    ISAAC_IMU_STANDIN,
    IsaacG1Transport,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

#: The MuJoCo Menagerie file DEX3_CLOSED_POSE was read off. Fetched, never committed.
_MENAGERIE_G1 = _REPO_ROOT / "assets/mujoco/unitree_g1/g1_with_hands.xml"

CONTROL_DT_S = 0.02
PHYSICS_DT = 1.0 / 500.0
SUBSTEPS = 10
TICK_PER_CONTROL_NS = 20_000_000

CANON_IDX = {name: i for i, (name, _) in enumerate(G1_JOINT_MAP)}
ELBOW = CANON_IDX["left_elbow"]


# -- fixtures / helpers ---------------------------------------------------------------------


def make_binding(**kwargs: Any) -> FakeIsaacBinding:
    return FakeIsaacBinding(**kwargs)


def make_transport(binding: FakeIsaacBinding | None = None, **kwargs: Any) -> IsaacG1Transport:
    """A transport with finger gains, since the fake articulation ships none.

    ``hand_gains`` is passed explicitly rather than left to the asset default because
    ``FakeIsaacBinding`` starts with zero gains everywhere — the fingers would not move and
    every gripper assertion below would be vacuous.
    """
    kwargs.setdefault("hand_gains", (400.0, 20.0))
    return IsaacG1Transport(binding if binding is not None else make_binding(), **kwargs)


@pytest.fixture
def transport() -> IsaacG1Transport:
    return make_transport()


def motor_cmd(
    transport: IsaacG1Transport, q: np.ndarray | None = None, *, kp: float = 400.0
) -> np.ndarray:
    """One well-formed 29-motor command; returns the q_target that was sent."""
    q_target = np.zeros(G1_NUM_MOTORS, dtype=np.float32) if q is None else q
    transport.write_motor_cmd(
        q_target,
        np.zeros(G1_NUM_MOTORS, dtype=np.float32),
        np.full(G1_NUM_MOTORS, kp, dtype=np.float32),
        np.full(G1_NUM_MOTORS, kp / 20.0, dtype=np.float32),
    )
    return q_target


def sim_counts(binding: FakeIsaacBinding) -> tuple[int, int, int, int]:
    """Everything that would change if a command reached the simulator."""
    return (
        len(binding.target_writes),
        len(binding.gain_writes),
        binding.step_calls,
        binding.get_physics_step_count(),
    )


def make_chunk(delta: float, steps: int, *, dt_s: float = CONTROL_DT_S) -> ActionChunk:
    return ActionChunk(
        targets=np.full((steps, G1_SPEC.num_joints), delta, dtype=np.float32),
        gripper_target=np.zeros(steps, dtype=np.float32),
        dt_s=dt_s,
        mode=ActionMode.JOINT_DELTA,
    )


class _TickStub:
    """A minimal :class:`IsaacBinding` whose tick this test controls.

    Only the members ``IsaacG1Transport.__init__`` and the tick path touch are implemented —
    a full fake would hide which of them the transport actually depends on.
    """

    def __init__(self, tick: Any) -> None:
        self._inner = FakeIsaacBinding()
        self.tick = tick

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def num_dofs(self) -> int:
        return self._inner.num_dofs

    @property
    def dof_names(self) -> tuple[str, ...]:
        return self._inner.dof_names

    @property
    def dof_indices(self) -> Any:
        return self._inner.dof_indices

    @property
    def physics_dt(self) -> float:
        return self._inner.physics_dt

    @property
    def camera_names(self) -> tuple[str, ...]:
        return self._inner.camera_names

    def get_physics_step_count(self) -> Any:
        return self.tick


# -- structural promises --------------------------------------------------------------------


def test_the_transport_satisfies_the_g1_transport_protocol(transport: IsaacG1Transport) -> None:
    """Four methods, no more: the seam every other backend implements."""
    assert isinstance(transport, G1Transport)


def test_the_robot_satisfies_robot_adapter_by_composition_not_inheritance() -> None:
    """``IsaacG1Robot`` HOLDS a ``G1Adapter``; a subclass would inherit ``connect()``'s
    DdsG1Transport fallback (a sim robot that can open a DDS socket is a hazard) and would be
    free to override the safety code this class exists to exercise unchanged."""
    robot = IsaacG1Robot(binding=make_binding())
    assert isinstance(robot, RobotAdapter)
    assert not isinstance(robot, G1Adapter)
    assert isinstance(robot.adapter, G1Adapter)


def test_the_registry_lists_isaac_g1_as_optional_and_constructs_it() -> None:
    """``get_robot`` is the single construction entry point (FR-06) — a backend reachable only
    by importing its module directly is a backend ``rollout.py`` cannot reach.

    OPTIONAL, not available: ``available_robots()`` means "constructible in ANY install", and
    this one needs Isaac Sim, which cannot share a venv with this repo's torch. Registering it
    lazily also keeps ``get_robot("mock")`` from importing the Isaac stack on machines that
    will never run it.
    """
    from wam.robot.registry import available_robots, get_robot, optional_robots

    assert "isaac_g1" in optional_robots()
    assert "isaac_g1" not in available_robots()
    robot = get_robot("isaac_g1", binding=make_binding())
    try:
        assert isinstance(robot, IsaacG1Robot)
        assert isinstance(robot, RobotAdapter)
        assert robot.read_state().q.shape == (G1_SPEC.num_joints,)
    finally:
        robot.close()


def test_rollout_in_the_wrong_venv_exits_with_the_topology_message_not_a_traceback() -> None:
    """The most likely first-run mistake, end to end through the actual CLI.

    ``rollout.py --robot isaac_g1`` in the WAM venv is a USAGE error — the operator is one
    paragraph away from the fix — so it must exit 1 with that paragraph, not dump a 20-line
    traceback that buries it. Run as a subprocess because that is the only way to see what an
    operator sees, exit code included.
    """
    result = subprocess.run(
        # --policy dummy so the robot is what fails: the default (checkpoint) refuses earlier,
        # on the missing PolicyContract, and would make this pass for the wrong reason.
        [
            sys.executable,
            str(_REPO_ROOT / "scripts/rollout.py"),
            "--robot",
            "isaac_g1",
            "--policy",
            "dummy",
            "--rollouts",
            "1",
        ],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=300,
        check=False,  # a non-zero exit is the thing under test
    )
    assert result.returncode == 1
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined
    assert "isaac-python" in combined
    assert "--policy remote" in combined


def test_the_robot_is_constructed_connected() -> None:
    """``get_robot(...)`` must hand back a readable robot, like ``MockRobot`` — no SDK, no
    socket, no hardware, and no ``connect()`` dance for the caller."""
    robot = IsaacG1Robot(binding=make_binding())
    assert robot.adapter.is_connected
    assert robot.read_state().q.shape == (G1_SPEC.num_joints,)


# -- the 43-DoF resolution: the failure this seam fears most --------------------------------


def test_every_motor_slot_reads_the_dof_that_carries_its_name() -> None:
    """The one assertion between a permuted map and a physical arm going somewhere nobody
    asked for.

    The fake's DOF order is deliberately scrambled (PhysX walks the articulation breadth-first
    from the base link), and each DOF is seeded with its own index as its position — so a
    transport that indexed positionally instead of by name reads back the wrong numbers here.
    """
    positions = np.arange(EXPECTED_NUM_DOFS, dtype=np.float64)
    binding = make_binding(initial_positions=positions)
    names = list(binding.dof_names)
    transport = make_transport(binding)

    q = transport.read_low_state()["q"]
    assert q.shape == (G1_NUM_MOTORS,)
    for slot, canonical in enumerate(G1_MOTOR_JOINT_NAMES):
        assert q[slot] == names.index(f"{canonical}_joint")


def test_a_motor_command_is_scattered_onto_the_dof_that_carries_its_name() -> None:
    """The write half of the same property. A permuted scatter is exactly as silent as a
    permuted gather, and it is the half that moves the arm."""
    binding = make_binding()
    names = list(binding.dof_names)
    transport = make_transport(binding)

    q_target = np.arange(G1_NUM_MOTORS, dtype=np.float32) * 0.01
    motor_cmd(transport, q_target)
    written = binding.target_writes[-1]
    for slot, canonical in enumerate(G1_MOTOR_JOINT_NAMES):
        assert written[names.index(f"{canonical}_joint")] == pytest.approx(q_target[slot])


def test_a_renamed_joint_fails_loudly_at_construction() -> None:
    """A resolution failure is a config/data finding, so it must name the joint and dump what
    the articulation actually has — and it must happen before anything is commanded."""
    names = list(fake_g1_dof_names())
    names[names.index("left_elbow_joint")] = "left_elbow_link_joint"
    with pytest.raises(ValueError) as excinfo:
        make_transport(make_binding(dof_names=names))
    message = str(excinfo.value)
    assert "left_elbow_joint" in message
    assert "left_elbow_link_joint" in message


def test_an_articulation_with_the_wrong_dof_count_is_refused() -> None:
    """43 = 29 body + 2 x 7 Dex3 fingers. Isaac Lab's shipped G1 cfg is a legacy 23-DoF model
    and every name would still resolve against a subset — the count is the separate guard."""
    names = list(fake_g1_dof_names())[:-1]
    with pytest.raises(ValueError, match="42 DOFs, expected 43"):
        make_transport(make_binding(dof_names=names))


def test_the_body_and_finger_maps_are_disjoint_and_cover_the_whole_articulation() -> None:
    """Every one of the 43 DoFs is addressed by exactly one of the two maps, checked through
    what the transport actually WRITES rather than through the binding's own bookkeeping.

    Overlap would make a gripper command move an arm joint; a gap would leave a DoF nobody
    ever commands, sitting at whatever the asset left in it.
    """
    binding = make_binding()
    transport = make_transport(binding)
    motor_cmd(transport, np.full(G1_NUM_MOTORS, 0.3, dtype=np.float32))
    transport.write_gripper_cmd(1.0, 1.0)
    written = binding.target_writes[-1]
    assert written.shape == (EXPECTED_NUM_DOFS,)
    assert np.count_nonzero(written) == G1_NUM_MOTORS + 2 * (len(DEX3_FINGER_JOINTS) - 1)


# -- the tick and the staleness contract ------------------------------------------------------


def test_the_tick_advances_by_exactly_one_control_period_per_motor_command(
    transport: IsaacG1Transport,
) -> None:
    """``control_dt_s / physics_dt`` physics steps, not "about that many": the adapter's
    ``dq_max * dt`` clip is a velocity limit only if dt is what we think it is."""
    assert transport.substeps == SUBSTEPS
    before = transport.tick_ns
    motor_cmd(transport)
    assert transport.tick_ns - before == TICK_PER_CONTROL_NS
    motor_cmd(transport)
    assert transport.tick_ns - before == 2 * TICK_PER_CONTROL_NS


def test_a_commands_own_physics_steps_execute_that_commands_target() -> None:
    """The temporal join between the tick tests and the addressing tests, which neither covers.

    Writing the target AFTER stepping — i.e. every command executing one full ``control_dt_s``
    late, always tracking its predecessor — leaves the tick exact, the DOF addressing right,
    the gains right and the readbacks in range. It is invisible to the whole suite, and any
    tracking or latency number measured on the box would then have been measured against a
    build nobody could distinguish from the intended one. So: from rest, ONE command must
    already have moved the arm toward its own target.
    """
    binding = make_binding()
    transport = make_transport(binding)
    body = binding.dof_indices.body_array()
    q_rest = binding.get_dof_positions()[body].copy()
    assert not q_rest.any()

    q_target = np.full(G1_NUM_MOTORS, 0.4, dtype=np.float32)
    motor_cmd(transport, q_target)

    moved = binding.get_dof_positions()[body] - q_rest
    assert moved.min() > 0.0, "the first command stepped physics against the previous target"


def test_neither_a_read_nor_a_gripper_command_advances_the_tick(
    transport: IsaacG1Transport,
) -> None:
    """Only physics moves the clock. A read that ticked would hide a stalled simulation behind
    a healthy-looking staleness check; a gripper command that stepped would double the
    simulated period behind the adapter's back (it issues one of each per step)."""
    motor_cmd(transport)
    before = transport.tick_ns
    transport.read_low_state()
    transport.read_low_state()
    transport.write_gripper_cmd(1.0, 1.0)
    assert transport.tick_ns == before


def test_the_tick_is_an_exact_int_and_a_float_tick_is_refused() -> None:
    """``G1Adapter`` compares ticks for EQUALITY. A float clock would make the watchdog fire at
    random a few million steps in, so it is a TypeError rather than a cast."""
    ok = make_transport(_TickStub(7))
    assert type(ok.tick_ns) is int
    assert ok.tick_ns == 7 * round(PHYSICS_DT * 1e9)

    ok.binding.tick = 7.0
    with pytest.raises(TypeError, match="must be an integer counter"):
        _ = ok.tick_ns
    ok.binding.tick = True
    with pytest.raises(TypeError, match="must be an integer counter"):
        _ = ok.tick_ns


def test_a_numpy_integer_tick_is_accepted() -> None:
    """A pybind counter may surface as ``numpy.int64``, which is not ``int`` but is still
    exact under ``==``. ``scripts/preflight_isaac.py`` applies the same rule (numbers.Integral),
    so the two agree on what "the tick is an integer" means."""
    stub = make_transport(_TickStub(np.int64(11)))
    assert type(stub.tick_ns) is int
    assert stub.tick_ns == 11 * round(PHYSICS_DT * 1e9)


def test_a_second_read_with_no_physics_in_between_is_stale() -> None:
    """The staleness signal the whole runtime rests on: the transport tick is unchanged, so
    every validity flag is cleared and the upstream safety layer rejects the state."""
    transport = make_transport()
    adapter = G1Adapter(G1Config(), transport)
    adapter.connect()
    motor_cmd(transport)

    fresh = adapter.read_state()
    assert (fresh.validity.q, fresh.validity.dq, fresh.validity.imu) == (True, True, True)
    stale = adapter.read_state()
    assert (stale.validity.q, stale.validity.dq, stale.validity.imu) == (False, False, False)
    assert stale.validity.gripper is False

    motor_cmd(transport)
    assert adapter.read_state().validity.q is True


def test_staleness_is_equality_not_direction_so_a_rewound_tick_reads_fresh() -> None:
    """EQUALITY, deliberately, and this is the test that pins it down.

    ``G1Adapter.read_state`` asks "is this the same sample I already had", not "did the clock
    move forward". A direction test (``tick <= last``) would mark a rewound clock stale — and a
    rewound clock is exactly what an episode reset or a reconnect produces, i.e. a genuinely
    NEW sample. ``forget_tick()`` exists for the case where the answer really is "no new
    sample", and nothing else may pretend to know.
    """
    stub = _TickStub(100)
    transport = make_transport(stub)
    adapter = G1Adapter(G1Config(), transport)
    adapter.connect()

    adapter.read_state()
    stub.tick = 40  # backwards: a different sample, therefore fresh
    assert adapter.read_state().validity.q is True
    stub.tick = 40  # unchanged: the same sample, therefore stale
    assert adapter.read_state().validity.q is False


def test_the_robot_reports_sim_time_from_the_physics_tick() -> None:
    """Recording and replay timestamp on this; it must advance only when physics does."""
    robot = IsaacG1Robot(binding=make_binding())
    assert robot.sim_time_ns == 0
    robot.execute(make_chunk(0.001, 3), 3)
    assert robot.sim_time_ns == 3 * TICK_PER_CONTROL_NS


def test_a_reset_does_not_rewind_the_tick_but_does_clear_the_stale_memory() -> None:
    """``get_num_physics_steps`` is a raw counter on both bindings, so ``reset()`` changes the
    POSE without moving the clock — the one case where "the tick did not advance" no longer
    means "no new sample", and the one case ``forget_tick()`` exists for."""
    robot = IsaacG1Robot(binding=make_binding())
    robot.execute(make_chunk(0.01, 2), 2)
    tick = robot.sim_time_ns
    robot.read_state()

    robot.reset()
    assert robot.sim_time_ns == tick
    assert robot.read_state().validity.q is True


# -- gains -------------------------------------------------------------------------------------


def test_the_callers_gains_reach_the_body_dofs_verbatim() -> None:
    """The caller owns the gains — nothing may silently re-tune them. (This is why the backend
    is raw Isaac Sim and not Isaac Lab, whose DCMotorCfg actuator model computes torque in
    Python and neutralises the sim PD gains.)"""
    binding = make_binding()
    transport = make_transport(binding)
    body = binding.dof_indices.body_array()

    kp = np.arange(G1_NUM_MOTORS, dtype=np.float32)
    kd = kp / 10.0
    transport.write_motor_cmd(
        np.zeros(G1_NUM_MOTORS, dtype=np.float32),
        np.zeros(G1_NUM_MOTORS, dtype=np.float32),
        kp,
        kd,
    )
    got_kp, got_kd = binding.get_dof_gains()
    np.testing.assert_allclose(got_kp[body], kp)
    np.testing.assert_allclose(got_kd[body], kd)


def test_zero_gains_are_written_and_not_floored() -> None:
    """``G1Adapter`` sends kp = kd = 0 for every unmapped motor (legs, waist roll/pitch) so the
    vendor controller keeps authority there. A floor would have this backend fight it."""
    binding = make_binding()
    transport = make_transport(binding)
    body = binding.dof_indices.body_array()
    motor_cmd(transport, kp=0.0)
    got_kp, got_kd = binding.get_dof_gains()
    assert not got_kp[body].any()
    assert not got_kd[body].any()


def test_the_finger_gains_come_from_the_asset_and_survive_a_motor_command() -> None:
    """``set_dof_gains`` writes all 43 DoFs at once, so this transport has to have a number for
    the 14 finger slots. It does not invent one: it reads the articulation's own back at
    construction and never touches them again."""
    binding = make_binding()
    binding.set_dof_gains(
        np.full(EXPECTED_NUM_DOFS, 7.0, dtype=np.float32),
        np.full(EXPECTED_NUM_DOFS, 0.5, dtype=np.float32),
    )
    transport = IsaacG1Transport(binding)  # no hand_gains override: the asset's own
    hand_kp, hand_kd = transport.hand_gains
    np.testing.assert_allclose(hand_kp, 7.0)
    np.testing.assert_allclose(hand_kd, 0.5)

    motor_cmd(transport, kp=300.0)
    fingers = np.concatenate([binding.dof_indices.hand_array(s) for s in ("left", "right")])
    got_kp, got_kd = binding.get_dof_gains()
    np.testing.assert_allclose(got_kp[fingers], 7.0)
    np.testing.assert_allclose(got_kd[fingers], 0.5)


def test_negative_gains_are_refused_by_the_transport_before_anything_is_written() -> None:
    """Refused HERE, at the seam — and the test has to prove it was here.

    ``IsaacBinding.set_dof_gains`` rejects negative gains too, one layer lower. Matching the
    bare message would therefore pass against a transport with NO guard at all, on the
    binding's exception, while the transport had already written the rejected command into its
    shadow target vector. So: match the transport's own prefixed message, and then flush that
    vector with a gripper command to show the rejected targets never got in.
    """
    binding = make_binding()
    transport = make_transport(binding)
    body = binding.dof_indices.body_array()
    motor_cmd(transport, np.arange(G1_NUM_MOTORS, dtype=np.float32) * 0.01)
    good_targets = binding.target_writes[-1][body].copy()
    before = sim_counts(binding)

    with pytest.raises(ValueError, match=r"^write_motor_cmd: kp/kd entries must be >= 0"):
        transport.write_motor_cmd(
            np.full(G1_NUM_MOTORS, 9.0, dtype=np.float32),
            np.zeros(G1_NUM_MOTORS, dtype=np.float32),
            np.full(G1_NUM_MOTORS, -1.0, dtype=np.float32),
            np.ones(G1_NUM_MOTORS, dtype=np.float32),
        )
    assert sim_counts(binding) == before

    transport.write_gripper_cmd(0.0, 0.0)
    np.testing.assert_array_equal(binding.target_writes[-1][body], good_targets)


def test_non_finite_input_is_rejected_at_the_seam() -> None:
    """A NaN reaching PhysX corrupts the articulation for the rest of the episode while the
    readbacks may stay finite — undetectable downstream, so it stops here, and it stops before
    any transport state is mutated.

    Same trap as the negative-gain test: the binding's own ``_batch`` also refuses non-finite
    values, so a bare ``match="non-finite"`` would be satisfied by the layer below while the
    transport quietly stored the NaN. The messages are matched by the name THIS layer gives the
    argument (``q_target``, ``left``), and the command that follows proves nothing was stored:
    with a NaN sitting in the shadow vector, the next legal command would raise too.
    """
    binding = make_binding()
    transport = make_transport(binding)
    before = sim_counts(binding)
    bad = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
    bad[3] = np.nan
    with pytest.raises(ValueError, match=r"^q_target: non-finite"):
        motor_cmd(transport, bad)
    with pytest.raises(ValueError, match=r"^left: non-finite"):
        transport.write_gripper_cmd(np.nan, 0.0)
    assert sim_counts(binding) == before

    motor_cmd(transport)
    assert np.isfinite(binding.target_writes[-1]).all()


def test_a_non_zero_velocity_feed_forward_is_refused_rather_than_dropped() -> None:
    """``IsaacBinding`` has no velocity-target and no effort-write channel, so a velocity
    feed-forward CANNOT be honoured. ``MujocoG1Transport`` applies it as a torque and
    ``DdsG1Transport`` puts it on the wire; here it raises, because silently discarding half a
    PD command is the kind of difference nobody discovers until a rollout looks wrong."""
    binding = make_binding()
    transport = make_transport(binding)
    dq = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
    dq[ELBOW] = 0.5
    before = sim_counts(binding)
    with pytest.raises(ValueError, match="dq_target must be all zeros"):
        transport.write_motor_cmd(
            np.zeros(G1_NUM_MOTORS, dtype=np.float32),
            dq,
            np.full(G1_NUM_MOTORS, 100.0, dtype=np.float32),
            np.full(G1_NUM_MOTORS, 5.0, dtype=np.float32),
        )
    assert sim_counts(binding) == before
    # G1Adapter always sends zeros, so the runtime never trips this.
    G1Adapter(G1Config(), transport).connect()
    motor_cmd(transport)


# -- the gripper -------------------------------------------------------------------------------


def test_the_gripper_round_trips_through_the_dex3_synergy() -> None:
    """One scalar per hand along a fixed open->closed line, and the readback inverts it. The
    canonical gripper channel is a scalar (OD-01); per-finger control is post-MVP."""
    binding = make_binding()
    transport = make_transport(binding)
    for commanded in (0.0, 0.35, 1.0):
        transport.write_gripper_cmd(commanded, commanded)
        for _ in range(60):  # the fingers move on motor commands, not gripper commands
            motor_cmd(transport)
        measured = transport.read_low_state()["gripper"]
        assert measured == pytest.approx([commanded, commanded], abs=0.02)


def test_the_gripper_readback_does_not_transpose_the_two_hands() -> None:
    """Left reads back as left.

    Every other gripper assertion here commands ``(x, x)``, so a transposed READ — ``out[1-i]``
    in ``_measure_gripper``'s projection loop — passes all of them and the whole suite. The
    write side is already guarded (``..._clipped_to_the_vendor_unit_range`` uses asymmetric
    input); this is the same guard one layer later, on the path
    ``read_low_state()["gripper"]`` -> ``G1Adapter.vendor_to_gripper`` -> the policy's
    observation. A bimanual chunk that closes one hand and opens the other would otherwise
    report the opposite to both the policy and the recorder, in range and without raising.
    """
    binding = make_binding()
    transport = make_transport(binding)
    transport.write_gripper_cmd(1.0, 0.0)
    for _ in range(80):
        motor_cmd(transport)

    measured = transport.read_low_state()["gripper"]
    assert measured[0] == pytest.approx(1.0, abs=0.02)
    assert measured[1] == pytest.approx(0.0, abs=0.02)
    # Anchored on the articulation too, so this cannot be satisfied by a projection that is
    # merely self-consistent: the closed hand's fingers left the open pose, the open one's did
    # not.
    q = binding.get_dof_positions()
    assert np.abs(q[binding.dof_indices.hand_array("left")]).max() > 0.1
    assert np.abs(q[binding.dof_indices.hand_array("right")]).max() < 0.02


def test_commanding_the_gripper_open_leaves_the_fingers_at_the_assets_open_pose() -> None:
    """"0 = open" is a claim about the ARTICULATION, and only this test makes it.

    ``_measure_gripper`` inverts whatever open->closed line it was built from, so the
    round-trip test passes unchanged against a wrong open pose — shifting the runtime
    ``_finger_open`` vector by 0.05 rad on every finger keeps every other assertion green.
    ``DEX3_OPEN_POSE`` the constant is pinned elsewhere; this pins the vector actually built
    from it, read off the DOFs rather than through the projection that would hide the error.
    """
    binding = make_binding()
    transport = make_transport(binding)
    transport.write_gripper_cmd(1.0, 1.0)
    for _ in range(80):
        motor_cmd(transport)
    assert np.abs(binding.get_dof_positions()[binding.dof_indices.hand_array("left")]).max() > 0.1

    transport.write_gripper_cmd(0.0, 0.0)
    for _ in range(80):
        motor_cmd(transport)
    q = binding.get_dof_positions()
    for side in ("left", "right"):
        np.testing.assert_allclose(
            q[binding.dof_indices.hand_array(side)],
            np.asarray(DEX3_OPEN_POSE, dtype=np.float64),
            atol=0.02,
        )


def test_a_gripper_command_only_moves_the_fingers() -> None:
    """The Dex3 slots and the 29 body slots share one 43-vector write; a gripper command must
    not disturb the body targets sitting in it."""
    binding = make_binding()
    transport = make_transport(binding)
    q_target = np.arange(G1_NUM_MOTORS, dtype=np.float32) * 0.01
    motor_cmd(transport, q_target)
    body = binding.dof_indices.body_array()
    after_motor = binding.target_writes[-1][body].copy()

    transport.write_gripper_cmd(1.0, 1.0)
    written = binding.target_writes[-1]
    np.testing.assert_allclose(written[body], after_motor)
    fingers = binding.dof_indices.hand_array("left")
    assert np.abs(written[fingers]).max() > 0.0


def test_the_gripper_command_is_clipped_to_the_vendor_unit_range() -> None:
    binding = make_binding()
    transport = make_transport(binding)
    transport.write_gripper_cmd(5.0, -5.0)
    left = binding.target_writes[-1][binding.dof_indices.hand_array("left")]
    right = binding.target_writes[-1][binding.dof_indices.hand_array("right")]
    np.testing.assert_allclose(left, transport.finger_synergy("left")[1])
    np.testing.assert_allclose(right, transport.finger_synergy("right")[0])


def test_the_two_hands_close_in_mirrored_directions() -> None:
    """The Dex3 sign mirroring is the MODEL's, not a convention imposed here — a table that
    lost it would curl one hand backwards through its own limits."""
    left_closed = np.asarray(DEX3_CLOSED_POSE["left"])
    right_closed = np.asarray(DEX3_CLOSED_POSE["right"])
    np.testing.assert_allclose(left_closed, -right_closed)
    assert np.asarray(DEX3_OPEN_POSE).any() == np.False_


@pytest.mark.skipif(
    not _MENAGERIE_G1.is_file(),
    reason="the vendor G1 model is not fetched (scripts/fetch_g1_model.py)",
)
def test_the_closed_finger_pose_matches_the_model_it_was_read_off() -> None:
    """PROVENANCE, checked rather than asserted in a comment.

    ``DEX3_CLOSED_POSE`` is a hard-coded table because ``IsaacBinding`` exposes no joint-limit
    getter — but the numbers came from somewhere: the joint-range endpoints farther from zero
    in the MuJoCo Menagerie ``g1_29dof_with_hand_rev_1_0``, which is the same kinematic model
    the Isaac USD converts. This re-derives them with plain XML parsing (no mujoco needed) so
    a hand-edited constant cannot quietly stop matching its source.

    It does NOT prove anything about the Isaac USD's limits — nothing here can; that is
    difference (5) in the ``isaac_transport`` module docstring.
    """
    curling = {"thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"}
    ranges = {
        joint.get("name"): tuple(float(v) for v in joint.get("range").split())
        for joint in ET.parse(_MENAGERIE_G1).getroot().iter("joint")
        if joint.get("name") and "_hand_" in joint.get("name") and joint.get("range")
    }
    for side in ("left", "right"):
        expected = []
        for finger in DEX3_FINGER_JOINTS:
            lo, hi = ranges[f"{side}_hand_{finger}_joint"]
            expected.append(0.0 if finger not in curling else (hi if abs(hi) >= abs(lo) else lo))
        np.testing.assert_allclose(DEX3_CLOSED_POSE[side], expected)


# -- the IMU stand-in ----------------------------------------------------------------------------


def test_the_imu_is_a_constant_stand_in_and_says_so() -> None:
    """Not a measurement, and the code must not pretend otherwise. ``IsaacBinding`` exposes
    joint state only — no root pose, no sensors — and the preflight checks no such symbol, so
    adding one here would smuggle an unverified assumption past the gate that exists to catch
    exactly that.

    The uncomfortable half is asserted too: ``G1Adapter`` sets ``validity.imu`` from freshness
    alone, so an Isaac rollout reports imu=True over a constant. That is a documented gap, not
    a bug in this file, and pinning it here is what stops it becoming folklore.
    """
    transport = make_transport()
    adapter = G1Adapter(G1Config(), transport)
    adapter.connect()
    assert transport.imu_is_measured is False

    for _ in range(20):
        motor_cmd(transport, np.full(G1_NUM_MOTORS, 0.3, dtype=np.float32))
    assert np.abs(transport.read_low_state()["dq"]).max() > 0.0  # the robot IS moving

    state = adapter.read_state()
    # Literals, not ISAAC_IMU_STANDIN. Comparing the reading against the constant it is read
    # from is a tautology: it holds for whatever that constant is edited to, so it could not
    # catch a stand-in that stopped being an identity pose at rest under 1 g.
    np.testing.assert_allclose(state.imu.orientation_wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.imu.angular_velocity, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(state.imu.linear_acceleration, [0.0, 0.0, 9.81])
    np.testing.assert_allclose(list(ISAAC_IMU_STANDIN["acc"]), [0.0, 0.0, 9.81])
    assert state.validity.imu is True  # the documented lie, pinned


def test_the_imu_stand_in_can_be_overridden() -> None:
    """The ``imu=`` argument exists and was never exercised — ignoring it entirely (always
    using ``ISAAC_IMU_STANDIN``) passed the suite. It is the seam a future scene that DOES
    carry a root-pose sensor would come in through, so it has to be known to work before
    anyone relies on it, and the payload has to be copied out rather than aliased."""
    tilted = {"quat_wxyz": (0.0, 1.0, 0.0, 0.0), "gyro": (0.1, 0.2, 0.3), "acc": (0.0, 0.0, 9.7)}
    transport = make_transport(imu=tilted)
    imu = transport.read_low_state()["imu"]
    np.testing.assert_allclose(imu["gyro"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(imu["acc"], [0.0, 0.0, 9.7])

    imu["gyro"][:] = 99.0  # a caller mutating what it was handed must not reach the transport
    np.testing.assert_allclose(transport.read_low_state()["imu"]["gyro"], [0.1, 0.2, 0.3])


# -- constructor guards --------------------------------------------------------------------------


def test_a_partial_grasp_closure_scales_the_closed_pose_and_its_range_is_guarded() -> None:
    """``grasp_closure`` shortens the synergy line so a "fully closed" command stops short of
    the joint limits — the knob an operator reaches for when the Dex3 crushes the object. Both
    halves were unexercised: dropping the multiplication and disabling the range check each
    passed the whole suite, which would have left the value silently ignored."""
    full = make_transport().finger_synergy("left")[1]
    half = make_transport(grasp_closure=0.5).finger_synergy("left")[1]
    np.testing.assert_allclose(half, full * 0.5)
    assert np.abs(half).max() > 0.0  # a scaled line is still a line, not a collapsed one

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="grasp_closure"):
            make_transport(grasp_closure=bad)


def test_the_closed_pose_override_is_validated_before_it_can_reach_physx() -> None:
    """``finger_closed`` is operator-supplied geometry. A non-finite entry would propagate
    through the synergy into every finger target as nan; the guard existed and nothing
    exercised it."""
    ok = {side: [0.1] * len(DEX3_FINGER_JOINTS) for side in ("left", "right")}
    np.testing.assert_allclose(make_transport(finger_closed=ok).finger_synergy("right")[1], 0.1)

    bad = {side: [0.1] * len(DEX3_FINGER_JOINTS) for side in ("left", "right")}
    bad["left"] = [float("nan")] + [0.1] * (len(DEX3_FINGER_JOINTS) - 1)
    with pytest.raises(ValueError, match="non-finite"):
        make_transport(finger_closed=bad)
    with pytest.raises(ValueError, match="missing the 'right' hand"):
        make_transport(finger_closed={"left": ok["left"]})


def test_a_negative_damp_gain_is_refused() -> None:
    """``kd < 0`` is not damping, it is a velocity amplifier — the e-stop would accelerate the
    arm. The guard was there and untested."""
    with pytest.raises(ValueError, match="damp_kd"):
        make_transport(damp_kd=-1.0)


def test_hand_gains_reports_left_then_right() -> None:
    """The diagnostic an operator reads to answer "does this USD ship finger gains at all".
    Reversing the two halves passed the suite, and a reversed answer to that question sends
    them looking in the wrong hand."""
    binding = make_binding()
    transport = make_transport(binding, hand_gains=(400.0, 20.0))
    kp, _ = transport.hand_gains
    n = len(DEX3_FINGER_JOINTS)
    assert kp.shape == (2 * n,)
    left, right = binding.dof_indices.hand_array("left"), binding.dof_indices.hand_array("right")
    # Distinguishable only through the DOF indices, since both halves carry the same gain:
    # write a marker into the articulation and check which half of the report moves.
    all_kp, all_kd = binding.get_dof_gains()
    all_kp[left] = 111.0
    all_kp[right] = 222.0
    binding.set_dof_gains(all_kp, all_kd)
    transport._kp_all[left] = 111.0  # mirroring the write the transport caches
    transport._kp_all[right] = 222.0
    kp, _ = transport.hand_gains
    np.testing.assert_allclose(kp[:n], 111.0)
    np.testing.assert_allclose(kp[n:], 222.0)


# -- the e-stop: the latch ---------------------------------------------------------------------


def test_a_main_thread_estop_damps_synchronously_and_lets_the_arm_settle() -> None:
    """On the main thread the Omniverse restriction does not apply, so the damp is immediate
    and the arm actually comes to rest instead of hanging in its last pose — the same shape
    ``MujocoG1Transport.emergency_damp()`` has."""
    binding = make_binding()
    transport = make_transport(binding, damp_kd=50.0)
    body = binding.dof_indices.body_array()

    for _ in range(10):
        motor_cmd(transport, np.full(G1_NUM_MOTORS, 1.0, dtype=np.float32))
    moving = float(np.abs(transport.read_low_state()["dq"]).max())
    assert moving > 0.0

    transport.emergency_damp()
    assert transport.damp_count == 1
    assert transport.damp_applied_count == 1
    assert transport.pending_damp is False
    assert transport.is_damping is True
    got_kp, got_kd = binding.get_dof_gains()
    assert not got_kp[body].any()
    np.testing.assert_allclose(got_kd[body], 50.0)
    assert float(np.abs(transport.read_low_state()["dq"]).max()) < 0.05 * moving


def test_no_motor_command_reaches_the_simulator_after_the_latch() -> None:
    """The property ``G1Adapter.estop()`` depends on, enforced at the transport rather than
    only in the adapter: an e-stop from a watchdog thread can land between the adapter's
    per-step check and this call."""
    binding = make_binding()
    transport = make_transport(binding)
    transport.emergency_damp()
    before = sim_counts(binding)

    motor_cmd(transport, np.full(G1_NUM_MOTORS, 1.0, dtype=np.float32))
    motor_cmd(transport, np.full(G1_NUM_MOTORS, 1.0, dtype=np.float32))
    transport.write_gripper_cmd(1.0, 1.0)

    assert sim_counts(binding) == before
    assert transport.blocked_motor_writes == 2
    assert transport.blocked_gripper_writes == 1


def test_reads_still_work_after_the_latch(transport: IsaacG1Transport) -> None:
    """An e-stop must not blind the operator; reading moves nothing."""
    transport.emergency_damp()
    low = transport.read_low_state()
    assert low["q"].shape == (G1_NUM_MOTORS,)


def test_the_latch_holds_across_a_full_adapter_execute() -> None:
    """End to end: after ``estop()`` the adapter refuses to stream and the transport refuses
    anything that slipped past it, so the joint positions stop changing."""
    binding = make_binding()
    transport = make_transport(binding)
    adapter = G1Adapter(
        G1Config(kp=(400.0,) * 15, kd=(20.0,) * 15),
        transport,
        clock=lambda: transport.tick_ns * 1e-9,
        sleep=transport.advance,
    )
    adapter.connect()
    adapter.execute(make_chunk(0.01, 5), 5)
    adapter.estop()

    q_after_estop = transport.read_low_state()["q"].copy()
    adapter.execute(make_chunk(0.01, 5), 5)
    np.testing.assert_array_equal(transport.read_low_state()["q"], q_after_estop)
    assert adapter.is_estopped


# -- the e-stop: the cross-thread path and its real limits --------------------------------------


def test_an_estop_from_another_thread_never_touches_isaac() -> None:
    """The Omniverse API is main-thread-only. ``FakeIsaacBinding`` enforces that rule for
    exactly this test: if ``emergency_damp()`` reached for the simulator from the watchdog
    thread it is called on, the binding would raise there and this would fail.
    """
    binding = make_binding()
    transport = make_transport(binding)
    before = sim_counts(binding)
    errors: list[BaseException] = []

    def watchdog() -> None:
        try:
            transport.emergency_damp()
        except BaseException as exc:  # noqa: BLE001 - the point is to capture it
            errors.append(exc)

    thread = threading.Thread(target=watchdog, name="watchdog")
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "emergency_damp() must never block a watchdog thread"

    assert errors == []
    assert transport.is_estopped is True
    assert transport.pending_damp is True
    assert transport.damp_applied_count == 0
    assert sim_counts(binding) == before  # nothing reached Isaac from that thread


def test_a_cross_thread_estop_drains_on_the_next_physics_step() -> None:
    """The documented latency floor: the damping lands at the next ``PHYSICS_PRE_STEP``, on the
    main thread, not when ``emergency_damp()`` returns."""
    binding = make_binding()
    transport = make_transport(binding, damp_kd=33.0)
    body = binding.dof_indices.body_array()
    motor_cmd(transport, kp=400.0)

    thread = threading.Thread(target=transport.emergency_damp, name="watchdog")
    thread.start()
    thread.join(timeout=5.0)
    kp_before, _ = binding.get_dof_gains()
    assert kp_before[body].any()  # still the caller's gains: nothing has stepped yet

    binding.step(1)  # the main loop's next step drains it
    got_kp, got_kd = binding.get_dof_gains()
    assert not got_kp[body].any()
    np.testing.assert_allclose(got_kd[body], 33.0)
    assert transport.damp_applied_count == 1
    assert transport.pending_damp is False


def test_a_wedged_main_thread_means_no_damp_at_all() -> None:
    """THE failure mode, asserted as it IS rather than as anyone would like it.

    If the main thread is blocked, deadlocked or simply not stepping, the pending flag is never
    drained and NOTHING happens in the simulator. On hardware
    ``DdsG1Transport.emergency_damp()`` would already have put damping on the DDS wire,
    independently of the control loop's health — that is the entire point of an e-stop, and
    this backend does not have it.

    What DOES survive a wedged main thread is the latch, in pure Python: no further command
    reaches the simulator. That is the half worth having and the half that is asserted.
    """
    binding = make_binding()
    transport = make_transport(binding)
    motor_cmd(transport, kp=400.0)
    body = binding.dof_indices.body_array()
    binding.wedge_main_thread()

    thread = threading.Thread(target=transport.emergency_damp, name="watchdog")
    thread.start()
    thread.join(timeout=5.0)
    assert not thread.is_alive()

    binding.step(500)  # the wedged main loop: no ticks, no callbacks, no integration
    kp_now, _ = binding.get_dof_gains()
    assert kp_now[body].any(), "documented: a wedged main loop never applies the damp"
    assert transport.damp_applied_count == 0
    assert transport.pending_damp is True

    assert transport.is_estopped is True
    before = sim_counts(binding)
    motor_cmd(transport)
    assert sim_counts(binding) == before

    binding.release_main_thread()
    binding.step(1)
    assert transport.damp_applied_count == 1


def test_a_failed_asynchronous_damp_is_recorded_and_retried() -> None:
    """The callback runs inside Isaac's C++ event dispatcher, where a propagating Python
    exception is undocumented behaviour — so a failure there cannot be raised at anyone. It is
    recorded on ``last_damp_error`` (the ONLY signal a harness gets) and retried on the next
    step, because a damp is idempotent and the next step is a free second chance."""
    binding = make_binding()
    transport = make_transport(binding)
    real_set_gains = binding.set_dof_gains
    failures = {"left": 1}

    def flaky(kp: np.ndarray, kd: np.ndarray) -> None:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("gain write refused")
        real_set_gains(kp, kd)

    thread = threading.Thread(target=transport.emergency_damp, name="watchdog")
    thread.start()
    thread.join(timeout=5.0)

    binding.set_dof_gains = flaky  # type: ignore[method-assign]
    binding.step(1)
    assert transport.damp_applied_count == 0
    assert isinstance(transport.last_damp_error, RuntimeError)
    assert transport.pending_damp is True

    binding.step(1)
    assert transport.damp_applied_count == 1
    assert transport.pending_damp is False


def test_last_damp_error_describes_the_LAST_attempt_and_is_not_sticky() -> None:
    """One transient failure must not poison the diagnostic for the rest of the process.

    This attribute is documented twice — the module docstring and ``emergency_damp``'s — as the
    ONLY way a harness can tell a failed asynchronous damp from a successful one, and harnesses
    are told to check it. Left sticky it answers "did a damp ever fail", not "did this one",
    so a harness that follows the instruction reads a stale exception forever while
    ``damp_applied_count`` climbs. ``MujocoG1Transport.emergency_damp`` clears it per attempt;
    this is that parity.
    """
    binding = make_binding()
    transport = make_transport(binding)
    real_set_gains = binding.set_dof_gains
    failures = {"left": 1}

    def flaky(kp: np.ndarray, kd: np.ndarray) -> None:
        if failures["left"] > 0:
            failures["left"] -= 1
            raise RuntimeError("transient")
        real_set_gains(kp, kd)

    binding.set_dof_gains = flaky  # type: ignore[method-assign]
    thread = threading.Thread(target=transport.emergency_damp, name="watchdog")
    thread.start()
    thread.join(timeout=5.0)

    binding.step(1)  # the failing attempt
    assert isinstance(transport.last_damp_error, RuntimeError)

    binding.step(1)  # the retry, which succeeds
    assert transport.damp_applied_count == 1
    assert transport.last_damp_error is None

    transport.clear_estop()
    transport.emergency_damp()  # main thread, synchronous, succeeds
    assert transport.damp_applied_count == 2
    assert transport.last_damp_error is None


def test_clearing_the_estop_discards_a_damp_that_never_drained() -> None:
    """The realistic asynchronous sequence, end to end, and the flag hygiene it needs.

    Watchdog e-stop with nothing stepping -> the damp stays pending and NEVER lands (the latch
    is precisely what stops the control loop calling ``step()``). Operator clears the latch.
    If ``clear_estop`` did not drop the pending flag, the first motor command after the resume
    would step physics, fire the drain and silently convert that command into damping mode —
    kp 0 on all 29 body motors, on a robot the operator has just deliberately re-enabled.
    """
    binding = make_binding()
    transport = make_transport(binding)
    body = binding.dof_indices.body_array()

    thread = threading.Thread(target=transport.emergency_damp, name="watchdog")
    thread.start()
    thread.join(timeout=5.0)
    assert transport.damp_count == 1
    assert transport.damp_applied_count == 0  # nothing stepped, so nothing drained
    assert transport.pending_damp is True

    transport.clear_estop()
    assert transport.pending_damp is False

    motor_cmd(transport, kp=400.0)
    kp_now = binding.get_dof_gains()[0]
    np.testing.assert_allclose(kp_now[body], 400.0)
    assert transport.is_damping is False
    assert transport.damp_applied_count == 0


def test_the_pre_physics_callback_never_raises_into_isaacs_dispatcher() -> None:
    """``_on_pre_physics`` MUST NOT RAISE: on the real binding it runs inside Isaac's C++ event
    dispatcher, where the behaviour of a propagating Python exception is undocumented.

    ``_drain`` already swallows on the asynchronous path, so no failure reachable through the
    normal e-stop path can distinguish the outer guard from nothing at all — which is exactly
    why this test reaches past it and makes the drain itself explode. What it pins is the
    guard, i.e. the part of the callback that survives a refactor moving code out of
    ``_drain``'s own ``try``.
    """
    binding = make_binding()
    transport = make_transport(binding)

    def explode(*, synchronous: bool) -> None:
        raise RuntimeError("drain exploded")

    transport._drain = explode  # type: ignore[method-assign]
    binding.step(1)  # must return normally: a callback that raises into C++ is worse
    assert isinstance(transport.last_damp_error, RuntimeError)
    assert str(transport.last_damp_error) == "drain exploded"


def test_a_failed_synchronous_damp_propagates_and_still_latches() -> None:
    """On the main thread there IS a caller, so a failed damp must not report success — the
    same contract ``MujocoG1Transport`` and ``DdsG1Transport`` hold. ``G1Adapter.estop()``
    latches in a ``finally``, so propagating can never leave the adapter willing to command."""
    binding = make_binding()
    transport = make_transport(binding)

    def refuse(kp: np.ndarray, kd: np.ndarray) -> None:
        raise RuntimeError("gain write refused")

    binding.set_dof_gains = refuse  # type: ignore[method-assign]
    adapter = G1Adapter(G1Config(), transport)
    adapter.connect()
    with pytest.raises(RuntimeError, match="gain write refused"):
        adapter.estop()
    assert adapter.is_estopped is True
    assert transport.is_estopped is True
    assert isinstance(transport.last_damp_error, RuntimeError)


def test_clearing_the_estop_needs_both_latches_and_the_robot_clears_both() -> None:
    """The adapter's latch makes ``execute()`` a no-op; the transport's makes every write a
    no-op. Clearing only one leaves a robot that accepts commands and silently drops them —
    which is why ``IsaacG1Robot.clear_estop()`` clears both."""
    robot = IsaacG1Robot(
        config=G1Config(kp=(400.0,) * 15, kd=(20.0,) * 15), binding=make_binding()
    )
    robot.estop()
    assert robot.is_estopped and robot.transport.is_estopped

    robot.adapter.clear_estop()  # only half the release
    q_before = robot.transport.read_low_state()["q"].copy()
    robot.execute(make_chunk(0.02, 5), 5)
    np.testing.assert_array_equal(robot.transport.read_low_state()["q"], q_before)

    robot.clear_estop()  # both
    assert robot.transport.is_estopped is False
    robot.execute(make_chunk(0.02, 5), 5)
    assert not np.array_equal(robot.transport.read_low_state()["q"], q_before)


def test_the_first_command_after_a_cleared_estop_restores_the_callers_gains() -> None:
    """The damping gains stay in force until then — same as ``MujocoG1Transport``, so an
    operator who clears the latch does not silently get a stiff arm back before commanding
    one."""
    binding = make_binding()
    transport = make_transport(binding, damp_kd=40.0)
    body = binding.dof_indices.body_array()
    transport.emergency_damp()
    transport.clear_estop()

    got_kp, got_kd = binding.get_dof_gains()
    assert not got_kp[body].any()
    np.testing.assert_allclose(got_kd[body], 40.0)

    motor_cmd(transport, kp=200.0)
    got_kp, got_kd = binding.get_dof_gains()
    np.testing.assert_allclose(got_kp[body], 200.0)
    np.testing.assert_allclose(got_kd[body], 10.0)
    assert transport.is_damping is False


def test_reset_restores_the_assets_gains_and_rebases_the_targets() -> None:
    """A fresh episode must not inherit the previous one's gains — nor a damping override.

    The targets are re-based on the MEASURED post-reset pose for the same reason
    ``G1Adapter.forget_command()`` exists: a target carried across a teleport describes a robot
    that is no longer there, and the first command afterwards would yank toward it.
    """
    binding = make_binding()
    transport = make_transport(binding)
    body = binding.dof_indices.body_array()
    for _ in range(20):
        motor_cmd(transport, np.full(G1_NUM_MOTORS, 0.5, dtype=np.float32), kp=400.0)
    assert binding.get_dof_gains()[0][body].any()
    assert transport.read_low_state()["q"].any()

    transport.reset()
    assert not binding.get_dof_gains()[0][body].any()  # back to the asset's own (zero here)
    np.testing.assert_allclose(
        binding.target_writes[-1], binding.get_dof_positions(), atol=1e-6
    )
    assert not transport.read_low_state()["q"].any()


def test_a_reset_drops_the_carried_command_and_not_only_the_stale_tick() -> None:
    """``IsaacG1Robot.reset()`` calls ``forget_tick()`` AND ``forget_command()``. Only the
    first is otherwise tested, and dropping the second is inert TODAY for one reason: both
    ``G1Config`` and ``configs/robot/isaac_g1.yaml`` ship ``q_track_window = 0``, where
    ``_carry_in`` collapses to the measured q and ``forget_command()`` is a no-op.

    ``isaac_g1.yaml`` explicitly instructs the operator to raise that window once they have
    measured tracking on the box. At that moment the untested branch goes live: the reset
    teleports the articulation, the carried commanded target survives it, and the first chunk
    afterwards commands a jump back toward the pre-reset pose — bounded by the window, and
    unasked for. So the window is turned ON here, which is the configuration this is written
    for, not the shipped one.
    """
    window = 0.3
    config = G1Config(q_track_window=tuple([window] * G1_SPEC.num_joints))
    binding = make_binding()
    robot = IsaacG1Robot(config=config, binding=binding)
    body = binding.dof_indices.body_array()

    robot.execute(make_chunk(0.03, 20), 20)
    carried = binding.target_writes[-1][body].copy()
    assert np.abs(carried).max() > window, "the carry has to exceed the window to be visible"

    robot.reset()
    q_after_reset = binding.get_dof_positions()[body].copy()
    assert not q_after_reset.any()  # the articulation teleported home

    robot.execute(make_chunk(0.0, 1), 1)  # a pure hold: it must hold where the robot IS
    commanded = binding.target_writes[-1][body]
    assert np.abs(commanded - q_after_reset).max() < 0.05, (
        "the first command after a reset yanked toward the pre-reset target"
    )


def test_a_latched_estop_survives_an_episode_reset() -> None:
    """``reset()`` is an EPISODE reset, not an e-stop release — mirroring ``MujocoG1Robot``."""
    robot = IsaacG1Robot(binding=make_binding())
    robot.estop()
    robot.reset()
    assert robot.is_estopped is True
    assert robot.transport.is_estopped is True


# -- sim-time pacing ------------------------------------------------------------------------------


def test_pacing_runs_on_sim_time_and_a_longer_chunk_period_steps_extra_physics() -> None:
    """``G1Adapter`` paces step ``i`` to ``t0 + i * dt_s``, which is what makes its
    ``dq_max * dt`` clip a real velocity limit. The injected clock is the physics tick and the
    injected sleep STEPS PHYSICS instead of blocking, so a chunk whose ``dt_s`` is two control
    periods advances two per step (one commanded, one holding).
    """
    robot = IsaacG1Robot(binding=make_binding())
    robot.execute(make_chunk(0.001, 3, dt_s=2 * CONTROL_DT_S), 3)
    # 3 commands + 2 hold gaps of one control period each.
    assert robot.sim_time_ns == 5 * TICK_PER_CONTROL_NS


# -- rendering --------------------------------------------------------------------------------------


def test_render_frames_returns_identical_frames_and_never_steps_physics() -> None:
    """The adapter owns the clock. A render that stepped behind its back would corrupt
    staleness detection AND silently widen the ``dq_max * dt`` clip."""
    robot = IsaacG1Robot(binding=make_binding(render_hw=(32, 48)))
    robot.execute(make_chunk(0.01, 2), 2)
    tick = robot.sim_time_ns

    frames = robot.render_frames(3)
    assert set(frames) == {"persp"}
    assert frames["persp"].shape == (3, 32, 48, 3)
    assert frames["persp"].dtype == np.uint8
    np.testing.assert_array_equal(frames["persp"][0], frames["persp"][2])
    assert robot.sim_time_ns == tick


def test_a_warmup_frame_is_retried_and_never_substituted_by_a_black_one() -> None:
    """Isaac's rgb annotator returns ``None`` for the first frames — up to 20 in NVIDIA's own
    test. A black frame passes the T-11 data-quality gates and poisons training silently, which
    is strictly worse than a crash, so this retries and then raises."""
    robot = IsaacG1Robot(binding=make_binding(warmup_frames=3), render_warmup_ticks=10)
    assert robot.render_frames(1)["persp"].shape[0] == 1

    stubborn = IsaacG1Robot(binding=make_binding(warmup_frames=50), render_warmup_ticks=4)
    with pytest.raises(RuntimeError, match="returned no frame after 4 render ticks"):
        stubborn.render_frames(1)


def test_an_unknown_camera_is_refused_at_construction() -> None:
    """A renamed camera must fail when the robot is built, not halfway through a rollout."""
    with pytest.raises(ValueError, match="is not among the binding's cameras"):
        IsaacG1Robot(binding=make_binding(), cameras=("head",))


# -- construction guards ------------------------------------------------------------------------------


def test_a_control_period_that_is_not_a_multiple_of_the_physics_step_is_refused() -> None:
    """Otherwise ``tick_ns`` would drift off the step grid and one control period would stop
    meaning one control period."""
    with pytest.raises(ValueError, match="not an integer multiple"):
        make_transport(control_dt_s=0.025)


def test_a_non_unit_gripper_range_is_refused() -> None:
    """The transport's gripper channel IS the Dex3 synergy fraction, so canonical and vendor
    units coincide. Any other range would command a fully-closed hand for every input above
    ~0.01, with no error at any layer."""
    with pytest.raises(ValueError, match=r"must be \(0.0, 1.0\)"):
        IsaacG1Robot(config=G1Config(gripper_vendor_max=100.0), binding=make_binding())


def test_asset_and_scene_path_are_the_same_knob_and_may_not_both_be_given() -> None:
    with pytest.raises(ValueError, match="either asset= or scene_path="):
        IsaacG1Robot(binding=make_binding(), asset="a.usd", scene_path="b.usd")


def test_a_transport_whose_control_period_disagrees_with_the_config_is_refused() -> None:
    """Sim-time pacing needs them equal; a mismatch would silently play the trajectory back at
    the wrong speed."""
    transport = make_transport(control_dt_s=0.04)
    with pytest.raises(ValueError, match="sim-time pacing needs them equal"):
        IsaacG1Robot(config=G1Config(control_dt_s=0.02), transport=transport)


def test_closing_the_robot_shuts_the_binding_down() -> None:
    """A leaked ``SimulationApp`` wedges the interpreter, so this is not housekeeping."""
    binding = make_binding()
    robot = IsaacG1Robot(binding=binding)
    robot.close()
    robot.close()
    assert binding.is_closed


# -- the torch-free promise -----------------------------------------------------------------------------


def test_the_isaac_modules_import_without_isaac_sim_and_without_torch() -> None:
    """The Isaac venv cannot contain this repo's torch (``isaacsim-core`` 6.0.1 pins 2.11.0,
    ``uv.lock`` resolves 2.13.0), so the whole Isaac side of the two-venv split must be
    torch-free — transport, binding and robot alike.

    Checked in a SUBPROCESS: this pytest process has torch imported by other test modules, so
    an in-process ``'torch' not in sys.modules`` would be testing the test runner. And this
    machine has no Isaac Sim at all, so the imports succeeding here is itself the proof that
    nothing at module scope reaches for the vendor stack.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed, so this check could not fail")
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from wam.robot.isaac_binding import FakeIsaacBinding\n"
        "from wam.robot.isaac_transport import IsaacG1Transport\n"
        "from wam.robot.isaac_g1 import IsaacG1Robot\n"
        "robot = IsaacG1Robot(binding=FakeIsaacBinding())\n"
        "robot.read_state()\n"
        "robot.hold()\n"
        "robot.render_frames(1)\n"
        "robot.estop()\n"
        "robot.close()\n"
        "leaked = sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=_REPO_ROOT, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
