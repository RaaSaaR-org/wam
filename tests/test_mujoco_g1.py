"""Tests for the MuJoCo simulation behind the G1 transport seam (T-21, FR-06, E2).

Same seam as tests/test_g1.py, one layer lower: there the adapter runs on FakeG1Transport
(kinematic first-order lag), here it runs on real contact physics and real rendered pixels.
``MujocoG1Transport`` is the THIRD ``G1Transport`` implementation, so the adapter under test
is the unmodified ``G1Adapter`` — nothing about mapping, clipping or the e-stop latch is
re-implemented for the sim, and these tests check exactly that.

The whole module skips (with an actionable reason) when ``mujoco`` is not installed or the
vendor G1 model has not been fetched — neither is a repo dependency: ``uv pip install mujoco``
and ``.venv/bin/python scripts/fetch_g1_model.py``.

Assertion style: physics is deterministic but not exact, so numeric assertions carry an
explicit tolerance and the reason for it. Rendered pixels are NOT bit-portable across GL
backends — image assertions are on variance and shape only, never on pixel values.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from wam.interfaces.protocols import RobotAdapter
from wam.interfaces.schema import ActionChunk, ActionMode
from wam.robot.g1 import G1_JOINT_MAP, G1_NUM_MOTORS, G1_SPEC, G1Config
from wam.robot.g1_transport import G1Transport
from wam.robot.mujoco_g1 import (
    DEFAULT_CAMERAS,
    SIM_DQ_MAX,
    SIM_KD,
    SIM_KP,
    VENDOR_MODEL,
    MujocoG1Robot,
    scene_critical_damping,
    scene_joint_limits,
)
from wam.robot.mujoco_transport import (
    DEFAULT_SCENE_PATH,
    DEX3_FINGER_JOINTS,
    G1_MOTOR_JOINT_NAMES,
    MJCF_JOINT_SUFFIX,
    MujocoG1Transport,
)
from wam.robot.registry import available_robots, get_robot, optional_robots

REPO_ROOT = Path(__file__).resolve().parents[1]

mujoco = pytest.importorskip(
    "mujoco",
    reason="MuJoCo sim tests need the optional 'sim' extra — `uv pip install mujoco`",
)

for _asset in (DEFAULT_SCENE_PATH, VENDOR_MODEL):
    if not (REPO_ROOT / _asset).is_file():
        pytest.skip(
            f"MuJoCo sim assets missing ({_asset}) — fetch the vendor G1 model with "
            "`.venv/bin/python scripts/fetch_g1_model.py`",
            allow_module_level=True,
        )

CANON_IDX = {name: i for i, (name, _) in enumerate(G1_JOINT_MAP)}
ELBOW = CANON_IDX["left_elbow"]
WAIST = CANON_IDX["waist_yaw"]
#: Motor slots the scene welds/locks: the 12 leg joints plus waist roll/pitch.
LOCKED_MOTORS = [*range(12), 13, 14]

CONTROL_DT_S = 0.02  # G1Config.control_dt_s == MujocoG1Transport.control_dt_s default


def make_chunk(
    deltas: np.ndarray, dt_s: float = CONTROL_DT_S, gripper: np.ndarray | None = None
) -> ActionChunk:
    deltas = np.asarray(deltas, dtype=np.float32)
    if gripper is None:
        gripper = np.zeros(deltas.shape[0], dtype=np.float32)
    return ActionChunk(
        mode=ActionMode.JOINT_DELTA,
        targets=deltas,
        gripper_target=np.asarray(gripper, dtype=np.float32),
        dt_s=dt_s,
    )


def joint_deltas(steps: int, **per_joint: float) -> np.ndarray:
    """[steps, 15] chunk targets with a constant per-step delta on the named joints only."""
    out = np.zeros((steps, len(G1_JOINT_MAP)), dtype=np.float32)
    for name, value in per_joint.items():
        out[:, CANON_IDX[name]] = value
    return out


@pytest.fixture(scope="module")
def sim() -> Iterator[MujocoG1Robot]:
    """One scene + one renderer for the whole module — loading the MJCF and building the
    offscreen GL context are the expensive parts (~0.2 s and ~0.4 s)."""
    robot = MujocoG1Robot()
    yield robot
    robot.close()


@pytest.fixture
def robot(sim: MujocoG1Robot) -> MujocoG1Robot:
    """The shared sim, rewound to the ``ready`` keyframe with the e-stop released."""
    sim.clear_estop()
    sim.reset()
    return sim


def body_pose(robot: MujocoG1Robot, name: str) -> tuple[np.ndarray, np.ndarray]:
    """(world position, world quaternion) of a named body — scene-level ground truth that
    does not pass through the canonical schema."""
    model = robot.transport.model
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert bid >= 0, f"scene has no body {name!r}"
    data = robot.transport.data
    return data.xpos[bid].copy(), data.xquat[bid].copy()


# -- seams: protocols and registry ---------------------------------------------------------


def test_mujoco_transport_and_robot_conform_to_the_protocols(sim: MujocoG1Robot) -> None:
    # The whole point of the design: MuJoCo is a THIRD G1Transport, not a second robot.
    assert isinstance(sim.transport, G1Transport)
    assert isinstance(sim, RobotAdapter)
    assert isinstance(sim.adapter, RobotAdapter)  # the wrapped, unmodified hardware adapter
    assert sim.adapter.is_connected
    assert sim.adapter.transport is sim.transport
    assert sim.spec is G1_SPEC
    assert sim.cameras == DEFAULT_CAMERAS


def test_registry_lists_mujoco_g1_as_optional_and_constructs_it() -> None:
    # available_robots() stays "constructible in ANY install" — mujoco is an extra.
    assert "mujoco_g1" in optional_robots()
    assert "mujoco_g1" not in available_robots()
    robot = get_robot("mujoco_g1")
    try:
        assert isinstance(robot, MujocoG1Robot)
        assert isinstance(robot, RobotAdapter)
    finally:
        robot.close()


def test_default_config_uses_sim_gains_and_the_scenes_own_joint_ranges(
    sim: MujocoG1Robot,
) -> None:
    cfg = sim.config
    assert cfg.kp == (SIM_KP,) * len(G1_JOINT_MAP)
    assert cfg.kd == SIM_KD  # per-joint critical damping, not a flat number
    # The no-config path must match the versioned config, not G1Config's 2.0 placeholder.
    assert cfg.dq_max == SIM_DQ_MAX
    q_min, q_max = scene_joint_limits(mujoco, sim.transport.model)
    np.testing.assert_allclose(sim.limits["q_min"], q_min, atol=1e-6)
    np.testing.assert_allclose(sim.limits["q_max"], q_max, atol=1e-6)
    # G1Config's +-1.5708 placeholder would clip real elbow flexion off (scene: 2.0944 rad).
    assert q_max[ELBOW] > G1Config().q_max[ELBOW]
    # ...and G1Config's dq_max placeholder is LOOSER than the shipped envelope.
    assert all(a <= b for a, b in zip(SIM_DQ_MAX, G1Config().dq_max))


def test_sim_kd_is_the_scenes_own_critical_damping(sim: MujocoG1Robot) -> None:
    """SIM_KD is frozen in code and duplicated in configs/robot/mujoco_g1.yaml, so it has to
    stay re-derivable from the scene: kd = 2*sqrt(SIM_KP*m_eff), with m_eff recovered from the
    vendor actuators' dampratio="1". A flat kd would be the wrong SHAPE — the wrist roll's
    effective inertia is a small fraction of the waist's."""
    derived = scene_critical_damping(mujoco, sim.transport.model, SIM_KP)
    np.testing.assert_allclose(SIM_KD, derived, atol=0.01)  # SIM_KD is rounded to 2 decimals
    assert max(SIM_KD) > 4.0 * min(SIM_KD), "critical damping cannot be a flat number here"


def test_a_non_unit_gripper_vendor_range_is_rejected_at_construction() -> None:
    """The transport's gripper channel IS the Dex3 synergy fraction, so vendor units are
    [0, 1]. Any other range would silently make G1Adapter.gripper_to_vendor command a fully
    closed hand for every input above ~0.01, and under-report closure on readback."""
    with pytest.raises(ValueError, match="gripper_vendor"):
        MujocoG1Robot(G1Config(gripper_vendor_min=0.0, gripper_vendor_max=100.0))


# -- motor slot resolution -----------------------------------------------------------------


def test_motor_slots_resolve_to_the_g1_joint_index_names_in_order(sim: MujocoG1Robot) -> None:
    names = sim.transport.motor_joint_names
    assert len(names) == G1_NUM_MOTORS
    assert len(set(names)) == G1_NUM_MOTORS
    assert names == tuple(n + MJCF_JOINT_SUFFIX for n in G1_MOTOR_JOINT_NAMES)
    # Spot-check the convention's boundaries: legs 0-11, waist 12-14, arms 15-21 / 22-28.
    assert names[0] == "left_hip_pitch_joint"
    assert names[11] == "right_ankle_roll_joint"
    assert names[12:15] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    assert names[15] == "left_shoulder_pitch_joint"
    assert names[21] == "left_wrist_yaw_joint"
    assert names[22] == "right_shoulder_pitch_joint"
    assert names[28] == "right_wrist_yaw_joint"
    # G1Adapter gathers canonical joints out of the 29-slot array by hard-coded index; if the
    # scene ever reordered a joint that gather would silently address the wrong motor.
    for canonical_name, motor_idx in G1_JOINT_MAP:
        assert names[motor_idx] == canonical_name + MJCF_JOINT_SUFFIX
    # ...and the names really are the scene's joints, resolved by name, not by raw index.
    model = sim.transport.model
    for slot, name in enumerate(names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert jid >= 0, f"slot {slot}: scene has no joint {name!r}"


def test_missing_joints_fail_at_construction_not_mid_rollout() -> None:
    # The vendor scene without hands has no Dex3 finger joints -> constructing must raise.
    handless = REPO_ROOT / "assets/mujoco/unitree_g1/scene.xml"
    if not handless.is_file():  # pragma: no cover - depends on the fetched tree
        pytest.skip("vendor scene.xml not fetched")
    with pytest.raises(ValueError, match="missing"):
        MujocoG1Transport(handless, camera_names=())


# -- read_low_state contract ---------------------------------------------------------------


def test_read_low_state_matches_the_documented_dict_contract(robot: MujocoG1Robot) -> None:
    low = robot.transport.read_low_state()
    assert set(low) == {"q", "dq", "imu", "gripper", "tick_ns"}
    for key in ("q", "dq"):
        assert low[key].shape == (G1_NUM_MOTORS,)
        assert low[key].dtype == np.float32
        assert np.isfinite(low[key]).all()
    imu = low["imu"]
    assert set(imu) == {"quat_wxyz", "gyro", "acc"}
    assert imu["quat_wxyz"].shape == (4,)
    assert imu["gyro"].shape == (3,)
    assert imu["acc"].shape == (3,)
    for value in imu.values():
        assert value.dtype == np.float32
        assert np.isfinite(value).all()
    np.testing.assert_allclose(np.linalg.norm(imu["quat_wxyz"]), 1.0, atol=1e-5)
    # Gravity is the dominant accelerometer term on a standing, welded robot.
    np.testing.assert_allclose(imu["acc"][2], 9.81, atol=0.05)
    assert low["gripper"].shape == (2,)
    assert low["gripper"].dtype == np.float32
    assert isinstance(low["tick_ns"], int)
    assert low["tick_ns"] >= 0


def test_read_low_state_returns_copies_not_live_views(robot: MujocoG1Robot) -> None:
    # A caller must not be able to corrupt qpos by writing into a returned array.
    low = robot.transport.read_low_state()
    low["q"][:] = 99.0
    low["imu"]["gyro"][:] = 99.0
    np.testing.assert_array_less(np.abs(robot.transport.read_low_state()["q"]), 10.0)
    np.testing.assert_array_less(np.abs(robot.transport.read_low_state()["imu"]["gyro"]), 10.0)


# -- tick semantics (the staleness signal G1Adapter turns into validity flags) --------------


def test_tick_advances_on_a_motor_write_and_never_without_one(robot: MujocoG1Robot) -> None:
    transport = robot.transport
    low = transport.read_low_state()
    start = low["tick_ns"]
    # Bare reads must NOT advance the tick: nothing stepped.
    for _ in range(3):
        assert transport.read_low_state()["tick_ns"] == start
    # A gripper command alone does not step either (the fingers move on the next motor cmd).
    transport.write_gripper_cmd(0.5, 0.5)
    assert transport.read_low_state()["tick_ns"] == start
    # Every motor command advances by exactly control_dt_s, even with zero gains.
    hold = np.asarray(low["q"], dtype=np.float64)
    zeros = np.zeros(G1_NUM_MOTORS)
    step_ns = round(CONTROL_DT_S * 1e9)
    for i in range(1, 4):
        transport.write_motor_cmd(hold, zeros, zeros, zeros)
        assert transport.read_low_state()["tick_ns"] == start + i * step_ns
    assert transport.control_dt_s == CONTROL_DT_S
    assert transport.substeps == round(CONTROL_DT_S / float(transport.model.opt.timestep))


def test_stale_tick_degrades_validity_through_the_unmodified_adapter(
    robot: MujocoG1Robot,
) -> None:
    fresh = robot.read_state()
    assert all(fresh.validity.as_dict().values())
    assert fresh.validate(G1_SPEC) == []
    stale = robot.read_state()  # no physics stepped in between -> same tick
    assert not any(stale.validity.as_dict().values())
    robot.execute(make_chunk(joint_deltas(1)), prefix_steps=1)
    recovered = robot.read_state()
    assert all(recovered.validity.as_dict().values())
    assert recovered.timestamp_ns == robot.sim_time_ns


# -- execute -------------------------------------------------------------------------------


def test_execute_moves_the_commanded_joint_and_leaves_the_others_alone(
    robot: MujocoG1Robot,
) -> None:
    q0 = robot.read_state().q.copy()
    steps = 12
    delta = -0.04  # == dq_max * dt, so nothing is velocity-clipped
    robot.execute(make_chunk(joint_deltas(steps, left_elbow=delta)), prefix_steps=steps)
    q1 = robot.read_state().q

    moved = q1 - q0
    commanded = steps * delta  # -0.48 rad, integrated exactly by the adapter
    # Right joint, right direction, and it tracked most of the commanded travel (measured
    # -0.390 rad: a position loop lags, it never leads).
    assert commanded - 0.02 <= moved[ELBOW] <= 0.5 * commanded
    # Everything else only reacts to the arm's inertia through the position loop: <= 0.05 rad.
    others = np.delete(np.abs(moved), ELBOW)
    assert others.max() < 0.05, f"unexpected coupling: {np.round(moved, 4)}"
    # Sim time advanced exactly one control period per executed step.
    assert robot.sim_time_ns == steps * round(CONTROL_DT_S * 1e9)


def test_execute_clips_an_over_large_delta_to_the_configured_limits() -> None:
    q_min, q_max = scene_joint_limits(mujoco, MujocoG1Transport(camera_names=()).model)
    tight_max = list(q_max)
    tight_max[ELBOW] = 0.6  # far inside the scene's own 2.0944 rad elbow range
    cfg = G1Config(
        q_min=q_min,
        q_max=tuple(tight_max),
        kp=(SIM_KP,) * len(G1_JOINT_MAP),
        kd=SIM_KD,
    )
    robot = MujocoG1Robot(cfg)
    try:
        q0 = robot.read_state().q.copy()
        huge = joint_deltas(30, left_elbow=10.0)  # 250x the per-step velocity budget

        # 1. Velocity clip: one step from rest can move at most dq_max * dt.
        robot.execute(make_chunk(huge[:1]), prefix_steps=1)
        q1 = robot.read_state().q
        budget = float(robot.limits["dq_max"][ELBOW]) * CONTROL_DT_S
        assert 0.0 < q1[ELBOW] - q0[ELBOW] <= budget + 1e-6

        # 2. Position clip: 29 more steps of the same delta saturate at the CONFIG limit,
        #    which only the adapter can enforce — MuJoCo's own ctrlrange is 3.5x wider.
        robot.execute(make_chunk(huge[1:]), prefix_steps=29)
        q2 = robot.read_state().q
        assert q2[ELBOW] <= tight_max[ELBOW] + 0.02  # 0.02 rad = steady-state tracking error
        assert q2[ELBOW] > 0.5  # it really did travel to the limit, it is not still climbing
        assert q2[ELBOW] < 0.5 * q_max[ELBOW]  # ...and stopped far short of the scene's range
        # Every canonical joint stays inside the configured box. The 0.02 rad slack is for
        # physics, not for the clip: a contacting joint can be pushed past its target because
        # the vendor wrist-pitch actuator only has +-5 Nm (mujoco_transport's CONTACT CAVEAT).
        assert (q2 <= np.asarray(cfg.q_max) + 0.02).all()
        assert (q2 >= np.asarray(cfg.q_min) - 0.02).all()
    finally:
        robot.close()


def test_execute_under_executes_every_delta_by_a_prefix_dependent_factor(
    robot: MujocoG1Robot,
) -> None:
    """PIN the known limitation, so it cannot silently drift.

    ``G1Adapter.execute()`` re-bases its target on the MEASURED q at every call, so the
    position loop's lag is discarded instead of caught up. The same total commanded travel
    therefore lands differently depending on ``prefix_steps`` — a rollout knob, not physics.
    Measured on ``left_shoulder_yaw`` (free space), 100 x 0.004 rad = 0.400 rad commanded:
    0.31 of it at prefix 1, 0.67 at prefix 5, 0.95 at prefix 25.

    This is architectural (no feed-forward, no integral action anywhere in the chain) and is
    NOT tunable away: even kp=4000 with critical damping reaches only ~0.86 at prefix 1. The
    bands below are wide enough for physics noise and narrow enough to fail if the sim ever
    starts executing a third — or all — of a commanded motion. See docs/sim.md.
    """
    joint = "left_shoulder_yaw"
    idx = CANON_IDX[joint]
    total, per_step = 100, 0.004
    commanded = total * per_step
    achieved: dict[int, float] = {}
    for prefix in (1, 5, 25):
        robot.clear_estop()
        robot.reset()
        q0 = robot.read_state().q[idx]
        done = 0
        while done < total:
            take = min(prefix, total - done)
            robot.execute(make_chunk(joint_deltas(total - done, **{joint: per_step})), take)
            done += take
        achieved[prefix] = float(robot.read_state().q[idx] - q0) / commanded

    # Monotone in prefix_steps: longer open-loop runs lose less to the re-basing.
    assert achieved[1] < achieved[5] < achieved[25]
    assert 0.20 <= achieved[1] <= 0.45, achieved
    assert 0.55 <= achieved[5] <= 0.80, achieved
    assert 0.85 <= achieved[25] <= 0.99, achieved
    # The whole point: the executed magnitude is NOT the commanded magnitude.
    assert achieved[25] < 1.0


def test_a_zero_delta_chunk_is_not_a_position_hold(robot: MujocoG1Robot) -> None:
    """The ratchet, same root cause as the test above: ``execute()`` re-reads q, so every
    cycle forgives the gravity droop of the previous one and the arm creeps monotonically.
    Measured max |q - keyframe|: 0.08 rad @ 2 s, 0.33 @ 10 s, still growing at 60 s. Anything
    that quotes the BOUNDED fixed-target droop (0.009 rad) as the sim's hold accuracy is
    measuring a protocol the runtime never uses."""
    q0 = robot.read_state().q.copy()
    zero = make_chunk(joint_deltas(1))
    drift = []
    for _ in range(2):  # 2 x 100 control periods = 2 x 2 s
        for _ in range(100):
            robot.execute(zero, prefix_steps=1)
        drift.append(float(np.abs(robot.read_state().q - q0).max()))
    assert drift[0] > 0.02, f"expected a visible ratchet, got {drift}"
    assert drift[1] > drift[0], f"the drift must still be growing, got {drift}"


def test_execute_rejects_ee_delta_and_a_bad_width(robot: MujocoG1Robot) -> None:
    chunk = make_chunk(joint_deltas(1))
    chunk.mode = ActionMode.EE_DELTA
    with pytest.raises(NotImplementedError, match="EE_DELTA"):
        robot.execute(chunk, prefix_steps=1)
    with pytest.raises(ValueError):
        robot.execute(make_chunk(np.zeros((1, 14), dtype=np.float32)), prefix_steps=1)
    assert robot.sim_time_ns == 0  # a rejected chunk stepped no physics


def test_hold_advances_sim_time_by_one_control_period(robot: MujocoG1Robot) -> None:
    # Unlike MockRobot.hold(), holding a real arm IS physics.
    robot.hold()
    assert robot.sim_time_ns == round(CONTROL_DT_S * 1e9)


def test_non_finite_commands_are_rejected_at_the_transport_seam(robot: MujocoG1Robot) -> None:
    """One NaN anywhere in ``data.ctrl`` makes MuJoCo zero ALL 43 control entries on every
    subsequent step: the whole robot slams to its zero pose in 0.2 s with no exception, finite
    read_low_state output and a normally advancing tick — invisible to every layer above. Both
    write paths must therefore reject non-finite input, not clip it."""
    transport = robot.transport
    finite = np.asarray(transport.read_low_state()["q"], dtype=np.float64)
    zeros = np.zeros(G1_NUM_MOTORS)
    for bad in (np.nan, np.inf):
        broken = finite.copy()
        broken[15] = bad
        with pytest.raises(ValueError, match="non-finite"):
            transport.write_motor_cmd(broken, zeros, zeros, zeros)
        # np.clip(nan, 0, 1) is nan, so clipping alone would NOT have caught this.
        with pytest.raises(ValueError, match="non-finite"):
            transport.write_gripper_cmd(bad, 0.0)
        with pytest.raises(ValueError, match="non-finite"):
            transport.write_gripper_cmd(0.0, bad)
    assert np.isfinite(transport.data.ctrl).all()
    assert transport.read_low_state()["tick_ns"] == 0  # nothing stepped


# -- e-stop --------------------------------------------------------------------------------


def test_estop_latches_and_clear_estop_releases(robot: MujocoG1Robot) -> None:
    damp_before = robot.transport.damp_count
    robot.estop()
    assert robot.is_estopped
    assert robot.transport.damp_count == damp_before + 1
    assert robot.transport.last_damp_error is None

    q_before = robot.read_state().q.copy()
    tick_before = robot.sim_time_ns
    robot.execute(make_chunk(joint_deltas(5, left_elbow=-0.04)), prefix_steps=5)
    robot.hold()
    # Latched: not one command reaches the transport, so not one substep is taken.
    assert robot.sim_time_ns == tick_before
    np.testing.assert_array_equal(robot.read_state().q, q_before)

    robot.clear_estop()
    assert not robot.is_estopped
    robot.execute(make_chunk(joint_deltas(5, left_elbow=-0.04)), prefix_steps=5)
    assert robot.sim_time_ns == tick_before + 5 * round(CONTROL_DT_S * 1e9)
    assert robot.read_state().q[ELBOW] < q_before[ELBOW] - 0.01


def test_a_failed_damp_is_observable_and_does_not_report_success() -> None:
    """A totally failed safe-stop must not be indistinguishable from a successful one.

    ``G1Adapter.estop()``'s contract: "the latch is set even if the transport raises while
    damping (the exception still propagates)". Recording the failure on ``last_damp_error``
    and returning normally would leave every layer above the seam believing the arm was
    damped, when its velocity is unchanged.
    """
    robot = MujocoG1Robot()
    try:
        robot.execute(make_chunk(joint_deltas(20, left_elbow=-0.04)), prefix_steps=20)
        moving = float(np.abs(robot.transport.read_low_state()["dq"]).max())
        assert moving > 0.1, "the arm has to be moving for this test to mean anything"

        class Boom(RuntimeError):
            pass

        real_mj = robot.transport._mj

        class FailingStep:
            def __getattr__(self, name: str) -> object:
                if name == "mj_step":

                    def _boom(model: object, data: object) -> None:
                        raise Boom("simulated solver failure")

                    return _boom
                return getattr(real_mj, name)

        robot.transport._mj = FailingStep()
        try:
            with pytest.raises(Boom):
                robot.estop()
        finally:
            robot.transport._mj = real_mj
        # The latch is still set (G1Adapter latches in a finally), and the failure is recorded.
        assert robot.is_estopped
        assert isinstance(robot.transport.last_damp_error, Boom)
    finally:
        robot.close()


def test_emergency_damp_is_not_a_freeze_and_advances_the_sim_clock() -> None:
    """``emergency_damp()`` steps ``damp_duration_s`` of physics, so it advances the tick by
    10 control periods and the arm keeps creeping under gravity during that interval. Callers
    that timestamp on sim time see a discontinuity at every e-stop."""
    robot = MujocoG1Robot()
    try:
        robot.execute(make_chunk(joint_deltas(10, left_elbow=-0.04)), prefix_steps=10)
        before_ns = robot.sim_time_ns
        q_before = robot.transport.read_low_state()["q"].copy()
        robot.estop()
        advanced_s = (robot.sim_time_ns - before_ns) * 1e-9
        np.testing.assert_allclose(advanced_s, 0.2, atol=1e-9)  # the default damp_duration_s
        crept = float(np.abs(robot.transport.read_low_state()["q"] - q_before).max())
        assert crept > 1e-3, "pure damping settles the arm, it does not freeze it"
    finally:
        robot.close()


def test_estop_from_another_thread_is_safe(sim: MujocoG1Robot) -> None:
    """``RobotAdapter.estop()`` promises "safe to call at any time, from any thread". MjData is
    not thread-safe: without the transport lock this segfaults the interpreter (reproduced 3/3
    before the fix), which no ``pytest.raises`` can catch."""
    sim.clear_estop()
    sim.reset()
    chunk = make_chunk(joint_deltas(40, left_elbow=-0.01, waist_yaw=0.01))
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker() -> None:
        try:
            while not stop.is_set():
                sim.execute(chunk, prefix_steps=40)
        except BaseException as exc:  # noqa: BLE001 - the point is to surface anything at all
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        for _ in range(50):
            sim.estop()
            sim.clear_estop()
    finally:
        stop.set()
        thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert errors == []
    assert np.isfinite(sim.read_state().q).all()
    sim.clear_estop()


def test_reset_is_an_episode_reset_not_an_estop_release(sim: MujocoG1Robot) -> None:
    sim.clear_estop()
    sim.reset()
    sim.estop()
    sim.reset()
    assert sim.is_estopped  # only a deliberate clear_estop() releases the latch
    sim.clear_estop()
    sim.reset()
    assert not sim.is_estopped


# -- gripper synergy -----------------------------------------------------------------------


def test_gripper_closure_is_monotonic_and_spans_open_to_closed(robot: MujocoG1Robot) -> None:
    commands = (0.0, 0.25, 0.5, 0.75, 1.0)
    settle = joint_deltas(25)  # 0.5 s per command: the Dex3 fingers settle in ~0.3 s
    measured = []
    for value in commands:
        chunk = make_chunk(settle, gripper=np.full(settle.shape[0], value, dtype=np.float32))
        robot.execute(chunk, prefix_steps=settle.shape[0])
        measured.append(robot.read_state().gripper_state.copy())
    closure = np.asarray(measured)  # [5, 2] canonical [0, 1] per hand

    assert closure.shape == (len(commands), 2)
    for hand in range(2):
        assert np.all(np.diff(closure[:, hand]) > 0.0), f"not monotonic: {closure[:, hand]}"
    # 0 -> open, 1 -> closed. The residual at 1.0 (measured 0.964) is joint-limit constraint
    # softness: the closed pose sits ON the limits, so the fingers stop just short.
    np.testing.assert_allclose(closure[0], 0.0, atol=1e-3)
    np.testing.assert_allclose(closure[-1], 1.0, atol=0.06)
    # Both hands follow the same scalar: the synergy is mirrored, not duplicated.
    np.testing.assert_allclose(closure[:, 0], closure[:, 1], atol=0.02)


def test_gripper_synergy_is_mirrored_between_the_hands(sim: MujocoG1Robot) -> None:
    left_open, left_closed = sim.transport.finger_synergy("left")
    right_open, right_closed = sim.transport.finger_synergy("right")
    assert left_open.shape == right_open.shape == (len(DEX3_FINGER_JOINTS),)
    np.testing.assert_array_equal(left_open, np.zeros(len(DEX3_FINGER_JOINTS)))
    np.testing.assert_array_equal(right_open, np.zeros(len(DEX3_FINGER_JOINTS)))
    np.testing.assert_allclose(left_closed, -right_closed, atol=1e-9)
    # thumb_0 is the opposition roll and contributes no travel; the other six do.
    thumb_0 = DEX3_FINGER_JOINTS.index("thumb_0")
    assert left_closed[thumb_0] == 0.0
    assert np.count_nonzero(left_closed) == len(DEX3_FINGER_JOINTS) - 1


# -- determinism ---------------------------------------------------------------------------


def test_two_freshly_built_sims_are_bit_identical() -> None:
    targets = joint_deltas(12, left_elbow=-0.03, left_shoulder_roll=0.02, waist_yaw=0.05)
    gripper = np.linspace(0.0, 1.0, targets.shape[0]).astype(np.float32)
    results = []
    for _ in range(2):
        robot = MujocoG1Robot()
        try:
            robot.execute(make_chunk(targets, gripper=gripper), prefix_steps=targets.shape[0])
            state = robot.read_state()
            results.append((state.q, state.gripper_state, robot.sim_time_ns))
        finally:
            robot.close()
    np.testing.assert_array_equal(results[0][0], results[1][0])
    np.testing.assert_array_equal(results[0][1], results[1][1])
    assert results[0][2] == results[1][2]
    assert not np.allclose(results[0][0], 0.0)  # the rollout actually did something


def test_reset_rewinds_to_a_bit_identical_state(robot: MujocoG1Robot) -> None:
    q0 = robot.read_state().q.copy()
    robot.execute(make_chunk(joint_deltas(6, left_elbow=-0.04)), prefix_steps=6)
    assert robot.sim_time_ns > 0
    robot.reset()
    assert robot.sim_time_ns == 0
    after = robot.read_state()
    np.testing.assert_array_equal(after.q, q0)
    assert all(after.validity.as_dict().values())  # a rewound clock must not read as stale


# -- rendering -----------------------------------------------------------------------------


def test_render_frames_shape_dtype_and_non_degenerate_images(robot: MujocoG1Robot) -> None:
    frames = robot.render_frames(2)
    assert set(frames) == set(DEFAULT_CAMERAS)
    height, width = robot.image_hw
    for name, arr in frames.items():
        assert arr.shape == (2, height, width, 3), name
        assert arr.dtype == np.uint8, name
        # Not bit-portable across GL backends: assert on variance, never on pixel values.
        assert float(arr[0].std()) > 10.0, f"{name}: flat image, camera sees nothing"
        assert len(np.unique(arr[0].reshape(-1, 3), axis=0)) > 100, f"{name}: near-uniform"
        # Rendering never steps physics, so the n frames of one call are identical copies.
        np.testing.assert_array_equal(arr[0], arr[1])
    assert robot.sim_time_ns == 0  # ...and the tick did not move behind the adapter's back

    before = robot.render_frames(1)
    robot.execute(make_chunk(joint_deltas(10, left_elbow=-0.04)), prefix_steps=10)
    after = robot.render_frames(1)
    for name in DEFAULT_CAMERAS:
        diff = np.abs(after[name].astype(np.int16) - before[name].astype(np.int16))
        assert diff.mean() > 1.0, f"{name}: the image did not follow the arm"

    with pytest.raises(ValueError):
        robot.render_frames(0)


# -- scene invariants ----------------------------------------------------------------------


def test_base_and_locked_joints_do_not_drift_over_a_rollout(robot: MujocoG1Robot) -> None:
    pos0, quat0 = body_pose(robot, "pelvis")
    locked0 = robot.transport.read_low_state()["q"][LOCKED_MOTORS].copy()

    robot.execute(
        make_chunk(joint_deltas(30, left_elbow=-0.04, waist_yaw=0.04, right_elbow=0.04)),
        prefix_steps=30,
    )

    pos1, quat1 = body_pose(robot, "pelvis")
    assert np.abs(pos1 - pos0).max() < 1e-3, f"pelvis translated {pos1 - pos0}"
    assert np.abs(quat1 - quat0).max() < 1e-3, f"pelvis rotated {quat1 - quat0}"
    locked1 = robot.transport.read_low_state()["q"][LOCKED_MOTORS]
    assert np.abs(locked1 - locked0).max() < 1e-3, f"locked joints drifted {locked1 - locked0}"
    # The rollout was not a no-op: the driven joints moved.
    assert robot.read_state().q[WAIST] > 0.1
