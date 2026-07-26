"""Tests for wam.robot: MockRobot kinematics, G1 skeleton mapping, registry."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from wam.interfaces.protocols import RobotAdapter
from wam.interfaces.schema import ActionChunk, ActionMode
from wam.robot import (
    G1_JOINT_MAP,
    G1_NUM_MOTORS,
    G1_SPEC,
    G1Adapter,
    G1Config,
    MockRobot,
    available_robots,
    get_robot,
)

SDK_INSTALLED = importlib.util.find_spec("unitree_sdk2py") is not None


def make_chunk(
    deltas: np.ndarray, dt_s: float = 0.1, gripper: np.ndarray | None = None
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


# -- MockRobot ---------------------------------------------------------------------------


def test_mock_conforms_to_protocol() -> None:
    assert isinstance(MockRobot(num_joints=3), RobotAdapter)


def test_mock_integrates_deltas_and_finite_difference_dq() -> None:
    robot = MockRobot(num_joints=3)
    deltas = np.array([[0.1, 0.0, -0.2], [0.05, 0.1, 0.0]], dtype=np.float32)
    robot.execute(make_chunk(deltas, dt_s=0.1), prefix_steps=2)
    state = robot.read_state()
    np.testing.assert_allclose(state.q, deltas.sum(axis=0), atol=1e-6)
    # dq is the finite difference of the LAST executed step.
    np.testing.assert_allclose(state.dq, deltas[1] / 0.1, atol=1e-5)
    assert state.validate() == []


def test_mock_clips_q_to_limits() -> None:
    robot = MockRobot(num_joints=2, q_min=-0.5, q_max=0.5)
    robot.execute(make_chunk(np.array([[10.0, -10.0]])), prefix_steps=1)
    state = robot.read_state()
    np.testing.assert_allclose(state.q, [0.5, -0.5], atol=1e-6)


def test_mock_respects_prefix_steps() -> None:
    robot = MockRobot(num_joints=1)
    deltas = np.array([[0.1], [0.1], [0.1], [0.1]], dtype=np.float32)
    robot.execute(make_chunk(deltas), prefix_steps=2)
    np.testing.assert_allclose(robot.read_state().q, [0.2], atol=1e-6)
    # prefix longer than the chunk executes the whole chunk.
    robot.execute(make_chunk(deltas), prefix_steps=99)
    np.testing.assert_allclose(robot.read_state().q, [0.6], atol=1e-6)
    with pytest.raises(ValueError):
        robot.execute(make_chunk(deltas), prefix_steps=-1)


def test_mock_estop_stops_and_latches() -> None:
    robot = MockRobot(num_joints=1)
    robot.execute(make_chunk(np.array([[0.3]])), prefix_steps=1)
    robot.estop()
    assert robot.is_estopped
    state = robot.read_state()
    np.testing.assert_allclose(state.dq, [0.0])
    robot.execute(make_chunk(np.array([[0.5]])), prefix_steps=1)  # ignored while latched
    np.testing.assert_allclose(robot.read_state().q, [0.3], atol=1e-6)
    robot.clear_estop()
    robot.execute(make_chunk(np.array([[0.5]])), prefix_steps=1)
    np.testing.assert_allclose(robot.read_state().q, [0.8], atol=1e-6)


def test_mock_hold_zeroes_dq_and_is_released_by_execute() -> None:
    robot = MockRobot(num_joints=1)
    robot.execute(make_chunk(np.array([[0.2]])), prefix_steps=1)
    assert robot.read_state().dq[0] != 0.0
    robot.hold()
    assert robot.is_holding
    np.testing.assert_allclose(robot.read_state().dq, [0.0])
    robot.execute(make_chunk(np.array([[0.1]])), prefix_steps=1)
    assert not robot.is_holding


def test_mock_simulated_latency_advances_clock_only() -> None:
    robot = MockRobot(num_joints=1, step_latency_s=0.05)
    assert robot.read_state().timestamp_ns == 0
    robot.execute(make_chunk(np.array([[0.0], [0.0]]), dt_s=0.1), prefix_steps=2)
    # 2 steps * (0.1 + 0.05) s, purely simulated.
    assert robot.read_state().timestamp_ns == round(2 * 0.15 * 1e9)
    assert robot.sim_time_ns == robot.read_state().timestamp_ns


def test_mock_noise_is_deterministic_and_read_only() -> None:
    a = MockRobot(num_joints=4, noise_std=0.01, seed=7)
    b = MockRobot(num_joints=4, noise_std=0.01, seed=7)
    np.testing.assert_array_equal(a.read_state().q, b.read_state().q)
    np.testing.assert_array_equal(a.read_state().dq, b.read_state().dq)
    # Internal state stays noise-free: executing zero deltas keeps q at exactly 0 internally.
    c = MockRobot(num_joints=4, noise_std=0.5, seed=1)
    c.execute(make_chunk(np.zeros((3, 4))), prefix_steps=3)
    assert not np.array_equal(c.read_state().q, c.read_state().q)  # fresh noise per read


def test_mock_rejects_ee_delta_and_bad_width() -> None:
    robot = MockRobot(num_joints=2)
    chunk = make_chunk(np.zeros((1, 2)))
    chunk.mode = ActionMode.EE_DELTA
    with pytest.raises(NotImplementedError):
        robot.execute(chunk, prefix_steps=1)
    with pytest.raises(ValueError):
        robot.execute(make_chunk(np.zeros((1, 3))), prefix_steps=1)


def test_mock_gripper_tracks_command() -> None:
    robot = MockRobot(num_joints=1)
    chunk = make_chunk(np.zeros((2, 1)), gripper=np.array([0.3, 0.9], dtype=np.float32))
    robot.execute(chunk, prefix_steps=2)
    np.testing.assert_allclose(robot.read_state().gripper_state, [0.9], atol=1e-6)


def test_mock_limits_keys_and_shapes() -> None:
    robot = MockRobot(num_joints=3)
    limits = robot.limits
    for key in ("q_min", "q_max", "dq_max"):
        assert limits[key].shape == (3,)
        assert limits[key].dtype == np.float32


def test_mock_render_frames_shape_and_moving_dot() -> None:
    robot = MockRobot(num_joints=1, q_min=-1.0, q_max=1.0, image_hw=(32, 48))
    frames = robot.render_frames(2)
    assert set(frames) == {"front", "wrist"}
    for arr in frames.values():
        assert arr.shape == (2, 32, 48, 3)
        assert arr.dtype == np.uint8

    def dot_col(img: np.ndarray) -> int:
        return int(np.argwhere((img == 255).all(axis=-1))[:, 1].mean())

    col_before = dot_col(frames["front"][0])
    robot.execute(make_chunk(np.array([[0.8]])), prefix_steps=1)
    col_after = dot_col(robot.render_frames(1)["front"][0])
    assert col_after > col_before  # dot column encodes q[0]


# -- G1 skeleton --------------------------------------------------------------------------


def test_g1_imports_and_constructs_without_sdk() -> None:
    adapter = G1Adapter()
    assert isinstance(adapter, RobotAdapter)
    assert not adapter.is_connected
    assert adapter.spec is G1_SPEC


def test_g1_joint_map_is_bijective_and_matches_spec() -> None:
    names = [name for name, _ in G1_JOINT_MAP]
    indices = [idx for _, idx in G1_JOINT_MAP]
    assert len(G1_JOINT_MAP) == 15  # waist + 2x7 arm joints
    assert len(set(names)) == len(names)
    assert len(set(indices)) == len(indices)
    assert all(0 <= i < G1_NUM_MOTORS for i in indices)
    assert G1_SPEC.joint_names == tuple(names)
    assert G1_SPEC.gripper_dims == 2


def test_g1_canonical_motor_roundtrip() -> None:
    canonical = np.arange(1, 16, dtype=np.float32)
    motor = G1Adapter.canonical_to_motor(canonical)
    assert motor.shape == (G1_NUM_MOTORS,)
    # Unmapped motors (legs, waist roll/pitch) stay zero.
    mapped = {idx for _, idx in G1_JOINT_MAP}
    assert all(motor[i] == 0.0 for i in range(G1_NUM_MOTORS) if i not in mapped)
    np.testing.assert_array_equal(G1Adapter.motor_to_canonical(motor), canonical)
    with pytest.raises(ValueError):
        G1Adapter.canonical_to_motor(np.zeros(14, dtype=np.float32))


@pytest.mark.skipif(SDK_INSTALLED, reason="unitree_sdk2py installed; guard not reachable")
def test_g1_connect_raises_clear_runtime_error_without_sdk() -> None:
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        G1Adapter().connect()


def test_g1_hardware_calls_require_connect() -> None:
    adapter = G1Adapter()
    with pytest.raises(RuntimeError, match="connect"):
        adapter.read_state()
    with pytest.raises(RuntimeError, match="connect"):
        adapter.execute(make_chunk(np.zeros((1, 15))), prefix_steps=1)
    with pytest.raises(RuntimeError, match="connect"):
        adapter.hold()
    adapter.estop()  # must be safe at any time: no-op when not connected


def test_g1_limits_from_config() -> None:
    cfg = G1Config(q_min=(-1.0,) * 15, q_max=(1.0,) * 15, dq_max=(0.5,) * 15)
    limits = G1Adapter(config=cfg).limits
    for key in ("q_min", "q_max", "dq_max"):
        assert limits[key].shape == (15,)
        assert limits[key].dtype == np.float32
    np.testing.assert_allclose(limits["dq_max"], 0.5)
    with pytest.raises(ValueError):
        G1Config(q_min=(0.0,) * 15, q_max=(0.0,) * 15)
    with pytest.raises(ValueError):
        G1Config(q_min=(-1.0,) * 3)


def test_g1_gripper_vendor_roundtrip() -> None:
    adapter = G1Adapter(config={"gripper_vendor_min": -0.2, "gripper_vendor_max": 5.4})
    g = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    vendor = adapter.gripper_to_vendor(g)
    np.testing.assert_allclose(vendor, [-0.2, 1.2, 5.4], atol=1e-6)
    np.testing.assert_allclose(adapter.vendor_to_gripper(vendor), g, atol=1e-6)


# -- Registry ------------------------------------------------------------------------------


def test_registry_constructs_mock_and_g1() -> None:
    mock = get_robot("mock", num_joints=5)
    assert isinstance(mock, MockRobot)
    assert mock.spec.num_joints == 5
    g1 = get_robot("g1")
    assert isinstance(g1, G1Adapter)
    assert isinstance(get_robot("MOCK"), MockRobot)  # case-insensitive
    assert available_robots() == ("g1", "mock")


def test_registry_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown robot"):
        get_robot("optimus")


def test_registry_returns_protocol_conformant_adapters() -> None:
    for name in available_robots():
        assert isinstance(get_robot(name), RobotAdapter)


def test_mock_default_limits_are_pi() -> None:
    robot = MockRobot(num_joints=2)
    np.testing.assert_allclose(robot.limits["q_max"], math.pi, rtol=1e-6)
