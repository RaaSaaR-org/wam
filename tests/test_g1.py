"""Tests for the completed G1 adapter against the swappable transport seam (T-21).

Everything runs against FakeG1Transport — no hardware, no unitree_sdk2py, deterministic.
The skeleton-era tests (mapping, config, guards) live in tests/test_robot.py and stay green.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from wam.interfaces.protocols import RobotAdapter
from wam.interfaces.schema import ActionChunk, ActionMode
from wam.robot.g1 import (
    G1_JOINT_MAP,
    G1_NUM_MOTORS,
    G1_SPEC,
    G1Adapter,
    G1Config,
)
from wam.robot.g1_transport import DdsG1Transport, FakeG1Transport, G1Transport

SDK_INSTALLED = importlib.util.find_spec("unitree_sdk2py") is not None

MOTOR_IDX = dict(G1_JOINT_MAP)
CANON_IDX = {name: i for i, (name, _) in enumerate(G1_JOINT_MAP)}
MAPPED = set(MOTOR_IDX.values())
UNMAPPED = sorted(set(range(G1_NUM_MOTORS)) - MAPPED)


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


class FakeTime:
    """Deterministic clock/sleep pair for execute() pacing: sleep(dt) advances the clock."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def connected_adapter(
    config: G1Config | dict | None = None,
    fake: FakeG1Transport | None = None,
    fake_time: FakeTime | None = None,
) -> tuple[G1Adapter, FakeG1Transport]:
    fake = fake if fake is not None else FakeG1Transport()
    ft = fake_time if fake_time is not None else FakeTime()
    adapter = G1Adapter(config=config, transport=fake, clock=ft.clock, sleep=ft.sleep)
    adapter.connect()
    return adapter, fake


# -- transport seam ------------------------------------------------------------------------


def test_fake_transport_conforms_to_protocol() -> None:
    assert isinstance(FakeG1Transport(), G1Transport)
    assert isinstance(DdsG1Transport(), G1Transport)


def test_fake_transport_first_order_lag_and_tick() -> None:
    fake = FakeG1Transport(lag=0.5, tick_step_ns=2_000_000)
    target = np.full(G1_NUM_MOTORS, 1.0)
    zeros = np.zeros(G1_NUM_MOTORS)
    fake.write_motor_cmd(target, zeros, zeros, zeros)
    np.testing.assert_allclose(fake.q, 0.5)
    fake.write_motor_cmd(target, zeros, zeros, zeros)
    np.testing.assert_allclose(fake.q, 0.75)
    assert fake.tick_ns == 4_000_000
    assert len(fake.motor_commands) == 2
    np.testing.assert_allclose(fake.motor_commands[0]["q_target"], 1.0)


def test_fake_transport_freeze_tick_and_noise_determinism() -> None:
    fake = FakeG1Transport(lag=1.0)
    fake.freeze_tick = True
    zeros = np.zeros(G1_NUM_MOTORS)
    fake.write_motor_cmd(zeros, zeros, zeros, zeros)
    assert fake.tick_ns == 0  # stalled controller
    a = FakeG1Transport(noise_std=0.01, seed=3)
    b = FakeG1Transport(noise_std=0.01, seed=3)
    np.testing.assert_array_equal(a.read_low_state()["q"], b.read_low_state()["q"])
    # Fresh noise per read; internal state stays noise-free.
    assert not np.array_equal(a.read_low_state()["q"], a.read_low_state()["q"])
    np.testing.assert_allclose(a.q, 0.0)


def test_fake_transport_rejects_bad_shapes() -> None:
    fake = FakeG1Transport()
    zeros = np.zeros(G1_NUM_MOTORS)
    with pytest.raises(ValueError):
        fake.write_motor_cmd(np.zeros(15), zeros, zeros, zeros)
    with pytest.raises(ValueError):
        FakeG1Transport(initial_q=np.zeros(28))


@pytest.mark.skipif(SDK_INSTALLED, reason="unitree_sdk2py installed; guard not reachable")
def test_dds_transport_constructs_but_raises_without_sdk() -> None:
    dds = DdsG1Transport(G1Config())  # constructor stores config only, never imports SDK
    assert dds.config is not None
    zeros = np.zeros(G1_NUM_MOTORS)
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        dds.open()
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        dds.read_low_state()
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        dds.write_motor_cmd(zeros, zeros, zeros, zeros)
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        dds.write_gripper_cmd(0.0, 0.0)
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        dds.emergency_damp()


# -- connect -------------------------------------------------------------------------------


def test_connect_with_injected_transport_needs_no_sdk() -> None:
    adapter, fake = connected_adapter()
    assert adapter.is_connected
    assert adapter.transport is fake
    assert isinstance(adapter, RobotAdapter)


@pytest.mark.skipif(SDK_INSTALLED, reason="unitree_sdk2py installed; guard not reachable")
def test_connect_default_builds_dds_transport_and_raises_without_sdk() -> None:
    adapter = G1Adapter()
    with pytest.raises(RuntimeError, match="unitree_sdk2py"):
        adapter.connect()
    assert not adapter.is_connected


def test_io_requires_connect_even_with_injected_transport() -> None:
    adapter = G1Adapter(transport=FakeG1Transport())
    with pytest.raises(RuntimeError, match="connect"):
        adapter.read_state()
    with pytest.raises(RuntimeError, match="connect"):
        adapter.execute(make_chunk(np.zeros((1, 15))), prefix_steps=1)
    with pytest.raises(RuntimeError, match="connect"):
        adapter.hold()


# -- read_state ----------------------------------------------------------------------------


def test_read_state_roundtrip_exact() -> None:
    motor_q = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
    for name, canon_i in CANON_IDX.items():
        motor_q[MOTOR_IDX[name]] = 0.01 * (canon_i + 1)
    motor_q[0] = 0.7  # leg motor, must NOT leak into canonical q
    fake = FakeG1Transport(initial_q=motor_q, initial_gripper=(-0.2, 5.4))
    fake.set_imu(quat_wxyz=(0.5, 0.5, 0.5, 0.5), gyro=(0.1, -0.2, 0.3), acc=(0.0, 0.0, 9.81))
    adapter, _ = connected_adapter(
        config={"gripper_vendor_min": -0.2, "gripper_vendor_max": 5.4}, fake=fake
    )
    state = adapter.read_state()
    assert state.validate(G1_SPEC) == []
    np.testing.assert_array_equal(state.q, G1Adapter.motor_to_canonical(motor_q))
    np.testing.assert_array_equal(state.dq, np.zeros(15, dtype=np.float32))
    np.testing.assert_array_equal(
        state.imu.orientation_wxyz, np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    )
    np.testing.assert_allclose(state.imu.angular_velocity, [0.1, -0.2, 0.3], atol=1e-7)
    # Gripper: vendor range [-0.2, 5.4] -> canonical [0, 1].
    np.testing.assert_allclose(state.gripper_state, [0.0, 1.0], atol=1e-6)
    assert state.timestamp_ns == fake.tick_ns
    assert all(state.validity.as_dict().values())


def test_read_state_stale_tick_degrades_validity() -> None:
    adapter, _fake = connected_adapter()
    fresh = adapter.read_state()
    assert all(fresh.validity.as_dict().values())
    stale = adapter.read_state()  # no writes in between -> tick unchanged
    assert not any(stale.validity.as_dict().values())
    adapter.execute(make_chunk(np.zeros((1, 15))), prefix_steps=1)  # writes advance the tick
    recovered = adapter.read_state()
    assert all(recovered.validity.as_dict().values())


def test_forget_tick_clears_the_staleness_memory_for_a_deliberate_clock_rewind() -> None:
    """The stale-tick detector is the runtime's only liveness signal, so it has exactly ONE
    owner: ``forget_tick()``. A simulator's episode reset rewinds the transport clock, and
    "the tick did not change" then no longer means "no new sample" — without this hook the
    caller would have to poke ``_last_tick_ns`` from outside the class."""
    adapter, _fake = connected_adapter()
    assert all(adapter.read_state().validity.as_dict().values())
    assert not any(adapter.read_state().validity.as_dict().values())  # same tick -> stale
    adapter.forget_tick()
    assert all(adapter.read_state().validity.as_dict().values())
    # It clears the memory once; it does not disable staleness detection.
    assert not any(adapter.read_state().validity.as_dict().values())


def test_read_state_reports_a_fresh_imu_as_valid_carrying_the_transports_gravity_vector(
) -> None:
    """The DEPLOY half of the T-31 validity divergence, pinned so it cannot be "fixed" here.

    ``FakeG1Transport`` defaults to ``acc = (0, 0, 9.81)`` and a real G1 publishes a real IMU,
    so ``imu=True`` with a gravity payload is the honest answer for this adapter. The T-16
    checkpoints trained on gr00t episodes that mark ``imu=False`` in every state, so their
    encoder only ever saw the learned ``missing['imu']`` vector — but reconciling those two
    facts belongs to ``PolicyContract``, not to a flag flipped here. Lying in this adapter
    would break every consumer that is not that one checkpoint.
    """
    adapter, _fake = connected_adapter()
    state = adapter.read_state()

    assert state.validity.imu is True
    np.testing.assert_allclose(state.imu.linear_acceleration, [0.0, 0.0, 9.81], atol=1e-6)


def test_a_policy_contract_masks_the_g1_imu_group_down_for_a_checkpoint_trained_without_it(
) -> None:
    """The repair, end to end on the real adapter: the policy sees the mask training used
    while the raw state — the one the safety layer judges — is left alone."""
    from wam.runtime.executor import PolicyContract, StateGroupUse

    adapter, _fake = connected_adapter()
    state = adapter.read_state()
    contract = PolicyContract(state_groups={"imu": StateGroupUse.NEVER})

    conformed, divergences = contract.conform(state)

    assert conformed.validity.imu is False
    assert state.validity.imu is True
    assert [d.group for d in divergences] == ["imu"]
    # The payload is untouched; only the mask the encoder reads changed.
    np.testing.assert_array_equal(
        conformed.imu.linear_acceleration, state.imu.linear_acceleration
    )


def test_read_state_missing_gripper_degrades_only_gripper() -> None:
    class NoGripperTransport(FakeG1Transport):
        def read_low_state(self) -> dict:
            low = super().read_low_state()
            del low["gripper"]
            return low

    fake = NoGripperTransport()
    adapter, _ = connected_adapter(fake=fake)
    state = adapter.read_state()
    assert state.validity.q and state.validity.dq and state.validity.imu
    assert not state.validity.gripper
    np.testing.assert_array_equal(state.gripper_state, np.zeros(2, dtype=np.float32))


# -- execute -------------------------------------------------------------------------------


def test_execute_sends_exactly_prefix_steps_with_correct_mapping() -> None:
    adapter, fake = connected_adapter()
    deltas = np.zeros((4, 15), dtype=np.float32)
    deltas[:, CANON_IDX["waist_yaw"]] = 0.01
    deltas[:, CANON_IDX["left_elbow"]] = 0.02
    deltas[:, CANON_IDX["right_shoulder_pitch"]] = -0.03
    adapter.execute(make_chunk(deltas, dt_s=0.1), prefix_steps=2)

    assert len(fake.motor_commands) == 2  # exactly prefix_steps, not the full chunk
    assert len(fake.gripper_commands) == 2
    first, second = fake.motor_commands
    # Spot-check waist + one joint per arm at their G1 motor indices (12 / 18 / 22).
    np.testing.assert_allclose(first["q_target"][MOTOR_IDX["waist_yaw"]], 0.01, atol=1e-6)
    np.testing.assert_allclose(first["q_target"][MOTOR_IDX["left_elbow"]], 0.02, atol=1e-6)
    np.testing.assert_allclose(
        first["q_target"][MOTOR_IDX["right_shoulder_pitch"]], -0.03, atol=1e-6
    )
    # Deltas integrate cumulatively from the position read once at execute() start.
    np.testing.assert_allclose(second["q_target"][MOTOR_IDX["waist_yaw"]], 0.02, atol=1e-6)
    np.testing.assert_allclose(second["q_target"][MOTOR_IDX["left_elbow"]], 0.04, atol=1e-6)
    # Gains: configured value at mapped indices, zero at unmapped ones.
    cfg = adapter.config
    assert first["kp"][MOTOR_IDX["waist_yaw"]] == cfg.kp[CANON_IDX["waist_yaw"]]
    assert first["kd"][MOTOR_IDX["left_elbow"]] == cfg.kd[CANON_IDX["left_elbow"]]
    assert all(first["kp"][i] == 0.0 and first["kd"][i] == 0.0 for i in UNMAPPED)
    np.testing.assert_array_equal(first["dq_target"], np.zeros(G1_NUM_MOTORS))


def test_execute_integrates_deltas_from_reread_position_not_zero() -> None:
    motor_q = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
    motor_q[MOTOR_IDX["waist_yaw"]] = 0.5
    adapter, fake = connected_adapter(fake=FakeG1Transport(initial_q=motor_q))
    deltas = np.zeros((1, 15), dtype=np.float32)
    deltas[0, CANON_IDX["waist_yaw"]] = 0.1
    adapter.execute(make_chunk(deltas), prefix_steps=1)
    np.testing.assert_allclose(
        fake.motor_commands[0]["q_target"][MOTOR_IDX["waist_yaw"]], 0.6, atol=1e-6
    )


def test_execute_unmapped_motors_hold_current_position() -> None:
    motor_q = np.zeros(G1_NUM_MOTORS, dtype=np.float32)
    motor_q[0] = 0.4  # left hip pitch
    motor_q[13] = -0.2  # waist roll (locked on the 23-DoF variant)
    adapter, fake = connected_adapter(fake=FakeG1Transport(initial_q=motor_q, lag=1.0))
    adapter.execute(make_chunk(np.full((2, 15), 0.05, dtype=np.float32)), prefix_steps=2)
    for cmd in fake.motor_commands:
        np.testing.assert_allclose(cmd["q_target"][0], 0.4, atol=1e-6)
        np.testing.assert_allclose(cmd["q_target"][13], -0.2, atol=1e-6)


def test_execute_clips_position_and_velocity_limits_before_sending() -> None:
    cfg = G1Config(dq_max=(0.5,) * 15)  # dt=0.1 -> max 0.05 rad per step
    adapter, fake = connected_adapter(config=cfg)
    deltas = np.full((40, 15), 10.0, dtype=np.float32)  # wildly over both limits
    adapter.execute(make_chunk(deltas, dt_s=0.1), prefix_steps=40)
    first = fake.motor_commands[0]["q_target"]
    mapped = sorted(MAPPED)
    np.testing.assert_allclose(first[mapped], 0.05, atol=1e-6)  # velocity-clipped
    last = fake.motor_commands[-1]["q_target"]
    assert np.all(last[mapped] <= np.asarray(cfg.q_max) + 1e-6)  # position-clipped
    np.testing.assert_allclose(np.max(last[mapped]), cfg.q_max[0], atol=1e-6)


def test_execute_gripper_converted_to_vendor_units() -> None:
    adapter, fake = connected_adapter(
        config={"gripper_vendor_min": -0.2, "gripper_vendor_max": 5.4}
    )
    chunk = make_chunk(
        np.zeros((2, 15), dtype=np.float32), gripper=np.array([0.0, 1.0], dtype=np.float32)
    )
    adapter.execute(chunk, prefix_steps=2)
    np.testing.assert_allclose(fake.gripper_commands[0], (-0.2, -0.2), atol=1e-6)
    np.testing.assert_allclose(fake.gripper_commands[1], (5.4, 5.4), atol=1e-6)


def test_execute_prefix_longer_than_chunk_and_rejects_negative() -> None:
    adapter, fake = connected_adapter()
    adapter.execute(make_chunk(np.zeros((3, 15))), prefix_steps=99)
    assert len(fake.motor_commands) == 3
    with pytest.raises(ValueError):
        adapter.execute(make_chunk(np.zeros((1, 15))), prefix_steps=-1)
    with pytest.raises(ValueError):
        adapter.execute(make_chunk(np.zeros((1, 14))), prefix_steps=1)  # wrong width


def test_execute_paces_steps_dt_apart_on_the_wall_clock() -> None:
    # The per-step dq_max * dt clip is a velocity limit ONLY if successive targets are
    # actually sent dt_s apart — execute() must pace the stream, never fire it in a burst.
    ft = FakeTime()
    adapter, fake = connected_adapter(fake_time=ft)
    adapter.execute(make_chunk(np.zeros((4, 15), dtype=np.float32), dt_s=0.1), prefix_steps=3)
    assert len(fake.motor_commands) == 3
    # 3 steps -> 2 inter-step delays of exactly dt_s (the fake clock advances only in sleep).
    assert ft.sleeps == pytest.approx([0.1, 0.1])
    # A single step needs no pacing delay.
    ft2 = FakeTime()
    adapter2, _ = connected_adapter(fake_time=ft2)
    adapter2.execute(make_chunk(np.zeros((1, 15), dtype=np.float32), dt_s=0.1), prefix_steps=1)
    assert ft2.sleeps == []


def test_execute_pacing_absorbs_elapsed_time() -> None:
    # If work between steps already consumed wall time, only the remainder is slept.
    ft = FakeTime()

    class SlowWriteTransport(FakeG1Transport):
        def write_gripper_cmd(self, left: float, right: float) -> None:
            super().write_gripper_cmd(left, right)
            ft.t += 0.04  # each step's writes take 40 ms of wall time

    adapter, _ = connected_adapter(fake=SlowWriteTransport(), fake_time=ft)
    adapter.execute(make_chunk(np.zeros((2, 15), dtype=np.float32), dt_s=0.1), prefix_steps=2)
    assert ft.sleeps == pytest.approx([0.06])  # 0.1 - 0.04 already elapsed


def test_execute_rejects_ee_delta_for_mvp() -> None:
    adapter, _ = connected_adapter()
    chunk = make_chunk(np.zeros((1, 15)))
    chunk.mode = ActionMode.EE_DELTA
    with pytest.raises(NotImplementedError, match="EE_DELTA"):
        adapter.execute(chunk, prefix_steps=1)


# -- hold / estop --------------------------------------------------------------------------


def test_hold_resends_current_q_with_zero_dq() -> None:
    motor_q = np.linspace(-0.3, 0.3, G1_NUM_MOTORS).astype(np.float32)
    adapter, fake = connected_adapter(fake=FakeG1Transport(initial_q=motor_q))
    adapter.hold()
    assert len(fake.motor_commands) == 1
    cmd = fake.motor_commands[0]
    np.testing.assert_allclose(cmd["q_target"], motor_q, atol=1e-6)
    np.testing.assert_array_equal(cmd["dq_target"], np.zeros(G1_NUM_MOTORS))
    assert cmd["kp"][MOTOR_IDX["waist_yaw"]] > 0.0


def test_estop_damps_latches_and_blocks_execute() -> None:
    adapter, fake = connected_adapter()
    adapter.estop()
    assert fake.damp_count == 1
    assert adapter.is_estopped
    adapter.execute(make_chunk(np.full((2, 15), 0.1, dtype=np.float32)), prefix_steps=2)
    assert fake.motor_commands == []  # latched -> no-op
    assert fake.gripper_commands == []
    adapter.hold()
    assert fake.motor_commands == []  # hold is a no-op too; damping stays in charge
    adapter.clear_estop()
    adapter.execute(make_chunk(np.zeros((1, 15))), prefix_steps=1)
    assert len(fake.motor_commands) == 1


def test_estop_latches_even_when_transport_damp_raises() -> None:
    # A DDS publish failure during emergency damping must NOT bypass the e-stop latch:
    # the exception propagates, but the adapter refuses all further motion commands.
    class DampFailTransport(FakeG1Transport):
        def emergency_damp(self) -> None:
            raise RuntimeError("DDS publish failed")

    fake = DampFailTransport()
    adapter, _ = connected_adapter(fake=fake)
    with pytest.raises(RuntimeError, match="DDS publish failed"):
        adapter.estop()
    assert adapter.is_estopped  # latched despite the failed damp
    adapter.execute(make_chunk(np.full((2, 15), 0.1, dtype=np.float32)), prefix_steps=2)
    assert fake.motor_commands == []  # execute is a no-op while latched
    assert fake.gripper_commands == []


def test_estop_without_transport_latches_safely() -> None:
    adapter = G1Adapter()
    adapter.estop()  # no transport attached: must not raise
    assert adapter.is_estopped
    adapter.clear_estop()
    assert not adapter.is_estopped


# -- config --------------------------------------------------------------------------------


def test_config_gains_validated() -> None:
    cfg = G1Config()
    assert len(cfg.kp) == 15 and len(cfg.kd) == 15
    assert all(g > 0 for g in cfg.kp)  # conservative but active defaults
    with pytest.raises(ValueError):
        G1Config(kp=(20.0,) * 3)
    with pytest.raises(ValueError):
        G1Config(kd=(-1.0,) * 15)
