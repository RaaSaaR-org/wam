"""Tests for wam.runtime.mock_loop (T-03: M0 exit criterion, no hardware)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    IMUState,
    JsonlRunLogger,
    Observation,
    Policy,
    RobotState,
    RunMetadata,
)
from wam.robot import MockRobot
from wam.runtime.mock_loop import DEFAULT_INSTRUCTION, DummyPolicy, run_mock_loop
from wam.safety import SafetyConfig, SafetyLayer, Watchdog

REPO_ROOT = Path(__file__).resolve().parent.parent
N_JOINTS = 6
SPEC = CanonicalSpaceSpec(joint_names=tuple(f"joint_{i}" for i in range(N_JOINTS)))
DT_S = 0.05


def make_safety_config(**overrides: object) -> SafetyConfig:
    base: dict[str, object] = {
        "q_min": (-3.0,) * N_JOINTS,
        "q_max": (3.0,) * N_JOINTS,
        "dq_max": (1.5,) * N_JOINTS,
        "ddq_max": (4.0,) * N_JOINTS,
        "workspace_min": (0.1, -0.6, 0.6),
        "workspace_max": (0.8, 0.6, 1.4),
        "ee_max_lin_vel_m_s": 0.5,
        "ee_max_step_m": 0.05,
        "gripper_rate_max": 2.0,
        "chunk_timeout_s": 0.5,
    }
    base.update(overrides)
    return SafetyConfig(**base)  # type: ignore[arg-type]


def make_robot() -> MockRobot:
    return MockRobot(spec=SPEC, q_min=-3.14, q_max=3.14, dq_max=2.0)


def make_logger(tmp_path: Path) -> JsonlRunLogger:
    metadata = RunMetadata.create("test-run", {"test": True}, git_commit="deadbeef")
    return JsonlRunLogger(tmp_path / "run.jsonl", metadata)


def make_state(t_ns: int = 0) -> RobotState:
    return RobotState(
        timestamp_ns=t_ns,
        q=np.zeros(N_JOINTS, dtype=np.float32),
        dq=np.zeros(N_JOINTS, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.zeros(3, dtype=np.float32),
        ),
        gripper_state=np.zeros(1, dtype=np.float32),
    )


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class WildPolicy:
    """Emits huge joint deltas + a gripper jump: must be projected, never rejected."""

    def predict(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.full((4, N_JOINTS), 10.0, dtype=np.float32),
            gripper_target=np.ones(4, dtype=np.float32),
            dt_s=DT_S,
        )


class NaNPolicy:
    """Emits NaN targets: must be replaced by a zero-delta HOLD chunk."""

    def predict(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.full((4, N_JOINTS), np.nan, dtype=np.float32),
            gripper_target=np.zeros(4, dtype=np.float32),
            dt_s=DT_S,
        )


# ---------------------------------------------------------------- DummyPolicy


def test_dummy_policy_conforms_to_policy_protocol() -> None:
    assert isinstance(DummyPolicy(SPEC), Policy)


def test_dummy_policy_deterministic_and_valid() -> None:
    policy = DummyPolicy(SPEC, steps=8, dt_s=DT_S)
    obs = Observation(images={}, state=make_state(123_000_000), instruction="x")
    c1 = policy.predict(obs)
    c2 = policy.predict(obs)
    assert c1.validate(SPEC) == []
    assert c1.mode is ActionMode.JOINT_DELTA
    assert c1.targets.shape == (8, N_JOINTS)
    assert c1.targets.dtype == np.float32
    np.testing.assert_array_equal(c1.targets, c2.targets)
    np.testing.assert_array_equal(c1.gripper_target, c2.gripper_target)
    assert float(c1.gripper_target.min()) >= 0.0
    assert float(c1.gripper_target.max()) <= 1.0
    assert float(np.abs(c1.targets).max()) > 0.0


def test_dummy_policy_phase_follows_state_timestamp() -> None:
    policy = DummyPolicy(SPEC)
    c0 = policy.predict(Observation(images={}, state=make_state(0), instruction=""))
    c1 = policy.predict(Observation(images={}, state=make_state(1_000_000_000), instruction=""))
    assert not np.array_equal(c0.targets, c1.targets)


def test_dummy_policy_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        DummyPolicy(SPEC, steps=0)
    with pytest.raises(ValueError):
        DummyPolicy(SPEC, dt_s=0.0)
    with pytest.raises(ValueError):
        DummyPolicy(SPEC, amplitude_rad=-0.1)


# ------------------------------------------------------------------ the loop


def test_loop_happy_path_executes_and_logs(tmp_path: Path) -> None:
    robot = make_robot()
    cfg = make_safety_config()
    logger = make_logger(tmp_path)
    with logger:
        result = run_mock_loop(
            robot,
            DummyPolicy(SPEC, steps=8, dt_s=DT_S),
            SafetyLayer(cfg, spec=SPEC),
            logger,
            5,
            4,
            watchdog=Watchdog.from_config(cfg),
        )
    assert result.iterations == 5
    assert result.executed_iterations == 5
    assert result.watchdog_timeouts == 0
    # Dummy policy defaults are clean-by-construction against the default limits.
    assert result.interventions_total == 0
    assert result.intervention_kinds == {}

    records = read_records(logger.path)
    assert len(records) == 5
    for i, rec in enumerate(records):
        assert rec["kind"] == "loop_iteration"
        assert rec["iteration"] == i
        assert rec["run_id"] == "test-run"
        assert rec["config_hash"] == logger.metadata.config_hash
        assert rec["instruction"] == DEFAULT_INSTRUCTION
        assert rec["executed"] is True
        assert rec["prefix_steps"] == 4
        assert rec["interventions"] == []
        assert rec["watchdog"] == {"enabled": True, "expired": False, "action": None}
        assert rec["chunk"]["mode"] == "joint_delta"
        assert rec["chunk"]["num_steps"] == 8
        assert rec["safe_chunk"]["targets_finite"] is True
        assert set(rec["timings_ms"]) == {"predict", "filter", "execute", "total"}
        assert all(v >= 0.0 for v in rec["timings_ms"].values())
        assert rec["state"]["validity"] == {"q": True, "dq": True, "imu": True, "gripper": True}
    # Receding horizon: 5 iterations x 4 executed steps x 0.05 s of simulated time.
    assert robot.sim_time_ns == 5 * 4 * 50_000_000
    assert float(np.abs(robot.read_state().q).max()) > 0.0


def test_stall_triggers_watchdog_hold_and_recovers(tmp_path: Path) -> None:
    robot = make_robot()
    cfg = make_safety_config()
    logger = make_logger(tmp_path)
    with logger:
        result = run_mock_loop(
            robot,
            DummyPolicy(SPEC, steps=8, dt_s=DT_S),
            SafetyLayer(cfg, spec=SPEC),
            logger,
            5,
            4,
            watchdog=Watchdog.from_config(cfg),
            stall_at={2},
        )
    assert result.watchdog_timeouts == 1
    assert result.executed_iterations == 4
    assert result.intervention_kinds["watchdog_timeout"] == 1
    # After the hold, dq is zero but the sinusoid phase implies motion: the safety layer
    # must ramp the velocity back up under the accel limit (projection, not rejection).
    assert result.intervention_kinds.get("accel_limit", 0) > 0

    records = read_records(logger.path)
    stalled = records[2]
    assert stalled["stalled"] is True
    assert stalled["executed"] is False
    assert stalled["safe_chunk"] is None
    assert stalled["watchdog"]["expired"] is True
    assert stalled["watchdog"]["action"] == "hold"
    assert [iv["kind"] for iv in stalled["interventions"]] == ["watchdog_timeout"]
    # Loop recovers: the following iterations execute normally.
    assert records[3]["executed"] is True
    assert records[4]["executed"] is True
    assert not robot.is_estopped


def test_stall_hold_reaches_robot(tmp_path: Path) -> None:
    robot = make_robot()
    cfg = make_safety_config()
    logger = make_logger(tmp_path)
    with logger:
        run_mock_loop(
            robot,
            DummyPolicy(SPEC),
            SafetyLayer(cfg, spec=SPEC),
            logger,
            3,
            4,
            watchdog=Watchdog.from_config(cfg),
            stall_at={2},  # stall on the LAST iteration -> hold must remain latched
        )
    assert robot.is_holding
    assert not robot.is_estopped


def test_stall_with_stop_policy_estops(tmp_path: Path) -> None:
    robot = make_robot()
    cfg = make_safety_config(timeout_policy="stop")
    logger = make_logger(tmp_path)
    with logger:
        result = run_mock_loop(
            robot,
            DummyPolicy(SPEC),
            SafetyLayer(cfg, spec=SPEC),
            logger,
            2,
            4,
            watchdog=Watchdog.from_config(cfg),
            stall_at={1},
        )
    assert result.watchdog_timeouts == 1
    assert robot.is_estopped
    records = read_records(logger.path)
    assert records[1]["watchdog"]["action"] == "stop"


def test_unsafe_chunks_projected_and_logged(tmp_path: Path) -> None:
    robot = make_robot()
    cfg = make_safety_config()
    logger = make_logger(tmp_path)
    with logger:
        result = run_mock_loop(robot, WildPolicy(), SafetyLayer(cfg, spec=SPEC), logger, 3, 2)
    # Projection, not rejection: every iteration still executes a safe chunk.
    assert result.executed_iterations == 3
    assert result.interventions_total > 0
    assert "accel_limit" in result.intervention_kinds
    assert "gripper_rate" in result.intervention_kinds
    records = read_records(logger.path)
    assert all(len(rec["interventions"]) > 0 for rec in records)
    # The robot never left the SAFETY position limits (tighter than its own).
    assert float(robot.read_state().q.max()) <= 3.0 + 1e-6


def test_nan_chunk_rejected_to_hold(tmp_path: Path) -> None:
    robot = make_robot()
    q_before = robot.read_state().q.copy()
    cfg = make_safety_config()
    logger = make_logger(tmp_path)
    with logger:
        result = run_mock_loop(robot, NaNPolicy(), SafetyLayer(cfg, spec=SPEC), logger, 1, 4)
    assert result.intervention_kinds == {"nan_reject": 1}
    assert result.executed_iterations == 1  # the HOLD chunk is executed, not the NaN one
    np.testing.assert_array_equal(robot.read_state().q, q_before)
    records = read_records(logger.path)
    assert records[0]["chunk"]["targets_finite"] is False
    assert records[0]["safe_chunk"]["targets_finite"] is True


def test_loop_argument_validation(tmp_path: Path) -> None:
    robot = make_robot()
    cfg = make_safety_config()
    safety = SafetyLayer(cfg, spec=SPEC)
    logger = make_logger(tmp_path)  # never opened: validation must fire first
    with pytest.raises(ValueError):
        run_mock_loop(robot, DummyPolicy(SPEC), safety, logger, 1, 0)
    with pytest.raises(ValueError):
        run_mock_loop(robot, DummyPolicy(SPEC), safety, logger, -1, 4)
    with pytest.raises(ValueError):
        run_mock_loop(robot, DummyPolicy(SPEC), safety, logger, 1, 4, stall_s=-1.0)


# --------------------------------------------------------------- top level


def test_top_level_exports() -> None:
    import wam

    assert wam.__version__
    assert wam.RobotState is RobotState
    assert wam.ActionChunk is ActionChunk
    assert wam.SafetyLayer is SafetyLayer
    assert callable(wam.get_robot)


# --------------------------------------------------------------------- CLI


def test_cli_end_to_end(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "run_mock_loop.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--iterations", "3", "--log-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("OK ")
    assert "iterations=3" in proc.stdout
    logs = list(tmp_path.glob("*.jsonl"))
    assert len(logs) == 1
    records = read_records(logs[0])
    assert len(records) == 4  # run_metadata + 3 iterations
    assert records[0]["kind"] == "run_metadata"
    assert [r["kind"] for r in records[1:]] == ["loop_iteration"] * 3
    run_id = records[0]["run_id"]
    assert run_id.startswith("mock-loop-")
    assert all(r["run_id"] == run_id for r in records)
    assert all(r["config_hash"] == records[0]["config_hash"] for r in records)


def test_cli_stall_flag_triggers_watchdog(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "run_mock_loop.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--iterations",
            "3",
            "--stall-at",
            "1",
            "--log-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "watchdog_timeouts=1" in proc.stdout
    assert "executed=2" in proc.stdout
