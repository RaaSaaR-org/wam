"""Tests for wam.runtime.executor + wam.runtime.policies (T-19, FR-05)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from wam.interfaces import (
    ActionChunk,
    CanonicalSpaceSpec,
    JsonlRunLogger,
    Observation,
    Policy,
    RobotState,
    RunMetadata,
)
from wam.robot import MockRobot
from wam.runtime import (
    ClosedLoopExecutor,
    DummyPolicy,
    ExecutorConfig,
    RolloutResult,
    run_rollouts,
)
from wam.safety import SafetyConfig, SafetyLayer, Watchdog

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = REPO_ROOT / "runs" / "d1-overfit-seed0" / "checkpoint.safetensors"

N_JOINTS = 6  # matches the d1-overfit-seed0 checkpoint's canonical space
SPEC = CanonicalSpaceSpec(joint_names=tuple(f"joint_{i}" for i in range(N_JOINTS)))
DT_S = 0.05

# SHARED ROLLOUT LOG CONTRACT key sets (+ kind and the JsonlRunLogger stamps).
STAMPS = {"kind", "run_id", "config_hash"}
CYCLE_KEYS = STAMPS | {
    "rollout_id",
    "cycle",
    "now_ns",
    "policy_latency_ms",
    "deadline_missed",
    "executed",
    "prefix_steps",
    "chunk_steps",
    "interventions",
    "watchdog",
}
SUMMARY_KEYS = STAMPS | {
    "rollout_id",
    "success",
    "task",
    "duration_s",
    "cycles",
    "executed_cycles",
    "interventions_total",
    "intervention_kinds",
    "watchdog_timeouts",
    "deadline_misses",
    "estopped",
    "policy_rate_hz",
}


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


def make_logger(tmp_path: Path) -> JsonlRunLogger:
    metadata = RunMetadata.create("test-exec", {"test": True}, git_commit="deadbeef")
    return JsonlRunLogger(tmp_path / "rollouts.jsonl", metadata)


def make_executor(
    tmp_path: Path,
    *,
    robot: MockRobot | None = None,
    policy: Policy | None = None,
    safety_cfg: SafetyConfig | None = None,
    watchdog: Watchdog | None = None,
    clock: object = None,
    **config_overrides: object,
) -> tuple[ClosedLoopExecutor, MockRobot, JsonlRunLogger]:
    robot = robot or MockRobot(spec=SPEC)
    cfg = safety_cfg or make_safety_config()
    config = ExecutorConfig(
        **{"prefix_steps": 4, "max_cycles": 5, **config_overrides}  # type: ignore[arg-type]
    )
    logger = make_logger(tmp_path)
    executor = ClosedLoopExecutor(
        robot,
        policy or DummyPolicy(SPEC, steps=8, dt_s=DT_S),
        SafetyLayer(cfg, spec=SPEC),
        watchdog,
        logger,
        config,
        clock=clock,  # type: ignore[arg-type]
    )
    return executor, robot, logger


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class FakeClock:
    """Deterministic injectable clock (seconds); advanced explicitly by tests/policies."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class SlowPolicy:
    """Wraps a policy and advances the fake clock during predict -> deadline miss."""

    def __init__(self, inner: Policy, clock: FakeClock, latency_s: float) -> None:
        self._inner = inner
        self._clock = clock
        self._latency_s = latency_s

    def predict(self, observation: Observation) -> ActionChunk:
        self._clock.advance(self._latency_s)
        return self._inner.predict(observation)


class EstopPolicy:
    """Presses the robot's e-stop while predicting at a given call index."""

    def __init__(self, inner: Policy, robot: MockRobot, at_call: int) -> None:
        self._inner = inner
        self._robot = robot
        self._at_call = at_call
        self.calls = 0

    def predict(self, observation: Observation) -> ActionChunk:
        if self.calls == self._at_call:
            self._robot.estop()
        self.calls += 1
        return self._inner.predict(observation)


class RecordingRobot(MockRobot):
    """MockRobot that records every (chunk_steps, prefix_steps) execute call."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.execute_calls: list[tuple[int, int]] = []

    def execute(self, chunk: ActionChunk, prefix_steps: int) -> None:
        self.execute_calls.append((chunk.num_steps, prefix_steps))
        super().execute(chunk, prefix_steps)


# ------------------------------------------------------------------- config


def test_executor_config_validation() -> None:
    with pytest.raises(ValidationError):
        ExecutorConfig(prefix_steps=0, max_cycles=5)
    with pytest.raises(ValidationError):
        ExecutorConfig(prefix_steps=4, max_cycles=0)
    with pytest.raises(ValidationError):
        ExecutorConfig(prefix_steps=4, max_cycles=5, policy_deadline_ms=0.0)
    with pytest.raises(ValidationError):
        ExecutorConfig(prefix_steps=4, max_cycles=5, min_policy_rate_hz=0.0)
    cfg = ExecutorConfig(prefix_steps=4, max_cycles=5)
    assert cfg.min_policy_rate_hz == 2.0
    assert cfg.stop_on_estop is True
    with pytest.raises(ValidationError):
        cfg.prefix_steps = 2  # type: ignore[misc]


# --------------------------------------------------------------- happy path


def test_happy_path_records_match_contract_exactly(tmp_path: Path) -> None:
    cfg = make_safety_config()
    executor, robot, logger = make_executor(
        tmp_path, safety_cfg=cfg, watchdog=Watchdog.from_config(cfg)
    )
    with logger:
        result = executor.run_rollout("r-happy")

    assert result.cycles == 5
    assert result.executed_cycles == 5
    assert result.success is False
    assert result.estopped is False
    assert result.watchdog_timeouts == 0
    assert result.deadline_misses == 0
    assert result.interventions_total == 0  # DummyPolicy defaults are clean
    assert result.intervention_kinds == {}
    assert result.policy_rate_hz > 2.0  # wall clock: microsecond predictions
    assert result.below_min_policy_rate is False
    # Receding horizon: 5 cycles x 4 executed steps x 0.05 s simulated time.
    assert robot.sim_time_ns == 5 * 4 * 50_000_000

    records = read_records(logger.path)
    assert [r["kind"] for r in records] == ["control_cycle"] * 5 + ["rollout_summary"]

    for i, rec in enumerate(records[:5]):
        assert set(rec) == CYCLE_KEYS  # contract keys, exactly
        assert rec["rollout_id"] == "r-happy"
        assert rec["cycle"] == i
        assert isinstance(rec["now_ns"], int)
        assert rec["now_ns"] == i * 4 * 50_000_000
        assert isinstance(rec["policy_latency_ms"], float)
        assert rec["policy_latency_ms"] >= 0.0
        assert rec["deadline_missed"] is False
        assert rec["executed"] is True
        assert rec["prefix_steps"] == 4
        assert rec["chunk_steps"] == 8
        assert rec["interventions"] == []
        assert rec["watchdog"] == {"expired": False, "action": None}
        assert rec["run_id"] == "test-exec"
        assert rec["config_hash"] == logger.metadata.config_hash

    summary = records[-1]
    assert set(summary) == SUMMARY_KEYS  # contract keys, exactly
    assert summary["rollout_id"] == "r-happy"
    assert summary["success"] is False
    assert summary["task"] == "pick_and_place"
    assert isinstance(summary["duration_s"], float) and summary["duration_s"] > 0.0
    assert summary["cycles"] == 5
    assert summary["executed_cycles"] == 5
    assert summary["interventions_total"] == 0
    assert summary["intervention_kinds"] == {}
    assert summary["watchdog_timeouts"] == 0
    assert summary["deadline_misses"] == 0
    assert summary["estopped"] is False
    assert isinstance(summary["policy_rate_hz"], float)


def test_safety_interventions_logged_and_aggregated(tmp_path: Path) -> None:
    class WildPolicy:
        def predict(self, observation: Observation) -> ActionChunk:
            from wam.interfaces import ActionMode

            return ActionChunk(
                mode=ActionMode.JOINT_DELTA,
                targets=np.full((8, N_JOINTS), 10.0, dtype=np.float32),
                gripper_target=np.ones(8, dtype=np.float32),
                dt_s=DT_S,
            )

    executor, _, logger = make_executor(tmp_path, policy=WildPolicy(), max_cycles=3)
    with logger:
        result = executor.run_rollout("r-wild")
    # Projection, not rejection: everything still executes, interventions are logged.
    assert result.executed_cycles == 3
    assert result.interventions_total > 0
    assert "accel_limit" in result.intervention_kinds
    records = read_records(logger.path)
    cycle_recs = [r for r in records if r["kind"] == "control_cycle"]
    total = 0
    for rec in cycle_recs:
        assert len(rec["interventions"]) > 0
        for iv in rec["interventions"]:
            assert set(iv) == {"kind", "detail", "timestamp_ns"}
        total += len(rec["interventions"])
    summary = records[-1]
    assert summary["interventions_total"] == total == result.interventions_total
    assert summary["intervention_kinds"] == result.intervention_kinds


# ------------------------------------------------------------ deadline miss


def test_deadline_miss_discards_chunk_and_holds(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = SlowPolicy(DummyPolicy(SPEC, steps=8, dt_s=DT_S), clock, latency_s=0.6)
    cfg = make_safety_config()
    executor, robot, logger = make_executor(
        tmp_path,
        policy=policy,
        safety_cfg=cfg,
        watchdog=Watchdog.from_config(cfg),
        clock=clock,
        max_cycles=3,
        policy_deadline_ms=500.0,
    )
    with logger:
        result = executor.run_rollout("r-late")

    assert result.deadline_misses == 3
    assert result.executed_cycles == 0
    assert result.intervention_kinds == {"deadline_miss": 3}
    assert result.watchdog_timeouts == 0  # robot time never advanced -> no expiry
    assert robot.is_holding  # late chunk discarded, hold commanded
    assert not robot.is_estopped
    assert robot.sim_time_ns == 0  # nothing executed, ever
    # 3 cycles over 1.8 s of injected clock -> 1.67 Hz < 2 Hz floor.
    assert result.policy_rate_hz == pytest.approx(3 / 1.8)
    assert result.below_min_policy_rate is True

    records = read_records(logger.path)
    for rec in records[:3]:
        assert set(rec) == CYCLE_KEYS
        assert rec["deadline_missed"] is True
        assert rec["executed"] is False
        assert rec["prefix_steps"] == 0
        assert rec["chunk_steps"] == 8
        assert rec["policy_latency_ms"] == pytest.approx(600.0)
        assert [iv["kind"] for iv in rec["interventions"]] == ["deadline_miss"]
        assert rec["watchdog"] == {"expired": False, "action": None}
    assert records[-1]["deadline_misses"] == 3


def test_fast_policy_meets_deadline_with_injected_clock(tmp_path: Path) -> None:
    clock = FakeClock()
    policy = SlowPolicy(DummyPolicy(SPEC, steps=8, dt_s=DT_S), clock, latency_s=0.1)
    executor, _robot, logger = make_executor(
        tmp_path, policy=policy, clock=clock, max_cycles=4, policy_deadline_ms=500.0
    )
    with logger:
        result = executor.run_rollout("r-fast")
    assert result.deadline_misses == 0
    assert result.executed_cycles == 4
    assert result.policy_rate_hz == pytest.approx(4 / 0.4)
    assert result.below_min_policy_rate is False


# -------------------------------------------------------------- replanning


def test_replanning_robot_sees_only_prefixes(tmp_path: Path) -> None:
    robot = RecordingRobot(spec=SPEC)
    executor, _, logger = make_executor(tmp_path, robot=robot, max_cycles=4, prefix_steps=2)
    with logger:
        result = executor.run_rollout("r-replan")

    # One fresh prediction per cycle; the full 8-step chunk reaches the adapter but only
    # the 2-step prefix is integrated — the remainder is discarded and replaced.
    assert result.cycles == 4
    assert robot.execute_calls == [(8, 2)] * 4
    assert robot.sim_time_ns == 4 * 2 * 50_000_000  # prefix time only, never chunk time
    records = read_records(logger.path)
    for rec in records[:4]:
        assert rec["prefix_steps"] == 2
        assert rec["chunk_steps"] == 8


# ------------------------------------------------------- estop and watchdog


def test_estop_stops_rollout(tmp_path: Path) -> None:
    robot = MockRobot(spec=SPEC)
    policy = EstopPolicy(DummyPolicy(SPEC, steps=8, dt_s=DT_S), robot, at_call=2)
    executor, _, logger = make_executor(tmp_path, robot=robot, policy=policy, max_cycles=10)
    with logger:
        result = executor.run_rollout("r-estop")
    assert result.estopped is True
    assert result.success is False
    assert result.cycles == 3  # stopped right after the e-stop cycle, not max_cycles
    assert robot.is_estopped
    records = read_records(logger.path)
    assert records[-1]["estopped"] is True
    assert records[-1]["cycles"] == 3


def test_estop_without_stop_on_estop_runs_to_max_cycles(tmp_path: Path) -> None:
    robot = MockRobot(spec=SPEC)
    policy = EstopPolicy(DummyPolicy(SPEC, steps=8, dt_s=DT_S), robot, at_call=1)
    executor, _, logger = make_executor(
        tmp_path, robot=robot, policy=policy, max_cycles=4, stop_on_estop=False
    )
    with logger:
        result = executor.run_rollout("r-estop-go-on")
    assert result.estopped is True
    assert result.cycles == 4


def test_watchdog_stop_estops_and_ends_rollout(tmp_path: Path) -> None:
    # prefix 4 x 0.05 s = 0.2 s of robot time per cycle > 0.1 s timeout -> expiry at
    # cycle 1; STOP policy -> e-stop -> rollout ends.
    cfg = make_safety_config(chunk_timeout_s=0.1, timeout_policy="stop")
    executor, robot, logger = make_executor(
        tmp_path, safety_cfg=cfg, watchdog=Watchdog.from_config(cfg), max_cycles=10
    )
    with logger:
        result = executor.run_rollout("r-wd-stop")
    assert result.watchdog_timeouts == 1
    assert result.executed_cycles == 1
    assert result.cycles == 2
    assert result.estopped is True
    assert result.intervention_kinds["watchdog_timeout"] == 1
    assert robot.is_estopped
    records = read_records(logger.path)
    assert records[1]["watchdog"] == {"expired": True, "action": "stop"}
    assert records[1]["executed"] is False
    assert [iv["kind"] for iv in records[1]["interventions"]] == ["watchdog_timeout"]


def test_watchdog_hold_recovers(tmp_path: Path) -> None:
    cfg = make_safety_config(chunk_timeout_s=0.1)  # timeout_policy hold (default)
    executor, robot, logger = make_executor(
        tmp_path, safety_cfg=cfg, watchdog=Watchdog.from_config(cfg), max_cycles=4
    )
    with logger:
        result = executor.run_rollout("r-wd-hold")
    # Cycle 0 executes (arm+feed at t=0), cycle 1 expires -> hold (time frozen), cycle 2
    # executes again, cycle 3 expires: hold alternates with execution, never e-stops.
    assert result.watchdog_timeouts == 2
    assert result.executed_cycles == 2
    assert result.estopped is False
    assert not robot.is_estopped
    records = read_records(logger.path)
    assert records[1]["watchdog"] == {"expired": True, "action": "hold"}
    assert records[2]["executed"] is True


def test_frozen_robot_clock_state_rejects_escalate_to_estop(tmp_path: Path) -> None:
    """Regression: a stalled vendor controller (frozen tick -> degraded validity ->
    state_reject/HOLD every cycle) must trip the watchdog escalation, not silently run to
    max_cycles. The robot clock is frozen (it IS the failing signal), so the executor
    times the reject streak on the host clock and escalates per timeout_policy='stop'."""
    from wam.robot.g1 import G1_SPEC, G1Adapter
    from wam.robot.g1_transport import FakeG1Transport

    clock = FakeClock()
    fake = FakeG1Transport()
    fake.freeze_tick = True  # stalled vendor controller from the start
    robot = G1Adapter(transport=fake, clock=clock, sleep=lambda s: clock.advance(s))
    robot.connect()

    n = G1_SPEC.num_joints
    cfg = make_safety_config(
        q_min=(-3.0,) * n,
        q_max=(3.0,) * n,
        dq_max=(2.0,) * n,
        ddq_max=(8.0,) * n,
        chunk_timeout_s=0.1,
        timeout_policy="stop",
    )
    policy = SlowPolicy(DummyPolicy(G1_SPEC, steps=8, dt_s=0.02), clock, latency_s=0.05)
    logger = make_logger(tmp_path)
    executor = ClosedLoopExecutor(
        robot,
        policy,
        SafetyLayer(cfg, spec=G1_SPEC),
        Watchdog.from_config(cfg),
        logger,
        ExecutorConfig(prefix_steps=2, max_cycles=25, policy_deadline_ms=500.0),
        clock=clock,
    )
    with logger:
        result = executor.run_rollout("r-frozen")

    assert result.watchdog_timeouts == 1
    assert result.estopped is True
    assert robot.is_estopped and fake.damp_count == 1
    assert result.intervention_kinds.get("state_reject", 0) >= 2  # frozen tick, every cycle
    assert result.intervention_kinds.get("watchdog_timeout", 0) == 1
    assert result.cycles < 25  # ended by the escalation, not by max_cycles


def test_state_reject_cycles_do_not_feed_watchdog(tmp_path: Path) -> None:
    from wam.robot.g1 import G1_SPEC, G1Adapter
    from wam.robot.g1_transport import FakeG1Transport

    clock = FakeClock()
    fake = FakeG1Transport()
    fake.freeze_tick = True
    robot = G1Adapter(transport=fake, clock=clock, sleep=lambda s: clock.advance(s))
    robot.connect()
    n = G1_SPEC.num_joints
    cfg = make_safety_config(
        q_min=(-3.0,) * n, q_max=(3.0,) * n, dq_max=(2.0,) * n, ddq_max=(8.0,) * n
    )
    watchdog = Watchdog.from_config(cfg)
    policy = SlowPolicy(DummyPolicy(G1_SPEC, steps=8, dt_s=0.02), clock, latency_s=0.01)
    logger = make_logger(tmp_path)
    executor = ClosedLoopExecutor(
        robot,
        policy,
        SafetyLayer(cfg, spec=G1_SPEC),
        watchdog,
        logger,
        ExecutorConfig(prefix_steps=2, max_cycles=4, policy_deadline_ms=500.0),
        clock=clock,
    )
    with logger:
        result = executor.run_rollout("r-nofeed")
    # Cycle 0 is fresh (first read) and feeds; every later cycle is a stale-state reject
    # and must NOT re-arm the watchdog ("watchdog food" contract, g1_transport docstring).
    assert result.intervention_kinds.get("state_reject", 0) == 3
    assert watchdog.last_feed_ns == 0  # only the initial arm/clean feed at robot t=0


# ------------------------------------------------------------- success_fn


def test_success_fn_early_exit(tmp_path: Path) -> None:
    # 3 executed cycles advance robot time to 3 x 4 x 0.05 s = 0.6 s.
    threshold_ns = 3 * 4 * 50_000_000

    def success_fn(state: RobotState) -> bool:
        return int(state.timestamp_ns) >= threshold_ns

    executor, robot, logger = make_executor(tmp_path, max_cycles=10)
    with logger:
        result = executor.run_rollout("r-success", success_fn=success_fn)
    assert result.success is True
    assert result.cycles == 3  # early exit, not max_cycles
    assert result.executed_cycles == 3
    assert robot.sim_time_ns == threshold_ns
    records = read_records(logger.path)
    assert records[-1]["success"] is True
    assert records[-1]["cycles"] == 3


# ------------------------------------------------------------ run_rollouts


def test_run_rollouts_helper(tmp_path: Path) -> None:
    executor, _, logger = make_executor(tmp_path, max_cycles=2)
    with logger:
        results = run_rollouts(executor, 3, rollout_id_prefix="ep")
    assert [r.rollout_id for r in results] == ["ep-0000", "ep-0001", "ep-0002"]
    assert all(isinstance(r, RolloutResult) for r in results)
    assert all(r.cycles == 2 for r in results)
    records = read_records(logger.path)
    summaries = [r for r in records if r["kind"] == "rollout_summary"]
    assert [s["rollout_id"] for s in summaries] == ["ep-0000", "ep-0001", "ep-0002"]
    assert len([r for r in records if r["kind"] == "control_cycle"]) == 6
    with pytest.raises(ValueError):
        run_rollouts(executor, -1)


# ------------------------------------------------------- checkpoint policy


def test_dummy_policy_reexport() -> None:
    from wam.runtime.mock_loop import DummyPolicy as MockLoopDummy
    from wam.runtime.policies import DummyPolicy as ReexportedDummy

    assert ReexportedDummy is MockLoopDummy


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="d1-overfit-seed0 checkpoint not present")
def test_checkpoint_policy_loads_and_predicts(tmp_path: Path) -> None:
    from wam.runtime import CheckpointPolicy  # lazy export; torch import lives here

    policy = CheckpointPolicy(CHECKPOINT, device="cpu")
    assert isinstance(policy, Policy)
    assert policy.metadata.run_id == "d1-overfit-seed0"
    assert policy.metadata.checkpoint_ref  # traceability, AC-04

    robot = MockRobot(spec=SPEC)  # 64x64 front/wrist cameras match the checkpoint config
    state = robot.read_state()
    images = {name: frames[0] for name, frames in robot.render_frames(1).items()}
    obs = Observation(images=images, state=state, instruction="Greife die rote Tasse.")
    c1 = policy.predict(obs)
    c2 = policy.predict(obs)

    assert c1.validate(SPEC) == []
    assert c1.targets.shape == (8, N_JOINTS)
    assert c1.targets.dtype == np.float32
    assert np.isfinite(c1.targets).all()
    np.testing.assert_array_equal(c1.targets, c2.targets)  # deterministic eval/no-grad
    np.testing.assert_array_equal(c1.gripper_target, c2.gripper_target)
    assert float(c1.gripper_target.min()) >= 0.0
    assert float(c1.gripper_target.max()) <= 1.0

    # End-to-end: the real checkpoint drives the closed loop through the safety layer.
    executor, loop_robot, logger = make_executor(tmp_path, policy=policy, max_cycles=2)
    with logger:
        result = executor.run_rollout("r-ckpt")
    assert result.executed_cycles == 2
    assert loop_robot.sim_time_ns > 0


# --------------------------------------------------------------- JointCheckpointPolicy (T-16)
#
# The world-action counterpart of the block above. These build their own checkpoint in
# ``tmp_path`` rather than leaning on a ``runs/`` artifact: the joint checkpoints that exist
# there are 15-joint G1 models on 120x160 frames, while everything else in this module runs on
# MockRobot's 6-joint / 64x64 space.


def _joint_checkpoint(tmp_path: Path, camera: str = "front"):
    """An untrained JointWorldActionModel checkpoint shaped for MockRobot(spec=SPEC)."""
    from wam.backbones.tiny import TinyBackboneConfig
    from wam.decoders import ActionHeadConfig
    from wam.encoders import ActionChunkEncoderConfig, StateMLPConfig
    from wam.training import JointTrainer, JointTrainingConfig

    config = JointTrainingConfig(
        state=StateMLPConfig(embedding_dim=8, hidden_dims=(16,), num_joints=N_JOINTS),
        backbone=TinyBackboneConfig(
            feature_dim=32,
            patch_size=16,
            depth=1,
            num_heads=4,
            num_frames=2,
            image_hw=(64, 64),  # MockRobot's render size
            state_embedding_dim=8,
        ),
        action_encoder=ActionChunkEncoderConfig(
            latent_dim=8, target_dim=N_JOINTS, hidden_dims=(16,)
        ),
        head=ActionHeadConfig(
            feature_dim=32, num_steps=8, target_dim=N_JOINTS, hidden_dims=(32,), dt_s=DT_S
        ),
        camera=camera,
        seed=0,
    )
    path = tmp_path / "joint.safetensors"
    JointTrainer(config).save_checkpoint(path, run_id="joint-test")
    return path


def _mock_observation(robot: MockRobot) -> Observation:
    images = {name: frames[0] for name, frames in robot.render_frames(1).items()}
    return Observation(images=images, state=robot.read_state(), instruction="Greife den Wuerfel.")


def test_joint_checkpoint_policy_loads_and_predicts(tmp_path: Path) -> None:
    from wam.runtime import JointCheckpointPolicy  # lazy export; torch import lives here

    policy = JointCheckpointPolicy(_joint_checkpoint(tmp_path), device="cpu")
    assert isinstance(policy, Policy)
    assert policy.metadata.run_id == "joint-test"
    assert policy.metadata.checkpoint_ref  # traceability, AC-04
    assert policy.camera == "front"

    obs = _mock_observation(MockRobot(spec=SPEC))
    c1 = policy.predict(obs)
    c2 = policy.predict(obs)

    assert c1.validate(SPEC) == []
    assert c1.targets.shape == (8, N_JOINTS)
    assert c1.targets.dtype == np.float32
    assert np.isfinite(c1.targets).all()
    np.testing.assert_array_equal(c1.targets, c2.targets)  # deterministic eval/no-grad
    np.testing.assert_array_equal(c1.gripper_target, c2.gripper_target)


def test_joint_policy_reads_the_overridden_camera_and_fails_loudly_on_a_missing_one(
    tmp_path: Path,
) -> None:
    """The override picks a DIFFERENT view, and an absent key raises instead of falling back.

    A policy silently reading the wrong camera would look healthy in every log line the
    executor writes — the chunk is finite, in-bounds and on time. It is only wrong.
    """
    from wam.interfaces import ActionMode
    from wam.runtime import JointCheckpointPolicy

    path = _joint_checkpoint(tmp_path, camera="front")
    robot = MockRobot(spec=SPEC)
    robot.execute(  # move q[0] so the two cameras' dot columns differ
        ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.full((4, N_JOINTS), 0.05, dtype=np.float32),
            gripper_target=np.zeros(4, dtype=np.float32),
            dt_s=DT_S,
        ),
        prefix_steps=4,
    )
    obs = _mock_observation(robot)

    trained = JointCheckpointPolicy(path).predict(obs)
    override = JointCheckpointPolicy(path, camera="wrist")
    assert override.camera == "wrist"
    assert not np.array_equal(trained.targets, override.predict(obs).targets)

    with pytest.raises(KeyError, match="no camera 'nose'"):
        JointCheckpointPolicy(path, camera="nose").predict(obs)


def test_joint_predict_runs_one_backbone_pass_at_the_clean_timestep(tmp_path: Path) -> None:
    """Representation-only readout: ONE forward_flow per predict, at t=1 (clean).

    Both halves are the latency claim. A second pass would mean test-time denoising, and any
    t < 1 would mean the policy reads a partially destroyed observation — the flow convention
    is ``x_t = (1-t)*x0 + t*x1`` with x1 clean.
    """
    import torch

    from wam.runtime import JointCheckpointPolicy

    policy = JointCheckpointPolicy(_joint_checkpoint(tmp_path))
    backbone = policy.model.backbone
    obs = _mock_observation(MockRobot(spec=SPEC))

    seen_t: list[float] = []
    original = backbone.forward_flow

    def counting(video_latents, t, text_ctx, state_ctx):
        seen_t.append(float(torch.as_tensor(t).reshape(-1)[0]))
        return original(video_latents, t, text_ctx, state_ctx)

    backbone.forward_flow = counting  # type: ignore[method-assign]
    try:
        chunk = policy.predict(obs)
    finally:
        backbone.forward_flow = original  # type: ignore[method-assign]

    assert seen_t == [1.0]

    # ... and t is load-bearing: the same readout at the noisy end gives a different chunk.
    model = policy.model
    image = torch.as_tensor(obs.images[model.config.camera])
    frames = image.unsqueeze(0).expand(model.config.backbone.num_frames, -1, -1, -1)
    latents = backbone.encode_video(frames.unsqueeze(0))
    ctx = (
        backbone.condition_text(obs.instruction),
        backbone.condition_state(model.state_encoder.encode(obs.state)),
    )
    with torch.no_grad():
        _, clean = backbone.forward_flow(latents, torch.ones(1), *ctx)
        _, noisy = backbone.forward_flow(latents, torch.zeros(1), *ctx)
    np.testing.assert_array_equal(chunk.targets, model.action_head.decode(clean[0]).targets)
    assert not np.array_equal(chunk.targets, model.action_head.decode(noisy[0]).targets)


def test_the_two_checkpoint_policies_reject_each_others_artifacts(tmp_path: Path) -> None:
    """Action-only and world-action checkpoints are different artifacts. Fail at load, loudly.

    The two failure modes differ because the mismatch is caught at different layers: a joint
    checkpoint carries tensors ``ActionOnlyModel`` has no slots for (state_dict), while an
    action-only checkpoint is missing whole config sections the joint model requires (pydantic).
    """
    from pydantic import ValidationError

    from wam.runtime import CheckpointPolicy, JointCheckpointPolicy

    with pytest.raises(RuntimeError, match="Unexpected key.*action_encoder"):
        CheckpointPolicy(_joint_checkpoint(tmp_path))

    if CHECKPOINT.exists():
        with pytest.raises(ValidationError, match="action_encoder"):
            JointCheckpointPolicy(CHECKPOINT)


def test_joint_policy_drives_the_closed_loop_through_the_safety_layer(tmp_path: Path) -> None:
    from wam.runtime import JointCheckpointPolicy

    policy = JointCheckpointPolicy(_joint_checkpoint(tmp_path))
    executor, loop_robot, logger = make_executor(tmp_path, policy=policy, max_cycles=2)
    with logger:
        result = executor.run_rollout("r-joint")
    assert result.executed_cycles == 2
    assert loop_robot.sim_time_ns > 0
