"""M4 end-to-end regression (T-19..T-23): checkpoint -> wire -> executor -> acceptance.

Compressed version of the M4 verification sequence:
- E2 static release gate with the trained D1 checkpoint on MockRobot.
- Closed loop OVER THE WIRE: in-process PolicyServer(CheckpointPolicy) + RemotePolicy
  driving the ClosedLoopExecutor — 5 sim:reach rollouts + 2 fault-injection rollouts via
  ``scripts/rollout.py`` (the E2/E3 entry point).
- Rollout logs verified against the SHARED ROLLOUT LOG CONTRACT.
- Acceptance evaluation over the small sample: AC-03-style logic (zero violations,
  insufficient n -> pending_data, injected violation -> fail), AC-04 provenance over the
  wire, AC-06 handled faults.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wam.evaluation import evaluate_acceptance, is_safety_violation, verify_backbone_swap
from wam.evaluation.acceptance import (
    FAULT_INJECTION_TASK,
    LIMIT_BREACH_KIND,
    SIM_TASK_PREFIX,
    load_rollout_summaries,
    load_run_metadata,
)
from wam.interfaces import ActionChunk, JsonlRunLogger, Observation, RunMetadata
from wam.robot import MockRobot
from wam.runtime import ClosedLoopExecutor, ExecutorConfig
from wam.safety import SafetyConfig, SafetyLayer

_ROOT = Path(__file__).resolve().parent.parent
_CHECKPOINT = _ROOT / "runs" / "d1-overfit-seed0" / "checkpoint.safetensors"
_SAFETY_YAML = _ROOT / "configs" / "safety" / "default.yaml"
_BACKBONES = ("flux3", "wan_i2v")  # torch-free registry subset for AC-05

pytestmark = pytest.mark.skipif(
    not _CHECKPOINT.exists(), reason="trained D1 checkpoint not available"
)

# Exact key sets of the SHARED ROLLOUT LOG CONTRACT (+ the JsonlRunLogger stamps).
_STAMPS = {"run_id", "config_hash"}
_CYCLE_KEYS = {
    "kind",
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
} | _STAMPS
_SUMMARY_KEYS = {
    "kind",
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
} | _STAMPS


def _load_rollout_cli():
    spec = importlib.util.spec_from_file_location(
        "rollout_cli_m4", _ROOT / "scripts" / "rollout.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rollout_cli():
    return _load_rollout_cli()


@pytest.fixture(scope="module")
def checkpoint_policy():
    from wam.runtime.policies import CheckpointPolicy

    return CheckpointPolicy(_CHECKPOINT)


@pytest.fixture(scope="module")
def server_uri(checkpoint_policy):
    from wam.runtime.server import PolicyServer

    server = PolicyServer(checkpoint_policy)
    thread, port = server.run_in_thread()
    yield f"ws://127.0.0.1:{port}"
    server.stop()
    thread.join(timeout=10.0)


@pytest.fixture(scope="module")
def rollout_logs(rollout_cli, server_uri, tmp_path_factory) -> dict[str, Path]:
    """5 sim:reach + 2 fault-injection rollouts through the in-process server.

    --skip-e2 for the reason the same flag is passed in tests/test_runtime.py: every test built on
    this fixture is about the closed loop, the wire or run provenance, and none is about the E2
    release gate. Asserting rc == 0 without it makes that gate a silent precondition of all four,
    so one known recipe gap (T-48) reports as four unrelated failures in the layer under test.
    The gate is asserted where it belongs -- test_e2_static_gate_passes_with_checkpoint, xfailed
    strict against the same T-48 entry -- and its threshold is untouched here.
    """
    out_dir = tmp_path_factory.mktemp("m4-rollouts")
    common = ["--policy", "remote", "--server-uri", server_uri, "--out-dir", str(out_dir),
              "--skip-e2"]
    rc_sim = rollout_cli.main(
        [*common, "--rollouts", "5", "--task", "sim:reach", "--run-id", "e2e-sim",
         "--e2-probes", "6"]
    )
    rc_fault = rollout_cli.main(
        [*common, "--rollouts", "2", "--fault-injection", "--policy-deadline-ms", "100",
         "--run-id", "e2e-fault", "--e2-probes", "6"]
    )
    assert rc_sim == 0 and rc_fault == 0
    return {"sim": out_dir / "e2e-sim.jsonl", "fault": out_dir / "e2e-fault.jsonl"}


# --------------------------------------------------------------------------- E2 static


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T-48 recipe gap, open since 2026-08-01 and reproduced at c59000e (the commit that "
        "introduced this test), so it is not a regression: the D1 head emits max |ddq| 18.29 "
        "rad/s^2 against the robot's declared ddq_max 8.0 (configs/robot/mock.yaml:27, which "
        "build_safety_config prefers over the 4.0 in configs/safety/default.yaml), so accel_limit "
        "fires on 4-5 of every probe's 8 steps and safety_intervention_rate is 1.000. The cause is "
        "ActionLossWeights(smoothness=0.0) at scripts/overfit_d1.py:239; the fix is a training run "
        "over a shared gitignored artifact and is the owner's call. Kept as xfail rather than "
        "skipped, and strict, so that repairing the recipe FAILS here and says to delete this "
        "marker -- a skip would go quiet instead. NO GATE VALUE IS RELAXED by this marker: the "
        "check still runs and still evaluates max_intervention_rate at 0.1. See TASKS.md's T-48 "
        "entry and its 2026-08-25 correction."
    ),
)
def test_e2_static_gate_passes_with_checkpoint(rollout_cli, checkpoint_policy) -> None:
    from wam.evaluation import e2_static_checks
    from wam.interfaces import CanonicalSpaceSpec, load_config

    robot_section = load_config(_ROOT / "configs" / "robot" / "mock.yaml")["robot"]
    spec = CanonicalSpaceSpec(**robot_section["canonical_space"])
    safety_cfg = rollout_cli.build_safety_config(
        SafetyConfig.from_yaml(_SAFETY_YAML), spec, robot_section["limits"]
    )
    robot = MockRobot(spec=spec, seed=0)
    report = e2_static_checks(
        checkpoint_policy,
        robot,
        SafetyLayer(safety_cfg, spec=spec),
        spec,
        n_probes=8,
        instruction="Greife die rote Tasse.",  # trained D1 instruction
    )
    assert report.passed, report.failed_gates()
    # 8 steps * 0.05 s = 0.4 s chunks: below the PRD band -> warn-only, never a failure
    assert any("PRD band" in w for w in report.warnings)


# ------------------------------------------------------------------- closed loop / wire


def test_remote_rollouts_succeed_and_match_contract(rollout_logs) -> None:
    records = [json.loads(line) for line in rollout_logs["sim"].read_text().splitlines()]
    cycles = [r for r in records if r.get("kind") == "control_cycle"]
    summaries = [r for r in records if r.get("kind") == "rollout_summary"]

    assert len(summaries) == 5
    assert all(s["success"] for s in summaries)  # sim:reach hit from jittered starts
    assert all(not s["estopped"] for s in summaries)
    assert all(s["watchdog_timeouts"] == 0 for s in summaries)
    assert all(s["task"] == "sim:reach" for s in summaries)
    # success requires genuine closed-loop tracking: several replan cycles, all executed
    assert all(s["cycles"] >= 3 for s in summaries)
    assert all(s["executed_cycles"] == s["cycles"] for s in summaries)

    assert cycles and all(set(c) == _CYCLE_KEYS for c in cycles)
    assert all(set(s) == _SUMMARY_KEYS for s in summaries)
    assert all(c["chunk_steps"] == 8 and c["prefix_steps"] == 4 for c in cycles)
    # every line stamped with the same run_id + config_hash (AC-04 traceability)
    assert {r["run_id"] for r in records} == {"e2e-sim"}
    assert len({r["config_hash"] for r in records}) == 1


def test_run_metadata_provenance_over_the_wire(rollout_logs) -> None:
    """AC-04: the remote client pulls checkpoint/dataset refs from the server's info."""
    metadata = load_run_metadata([rollout_logs["sim"]])
    assert metadata is not None
    assert metadata["checkpoint_ref"] == str(_CHECKPOINT.resolve())
    assert str(metadata["dataset_snapshot_ref"]).startswith("sha256:")
    assert metadata["config_hash"]


def test_fault_rollouts_are_handled(rollout_logs) -> None:
    summaries = load_rollout_summaries([rollout_logs["fault"]])
    assert len(summaries) == 2
    for s in summaries:
        # Mock rollouts are SIM evidence: the task label must carry the 'sim:' prefix so
        # AC-06 reports pending_hardware instead of claiming real-robot safe stops.
        assert s["task"] == f"{SIM_TASK_PREFIX}{FAULT_INJECTION_TASK}"
        assert not s["success"]
        assert not s["estopped"]
        kinds = s["intervention_kinds"]
        assert kinds.get("nan_reject", 0) > 0  # injected NaN chunks were rejected
        assert kinds.get("deadline_miss", 0) > 0  # injected stalls tripped the deadline
        assert s["deadline_misses"] > 0
        assert kinds.get(LIMIT_BREACH_KIND, 0) == 0
        assert s["interventions_total"] == sum(kinds.values())


# ------------------------------------------------------------------------- acceptance


def test_acceptance_logic_on_small_sample(rollout_logs) -> None:
    logs = [rollout_logs["sim"], rollout_logs["fault"]]
    summaries = load_rollout_summaries(logs)
    metadata = load_run_metadata(logs)
    swap = verify_backbone_swap(_BACKBONES)

    report = evaluate_acceptance(
        summaries, metadata_line=metadata, known_task="reach", backbone_check=swap
    )

    # AC-03-style logic: zero violations in the 5 sim rollouts, but n < 100 -> pending.
    sim_only = [s for s in summaries if s["task"] == "sim:reach"]
    assert len(sim_only) == 5 and not any(is_safety_violation(s) for s in sim_only)
    ac03 = report.criterion("AC-03")
    assert ac03.status == "pending_data" and ac03.insufficient_n
    assert (ac03.n, ac03.successes) == (5, 5)
    assert ac03.metrics["population"] == "sim" and ac03.metrics["violations"] == 0

    # one injected violation must flip AC-03 to a hard FAIL regardless of n
    violating = dict(sim_only[0], rollout_id="bad", intervention_kinds={"nan_reject": 1})
    flipped = evaluate_acceptance(
        [*summaries, violating], metadata_line=metadata, known_task="reach",
        backbone_check=swap,
    )
    assert flipped.criterion("AC-03").status == "fail"

    assert report.criterion("AC-01").status == "pending_hardware"
    assert report.criterion("AC-01").metrics["sim_n"] == 5
    assert report.criterion("AC-02").status == "pending_data"
    assert report.criterion("AC-04").status == "pass"  # provenance survived the wire
    assert report.criterion("AC-05").status == "pass"
    ac06 = report.criterion("AC-06")
    # Sim-only fault injections are handled but must NOT pass as real-robot evidence.
    assert ac06.status == "pending_hardware" and (ac06.n, ac06.successes) == (2, 2)
    assert ac06.metrics["sim_n"] == 2 and ac06.metrics["real_n"] == 0
    assert report.criterion("AC-07").status == "pending_data"
    assert report.overall_status == "pending"
    assert not report.failed_criteria()


# ------------------------------------------------------- executor timeout (T-19/T-20)


class _TimeoutPolicy:
    """Simulates the RemotePolicy client timeout contract: predict raises TimeoutError."""

    def predict(self, observation: Observation) -> ActionChunk:
        raise TimeoutError("simulated remote timeout after 1.0 s")


def test_executor_treats_policy_timeout_as_deadline_miss(tmp_path) -> None:
    robot = MockRobot(num_joints=6, seed=0)
    safety = SafetyLayer(SafetyConfig.from_yaml(_SAFETY_YAML), robot.spec)
    metadata = RunMetadata.create("timeout-run", {"test": "timeout"})
    log_path = tmp_path / "timeout.jsonl"
    config = ExecutorConfig(prefix_steps=2, max_cycles=3, policy_deadline_ms=50.0)

    with JsonlRunLogger(log_path, metadata) as logger:
        executor = ClosedLoopExecutor(
            robot=robot,
            policy=_TimeoutPolicy(),
            safety=safety,
            watchdog=None,
            logger=logger,
            config=config,
        )
        result = executor.run_rollout("timeout-0000")

    assert result.deadline_misses == 3 and result.executed_cycles == 0
    assert not result.estopped
    assert result.intervention_kinds == {"deadline_miss": 3}
    assert robot.is_holding  # every timed-out cycle commanded a hold
    assert np.allclose(robot.read_state().q, 0.0)  # nothing was ever executed

    cycles = [
        json.loads(line)
        for line in log_path.read_text().splitlines()
        if json.loads(line).get("kind") == "control_cycle"
    ]
    assert len(cycles) == 3
    for c in cycles:
        assert c["deadline_missed"] and not c["executed"]
        assert c["chunk_steps"] == 0 and c["prefix_steps"] == 0
        assert c["interventions"][0]["kind"] == "deadline_miss"
        assert "timed out" in c["interventions"][0]["detail"]
