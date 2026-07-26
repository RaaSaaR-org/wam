"""T-22 + T-23: E2 kinematic/sim checks and the AC-01..AC-07 acceptance harness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wam.evaluation import (
    VERDICT_HELPS,
    AblationReport,
    AcceptanceReport,
    MetricDelta,
    e2_sim_rollout_checks,
    e2_static_checks,
    evaluate_acceptance,
    is_safety_violation,
    load_rollout_summaries,
    load_run_metadata,
    verify_backbone_swap,
)
from wam.evaluation.acceptance import (
    AC01_MIN_N,
    AC03_MIN_N,
    DEFAULT_GENERALIZATION_TASK,
    DEFAULT_KNOWN_TASK,
    LIMIT_BREACH_KIND,
)
from wam.evaluation.e2_checks import (
    E2_GATE_DETERMINISM,
    E2_GATE_DURATION_BAND,
    E2_GATE_INTERVENTION_RATE,
    E2_GATE_POLICY_RATE,
    E2_GATE_ROLLOUTS,
    E2_GATE_TARGETS_FINITE,
    E2_GATE_ZERO_ESTOPS,
    E2_GATE_ZERO_WATCHDOG,
    E2_STATIC_GATES,
)
from wam.interfaces import ActionChunk, ActionMode, CanonicalSpaceSpec, Observation
from wam.robot import MockRobot
from wam.runtime import DummyPolicy
from wam.safety import SafetyConfig, SafetyLayer

_ROOT = Path(__file__).resolve().parent.parent
_SAFETY_YAML = _ROOT / "configs" / "safety" / "default.yaml"
_SCRIPT = _ROOT / "scripts" / "run_acceptance.py"

# torch-free registry subset — keeps the AC-05 smoke check fast in tests
_BACKBONES = ("flux3", "wan_i2v")
_SWAP = verify_backbone_swap(_BACKBONES)

METADATA = {
    "kind": "run_metadata",
    "run_id": "run-1",
    "config_hash": "cafebabe01",
    "git_commit": "deadbeef",
    "schema_version": "0.1.0",
    "interfaces_version": "0.1.0",
    "checkpoint_ref": "runs/d1-overfit-seed0/checkpoint.safetensors",
    "dataset_snapshot_ref": "datasets/mock-d1",
    "created_at": "2026-07-26T00:00:00+00:00",
}


def summary(
    task: str = DEFAULT_KNOWN_TASK,
    success: bool = True,
    *,
    rollout_id: str = "r0",
    estopped: bool = False,
    kinds: dict[str, int] | None = None,
    watchdog_timeouts: int = 0,
    policy_rate_hz: float = 5.0,
    cycles: int = 20,
    **extra,
) -> dict:
    kinds = dict(kinds or {})
    record = {
        "kind": "rollout_summary",
        "rollout_id": rollout_id,
        "success": bool(success),
        "task": task,
        "duration_s": 4.0,
        "cycles": cycles,
        "executed_cycles": cycles,
        "interventions_total": sum(kinds.values()),
        "intervention_kinds": kinds,
        "watchdog_timeouts": watchdog_timeouts,
        "deadline_misses": 0,
        "estopped": bool(estopped),
        "policy_rate_hz": policy_rate_hz,
        "run_id": "run-1",
        "config_hash": "cafebabe01",
    }
    record.update(extra)
    return record


def batch(n: int, task: str, successes: int, prefix: str = "r") -> list[dict]:
    return [summary(task, i < successes, rollout_id=f"{prefix}{i}") for i in range(n)]


def full_pass_summaries() -> list[dict]:
    """60 known + 40 generalization (= 100 for AC-03) + handled fault injections."""
    rollouts = batch(60, DEFAULT_KNOWN_TASK, 55, prefix="k")
    rollouts += batch(40, DEFAULT_GENERALIZATION_TASK, 25, prefix="g")
    rollouts += [
        summary("fault_injection", False, rollout_id=f"f{i}", kinds={"nan_reject": 2})
        for i in range(3)
    ]
    return rollouts


def write_ablation_json(path: Path) -> Path:
    def delta(hib: bool, base: float, cand: float, pct: float) -> MetricDelta:
        return MetricDelta(
            higher_is_better=hib,
            baseline=base,
            candidate=cand,
            delta=cand - base,
            improvement_pct=pct,
        )

    report = AblationReport(
        baseline_name="action_only",
        candidate_name="world_action",
        threshold_pct=5.0,
        metrics={
            "mse": delta(False, 1.0, 0.5, 50.0),
            "mae": delta(False, 0.8, 0.6, 25.0),
            "gripper_accuracy": delta(True, 0.8, 0.9, 12.5),
            "smoothness_pred": delta(False, 0.1, 0.08, 20.0),
        },
        verdict=VERDICT_HELPS,
    )
    path.write_text(report.to_json(), encoding="utf-8")
    return path


def evaluate(summaries: list[dict], **kwargs) -> AcceptanceReport:
    kwargs.setdefault("backbone_check", _SWAP)
    kwargs.setdefault("metadata_line", METADATA)
    return evaluate_acceptance(summaries, **kwargs)


# ---------------------------------------------------------------------------- log loading


def test_load_rollout_summaries_filters_kinds(tmp_path: Path) -> None:
    log = tmp_path / "rollouts.jsonl"
    records = [
        METADATA,
        {"kind": "control_cycle", "rollout_id": "r0", "cycle": 0},
        summary(rollout_id="r0"),
        summary(rollout_id="r1", success=False),
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    loaded = load_rollout_summaries(log)
    assert [s["rollout_id"] for s in loaded] == ["r0", "r1"]
    assert all(s["kind"] == "rollout_summary" for s in loaded)

    log2 = tmp_path / "second.jsonl"
    log2.write_text(json.dumps(summary(rollout_id="r2")) + "\n", encoding="utf-8")
    loaded = load_rollout_summaries([log, log2])
    assert [s["rollout_id"] for s in loaded] == ["r0", "r1", "r2"]

    metadata = load_run_metadata([log2, log])
    assert metadata is not None and metadata["run_id"] == "run-1"
    assert load_run_metadata(log2) is None


def test_load_rollout_summaries_rejects_bad_json(tmp_path: Path) -> None:
    log = tmp_path / "broken.jsonl"
    log.write_text('{"kind": "rollout_summary"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="broken.jsonl:2"):
        load_rollout_summaries(log)


# ---------------------------------------------------------------------------- AC-01 / AC-02


def test_ac01_pass_with_fields() -> None:
    report = evaluate(batch(60, DEFAULT_KNOWN_TASK, 55))
    c = report.criterion("AC-01")
    assert c.status == "pass" and c.passed
    assert (c.n, c.successes, c.required_n) == (60, 55, AC01_MIN_N)
    assert c.rate == pytest.approx(55 / 60)
    assert not c.insufficient_n


def test_ac01_fail_on_low_rate() -> None:
    c = evaluate(batch(50, DEFAULT_KNOWN_TASK, 30)).criterion("AC-01")
    assert c.status == "fail"
    assert c.rate == pytest.approx(0.6)
    assert not c.insufficient_n


def test_ac01_insufficient_n_is_pending_data() -> None:
    c = evaluate(batch(10, DEFAULT_KNOWN_TASK, 10)).criterion("AC-01")
    assert c.status == "pending_data"
    assert c.insufficient_n
    assert c.n == 10


def test_ac01_sim_only_is_pending_hardware() -> None:
    sims = batch(60, f"sim:{DEFAULT_KNOWN_TASK}", 55)
    c = evaluate(sims).criterion("AC-01")
    assert c.status == "pending_hardware"
    assert c.metrics["sim_n"] == 60
    assert c.metrics["sim_rate"] == pytest.approx(55 / 60)


def test_ac01_no_data_is_pending_data() -> None:
    c = evaluate([]).criterion("AC-01")
    assert c.status == "pending_data"
    assert c.n == 0


def test_ac02_uses_separate_task_label() -> None:
    known_only = evaluate(batch(60, DEFAULT_KNOWN_TASK, 55))
    assert known_only.criterion("AC-02").status == "pending_data"

    with_gen = evaluate(
        batch(60, DEFAULT_KNOWN_TASK, 55) + batch(30, DEFAULT_GENERALIZATION_TASK, 20, prefix="g")
    )
    c = with_gen.criterion("AC-02")
    assert c.status == "pass"
    assert (c.n, c.successes, c.required_n) == (30, 20, 30)
    assert c.rate == pytest.approx(20 / 30)


# ---------------------------------------------------------------------------- AC-03


def test_violation_definition_edge_cases() -> None:
    assert is_safety_violation(summary(estopped=True))
    assert is_safety_violation(summary(kinds={"nan_reject": 1}))
    assert is_safety_violation(summary(kinds={"schema_reject": 1}))
    assert is_safety_violation(summary(kinds={"state_reject": 1}))
    assert is_safety_violation(summary(watchdog_action="stop"))
    # a recorded limit breach is evidence a hard limit was ACTUALLY exceeded -> violation,
    # independent of e-stop status
    assert is_safety_violation(summary(kinds={LIMIT_BREACH_KIND: 1}))
    assert not is_safety_violation(summary(kinds={LIMIT_BREACH_KIND: 0}))
    # clamping kinds are interventions doing their job, NOT violations
    assert not is_safety_violation(summary(kinds={"joint_limit": 5, "velocity_limit": 3}))
    assert not is_safety_violation(summary(kinds={"nan_reject": 0}))
    assert not is_safety_violation(summary(watchdog_action="hold", watchdog_timeouts=1))


def test_ac03_pass_with_clamps_only() -> None:
    rollouts = [summary(rollout_id=f"r{i}", kinds={"joint_limit": 1}) for i in range(AC03_MIN_N)]
    c = evaluate(rollouts).criterion("AC-03")
    assert c.status == "pass"
    assert c.n == AC03_MIN_N


def test_ac03_fails_on_single_violation() -> None:
    rollouts = batch(AC03_MIN_N - 1, DEFAULT_KNOWN_TASK, AC03_MIN_N - 1)
    rollouts.append(summary(rollout_id="bad", kinds={"nan_reject": 1}))
    c = evaluate(rollouts).criterion("AC-03")
    assert c.status == "fail"
    assert c.metrics["violations"] == 1
    assert "bad" in c.metrics["violating_rollouts"]


def test_ac03_fails_on_estop_even_below_n() -> None:
    c = evaluate([summary(rollout_id="e", estopped=True)]).criterion("AC-03")
    assert c.status == "fail"


def test_ac03_fails_on_limit_breach() -> None:
    # 100 real rollouts each recording an actual limit breach must NEVER pass the
    # 'Zero safety violations' gate.
    rollouts = [
        summary(rollout_id=f"r{i}", kinds={LIMIT_BREACH_KIND: 1}) for i in range(AC03_MIN_N)
    ]
    c = evaluate(rollouts).criterion("AC-03")
    assert c.status == "fail"
    assert c.metrics["violations"] == AC03_MIN_N
    assert c.metrics["violation_kinds"].get(LIMIT_BREACH_KIND) == AC03_MIN_N


def test_ac03_sim_violation_fails_despite_clean_real_population() -> None:
    # Module contract: 'a violation in sim fails AC-03 immediately' — real rollouts define
    # the sufficiency basis, but sim violation evidence is never discarded.
    rollouts = batch(AC03_MIN_N, DEFAULT_KNOWN_TASK, AC03_MIN_N)
    rollouts.append(
        summary(f"sim:{DEFAULT_KNOWN_TASK}", rollout_id="sv", kinds={"nan_reject": 1})
    )
    c = evaluate(rollouts).criterion("AC-03")
    assert c.status == "fail"
    assert c.metrics["population"] == "real"
    assert c.metrics["sim_violations"] == 1
    assert "sv" in c.metrics["violating_rollouts"]
    assert "sim" in c.detail


def test_ac03_excludes_fault_injection_rollouts() -> None:
    rollouts = batch(AC03_MIN_N, DEFAULT_KNOWN_TASK, AC03_MIN_N)
    rollouts += [summary("fault_injection", False, rollout_id="f0", kinds={"nan_reject": 3})]
    c = evaluate(rollouts).criterion("AC-03")
    assert c.status == "pass"
    assert c.n == AC03_MIN_N


def test_ac03_insufficient_and_sim_only() -> None:
    c = evaluate(batch(50, DEFAULT_KNOWN_TASK, 50)).criterion("AC-03")
    assert c.status == "pending_data"
    assert c.insufficient_n

    # Controlled sim rollouts count as AC-03 evidence (M4/T-23): >= 100 clean sim
    # rollouts pass; a sim violation fails immediately; below the floor -> pending_data.
    c = evaluate(batch(120, f"sim:{DEFAULT_KNOWN_TASK}", 120)).criterion("AC-03")
    assert c.status == "pass"
    assert c.metrics["sim_n"] == 120 and c.metrics["population"] == "sim"

    c = evaluate(batch(20, f"sim:{DEFAULT_KNOWN_TASK}", 20)).criterion("AC-03")
    assert c.status == "pending_data" and c.insufficient_n

    bad = batch(120, f"sim:{DEFAULT_KNOWN_TASK}", 120)
    bad.append(summary(f"sim:{DEFAULT_KNOWN_TASK}", rollout_id="v", kinds={"nan_reject": 1}))
    assert evaluate(bad).criterion("AC-03").status == "fail"


# ---------------------------------------------------------------------------- AC-04


def test_ac04_reproducibility() -> None:
    assert evaluate([]).criterion("AC-04").status == "pass"

    incomplete = dict(METADATA, checkpoint_ref="")
    c = evaluate([], metadata_line=incomplete).criterion("AC-04")
    assert c.status == "fail"
    assert "checkpoint_ref" in c.detail

    c = evaluate([], metadata_line=None).criterion("AC-04")
    assert c.status == "fail"
    assert "run_metadata" in c.detail


# ---------------------------------------------------------------------------- AC-05


def test_ac05_backbone_swap_check() -> None:
    assert _SWAP.passed
    assert set(_BACKBONES) <= set(_SWAP.conformant)
    assert len(_SWAP.available) >= 2
    assert all(dim >= 1 for dim in _SWAP.feature_dims.values())

    c = evaluate([]).criterion("AC-05")
    assert c.status == "pass"
    assert c.n >= 2

    single = verify_backbone_swap(("flux3",))
    assert not single.passed
    assert evaluate([], backbone_check=single).criterion("AC-05").status == "fail"

    unknown = verify_backbone_swap(("flux3", "does_not_exist"))
    assert not unknown.passed
    assert "does_not_exist" in unknown.errors


# ---------------------------------------------------------------------------- AC-06


def test_ac06_handled_faults_pass() -> None:
    rollouts = [
        summary("fault_injection", False, rollout_id="f0", kinds={"nan_reject": 1}),
        summary("fault_injection", False, rollout_id="f1", watchdog_timeouts=1),
        summary("fault_injection", False, rollout_id="f2", estopped=True),  # clean e-stop
    ]
    c = evaluate(rollouts).criterion("AC-06")
    assert c.status == "pass"
    assert (c.n, c.successes) == (3, 3)


def test_ac06_unhandled_or_breached_faults_fail() -> None:
    silent = [summary("fault_injection", False, rollout_id="f0")]  # no reaction at all
    c = evaluate(silent).criterion("AC-06")
    assert c.status == "fail"
    assert "f0" in c.detail

    breach = [summary("fault_injection", False, rollout_id="f1", kinds={LIMIT_BREACH_KIND: 1})]
    assert evaluate(breach).criterion("AC-06").status == "fail"


def test_ac06_pending_states() -> None:
    assert evaluate([]).criterion("AC-06").status == "pending_data"

    sims = [summary("sim:fault_injection", False, rollout_id="f0", kinds={"nan_reject": 1})]
    c = evaluate(sims).criterion("AC-06")
    assert c.status == "pending_hardware"
    assert c.metrics["sim_n"] == 1


# ---------------------------------------------------------------------------- AC-07


def test_ac07_pending_without_ablation() -> None:
    c = evaluate([]).criterion("AC-07")
    assert c.status == "pending_data"
    assert "pending real D2 data" in c.detail


def test_ac07_with_ablation_report(tmp_path: Path) -> None:
    path = write_ablation_json(tmp_path / "ablation.json")
    c = evaluate([], ablation_json=path).criterion("AC-07")
    assert c.status == "pass"
    assert VERDICT_HELPS in c.detail
    assert c.metrics["verdict"] == VERDICT_HELPS

    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert evaluate([], ablation_json=bad).criterion("AC-07").status == "fail"


# ---------------------------------------------------------------------------- report


def test_full_pass_report_and_markdown(tmp_path: Path) -> None:
    ablation = write_ablation_json(tmp_path / "ablation.json")
    report = evaluate(full_pass_summaries(), ablation_json=ablation)

    assert report.passed
    assert report.overall_status == "pass"
    assert [c.criterion for c in report.criteria] == [f"AC-0{i}" for i in range(1, 8)]
    assert report.n_rollouts == 103 and report.n_sim == 0
    assert report.run_ids == ("run-1",)

    markdown = report.render_markdown()
    for i in range(1, 8):
        assert f"AC-0{i}" in markdown
    assert "**PASS**" in markdown

    roundtrip = AcceptanceReport.from_json(report.to_json())
    assert roundtrip == report


def test_mixed_report_statuses_render() -> None:
    rollouts = batch(50, DEFAULT_KNOWN_TASK, 30)  # AC-01 fail
    rollouts += batch(5, f"sim:{DEFAULT_GENERALIZATION_TASK}", 3)  # AC-02 pending_hardware
    report = evaluate(rollouts, metadata_line=None)  # AC-04 fail, AC-07 pending_data

    assert not report.passed
    assert report.overall_status == "fail"
    assert set(report.failed_criteria()) >= {"AC-01", "AC-04"}
    assert "AC-02" in report.pending_criteria()

    markdown = report.render_markdown()
    assert "FAIL" in markdown
    assert "PENDING-DATA" in markdown
    assert "PENDING-HARDWARE" in markdown


def test_e1_reports_attached(tmp_path: Path) -> None:
    from wam.evaluation import E1Report

    e1 = E1Report(
        mode="joint_delta",
        num_predictions=4,
        num_episodes=1,
        horizon_steps=8,
        target_dim=7,
        mse=0.01,
        mae=0.05,
        per_joint_mse={},
        per_joint_mae={},
        per_step_mse=(),
        per_step_mae=(),
        gripper_accuracy=1.0,
        smoothness_pred=0.0,
        smoothness_target=0.0,
        per_episode={},
    )
    report = evaluate([], e1_reports={"holdout": e1})
    assert report.e1_summaries["holdout"]["mse"] == pytest.approx(0.01)
    assert "holdout" in report.render_markdown()


# ---------------------------------------------------------------------------- E2 static


# Slow sinusoid: zero safety interventions even on phase-shifted synthetic probe states
# (default periods trip the gripper rate limit when the probe timestamp jumps the phase).
_CALM = {"amplitude_rad": 0.02, "period_s": 1e5, "gripper_period_s": 1e6}


def _e2_fixture(**policy_kwargs):
    robot = MockRobot(num_joints=6, seed=0)  # matches configs/safety/default.yaml
    spec = robot.spec
    safety = SafetyLayer(SafetyConfig.from_yaml(_SAFETY_YAML), spec)
    policy = DummyPolicy(spec, **policy_kwargs)
    return policy, robot, safety, spec


class NaNPolicy:
    def __init__(self, spec: CanonicalSpaceSpec) -> None:
        self._spec = spec

    def predict(self, observation: Observation) -> ActionChunk:
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.full((8, self._spec.num_joints), np.nan, dtype=np.float32),
            gripper_target=np.zeros(8, dtype=np.float32),
            dt_s=0.05,
        )


class DriftingPolicy:
    """Non-deterministic on purpose: output depends on call count, not the observation."""

    def __init__(self, spec: CanonicalSpaceSpec) -> None:
        self._spec = spec
        self._calls = 0

    def predict(self, observation: Observation) -> ActionChunk:
        self._calls += 1
        value = 1e-5 * self._calls
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.full((16, self._spec.num_joints), value, dtype=np.float32),
            gripper_target=np.zeros(16, dtype=np.float32),
            dt_s=0.05,
        )


def test_e2_static_checks_pass() -> None:
    # 16 * 0.05 s = 0.8 s: inside the PRD chunk band
    policy, robot, safety, spec = _e2_fixture(steps=16, dt_s=0.05, **_CALM)
    report = e2_static_checks(policy, robot, safety, spec, n_probes=8)

    assert report.check == "static"
    assert report.n == 8
    assert tuple(g.name for g in report.gates) == E2_STATIC_GATES
    assert report.passed, report.failed_gates()
    assert report.warnings == []
    latency = next(g for g in report.gates if g.name == "policy_latency")
    assert latency.metrics["mean_ms"] > 0.0


def test_e2_static_checks_is_deterministic() -> None:
    policy, robot, safety, spec = _e2_fixture(steps=16, dt_s=0.05, **_CALM)
    a = e2_static_checks(policy, robot, safety, spec, n_probes=6, seed=7)
    policy2, robot2, safety2, spec2 = _e2_fixture(steps=16, dt_s=0.05, **_CALM)
    b = e2_static_checks(policy2, robot2, safety2, spec2, n_probes=6, seed=7)
    strip = {"gates": {"__all__": {"metrics"}}}  # timing metrics differ between runs
    assert a.model_dump(exclude=strip) == b.model_dump(exclude=strip)


def test_e2_static_duration_band_warns_only() -> None:
    policy, robot, safety, spec = _e2_fixture(**_CALM)  # defaults: 8 * 0.05 = 0.4 s < 0.5 s
    report = e2_static_checks(policy, robot, safety, spec, n_probes=4)
    assert report.passed  # warn gate must not fail the report
    assert report.warnings and "PRD band" in report.warnings[0]
    duration = next(g for g in report.gates if g.name == E2_GATE_DURATION_BAND)
    assert duration.passed and duration.metrics["out_of_band"] == 4


def test_e2_static_flags_nan_policy() -> None:
    _, robot, safety, spec = _e2_fixture()
    report = e2_static_checks(NaNPolicy(spec), robot, safety, spec, n_probes=4)
    assert not report.passed
    failed = report.failed_gates()
    assert E2_GATE_TARGETS_FINITE in failed
    assert E2_GATE_INTERVENTION_RATE in failed  # every chunk triggers nan_reject
    assert E2_GATE_DETERMINISM not in failed  # identical NaN chunks are still deterministic
    rate = next(g for g in report.gates if g.name == E2_GATE_INTERVENTION_RATE)
    assert rate.metrics["intervention_kinds"].get("nan_reject", 0) >= 4


def test_e2_static_flags_nondeterminism() -> None:
    _, robot, safety, spec = _e2_fixture()
    report = e2_static_checks(DriftingPolicy(spec), robot, safety, spec, n_probes=4)
    assert E2_GATE_DETERMINISM in report.failed_gates()


def test_e2_static_rejects_zero_probes() -> None:
    policy, robot, safety, spec = _e2_fixture()
    with pytest.raises(ValueError, match="n_probes"):
        e2_static_checks(policy, robot, safety, spec, n_probes=0)


# ---------------------------------------------------------------------------- E2 sim


def test_e2_sim_rollout_checks_pass() -> None:
    sims = [summary(f"sim:{DEFAULT_KNOWN_TASK}", rollout_id=f"s{i}") for i in range(10)]
    report = e2_sim_rollout_checks(sims)
    assert report.check == "sim_rollout"
    assert report.passed and report.n == 10


def test_e2_sim_rollout_checks_gate_failures() -> None:
    base = [summary(f"sim:{DEFAULT_KNOWN_TASK}", rollout_id=f"s{i}") for i in range(4)]

    estopped = base + [summary("sim:x", rollout_id="e", estopped=True)]
    assert E2_GATE_ZERO_ESTOPS in e2_sim_rollout_checks(estopped).failed_gates()

    watchdog = base + [summary("sim:x", rollout_id="w", watchdog_timeouts=2)]
    assert E2_GATE_ZERO_WATCHDOG in e2_sim_rollout_checks(watchdog).failed_gates()

    slow = base + [summary("sim:x", rollout_id="s", policy_rate_hz=1.0)]
    assert E2_GATE_POLICY_RATE in e2_sim_rollout_checks(slow).failed_gates()

    noisy = [
        summary("sim:x", rollout_id=f"n{i}", kinds={"joint_limit": 10}, cycles=20) for i in range(4)
    ]
    assert E2_GATE_INTERVENTION_RATE in e2_sim_rollout_checks(noisy).failed_gates()


def test_e2_sim_rollout_checks_empty() -> None:
    report = e2_sim_rollout_checks([])
    assert not report.passed
    assert report.failed_gates() == [E2_GATE_ROLLOUTS]


# ---------------------------------------------------------------------------- CLI


def _load_cli():
    spec = importlib.util.spec_from_file_location("run_acceptance_cli", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_log(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_cli_full_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_log(tmp_path / "rollouts.jsonl", [METADATA, *full_pass_summaries()])
    ablation = write_ablation_json(tmp_path / "ablation.json")
    out_dir = tmp_path / "acceptance"

    cli = _load_cli()
    rc = cli.main(
        [
            "--rollout-logs",
            str(tmp_path / "*.jsonl"),
            "--ablation-json",
            str(ablation),
            "--out-dir",
            str(out_dir),
            "--backbones",
            ",".join(_BACKBONES),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    report = AcceptanceReport.from_json(
        (out_dir / "acceptance_report.json").read_text(encoding="utf-8")
    )
    assert report.passed
    markdown = (out_dir / "acceptance_report.md").read_text(encoding="utf-8")
    assert "AC-07" in markdown
    assert "AC-01" in captured.out


def test_cli_fails_on_failed_criterion(tmp_path: Path) -> None:
    # AC-01: enough rollouts, but success rate below 0.80 -> hard FAIL -> exit 1
    _write_log(tmp_path / "rollouts.jsonl", [METADATA, *batch(50, DEFAULT_KNOWN_TASK, 30)])
    cli = _load_cli()
    rc = cli.main(
        [
            "--rollout-logs",
            str(tmp_path / "*.jsonl"),
            "--out-dir",
            str(tmp_path / "acceptance"),
            "--backbones",
            ",".join(_BACKBONES),
        ]
    )
    assert rc == 1


def test_cli_no_matching_logs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli = _load_cli()
    rc = cli.main(["--rollout-logs", str(tmp_path / "nothing-*.jsonl")])
    assert rc == 1
    assert "no rollout logs" in capsys.readouterr().err
