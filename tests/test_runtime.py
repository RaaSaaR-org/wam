"""Train/deploy divergence guard (T-31): PolicyContract, executor enforcement, E2 gates.

Both defects these tests pin are silent at every layer that already exists — the predicted
chunk stays finite, in-bounds and on time whether the policy was fed the inputs it trained on
or not — so nothing here asserts on chunk values. Every test asserts on what REACHED the
policy, what refused to start, or what was written down.

1. The validity mask. ``scripts/convert_lerobot_g1.py`` writes ``imu=False`` for every gr00t
   state; ``G1Adapter``/``MujocoG1Robot``/``MockRobot`` all report ``imu=True``. Measured on
   the shipped ``runs/t16-lora-seed0`` state encoder over all 590 states of
   ``gr00t-apple-000000``, flipping that flag with ``FakeG1Transport``'s gravity payload moves
   the embedding by 2.01 mean / 2.27 max against an embedding norm of 2.45 and a maximum
   TRAIN-to-TRAIN distance of 3.55.
2. The instruction. Every gr00t manifest carries one unique string; the closed loop defaults to
   a ``datasets/mock-d1`` string instead.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from wam.data.episode import EpisodeWriter
from wam.evaluation import e2_static_checks
from wam.evaluation.e2_checks import (
    E2_GATE_INSTRUCTION,
    E2_GATE_STATE_GROUPS,
    E2_STATIC_GATES,
    E2Report,
)
from wam.interfaces import (
    ActionChunk,
    ActionMode,
    CanonicalSpaceSpec,
    JsonlRunLogger,
    Observation,
    RobotState,
    RunMetadata,
)
from wam.interfaces.schema import IMUState, ValidityMask
from wam.robot import MockRobot
from wam.runtime import ClosedLoopExecutor, DummyPolicy, ExecutorConfig
from wam.runtime.executor import (
    POLICY_CONTRACT_VERSION,
    PolicyContract,
    StateGroupUse,
)
from wam.runtime.mock_loop import DEFAULT_INSTRUCTION
from wam.safety import SafetyConfig, SafetyLayer

REPO_ROOT = Path(__file__).resolve().parent.parent
D1_CHECKPOINT = REPO_ROOT / "runs" / "d1-overfit-seed0" / "checkpoint.safetensors"
needs_d1_checkpoint = pytest.mark.skipif(
    not D1_CHECKPOINT.exists(), reason="d1-overfit-seed0 checkpoint not present (gitignored)"
)

N_JOINTS = 6
SPEC = CanonicalSpaceSpec(joint_names=tuple(f"joint_{i}" for i in range(N_JOINTS)))
DT_S = 0.05

TRAINED_INSTRUCTION = "move the apple to the plate"

#: The gr00t training contract, spelled out rather than read off disk so the test states the
#: fact it is guarding instead of inheriting it from a dataset that could change.
GR00T_CONTRACT = PolicyContract(
    instructions=(TRAINED_INSTRUCTION,),
    state_groups={
        "q": StateGroupUse.ALWAYS,
        "dq": StateGroupUse.ALWAYS,
        "imu": StateGroupUse.NEVER,
        "gripper": StateGroupUse.ALWAYS,
    },
    source="test",
)


def make_safety_config() -> SafetyConfig:
    return SafetyConfig(
        q_min=(-3.0,) * N_JOINTS,
        q_max=(3.0,) * N_JOINTS,
        dq_max=(1.5,) * N_JOINTS,
        ddq_max=(4.0,) * N_JOINTS,
        workspace_min=(0.1, -0.6, 0.6),
        workspace_max=(0.8, 0.6, 1.4),
        ee_max_lin_vel_m_s=0.5,
        ee_max_step_m=0.05,
        gripper_rate_max=2.0,
        chunk_timeout_s=0.5,
    )


def make_state(*, imu_valid: bool = True, acc_z: float = 9.81, gripper_valid: bool = True):
    """A deploy-shaped state: real IMU payload, adapter-declared validity."""
    return RobotState(
        timestamp_ns=1_000_000,
        q=np.zeros(N_JOINTS, dtype=np.float32),
        dq=np.zeros(N_JOINTS, dtype=np.float32),
        imu=IMUState(
            orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            linear_acceleration=np.array([0.0, 0.0, acc_z], dtype=np.float32),
        ),
        gripper_state=np.zeros(1, dtype=np.float32),
        validity=ValidityMask(q=True, dq=True, imu=imu_valid, gripper=gripper_valid),
    )


class RecordingPolicy:
    """A ``Policy`` that keeps every observation it was handed. Chunks are constant zeros."""

    def __init__(self, spec: CanonicalSpaceSpec, steps: int = 8) -> None:
        self.observations: list[Observation] = []
        self._spec = spec
        self._steps = steps

    def predict(self, observation: Observation) -> ActionChunk:
        self.observations.append(observation)
        return ActionChunk(
            mode=ActionMode.JOINT_DELTA,
            targets=np.zeros((self._steps, self._spec.num_joints), dtype=np.float32),
            gripper_target=np.zeros(self._steps, dtype=np.float32),
            dt_s=DT_S,
        )


class RecordingSafety:
    """A ``SafetyFilter`` that records the state it was asked to judge and passes chunks."""

    def __init__(self, inner: SafetyLayer) -> None:
        self._inner = inner
        self.states: list[RobotState] = []

    def filter(self, state: RobotState, chunk: ActionChunk):
        self.states.append(state)
        return self._inner.filter(state, chunk)


def make_logger(tmp_path: Path, name: str = "rollouts.jsonl") -> JsonlRunLogger:
    metadata = RunMetadata.create("test-contract", {"test": True}, git_commit="deadbeef")
    return JsonlRunLogger(tmp_path / name, metadata)


def read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# -- derivation from the data ------------------------------------------------------------


def _write_episode(
    root: Path, name: str, instruction: str, masks: list[ValidityMask], *, subdir: str = ""
) -> None:
    """One minimal episode with an explicit per-row validity mask sequence.

    ``subdir`` puts the episode below ``root`` the way a chunked LeRobot conversion does, which
    is the only layout in which an episode's path and its directory name differ.
    """
    spec = CanonicalSpaceSpec(joint_names=("a", "b"), gripper_dims=1)
    with EpisodeWriter(
        root / subdir / name, episode_id=name, spec=spec, fps=10.0, instruction=instruction
    ) as w:
        for i, mask in enumerate(masks):
            w.add_state(
                RobotState(
                    timestamp_ns=i * 1_000_000,
                    q=np.zeros(2, dtype=np.float32),
                    dq=np.zeros(2, dtype=np.float32),
                    imu=IMUState(
                        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                        angular_velocity=np.zeros(3, dtype=np.float32),
                        linear_acceleration=np.zeros(3, dtype=np.float32),
                    ),
                    gripper_state=np.zeros(1, dtype=np.float32),
                    validity=mask,
                )
            )


def test_contract_from_dataset_marks_a_group_never_when_no_recorded_state_declared_it_valid(
    tmp_path: Path,
) -> None:
    """The exact gr00t shape: imu=False in every row -> the encoder only saw missing[imu]."""
    root = tmp_path / "ds"
    _write_episode(root, "ep-0", "grab it", [ValidityMask(imu=False)] * 3)
    _write_episode(root, "ep-1", "grab it", [ValidityMask(imu=False)] * 3)

    contract = PolicyContract.from_dataset(root)

    assert contract.state_groups["imu"] is StateGroupUse.NEVER
    assert contract.state_groups["q"] is StateGroupUse.ALWAYS
    assert contract.instructions == ("grab it",)


def test_contract_from_dataset_marks_a_group_mixed_when_rows_disagree_so_neither_value_alarms(
    tmp_path: Path,
) -> None:
    """A group seen both ways in training puts BOTH deployed values in distribution."""
    root = tmp_path / "ds"
    _write_episode(root, "ep-0", "grab it", [ValidityMask(imu=True), ValidityMask(imu=False)])

    contract = PolicyContract.from_dataset(root)

    assert contract.state_groups["imu"] is StateGroupUse.MIXED
    assert contract.state_divergences(make_state(imu_valid=True)) == ()
    assert contract.state_divergences(make_state(imu_valid=False)) == ()


def test_contract_from_dataset_collects_every_distinct_instruction_across_episodes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ds"
    _write_episode(root, "ep-0", "grab the cup", [ValidityMask()])
    _write_episode(root, "ep-1", "grab the cube", [ValidityMask()])
    _write_episode(root, "ep-2", "grab the cup", [ValidityMask()])

    contract = PolicyContract.from_dataset(root)

    assert set(contract.instructions) == {"grab the cup", "grab the cube"}


def test_contract_from_dataset_restricted_to_a_split_ignores_the_episodes_outside_it(
    tmp_path: Path,
) -> None:
    """A holdout's episodes never touched the weights, so they must not shape the contract."""
    root = tmp_path / "ds"
    _write_episode(root, "train-0", "trained text", [ValidityMask(imu=False)])
    _write_episode(root, "holdout-0", "holdout text", [ValidityMask(imu=True)])

    contract = PolicyContract.from_dataset(root, episode_ids=["train-0"])

    assert contract.instructions == ("trained text",)
    assert contract.state_groups["imu"] is StateGroupUse.NEVER


def test_contract_from_dataset_hashes_the_same_episodes_the_trainer_hashes(
    tmp_path: Path,
) -> None:
    """dataset_snapshot_ref must be comparable to the checkpoint's, or binding is guesswork.

    Two properties of that convention, neither of which a flat root can show. The episodes live
    in different chunk directories, so what is hashed is each episode's path RELATIVE TO THE
    ROOT and not its directory name — the shape ``convert_lerobot_g1`` writes, and the only one
    in which the two differ. And the restricted contract is hashed as well: ``train_t16_lora``
    narrows its hash to the episodes it actually trained on and ``eval_t16`` refuses a
    checkpoint whose ref disagrees, so a contract that hashed the whole root would name a
    training set that trained nothing.
    """
    import hashlib

    from wam.data.episode import MANIFEST_FILENAME, list_episodes

    root = tmp_path / "ds"
    _write_episode(root, "ep-0", "x", [ValidityMask()], subdir="chunk-000")
    _write_episode(root, "ep-1", "x", [ValidityMask()], subdir="chunk-001")

    def snapshot(dirs: list[Path], key=lambda d: str(d.relative_to(root))) -> str:
        """``train_t16_lora._dataset_snapshot_hash`` / ``eval_t16.dataset_snapshot_hash``."""
        digest = hashlib.sha256()
        for episode_dir in dirs:
            digest.update(key(episode_dir).encode("utf-8"))
            digest.update((episode_dir / MANIFEST_FILENAME).read_bytes())
        return f"sha256:{digest.hexdigest()}"

    episodes = list_episodes(root)
    assert len(episodes) == 2
    # Without this the "relative path" half of the convention is unobservable.
    assert snapshot(episodes) != snapshot(episodes, key=lambda d: d.name)

    assert PolicyContract.from_dataset(root).dataset_snapshot_ref == snapshot(episodes)

    trained = PolicyContract.from_dataset(root, episode_ids=["ep-0"])
    assert trained.dataset_snapshot_ref == snapshot(episodes[:1])
    assert trained.dataset_snapshot_ref != snapshot(episodes)


def test_contract_from_dataset_refuses_a_root_with_no_episodes_instead_of_declaring_nothing(
    tmp_path: Path,
) -> None:
    """An empty scan would produce a contract that says every group is NEVER valid — a
    fabricated one that would mask every input the policy has."""
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no episode directories"):
        PolicyContract.from_dataset(tmp_path / "empty")


# -- conform semantics -------------------------------------------------------------------


def test_conform_masks_down_the_group_the_checkpoint_only_ever_saw_as_missing() -> None:
    """The core repair: deployed imu=True + trained NEVER -> the encoder gets missing[imu],
    which is byte-for-byte the input all 20 000 training steps used."""
    conformed, divergences = GR00T_CONTRACT.conform(make_state(imu_valid=True))

    assert conformed.validity.imu is False
    assert [d.group for d in divergences] == ["imu"]
    assert divergences[0].repaired is True


def test_conform_touches_nothing_but_the_validity_mask() -> None:
    """The masked group's payload, q, dq, gripper and the timestamp must survive untouched:
    the repair is an input-DISTRIBUTION fix, not a state rewrite."""
    state = make_state(imu_valid=True, acc_z=9.81)
    conformed, _ = GR00T_CONTRACT.conform(state)

    assert conformed.timestamp_ns == state.timestamp_ns
    assert np.array_equal(conformed.q, state.q)
    assert np.array_equal(conformed.dq, state.dq)
    assert np.array_equal(conformed.gripper_state, state.gripper_state)
    assert np.array_equal(conformed.imu.linear_acceleration, state.imu.linear_acceleration)
    assert conformed.validity.as_dict() == {"q": True, "dq": True, "imu": False, "gripper": True}
    # The input state is not mutated in place — a caller still holds what the robot said.
    assert state.validity.imu is True


def test_conform_is_a_no_op_when_the_deployed_mask_already_matches_training() -> None:
    state = make_state(imu_valid=False)
    conformed, divergences = GR00T_CONTRACT.conform(state)

    assert conformed is state
    assert divergences == ()


def test_conform_reports_but_cannot_repair_a_group_that_training_always_had() -> None:
    """Trained ALWAYS + deployed invalid has no repair: missing[gripper] is untrained and
    there is no measurement to substitute. It must be reported unrepaired, not masked."""
    conformed, divergences = GR00T_CONTRACT.conform(
        make_state(imu_valid=False, gripper_valid=False)
    )

    assert [d.group for d in divergences] == ["gripper"]
    assert divergences[0].repaired is False
    assert conformed.validity.gripper is False  # untouched: nothing to repair it with


def test_contract_rejects_a_validity_group_name_the_state_encoder_does_not_have() -> None:
    """A typo'd group would silently never match, i.e. a contract that checks nothing."""
    with pytest.raises(ValueError, match="unknown validity group"):
        PolicyContract(state_groups={"imu_": StateGroupUse.NEVER})


def test_contract_json_roundtrip_preserves_every_declared_field() -> None:
    contract = PolicyContract(
        instructions=(TRAINED_INSTRUCTION,),
        state_groups={"imu": StateGroupUse.NEVER},
        camera="ego",
        dataset_snapshot_ref="sha256:abc",
        checkpoint_config_hash="deadbeef",
        source="datasets/x (2 episodes)",
    )
    assert PolicyContract.from_json(contract.to_json()) == contract
    assert contract.contract_version == POLICY_CONTRACT_VERSION


def test_a_contract_that_declares_no_instructions_skips_the_check_rather_than_failing_it() -> None:
    """'Not declared' must not read as 'not trained on' — callers report the gap instead."""
    contract = PolicyContract(state_groups={"imu": StateGroupUse.NEVER})

    assert contract.declares_instructions is False
    assert contract.instruction_seen("anything at all") is True


# -- executor enforcement ------------------------------------------------------------------


def build_executor(tmp_path: Path, **kwargs):
    """(executor, robot, policy, safety, logger) with a contract-aware closed loop."""
    contract = kwargs.pop("contract", GR00T_CONTRACT)
    instruction = kwargs.pop("instruction", TRAINED_INSTRUCTION)
    robot = kwargs.pop("robot", None) or MockRobot(spec=SPEC)
    policy = RecordingPolicy(SPEC)
    safety = RecordingSafety(SafetyLayer(make_safety_config(), spec=SPEC))
    logger = make_logger(tmp_path)
    config = ExecutorConfig(
        prefix_steps=2,
        max_cycles=kwargs.pop("max_cycles", 3),
        instruction=instruction,
        **kwargs,
    )
    executor = ClosedLoopExecutor(
        robot, policy, safety, None, logger, config, contract=contract
    )
    return executor, robot, policy, safety, logger


def test_executor_refuses_to_construct_on_an_instruction_the_checkpoint_never_trained_on(
    tmp_path: Path,
) -> None:
    """The refusal happens before a single read_state, so nothing has moved when it fires."""
    with pytest.raises(ValueError, match="not one the checkpoint was"):
        build_executor(tmp_path, instruction=DEFAULT_INSTRUCTION)


def test_executor_accepts_an_unseen_instruction_only_behind_the_explicit_override_flag(
    tmp_path: Path,
) -> None:
    executor, _, _, _, logger = build_executor(
        tmp_path, instruction=DEFAULT_INSTRUCTION, allow_unseen_instruction=True
    )
    with logger:
        executor.run_rollout("r-0000")

    contract_records = [
        r for r in read_records(logger.path) if r["kind"] == "policy_contract"
    ]
    assert contract_records[0]["instruction_seen"] is False
    assert contract_records[0]["allow_unseen_instruction"] is True
    assert contract_records[0]["contract"]["instructions"] == [TRAINED_INSTRUCTION]


def test_executor_masks_the_imu_group_before_the_policy_sees_the_observation(
    tmp_path: Path,
) -> None:
    """MockRobot declares imu=True; a gr00t-trained policy must be handed imu=False."""
    executor, robot, policy, _, logger = build_executor(tmp_path)
    assert robot.read_state().validity.imu is True
    with logger:
        executor.run_rollout("r-0000")

    assert policy.observations
    assert all(obs.state.validity.imu is False for obs in policy.observations)


def test_executor_never_lets_the_contract_change_what_the_safety_filter_judges(
    tmp_path: Path,
) -> None:
    """Masking is an input repair for the POLICY. The deterministic layer keeps seeing the
    state the robot reported, or a contract could be used to talk a limit check out of
    looking at something (FR-07)."""
    executor, _, _, safety, logger = build_executor(tmp_path)
    with logger:
        executor.run_rollout("r-0000")

    assert safety.states
    assert all(state.validity.imu is True for state in safety.states)


def test_executor_writes_one_policy_contract_record_for_a_rollout_with_stable_validity(
    tmp_path: Path,
) -> None:
    """One line, not one per cycle: a per-cycle repeat would be noise nobody reads."""
    executor, _, _, _, logger = build_executor(tmp_path, max_cycles=4)
    with logger:
        executor.run_rollout("r-0000")

    records = read_records(logger.path)
    contract_records = [r for r in records if r["kind"] == "policy_contract"]
    assert len(contract_records) == 1
    assert contract_records[0]["cycle"] == 0
    assert [d["group"] for d in contract_records[0]["divergences"]] == ["imu"]
    assert contract_records[0]["divergences"][0]["repaired"] is True


def test_executor_writes_another_policy_contract_record_when_validity_changes_mid_rollout(
    tmp_path: Path,
) -> None:
    """A sensor dropping out halfway through is a NEW divergence and must get its own line —
    otherwise the only record of it is a first-cycle snapshot that was still true."""

    class DroppingRobot(MockRobot):
        """Reports gripper invalid from the third read on (a mid-rollout sensor failure)."""

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.reads = 0

        def read_state(self) -> RobotState:
            state = super().read_state()
            self.reads += 1
            if self.reads > 2:
                return replace(
                    state, validity=ValidityMask(q=True, dq=True, imu=True, gripper=False)
                )
            return state

    executor, _, _, _, logger = build_executor(
        tmp_path, robot=DroppingRobot(spec=SPEC), max_cycles=4
    )
    with logger:
        executor.run_rollout("r-0000")

    contract_records = [r for r in read_records(logger.path) if r["kind"] == "policy_contract"]
    assert len(contract_records) == 2
    groups = [{d["group"] for d in r["divergences"]} for r in contract_records]
    assert groups == [{"imu"}, {"imu", "gripper"}]
    unrepaired = [d for d in contract_records[1]["divergences"] if not d["repaired"]]
    assert [d["group"] for d in unrepaired] == ["gripper"]


def test_executor_without_a_contract_emits_the_unchanged_shared_rollout_log_record_stream(
    tmp_path: Path,
) -> None:
    """Opt-in means opt-in: archived consumers must not start seeing a new record kind just
    because the guard exists."""
    robot = MockRobot(spec=SPEC)
    logger = make_logger(tmp_path)
    executor = ClosedLoopExecutor(
        robot,
        DummyPolicy(SPEC, steps=8, dt_s=DT_S),
        SafetyLayer(make_safety_config(), spec=SPEC),
        None,
        logger,
        ExecutorConfig(prefix_steps=2, max_cycles=3),
    )
    with logger:
        executor.run_rollout("r-0000")

    kinds = [r["kind"] for r in read_records(logger.path)]
    assert kinds == ["control_cycle"] * 3 + ["rollout_summary"]


def test_executor_config_defaults_to_refusing_an_unseen_instruction() -> None:
    """A new flag that changes what is measured is OFF by default (house rule)."""
    assert ExecutorConfig(prefix_steps=1, max_cycles=1).allow_unseen_instruction is False


def test_archived_executor_config_without_the_override_field_still_parses() -> None:
    """Re-scoring an archived rollout's config_record must not need the new key."""
    archived = {
        "prefix_steps": 4,
        "max_cycles": 12,
        "policy_deadline_ms": 500.0,
        "min_policy_rate_hz": 2.0,
        "instruction": DEFAULT_INSTRUCTION,
        "task": "sim:reach",
        "stop_on_estop": True,
    }
    assert ExecutorConfig.model_validate(archived).allow_unseen_instruction is False


# -- E2 gates --------------------------------------------------------------------------------


def run_e2(contract: PolicyContract | None, instruction: str, **kwargs) -> E2Report:
    robot = kwargs.pop("robot", None) or MockRobot(spec=SPEC)
    return e2_static_checks(
        DummyPolicy(SPEC, dt_s=DT_S, amplitude_rad=0.05, period_s=8.0, gripper_period_s=60.0),
        robot,
        SafetyLayer(make_safety_config(), spec=SPEC),
        SPEC,
        n_probes=4,
        instruction=instruction,
        contract=contract,
        **kwargs,
    )


def test_e2_fails_the_instruction_gate_when_the_prompt_was_never_trained_on() -> None:
    report = run_e2(GR00T_CONTRACT, DEFAULT_INSTRUCTION)

    assert not report.passed
    assert E2_GATE_INSTRUCTION in report.failed_gates()


def test_e2_passes_the_instruction_gate_for_a_trained_prompt() -> None:
    report = run_e2(GR00T_CONTRACT, TRAINED_INSTRUCTION)

    assert E2_GATE_INSTRUCTION not in report.failed_gates()


def test_e2_warns_but_does_not_fail_on_a_validity_divergence_the_runtime_repairs() -> None:
    """MockRobot reports imu=True against a NEVER-trained group: repairable, so failing here
    would block a run the executor has already made correct — but it must still be said."""
    report = run_e2(GR00T_CONTRACT, TRAINED_INSTRUCTION)

    assert E2_GATE_STATE_GROUPS not in report.failed_gates()
    assert any("masking down" in w for w in report.warnings)
    gate = next(g for g in report.gates if g.name == E2_GATE_STATE_GROUPS)
    assert gate.metrics["repaired_groups"] == ["imu"]


def test_e2_fails_the_state_group_gate_when_a_group_trained_always_valid_is_deployed_invalid(
) -> None:
    """No repair exists for this direction: the encoder's missing vector is untrained."""

    class NoGripperRobot(MockRobot):
        def read_state(self) -> RobotState:
            return replace(
                super().read_state(),
                validity=ValidityMask(q=True, dq=True, imu=False, gripper=False),
            )

    report = run_e2(GR00T_CONTRACT, TRAINED_INSTRUCTION, robot=NoGripperRobot(spec=SPEC))

    assert E2_GATE_STATE_GROUPS in report.failed_gates()
    gate = next(g for g in report.gates if g.name == E2_GATE_STATE_GROUPS)
    assert gate.metrics["unrepairable_groups"] == ["gripper"]


def test_e2_records_whether_the_train_deploy_gates_ran_at_all() -> None:
    """A passing report with no contract says 'nobody looked', which is a different claim
    from 'no divergence' — and the difference must not have to be inferred."""
    with_contract = run_e2(GR00T_CONTRACT, TRAINED_INSTRUCTION)
    without = run_e2(None, TRAINED_INSTRUCTION)

    assert with_contract.contract_checked is True
    assert without.contract_checked is False
    assert tuple(g.name for g in without.gates) == E2_STATIC_GATES
    assert without.passed


def test_e2_records_the_instruction_the_probes_actually_carried() -> None:
    """Latency and determinism were measured under THIS text conditioning; until now the
    report did not say which string that was."""
    assert run_e2(None, TRAINED_INSTRUCTION).instruction == TRAINED_INSTRUCTION


def test_archived_e2_report_without_the_new_fields_still_parses_and_rescores() -> None:
    archived = {
        "report_version": "0.1.0",
        "check": "static",
        "n": 8,
        "policy": "DummyPolicy",
        "robot": "MockRobot",
        "gates": [{"name": "determinism", "passed": True, "detail": "", "metrics": {}}],
        "warnings": [],
    }
    report = E2Report.model_validate(archived)

    assert report.passed
    assert report.contract_checked is False
    assert report.instruction == ""


def test_the_e2_probe_instruction_default_is_one_that_exists_in_a_dataset_in_this_repo() -> None:
    """It used to default to a THIRD string ("Greife den Würfel und lege ihn ab.") present in
    no dataset here, so the gate probed policies under conditioning nothing was trained on."""
    import inspect

    default = inspect.signature(e2_static_checks).parameters["instruction"].default
    assert default == DEFAULT_INSTRUCTION
    mock_d1 = PolicyContract.from_dataset(REPO_ROOT / "datasets" / "mock-d1")
    assert default in mock_d1.instructions


# -- the rollout CLI -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rollout_cli():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rollout_cli_contract", REPO_ROOT / "scripts" / "rollout.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@needs_d1_checkpoint
def test_rollout_cli_refuses_a_trained_checkpoint_that_carries_no_contract(
    rollout_cli, tmp_path: Path
) -> None:
    """The default has to be refusal: the two divergences are invisible in every artifact a
    rollout produces, so 'it ran and the numbers looked plausible' is not evidence."""
    with pytest.raises(SystemExit, match="no PolicyContract"):
        rollout_cli.main(
            [
                "--robot", "mock", "--policy", "checkpoint", "--rollouts", "1",
                "--max-cycles", "2", "--e2-probes", "2", "--out-dir", str(tmp_path),
            ]
        )


@needs_d1_checkpoint
def test_rollout_cli_runs_a_trained_checkpoint_once_a_contract_is_derived_from_its_dataset(
    rollout_cli, tmp_path: Path
) -> None:
    """--skip-e2 on purpose: this test is about the PolicyContract record, and asserting
    rc == 0 without it silently makes the E2 release gate a precondition of a contract
    test. The checkpoint does not clear that gate (T-48: accel_limit, jerk from a
    smoothness weight of 0.0), which is a recipe gap recorded in TASKS.md, not a
    contract defect. The gate itself is untouched."""
    rc = rollout_cli.main(
        [
            "--robot", "mock", "--policy", "checkpoint", "--rollouts", "1",
            "--max-cycles", "2", "--e2-probes", "2", "--skip-e2",
            "--contract-from-dataset", str(REPO_ROOT / "datasets" / "mock-d1"),
            "--run-id", "contract-ok", "--out-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    records = read_records(tmp_path / "contract-ok.jsonl")
    declared = next(r for r in records if r["kind"] == "policy_contract")["contract"]
    assert declared["state_groups"]["imu"] == "always"  # mock-d1 recorded a valid IMU
    assert DEFAULT_INSTRUCTION in declared["instructions"]


@needs_d1_checkpoint
def test_rollout_cli_only_skips_the_contract_when_told_to_in_so_many_words(
    rollout_cli, tmp_path: Path
) -> None:
    """--skip-e2 on purpose: this test is about the PolicyContract record, and asserting
    rc == 0 without it silently makes the E2 release gate a precondition of a contract
    test. The checkpoint does not clear that gate (T-48: accel_limit, jerk from a
    smoothness weight of 0.0), which is a recipe gap recorded in TASKS.md, not a
    contract defect. The gate itself is untouched."""
    rc = rollout_cli.main(
        [
            "--robot", "mock", "--policy", "checkpoint", "--rollouts", "1",
            "--max-cycles", "2", "--e2-probes", "2", "--no-policy-contract", "--skip-e2",
            "--run-id", "contract-off", "--out-dir", str(tmp_path),
        ]
    )
    assert rc == 0
    records = read_records(tmp_path / "contract-off.jsonl")
    # Recorded as unchecked rather than left silent: a missing record would be
    # indistinguishable from a run that was checked and came back clean.
    unchecked = next(r for r in records if r["kind"] == "policy_contract")
    assert unchecked["contract"] is None
    assert unchecked["instruction_seen"] is None


@needs_d1_checkpoint
def test_rollout_cli_refuses_a_contract_bound_to_a_different_checkpoint(
    rollout_cli, tmp_path: Path
) -> None:
    """A contract paired with the wrong checkpoint is worse than none: the gates would report
    on a model that is not in the loop."""
    contract_path = tmp_path / "policy_contract.json"
    contract_path.write_text(
        PolicyContract(
            instructions=(DEFAULT_INSTRUCTION,),
            state_groups={"imu": StateGroupUse.ALWAYS},
            checkpoint_config_hash="0" * 64,
        ).to_json()
    )
    with pytest.raises(SystemExit, match="bound to checkpoint config_hash"):
        rollout_cli.main(
            [
                "--robot", "mock", "--policy", "checkpoint", "--rollouts", "1",
                "--max-cycles", "2", "--e2-probes", "2",
                "--policy-contract", str(contract_path), "--out-dir", str(tmp_path),
            ]
        )


def test_rollout_cli_discovers_a_contract_sitting_next_to_the_checkpoint(
    rollout_cli, tmp_path: Path
) -> None:
    """The contract has to travel with the artifact; a flag somebody must remember to retype
    is exactly how the camera key stayed unchecked for as long as it did."""
    checkpoint = REPO_ROOT / "runs" / "d1-overfit-seed0" / "checkpoint.safetensors"
    assert rollout_cli.find_policy_contract(checkpoint) is None

    run_dir = tmp_path / "runs" / "fake"
    (run_dir / "checkpoints").mkdir(parents=True)
    fake_ckpt = run_dir / "checkpoints" / "model.safetensors"
    fake_ckpt.touch()
    contract_path = run_dir / "policy_contract.json"
    contract_path.write_text(PolicyContract().to_json())

    assert rollout_cli.find_policy_contract(fake_ckpt) == contract_path


def test_rollout_cli_notes_a_policy_whose_trained_chunk_period_is_not_the_robots(
    rollout_cli,
) -> None:
    """The camera's third sibling: ActionHeadConfig.dt_s reaches the robot with nothing
    comparing it. A note, not a gate — pacing follows the chunk, so the loop stays
    self-consistent; it is the sim's time-per-step that quietly changes."""

    class FakeHead:
        dt_s = 0.0333333

    class FakeConfig:
        head = FakeHead()

    class FakeModel:
        config = FakeConfig()

    class FakePolicy:
        model = FakeModel()

    assert rollout_cli.policy_period_note(FakePolicy(), 0.02) is not None
    assert rollout_cli.policy_period_note(FakePolicy(), 0.0333333) is None
    # A policy that exposes no head (DummyPolicy, RemotePolicy) must not produce a note.
    assert rollout_cli.policy_period_note(DummyPolicy(SPEC, dt_s=DT_S), 0.02) is None
