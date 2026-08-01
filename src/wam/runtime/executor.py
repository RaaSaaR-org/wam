"""Closed-loop runtime executor (T-19, FR-05): receding horizon with replanning.

REPLANNING SEMANTIC (FR-05): every cycle produces a FRESH action chunk from a fresh
observation; only the first ``prefix_steps`` of the (safety-filtered) chunk are executed
and the unexecuted remainder is DISCARDED — the next prediction replaces it. The robot
therefore only ever sees prefixes; no stale chunk tail survives a replan.

Contracts:
- Torch-free. Model inference is behind the ``Policy`` protocol
  (:class:`wam.runtime.policies.CheckpointPolicy` for trained checkpoints).
- Every policy output passes the deterministic safety filter before the robot adapter
  sees it (FR-07). The learned model never commands the robot directly.
- Deadline (PRD §11.1): a prediction arriving later than ``policy_deadline_ms`` is
  DISCARDED — never executed late — and ``robot.hold()`` is commanded instead. The
  watchdog is intentionally NOT fed on that path, so a persistently late policy trips it.
- Rejected states are NOT watchdog food: when the safety filter rejects the cycle
  (``nan_reject``/``schema_reject``/``state_reject`` -> HOLD chunk), the watchdog is not
  fed — a robot that keeps serving stale/unusable states must not keep the watchdog armed.
  Because a stalled robot also freezes ``state.timestamp_ns`` (the watchdog's clock), the
  executor additionally measures an uninterrupted reject streak on the HOST clock and
  escalates per the watchdog's HOLD/STOP action once it exceeds the watchdog timeout.
- Watchdog expiry: stale loop -> HOLD or STOP (e-stop) per its configured action; the
  watchdog is re-armed only after the safe state was commanded (PRD §11.2 recovery).
- Time: policy latency is measured with ``time.monotonic`` by default; a ``clock``
  callable (seconds) can be injected so tests enforce deadlines deterministically.
  ``now_ns`` and the watchdog's normal feed/expiry run on ROBOT time
  (``state.timestamp_ns`` — simulated for the mock, wall-clock-ish on hardware); the
  stale-state reject-streak escalation above runs on the HOST clock because a stalled
  robot freezes its own timestamps.
- Logging: writes exactly the SHARED ROLLOUT LOG CONTRACT records — one
  ``kind="control_cycle"`` line per cycle and one ``kind="rollout_summary"`` line per
  rollout, stamped run_id + config_hash by :class:`JsonlRunLogger` (AC-04). The
  below-min-rate flag lives ONLY in :class:`RolloutResult` (``below_min_policy_rate``);
  the summary record keeps the fixed contract keys and consumers derive the flag from
  ``policy_rate_hz``. An attached :class:`PolicyContract` adds a THIRD kind,
  ``kind="policy_contract"``; without one the record stream is byte-identical to before.
- Train/deploy divergence: an attached :class:`PolicyContract` states what the checkpoint
  was trained on and the executor makes the observation match it, or refuses. See that
  class for the two divergences this exists to prevent.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from wam.interfaces import (
    JsonlRunLogger,
    Observation,
    Policy,
    RobotAdapter,
    RobotState,
    SafetyFilter,
    SafetyIntervention,
)
from wam.interfaces.schema import ValidityMask
from wam.runtime.mock_loop import DEFAULT_INSTRUCTION
from wam.safety import Watchdog, WatchdogAction

__all__ = [
    "POLICY_CONTRACT_VERSION",
    "STATE_GROUPS",
    "ClosedLoopExecutor",
    "ExecutorConfig",
    "PolicyContract",
    "RolloutResult",
    "StateDivergence",
    "StateGroupUse",
    "policy_contract_record",
    "run_rollouts",
]

SuccessFn = Callable[[RobotState], bool]

# Safety-filter rejection kinds: cycles carrying one of these executed only a HOLD chunk
# built from an unusable prediction/state and therefore never feed the watchdog.
_REJECT_KINDS = frozenset({"nan_reject", "schema_reject", "state_reject"})

_NS_PER_S = 1_000_000_000

POLICY_CONTRACT_VERSION = "0.1.0"

#: Validity-mask group order, identical to ``StateMLP``'s ``_GROUP_ORDER``. The encoder
#: substitutes a learned ``missing[<group>]`` vector wherever the mask says invalid, so
#: "which groups were valid during training" is a real property of the trained weights.
STATE_GROUPS: tuple[str, ...] = ("q", "dq", "imu", "gripper")


class StateGroupUse(str, Enum):
    """How a validity group appeared across the states a checkpoint was trained on.

    ``ALWAYS``  — every training state marked it valid; the encoder's ``missing`` vector for
                  this group is untrained, so deploying it invalid feeds an unlearned input.
    ``NEVER``   — no training state marked it valid; the encoder saw ONLY the learned
                  ``missing`` vector there and the raw input columns are unconstrained, so
                  deploying it valid feeds an unlearned input.
    ``MIXED``   — both occurred, so either deployed value is in distribution.
    """

    ALWAYS = "always"
    NEVER = "never"
    MIXED = "mixed"


@dataclass(frozen=True)
class StateDivergence:
    """One validity group whose deployed flag does not match how it was trained.

    ``repaired`` is True only for the direction that has a FAITHFUL repair: training never
    saw the group valid, so forcing the deployed flag to invalid reproduces the exact input
    the encoder was fitted on. The other direction (trained ALWAYS, deployed invalid) has no
    repair — there is no measurement to invent — and is reported unrepaired.
    """

    group: str
    trained: StateGroupUse
    deployed_valid: bool
    repaired: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "trained": self.trained.value,
            "deployed_valid": self.deployed_valid,
            "repaired": self.repaired,
            "detail": self.detail,
        }


class PolicyContract(BaseModel):
    """What a checkpoint was trained on, as data the runtime can check before it moves a motor.

    Two confirmed train/deploy divergences motivate this, both silent — the predicted chunk
    stays finite, in-bounds and on time in either case, so nothing downstream notices:

    1. **Validity mask.** ``scripts/convert_lerobot_g1.py`` writes ``imu=False`` with a zero
       IMU payload for every gr00t state, so all 20 000 T-16 training steps fed the learned
       ``missing['imu']`` vector. Every deployment adapter declares ``imu=True`` — honestly,
       because a G1 (and the MuJoCo scene) really does have an IMU. Measured on the shipped
       ``runs/t16-lora-seed0`` encoder over all 590 states of ``gr00t-apple-000000``:
       flipping the flag with the transport's gravity payload moves the 32-d state embedding
       by ``||delta||`` 2.01 mean / 2.27 max, against an embedding norm of 2.45 and a maximum
       distance between two TRAINING states of 3.55 — ~1.75x the RMS radius of the training
       cloud, along a direction the encoder never saw. The adapters are not wrong; the
       policy's expectations were never written down. This class writes them down.
    2. **Instruction.** Every gr00t manifest says "move the apple to the plate" (one unique
       string, so training conditioned on ONE frozen umT5 context). The closed loop defaults
       to ``DEFAULT_INSTRUCTION`` — a real ``datasets/mock-d1`` instruction, correct for D1
       checkpoints and wrong for the gr00t-trained ones. The repo's own probe
       (``runs/wan_ablation/2026-07-26-zerogpu-5b.json``) puts an instruction swap at
       relative L2 0.021 / 0.034 on readout blocks 2 / 10 — ~2.5x the influence of the whole
       state input, well under a tenth of frame-order. A real distribution shift on the
       readout the ActionHead was fitted to, not an output-destroying failure.

    The contract is DECLARED, not guessed: ``from_dataset`` derives it from the episodes a
    run actually trained on, so it cannot drift from the data the way a hand-written note
    would. ``dataset_snapshot_ref`` / ``checkpoint_config_hash`` bind it to one artifact;
    a caller that checks them (``scripts/rollout.py`` does) cannot pair a contract with the
    wrong checkpoint.

    Empty ``instructions`` means "not declared" — the instruction check is then skipped, and
    the caller is expected to say so out loud rather than silently pass.
    """

    model_config = ConfigDict(frozen=True)

    contract_version: str = POLICY_CONTRACT_VERSION
    instructions: tuple[str, ...] = ()
    state_groups: dict[str, StateGroupUse] = Field(default_factory=dict)
    camera: str | None = None
    dataset_snapshot_ref: str | None = None
    checkpoint_config_hash: str | None = None
    #: Free-text provenance of the contract itself (which dataset root it was derived from).
    source: str = ""

    @field_validator("state_groups")
    @classmethod
    def _known_groups(cls, v: dict[str, StateGroupUse]) -> dict[str, StateGroupUse]:
        unknown = sorted(set(v) - set(STATE_GROUPS))
        if unknown:
            raise ValueError(f"state_groups: unknown validity group(s) {unknown}")
        return v

    # -- derivation / serialization ---------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        root: str | Path,
        *,
        episode_ids: Iterable[str] | None = None,
        camera: str | None = None,
        dataset_snapshot_ref: str | None = None,
        checkpoint_config_hash: str | None = None,
    ) -> PolicyContract:
        """Derive the contract from the episodes on disk — the only source that cannot drift.

        Reads each episode's ``manifest.json`` for the instruction and the four ``valid_*``
        columns of ``states.parquet`` for the mask. Deliberately NOT the manifest's declared
        spec: the mask is per-row, and a group that is valid in some rows and not others is
        ``MIXED`` — the one case where no deployed value is out of distribution.

        ``episode_ids`` restricts the scan to a training split (a holdout's episodes were
        never seen by the weights, so they do not belong in the contract).

        ``dataset_snapshot_ref`` is computed over the scanned episodes with the SAME
        convention ``train_t16_lora``/``eval_t16`` use (episode-relative path + manifest
        bytes, manifests carry sha256s of their own data files), so a caller can prove the
        contract came from the checkpoint's own training set instead of assuming it.

        Imports pyarrow and ``wam.data`` lazily: the executor's import surface stays numpy +
        pydantic so a control loop does not pay for the dataset stack.
        """
        import hashlib

        import pyarrow.parquet as pq  # local: keeps the runtime import surface small

        from wam.data.episode import MANIFEST_FILENAME, list_episodes

        root = Path(root)
        wanted = None if episode_ids is None else {str(e) for e in episode_ids}
        instructions: list[str] = []
        seen_valid: dict[str, bool] = {g: False for g in STATE_GROUPS}
        seen_invalid: dict[str, bool] = {g: False for g in STATE_GROUPS}
        digest = hashlib.sha256()
        n_episodes = 0
        for episode_dir in list_episodes(root):
            if wanted is not None and episode_dir.name not in wanted:
                continue
            n_episodes += 1
            manifest_bytes = (episode_dir / MANIFEST_FILENAME).read_bytes()
            digest.update(str(episode_dir.relative_to(root)).encode("utf-8"))
            digest.update(manifest_bytes)
            instruction = json.loads(manifest_bytes).get("instruction")
            if isinstance(instruction, str) and instruction not in instructions:
                instructions.append(instruction)
            table = pq.read_table(
                episode_dir / "states.parquet",
                columns=[f"valid_{g}" for g in STATE_GROUPS],
            )
            for group, column in zip(STATE_GROUPS, table.columns):
                values = set(column.to_pylist())
                seen_valid[group] = seen_valid[group] or (True in values)
                seen_invalid[group] = seen_invalid[group] or (False in values)
        if n_episodes == 0:
            raise ValueError(f"{root}: no episode directories with a {MANIFEST_FILENAME}")

        groups: dict[str, StateGroupUse] = {}
        for group in STATE_GROUPS:
            if seen_valid[group] and seen_invalid[group]:
                groups[group] = StateGroupUse.MIXED
            elif seen_valid[group]:
                groups[group] = StateGroupUse.ALWAYS
            else:
                groups[group] = StateGroupUse.NEVER
        return cls(
            instructions=tuple(instructions),
            state_groups=groups,
            camera=camera,
            dataset_snapshot_ref=dataset_snapshot_ref or f"sha256:{digest.hexdigest()}",
            checkpoint_config_hash=checkpoint_config_hash,
            source=f"{root} ({n_episodes} episodes)",
        )

    @classmethod
    def from_json(cls, text: str) -> PolicyContract:
        return cls.model_validate(json.loads(text))

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    # -- checks ------------------------------------------------------------------------------

    @property
    def declares_instructions(self) -> bool:
        """False == the instruction check cannot run; callers must say so, not stay quiet."""
        return bool(self.instructions)

    def instruction_seen(self, instruction: str) -> bool:
        """True when the checkpoint conditioned on exactly this string during training."""
        return not self.declares_instructions or instruction in self.instructions

    def state_divergences(self, state: RobotState) -> tuple[StateDivergence, ...]:
        """Every validity group whose deployed flag contradicts how it was trained."""
        flags = state.validity.as_dict()
        out: list[StateDivergence] = []
        for group in STATE_GROUPS:
            trained = self.state_groups.get(group)
            if trained is None or trained is StateGroupUse.MIXED:
                continue
            deployed = bool(flags[group])
            if trained is StateGroupUse.NEVER and deployed:
                out.append(
                    StateDivergence(
                        group=group,
                        trained=trained,
                        deployed_valid=True,
                        repaired=True,
                        detail=(
                            f"{group}: no training state marked it valid, so the encoder only "
                            f"ever saw the learned missing[{group}] vector; the deployed state "
                            "declares it valid. Masked back to invalid to reproduce training."
                        ),
                    )
                )
            elif trained is StateGroupUse.ALWAYS and not deployed:
                out.append(
                    StateDivergence(
                        group=group,
                        trained=trained,
                        deployed_valid=False,
                        repaired=False,
                        detail=(
                            f"{group}: every training state marked it valid, so missing[{group}] "
                            "is untrained; the deployed state declares it invalid and there is "
                            "no measurement to substitute. NOT repairable."
                        ),
                    )
                )
        return tuple(out)

    def conform(self, state: RobotState) -> tuple[RobotState, tuple[StateDivergence, ...]]:
        """Return ``(state the policy should see, divergences found)``.

        The only edit made is masking DOWN a group the checkpoint never trained with — an
        exact reproduction of the training input, not an approximation. Nothing else about
        the state is touched, and the safety layer still sees the state the ROBOT reported
        (the executor filters on the raw state), so this can never relax a limit check.
        """
        divergences = self.state_divergences(state)
        to_mask = [d.group for d in divergences if d.repaired]
        if not to_mask:
            return state, divergences
        flags = state.validity.as_dict()
        for group in to_mask:
            flags[group] = False
        return replace(state, validity=ValidityMask(**flags)), divergences


def policy_contract_record(
    contract: PolicyContract | None,
    *,
    instruction: str,
    allow_unseen_instruction: bool = False,
    rollout_id: str | None = None,
    cycle: int | None = None,
    divergences: Iterable[StateDivergence] = (),
) -> dict[str, Any]:
    """One ``kind="policy_contract"`` log line, with ONE shape whether or not there is a
    contract.

    ``contract=None`` is a first-class case, not an omission: a rollout that deliberately ran
    unchecked (``rollout.py --no-policy-contract``) has to be distinguishable in the archive
    from one that was checked and found clean. If the "unchecked" case were simply the absence
    of a record, those two would look identical to every later reader.
    """
    return {
        "kind": "policy_contract",
        "rollout_id": rollout_id,
        "cycle": cycle,
        "instruction": instruction,
        "instruction_seen": None if contract is None else contract.instruction_seen(instruction),
        "allow_unseen_instruction": bool(allow_unseen_instruction),
        "contract": None if contract is None else contract.model_dump(mode="json"),
        "divergences": [d.as_dict() for d in divergences],
    }


class ExecutorConfig(BaseModel):
    """Closed-loop executor parameters (FR-05, PRD §11.1).

    - ``prefix_steps``: steps of each safe chunk to execute before replanning.
    - ``max_cycles``: hard rollout length bound (cycles == policy predictions).
    - ``policy_deadline_ms``: predictions later than this are discarded -> hold.
    - ``min_policy_rate_hz``: MVP floor (>= 2 Hz, PRD §11.1); rollouts below it are
      flagged in :attr:`RolloutResult.below_min_policy_rate`.
    - ``stop_on_estop``: end the rollout as soon as the robot reports an e-stop.
    - ``allow_unseen_instruction``: OFF by default. With a :class:`PolicyContract` attached,
      an instruction the checkpoint never trained on refuses to start; this is the explicit
      "I know, run it anyway" the operator has to type. It is a config field rather than a
      constructor argument so it lands in ``config_hash`` — a run that overrode the check is
      a distinguishable artifact forever after.
    """

    model_config = ConfigDict(frozen=True)

    prefix_steps: int = Field(ge=1)
    max_cycles: int = Field(ge=1)
    policy_deadline_ms: float = Field(default=500.0, gt=0)
    min_policy_rate_hz: float = Field(default=2.0, gt=0)
    instruction: str = DEFAULT_INSTRUCTION
    task: str = "pick_and_place"
    stop_on_estop: bool = True
    allow_unseen_instruction: bool = False


@dataclass
class RolloutResult:
    """Outcome of one rollout; mirrors the ``rollout_summary`` log-record contract.

    ``below_min_policy_rate`` is a derived convenience flag (``policy_rate_hz <
    ExecutorConfig.min_policy_rate_hz``) and is deliberately NOT part of the shared
    summary record.
    """

    rollout_id: str
    success: bool
    task: str
    duration_s: float
    cycles: int
    executed_cycles: int
    interventions_total: int
    intervention_kinds: dict[str, int]
    watchdog_timeouts: int
    deadline_misses: int
    estopped: bool
    policy_rate_hz: float
    below_min_policy_rate: bool = field(default=False, compare=False)

    def summary_record(self) -> dict[str, Any]:
        """Exactly the SHARED ROLLOUT LOG CONTRACT ``rollout_summary`` payload."""
        return {
            "kind": "rollout_summary",
            "rollout_id": self.rollout_id,
            "success": self.success,
            "task": self.task,
            "duration_s": self.duration_s,
            "cycles": self.cycles,
            "executed_cycles": self.executed_cycles,
            "interventions_total": self.interventions_total,
            "intervention_kinds": dict(self.intervention_kinds),
            "watchdog_timeouts": self.watchdog_timeouts,
            "deadline_misses": self.deadline_misses,
            "estopped": self.estopped,
            "policy_rate_hz": self.policy_rate_hz,
        }


def _render_images(robot: RobotAdapter) -> dict[str, np.ndarray]:
    """One [H, W, 3] frame per camera if the adapter exposes ``render_frames``."""
    render = getattr(robot, "render_frames", None)
    if not callable(render):
        return {}
    return {name: frames[0] for name, frames in render(1).items()}


def _is_estopped(robot: RobotAdapter) -> bool:
    return bool(getattr(robot, "is_estopped", False))


class ClosedLoopExecutor:
    """Receding-horizon closed loop: observe -> predict -> filter -> execute prefix.

    Per cycle: ``read_state`` -> render frames (if the adapter can) -> ``Observation``
    -> timed ``policy.predict`` -> watchdog/deadline gates -> ``safety.filter`` ->
    ``robot.execute(safe_chunk, prefix_steps)`` -> feed watchdog -> success check.
    Only the prefix executes; the remainder is replaced by the next prediction (FR-05).

    ``contract`` is the train/deploy guard (:class:`PolicyContract`). With one attached the
    executor refuses to construct on an instruction the checkpoint never saw (unless
    ``config.allow_unseen_instruction``), masks every observation down to the validity groups
    training used, and writes a ``kind="policy_contract"`` record whenever the set of
    divergences changes — including the first cycle, so a conforming run still says what it
    conformed. With ``contract=None`` the executor behaves exactly as before, down to the
    emitted record stream.
    """

    def __init__(
        self,
        robot: RobotAdapter,
        policy: Policy,
        safety: SafetyFilter,
        watchdog: Watchdog | None,
        logger: JsonlRunLogger,
        config: ExecutorConfig,
        clock: Callable[[], float] | None = None,
        contract: PolicyContract | None = None,
    ) -> None:
        self._robot = robot
        self._policy = policy
        self._safety = safety
        self._watchdog = watchdog
        self._logger = logger
        self._config = config
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._contract = contract
        if (
            contract is not None
            and not config.allow_unseen_instruction
            and not contract.instruction_seen(config.instruction)
        ):
            # Refuse HERE, at construction, not at the first cycle: nothing has been read from
            # the robot yet and no motor has moved. The alternative — a warning — would be
            # indistinguishable from noise in a log nobody reads until the numbers look wrong.
            raise ValueError(
                f"instruction {config.instruction!r} is not one the checkpoint was "
                f"trained on ({list(contract.instructions)}). The trained text context "
                "conditions cross-attention in every backbone block, so this is a real "
                "distribution shift on the readout the ActionHead was fitted to. Pass a "
                "trained instruction, or set allow_unseen_instruction=True to accept it "
                "deliberately (it is recorded in config_hash)."
            )
        # Last emitted divergence fingerprint, so a stable run logs ONE contract record and a
        # mid-rollout validity change (a sensor dropping out) still gets its own line.
        self._last_divergence_key: tuple[tuple[str, bool, bool], ...] | None = None

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    @property
    def contract(self) -> PolicyContract | None:
        return self._contract

    def _conform(self, rollout_id: str, cycle: int, state: RobotState) -> RobotState:
        """Apply the policy contract to one state, logging every change of divergence set."""
        contract = self._contract
        if contract is None:
            return state
        conformed, divergences = contract.conform(state)
        key = tuple((d.group, d.deployed_valid, d.repaired) for d in divergences)
        if key != self._last_divergence_key:
            self._last_divergence_key = key
            # The whole contract goes in the line, not a reference to it: the run's config
            # reaches the log only as a config_hash, so without this the log could not say
            # WHAT the policy was held to — only that something was checked (AC-04).
            self._logger.log(
                policy_contract_record(
                    contract,
                    instruction=self._config.instruction,
                    allow_unseen_instruction=self._config.allow_unseen_instruction,
                    rollout_id=rollout_id,
                    cycle=cycle,
                    divergences=divergences,
                )
            )
        return conformed

    def run_rollout(self, rollout_id: str, success_fn: SuccessFn | None = None) -> RolloutResult:
        """Run one rollout of up to ``max_cycles`` cycles; one JSONL record per cycle.

        ``success_fn(state)`` is evaluated on the freshly read post-action state each
        cycle; returning True ends the rollout early with ``success=True``. An e-stop
        ends it with ``success=False`` when ``stop_on_estop`` is set.
        """
        cfg = self._config
        robot, policy, safety, watchdog = self._robot, self._policy, self._safety, self._watchdog
        t_start = self._clock()

        kinds: dict[str, int] = {}
        cycles = 0
        executed_cycles = 0
        watchdog_timeouts = 0
        deadline_misses = 0
        success = False
        estopped = False
        # Host-clock start of the current uninterrupted safety-reject streak (None when
        # the last cycle was clean). The watchdog itself runs on robot time, which FREEZES
        # exactly when the robot stalls — so stale-state escalation is timed on the host.
        reject_streak_start_s: float | None = None

        for cycle in range(cfg.max_cycles):
            cycles = cycle + 1
            state = robot.read_state()
            now_ns = int(state.timestamp_ns)
            if watchdog is not None and watchdog.last_feed_ns is None:
                watchdog.feed(now_ns)  # arm; never-fed == expired would trip immediately

            # The POLICY sees the contract-conformed state; the SAFETY filter below sees the
            # raw one the robot reported. Masking is an input-distribution repair, never a
            # way to make a limit check look at something the robot did not say (FR-07).
            observation = Observation(
                images=_render_images(robot),
                state=self._conform(rollout_id, cycle, state),
                instruction=cfg.instruction,
            )
            t0 = self._clock()
            timeout_detail: str | None = None
            try:
                chunk = policy.predict(observation)
            except TimeoutError as exc:
                # RemotePolicy contract (T-20): a client-side timeout surfaces as a
                # deadline miss — no chunk arrived, so there is nothing to execute.
                chunk = None
                timeout_detail = f"policy timed out: {exc}"
            policy_latency_ms = (self._clock() - t0) * 1e3
            deadline_missed = chunk is None or policy_latency_ms > cfg.policy_deadline_ms
            if deadline_missed:
                deadline_misses += 1

            interventions: list[SafetyIntervention] = []
            decision: WatchdogAction | None = None
            executed = False
            prefix_executed = 0
            expired = watchdog is not None and watchdog.expired(now_ns)

            if expired and watchdog is not None:
                # Stale loop: the fresh chunk is discarded, HOLD/STOP is commanded.
                decision = watchdog.decide(now_ns)
                wd_intervention = watchdog.intervention(now_ns)
                if wd_intervention is not None:
                    interventions.append(wd_intervention)
                if decision is WatchdogAction.STOP:
                    robot.estop()
                else:
                    robot.hold()
                watchdog.feed(now_ns)  # re-arm only after the safe state was commanded
                watchdog_timeouts += 1
            elif deadline_missed:
                # Late prediction: discard, hold. Watchdog NOT fed -> chronic lateness
                # eventually trips it (PRD §11.1: never execute a stale action).
                robot.hold()
                interventions.append(
                    SafetyIntervention(
                        kind="deadline_miss",
                        detail=timeout_detail
                        or (
                            f"policy latency {policy_latency_ms:.3f} ms > deadline "
                            f"{cfg.policy_deadline_ms:.3f} ms; chunk discarded, hold commanded"
                        ),
                        timestamp_ns=now_ns,
                    )
                )
            else:
                assert chunk is not None  # deadline_missed covers the timeout path
                safe_chunk, filter_interventions = safety.filter(state, chunk)
                interventions.extend(filter_interventions)
                rejected = any(iv.kind in _REJECT_KINDS for iv in filter_interventions)
                host_now_s = self._clock()
                if rejected and reject_streak_start_s is None:
                    reject_streak_start_s = host_now_s
                elif not rejected:
                    reject_streak_start_s = None
                if (
                    rejected
                    and watchdog is not None
                    and reject_streak_start_s is not None
                    and (host_now_s - reject_streak_start_s) * _NS_PER_S > watchdog.timeout_ns
                ):
                    # Frozen/unusable robot state persisting past the watchdog timeout: the
                    # robot clock (the watchdog's time source) is frozen too, so escalate
                    # here on host time — HOLD or STOP per the watchdog's configured action.
                    decision = watchdog.action
                    interventions.append(
                        SafetyIntervention(
                            kind="watchdog_timeout",
                            detail=(
                                f"safety filter rejected the state for "
                                f"{host_now_s - reject_streak_start_s:.3f} s (host clock) > "
                                f"timeout {watchdog.timeout_ns} ns; robot time frozen; "
                                f"decision={decision.value}"
                            ),
                            timestamp_ns=now_ns,
                        )
                    )
                    if decision is WatchdogAction.STOP:
                        robot.estop()
                    else:
                        robot.hold()
                    # Re-arm the streak only after the safe state was commanded.
                    reject_streak_start_s = host_now_s
                    watchdog_timeouts += 1
                    expired = True
                else:
                    prefix_executed = min(cfg.prefix_steps, safe_chunk.num_steps)
                    # Receding horizon: only the prefix runs; the remainder is discarded
                    # and replaced by the next cycle's prediction (FR-05). A rejected
                    # cycle executes the safety layer's HOLD chunk but does NOT feed the
                    # watchdog: stale robot data must not keep it armed.
                    robot.execute(safe_chunk, cfg.prefix_steps)
                    if watchdog is not None and not rejected:
                        watchdog.feed(now_ns)
                    executed = True
                    executed_cycles += 1

            for intervention in interventions:
                kinds[intervention.kind] = kinds.get(intervention.kind, 0) + 1
            if _is_estopped(robot):
                estopped = True

            self._logger.log(
                {
                    "kind": "control_cycle",
                    "rollout_id": rollout_id,
                    "cycle": cycle,
                    "now_ns": now_ns,
                    "policy_latency_ms": float(policy_latency_ms),
                    "deadline_missed": bool(deadline_missed),
                    "executed": bool(executed),
                    "prefix_steps": int(prefix_executed),
                    "chunk_steps": int(chunk.num_steps) if chunk is not None else 0,
                    "interventions": [
                        {"kind": iv.kind, "detail": iv.detail, "timestamp_ns": int(iv.timestamp_ns)}
                        for iv in interventions
                    ],
                    "watchdog": {
                        "expired": bool(expired),
                        "action": decision.value if decision is not None else None,
                    },
                }
            )

            if estopped and cfg.stop_on_estop:
                break
            if success_fn is not None and success_fn(robot.read_state()):
                success = True
                break

        duration_s = max(self._clock() - t_start, 0.0)
        policy_rate_hz = cycles / duration_s if duration_s > 0 else 0.0
        result = RolloutResult(
            rollout_id=rollout_id,
            success=success,
            task=cfg.task,
            duration_s=duration_s,
            cycles=cycles,
            executed_cycles=executed_cycles,
            interventions_total=sum(kinds.values()),
            intervention_kinds=kinds,
            watchdog_timeouts=watchdog_timeouts,
            deadline_misses=deadline_misses,
            estopped=estopped,
            policy_rate_hz=policy_rate_hz,
            below_min_policy_rate=policy_rate_hz < cfg.min_policy_rate_hz,
        )
        self._logger.log(result.summary_record())
        return result


def run_rollouts(
    executor: ClosedLoopExecutor,
    n: int,
    rollout_id_prefix: str = "rollout",
    success_fn: SuccessFn | None = None,
) -> list[RolloutResult]:
    """Run ``n`` sequential rollouts with ids ``<prefix>-0000`` ... ``<prefix>-<n-1>``."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return [
        executor.run_rollout(f"{rollout_id_prefix}-{i:04d}", success_fn=success_fn)
        for i in range(n)
    ]
