"""E2 kinematic/sim checks (T-22, PRD §12.1) — the deterministic gate before E3 real-robot runs.

Two entry points, both torch-free:

- :func:`e2_static_checks` probes a live ``Policy`` against synthetic observations derived
  from a ``RobotAdapter`` (no execution, nothing is sent to the robot): chunk schema
  validity, finiteness, PRD duration band (warn-only), safety-filter intervention rate,
  determinism, policy latency and — when a ``PolicyContract`` is supplied — whether this
  deployment feeds the policy inputs it was never trained on.
- :func:`e2_sim_rollout_checks` gates aggregated sim rollouts using the shared rollout-log
  contract (``kind == 'rollout_summary'`` dicts): zero e-stops, zero watchdog timeouts,
  policy rate >= 2 Hz (FR-05) and a bounded intervention rate.

Every check is reported as a :class:`wam.data.validation.GateResult`; a report ``passed``
iff every gate passed. The chunk-duration gate never fails — out-of-band durations are
surfaced as warnings only, because the PRD band (0.5–2.0 s) is a design target, not a
safety limit.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

from wam.data.validation import GateResult
from wam.interfaces import (
    ActionChunk,
    CanonicalSpaceSpec,
    Observation,
    Policy,
    RobotAdapter,
    SafetyFilter,
)
from wam.runtime.executor import PolicyContract
from wam.runtime.mock_loop import DEFAULT_INSTRUCTION

E2_VERSION = "0.1.0"

# PRD FR-05: action chunks cover 0.5-2.0 s; policy replans at >= 2 Hz.
PRD_CHUNK_MIN_S = 0.5
PRD_CHUNK_MAX_S = 2.0
MIN_POLICY_RATE_HZ = 2.0

E2_GATE_CHUNK_VALID = "chunk_valid"
E2_GATE_TARGETS_FINITE = "targets_finite"
E2_GATE_DURATION_BAND = "chunk_duration_band"
E2_GATE_INTERVENTION_RATE = "safety_intervention_rate"
E2_GATE_DETERMINISM = "determinism"
E2_GATE_LATENCY = "policy_latency"
E2_STATIC_GATES = (
    E2_GATE_CHUNK_VALID,
    E2_GATE_TARGETS_FINITE,
    E2_GATE_DURATION_BAND,
    E2_GATE_INTERVENTION_RATE,
    E2_GATE_DETERMINISM,
    E2_GATE_LATENCY,
)

# Contract gates. Conditional on a PolicyContract being supplied, hence NOT in
# E2_STATIC_GATES: a report without a contract is missing them, and that absence is itself
# reported (a warning) rather than silently reading as "checked and fine".
E2_GATE_INSTRUCTION = "policy_instruction_seen"
E2_GATE_STATE_GROUPS = "policy_state_groups"
E2_CONTRACT_GATES = (E2_GATE_INSTRUCTION, E2_GATE_STATE_GROUPS)

E2_GATE_ROLLOUTS = "rollouts_present"
E2_GATE_ZERO_ESTOPS = "zero_estops"
E2_GATE_ZERO_WATCHDOG = "zero_watchdog_timeouts"
E2_GATE_POLICY_RATE = "policy_rate"
E2_SIM_GATES = (
    E2_GATE_ROLLOUTS,
    E2_GATE_ZERO_ESTOPS,
    E2_GATE_ZERO_WATCHDOG,
    E2_GATE_POLICY_RATE,
    E2_GATE_INTERVENTION_RATE,
)

__all__ = [
    "E2_CONTRACT_GATES",
    "E2_GATE_CHUNK_VALID",
    "E2_GATE_DETERMINISM",
    "E2_GATE_DURATION_BAND",
    "E2_GATE_INSTRUCTION",
    "E2_GATE_INTERVENTION_RATE",
    "E2_GATE_LATENCY",
    "E2_GATE_POLICY_RATE",
    "E2_GATE_ROLLOUTS",
    "E2_GATE_STATE_GROUPS",
    "E2_GATE_TARGETS_FINITE",
    "E2_GATE_ZERO_ESTOPS",
    "E2_GATE_ZERO_WATCHDOG",
    "E2_SIM_GATES",
    "E2_STATIC_GATES",
    "E2_VERSION",
    "MIN_POLICY_RATE_HZ",
    "PRD_CHUNK_MAX_S",
    "PRD_CHUNK_MIN_S",
    "E2Report",
    "e2_sim_rollout_checks",
    "e2_static_checks",
]


class E2Report(BaseModel):
    """All E2 gate results for one check run (static probes or sim rollout aggregate)."""

    model_config = ConfigDict(frozen=True)

    report_version: str = E2_VERSION
    check: Literal["static", "sim_rollout"]
    n: int
    policy: str = ""
    robot: str = ""
    #: The instruction the probes carried. Defaulted so archived reports keep parsing; it is
    #: recorded because latency and determinism were measured under THIS text conditioning,
    #: and until the contract gates existed nothing said which string that was.
    instruction: str = ""
    #: Whether the train/deploy gates ran at all. A passing report with this False says
    #: "nobody looked", which is a different claim from "no divergence" — and the difference
    #: is invisible if it has to be inferred from the gate list. False in archived reports,
    #: which is the truth: they predate the check.
    contract_checked: bool = False
    gates: list[GateResult] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    def failed_gates(self) -> list[str]:
        return [gate.name for gate in self.gates if not gate.passed]

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2)


def _chunks_equal(a: ActionChunk, b: ActionChunk) -> bool:
    """Bitwise chunk equality — the determinism contract for a stateless policy."""
    return (
        a.mode == b.mode
        and a.dt_s == b.dt_s
        and np.asarray(a.targets).shape == np.asarray(b.targets).shape
        and np.asarray(a.gripper_target).shape == np.asarray(b.gripper_target).shape
        and np.array_equal(np.asarray(a.targets), np.asarray(b.targets), equal_nan=True)
        and np.array_equal(
            np.asarray(a.gripper_target), np.asarray(b.gripper_target), equal_nan=True
        )
    )


def _chunk_finite(chunk: ActionChunk) -> bool:
    return (
        bool(np.isfinite(np.asarray(chunk.targets, dtype=np.float64)).all())
        and bool(np.isfinite(np.asarray(chunk.gripper_target, dtype=np.float64)).all())
        and bool(np.isfinite(float(chunk.dt_s)))
    )


def _probe_images(robot: RobotAdapter) -> dict[str, np.ndarray]:
    """One RGB frame per camera; zero-frames if the adapter cannot render offline."""
    render = getattr(robot, "render_frames", None)
    if callable(render):
        frames = render(1)
        return {camera: np.asarray(stack[0]) for camera, stack in frames.items()}
    return {"front": np.zeros((64, 64, 3), dtype=np.uint8)}


def e2_static_checks(
    policy: Policy,
    robot: RobotAdapter,
    safety: SafetyFilter,
    spec: CanonicalSpaceSpec,
    n_probes: int = 16,
    *,
    seed: int = 0,
    perturbation_rad: float = 0.1,
    max_intervention_rate: float = 0.1,
    max_mean_latency_ms: float = 1000.0 / MIN_POLICY_RATE_HZ,
    determinism_probes: int = 3,
    instruction: str = DEFAULT_INSTRUCTION,
    contract: PolicyContract | None = None,
) -> E2Report:
    """Probe ``policy`` on ``n_probes`` synthetic observations; nothing is executed.

    Observations reuse the robot's current state with deterministic joint perturbations
    (seeded, clipped to the adapter's position limits) and monotonic timestamps, so the
    whole check is reproducible. The safety filter runs on every predicted chunk but its
    output is discarded — this is a read-only gate.

    ``instruction`` defaults to the closed loop's :data:`DEFAULT_INSTRUCTION`. It used to
    default to a THIRD string ("Greife den Würfel und lege ihn ab.") that appears in no
    dataset in this repo, which meant the gate's latency, determinism and intervention
    numbers were measured under text conditioning no checkpoint had ever seen. Who that
    default change reached, stated precisely rather than waved at — an earlier revision of
    this docstring claimed "every caller in-tree passes the argument", which is false:

    - **No archived report changes.** ``scripts/rollout.py`` is the only in-tree writer of a
      persisted :class:`E2Report`, and it has always passed ``instruction=args.instruction``.
      Everything under ``runs/`` was therefore measured under a string somebody chose.
    - **Six test call sites DID change what they probe with**: ``tests/test_acceptance.py``
      lines 536, 549, 551, 558, 567 and 579 call this without ``instruction=`` (line 586 does
      too but raises on ``n_probes=0`` before reaching it). Their policies — ``DummyPolicy``,
      ``NaNPolicy``, ``DriftingPolicy`` — ignore ``Observation.instruction`` entirely, so no
      measured value there moves; what moves is the ``instruction`` field those reports carry,
      which is the point. ``tests/test_e2e_m4.py`` and ``tests/test_runtime.py`` pass it.

    The default is now honest rather than quietly novel, and the change is visible rather than
    silent because :attr:`E2Report.instruction` records the resolved string. If a future caller
    needs the gate to prove which text it conditioned on, make this parameter required — that
    turns a defaulted probe into a compile-time event instead of a docstring promise.

    ``contract`` adds the train/deploy gates (:data:`E2_CONTRACT_GATES`). Probe states come
    from the real adapter, so the state-group gate is testing exactly what a rollout would
    feed the policy. States are contract-conformed before probing, matching what
    :class:`wam.runtime.executor.ClosedLoopExecutor` does, so the latency and determinism
    numbers are measured on the inputs the rollout will actually use.
    """
    if n_probes < 1:
        raise ValueError(f"e2_static_checks: n_probes must be >= 1, got {n_probes}")

    base_state = robot.read_state()
    limits = robot.limits
    q_min = np.asarray(limits["q_min"], dtype=np.float64)
    q_max = np.asarray(limits["q_max"], dtype=np.float64)
    images = _probe_images(robot)
    rng = np.random.default_rng(seed)

    observations: list[Observation] = []
    raw_states = []
    for i in range(n_probes):
        delta = rng.uniform(-perturbation_rad, perturbation_rad, size=base_state.q.shape)
        q = np.clip(base_state.q.astype(np.float64) + delta, q_min, q_max).astype(np.float32)
        state = replace(base_state, timestamp_ns=base_state.timestamp_ns + i * 50_000_000, q=q)
        raw_states.append(state)
        if contract is not None:
            state, _ = contract.conform(state)
        observations.append(Observation(images=images, state=state, instruction=instruction))

    latencies_ms: list[float] = []
    validity_errors: list[str] = []
    nonfinite = 0
    durations: list[float] = []
    intervention_hits = 0
    intervention_kinds: dict[str, int] = {}
    warnings: list[str] = []

    for i, obs in enumerate(observations):
        start = time.perf_counter()
        chunk = policy.predict(obs)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

        errors = chunk.validate(spec)
        if errors:
            validity_errors.append(f"probe {i}: {errors[0]}")
        if not _chunk_finite(chunk):
            nonfinite += 1
        durations.append(float(chunk.duration))

        # The RAW state, not the conformed one: the safety layer judges what the robot
        # reported. Contract masking is an input-distribution repair for the policy and must
        # never change what a limit check looks at (FR-07).
        _, interventions = safety.filter(raw_states[i], chunk)
        if interventions:
            intervention_hits += 1
            for intervention in interventions:
                kind = intervention.kind
                intervention_kinds[kind] = intervention_kinds.get(kind, 0) + 1

    gates: list[GateResult] = []
    gates.append(
        GateResult(
            name=E2_GATE_CHUNK_VALID,
            passed=not validity_errors,
            detail="; ".join(validity_errors[:3]),
            metrics={"invalid_probes": len(validity_errors), "n_probes": n_probes},
        )
    )
    gates.append(
        GateResult(
            name=E2_GATE_TARGETS_FINITE,
            passed=nonfinite == 0,
            detail="" if nonfinite == 0 else f"{nonfinite}/{n_probes} chunks with NaN/Inf",
            metrics={"nonfinite_probes": nonfinite},
        )
    )

    out_of_band = [d for d in durations if not PRD_CHUNK_MIN_S <= d <= PRD_CHUNK_MAX_S]
    if out_of_band:
        warnings.append(
            f"chunk duration outside PRD band [{PRD_CHUNK_MIN_S}, {PRD_CHUNK_MAX_S}] s "
            f"for {len(out_of_band)}/{n_probes} probes (e.g. {out_of_band[0]:.3f} s)"
        )
    gates.append(
        GateResult(
            name=E2_GATE_DURATION_BAND,
            passed=True,  # warn-only gate: PRD band is a design target, not a safety limit
            detail=warnings[-1] if out_of_band else "",
            metrics={
                "min_duration_s": min(durations),
                "max_duration_s": max(durations),
                "out_of_band": len(out_of_band),
            },
        )
    )

    intervention_rate = intervention_hits / n_probes
    gates.append(
        GateResult(
            name=E2_GATE_INTERVENTION_RATE,
            passed=intervention_rate <= max_intervention_rate,
            detail=(
                ""
                if intervention_rate <= max_intervention_rate
                else f"intervention rate {intervention_rate:.3f} > {max_intervention_rate}"
            ),
            metrics={
                "intervention_rate": intervention_rate,
                "max_intervention_rate": max_intervention_rate,
                "intervention_kinds": dict(intervention_kinds),
            },
        )
    )

    mismatches = 0
    checked = min(max(determinism_probes, 1), n_probes)
    for obs in observations[:checked]:
        if not _chunks_equal(policy.predict(obs), policy.predict(obs)):
            mismatches += 1
    gates.append(
        GateResult(
            name=E2_GATE_DETERMINISM,
            passed=mismatches == 0,
            detail=(
                ""
                if mismatches == 0
                else f"{mismatches}/{checked} probes returned different chunks for the same obs"
            ),
            metrics={"checked": checked, "mismatches": mismatches},
        )
    )

    mean_ms = float(np.mean(latencies_ms))
    max_ms = float(np.max(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    gates.append(
        GateResult(
            name=E2_GATE_LATENCY,
            passed=mean_ms <= max_mean_latency_ms,
            detail=(
                ""
                if mean_ms <= max_mean_latency_ms
                else f"mean latency {mean_ms:.1f} ms > {max_mean_latency_ms:.1f} ms"
            ),
            metrics={
                "mean_ms": mean_ms,
                "max_ms": max_ms,
                "p95_ms": p95_ms,
                "implied_rate_hz": 1000.0 / mean_ms if mean_ms > 0 else None,
                "max_mean_latency_ms": max_mean_latency_ms,
            },
        )
    )

    if contract is not None:
        gates.extend(_contract_gates(contract, instruction, raw_states, warnings))

    return E2Report(
        check="static",
        n=n_probes,
        policy=type(policy).__name__,
        robot=type(robot).__name__,
        instruction=instruction,
        contract_checked=contract is not None,
        gates=gates,
        warnings=warnings,
    )


def _contract_gates(
    contract: PolicyContract,
    instruction: str,
    states: Sequence[Any],
    warnings: list[str],
) -> list[GateResult]:
    """The two train/deploy gates. Split out so the probe loop stays about the policy.

    Failure policy, decided per divergence direction rather than uniformly:

    - **Instruction — FAIL.** There is no repair the runtime can apply and no reading of the
      run that is still valid: the checkpoint conditioned on one frozen text context, and a
      different string moves the readout blocks the ActionHead was fitted to. The operator
      overrides it deliberately (``ExecutorConfig.allow_unseen_instruction``), not by
      ignoring a warning.
    - **State groups — FAIL only where no repair exists.** A group trained NEVER-valid but
      deployed valid IS repairable (mask it down and the encoder sees the exact training
      input), so it is a loud warning, not a gate failure — failing would block a run that
      the executor has already made correct. A group trained ALWAYS-valid but deployed
      invalid has no repair: the encoder's ``missing`` vector for it is untrained, so the
      policy is reading an unlearned input and the gate fails.
    """
    gates: list[GateResult] = []

    seen = contract.instruction_seen(instruction)
    if not contract.declares_instructions:
        warnings.append(
            "PolicyContract declares no instructions: the deployed instruction "
            f"{instruction!r} could NOT be checked against training"
        )
    gates.append(
        GateResult(
            name=E2_GATE_INSTRUCTION,
            passed=seen,
            detail=(
                ""
                if seen
                else (
                    f"deployed instruction {instruction!r} is not one of the "
                    f"{len(contract.instructions)} the checkpoint trained on "
                    f"(e.g. {contract.instructions[0]!r})"
                )
            ),
            metrics={
                "declared": contract.declares_instructions,
                "num_trained_instructions": len(contract.instructions),
            },
        )
    )

    repairable: dict[str, str] = {}
    unrepairable: dict[str, str] = {}
    for state in states:
        for divergence in contract.state_divergences(state):
            target = repairable if divergence.repaired else unrepairable
            target[divergence.group] = divergence.detail
    if repairable:
        warnings.append(
            "validity-mask divergence repaired by masking down to the trained mask: "
            + "; ".join(repairable.values())
        )
    gates.append(
        GateResult(
            name=E2_GATE_STATE_GROUPS,
            passed=not unrepairable,
            detail="; ".join(unrepairable.values()),
            metrics={
                "repaired_groups": sorted(repairable),
                "unrepairable_groups": sorted(unrepairable),
                "trained": {k: v.value for k, v in contract.state_groups.items()},
            },
        )
    )
    return gates


def e2_sim_rollout_checks(
    rollout_summaries: Sequence[Mapping[str, Any]],
    *,
    min_policy_rate_hz: float = MIN_POLICY_RATE_HZ,
    max_intervention_rate: float = 0.1,
) -> E2Report:
    """Gate aggregated sim rollouts (``kind == 'rollout_summary'`` dicts, shared contract).

    Gates: rollouts present, zero e-stops, zero watchdog timeouts, every rollout's policy
    rate >= ``min_policy_rate_hz`` (FR-05), interventions per control cycle bounded by
    ``max_intervention_rate``. Missing keys are treated as 0/False.
    """
    n = len(rollout_summaries)
    if n == 0:
        gate = GateResult(
            name=E2_GATE_ROLLOUTS, passed=False, detail="no rollout summaries", metrics={"n": 0}
        )
        return E2Report(check="sim_rollout", n=0, gates=[gate])

    estops = sum(1 for s in rollout_summaries if bool(s.get("estopped")))
    watchdog = sum(int(s.get("watchdog_timeouts", 0)) for s in rollout_summaries)
    rates = [float(s.get("policy_rate_hz", 0.0)) for s in rollout_summaries]
    min_rate = min(rates)
    cycles = sum(int(s.get("cycles", 0)) for s in rollout_summaries)
    interventions = sum(int(s.get("interventions_total", 0)) for s in rollout_summaries)
    intervention_rate = interventions / cycles if cycles > 0 else 0.0

    gates = [
        GateResult(name=E2_GATE_ROLLOUTS, passed=True, metrics={"n": n}),
        GateResult(
            name=E2_GATE_ZERO_ESTOPS,
            passed=estops == 0,
            detail="" if estops == 0 else f"{estops}/{n} rollouts e-stopped",
            metrics={"estops": estops},
        ),
        GateResult(
            name=E2_GATE_ZERO_WATCHDOG,
            passed=watchdog == 0,
            detail="" if watchdog == 0 else f"{watchdog} watchdog timeouts",
            metrics={"watchdog_timeouts": watchdog},
        ),
        GateResult(
            name=E2_GATE_POLICY_RATE,
            passed=min_rate >= min_policy_rate_hz,
            detail=(
                ""
                if min_rate >= min_policy_rate_hz
                else f"min policy rate {min_rate:.2f} Hz < {min_policy_rate_hz} Hz"
            ),
            metrics={"min_rate_hz": min_rate, "mean_rate_hz": float(np.mean(rates))},
        ),
        GateResult(
            name=E2_GATE_INTERVENTION_RATE,
            passed=intervention_rate <= max_intervention_rate,
            detail=(
                ""
                if intervention_rate <= max_intervention_rate
                else f"{intervention_rate:.3f} interventions/cycle > {max_intervention_rate}"
            ),
            metrics={
                "intervention_rate": intervention_rate,
                "interventions_total": interventions,
                "cycles": cycles,
            },
        ),
    ]
    return E2Report(check="sim_rollout", n=n, gates=gates)
