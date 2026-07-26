"""MVP acceptance harness AC-01..AC-07 (T-23, PRD §12.2).

Consumes rollout logs written via ``JsonlRunLogger`` per the shared rollout-log contract
(``kind == 'rollout_summary'`` records; provenance via the ``kind == 'run_metadata'``
line). Torch-free: logs and report JSONs in, :class:`AcceptanceReport` out.

Criterion statuses:

- ``pass`` / ``fail`` — criterion evaluated on sufficient real-robot data.
- ``pending_data`` — no (or not enough) matching rollouts / artifacts yet.
- ``pending_hardware`` — only sim rollouts exist (task label prefix ``'sim:'``) for a
  hardware-dependent criterion (AC-01/02, real fault injection for AC-06); sim stats are
  still reported in metrics.

AC-03 counts CONTROLLED SIM ROLLOUTS as evidence (M4/T-23): the deterministic safety
chain (SafetyLayer, watchdog, executor gates) runs identically in E2 sim rollouts and on
hardware, so a violation in sim fails AC-03 immediately and >= 100 clean rollouts pass it
regardless of platform. Real rollouts take precedence when both exist; the evaluation
basis is recorded in the criterion metrics (``population``).

Safety-violation definition (AC-03): a rollout is a violation iff ``estopped == True``,
or any intervention of kind ``nan_reject`` / ``schema_reject`` / ``state_reject`` /
``limit_breach`` occurred, or the watchdog escalated to ``stop`` (surfaced as
``estopped`` in the summary contract; an optional ``watchdog_action == 'stop'`` key is
honored defensively). ``limit_breach`` is external evidence that a hard limit was
ACTUALLY exceeded (the deterministic chain never emits it), so it is always a violation.
Clamping kinds (``joint_limit``, ``velocity_limit``, ``accel_limit``, workspace/gripper
clamps, …) are safety *interventions* doing their job — they are NOT violations.
``task == 'fault_injection'`` rollouts are excluded from AC-03 (they intentionally
trigger rejects) and evaluated by AC-06 instead. When real rollouts define the AC-03
evaluation basis, violations in the accompanying sim rollouts STILL fail the criterion
(the safety chain is platform-independent — sim violations are never discarded).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

from wam.evaluation.ablation import AblationReport
from wam.evaluation.offline import E1Report

ACCEPTANCE_VERSION = "0.1.0"

LIMIT_BREACH_KIND = "limit_breach"
VIOLATION_KINDS = frozenset({"nan_reject", "schema_reject", "state_reject", LIMIT_BREACH_KIND})
SIM_TASK_PREFIX = "sim:"
FAULT_INJECTION_TASK = "fault_injection"

DEFAULT_KNOWN_TASK = "pick_place_known"
DEFAULT_GENERALIZATION_TASK = "pick_place_generalization"

AC01_MIN_RATE = 0.80
AC01_MIN_N = 50
AC02_MIN_RATE = 0.50
AC02_MIN_N = 30
AC03_MIN_N = 100

CriterionStatus = Literal["pass", "fail", "pending_data", "pending_hardware"]

_STATUS_LABELS = {
    "pass": "PASS",
    "fail": "FAIL",
    "pending_data": "PENDING-DATA",
    "pending_hardware": "PENDING-HARDWARE",
}

_TITLES = {
    "AC-01": "Success rate on known task",
    "AC-02": "Generalization success rate",
    "AC-03": "Zero safety violations",
    "AC-04": "Reproducibility (run metadata)",
    "AC-05": "Backbone swap",
    "AC-06": "Safe stop on induced failures",
    "AC-07": "World-action ablation",
}

__all__ = [
    "AC01_MIN_N",
    "AC01_MIN_RATE",
    "AC02_MIN_N",
    "AC02_MIN_RATE",
    "AC03_MIN_N",
    "ACCEPTANCE_VERSION",
    "DEFAULT_GENERALIZATION_TASK",
    "DEFAULT_KNOWN_TASK",
    "FAULT_INJECTION_TASK",
    "LIMIT_BREACH_KIND",
    "SIM_TASK_PREFIX",
    "VIOLATION_KINDS",
    "AcceptanceCriterion",
    "AcceptanceReport",
    "BackboneSwapCheck",
    "evaluate_acceptance",
    "is_safety_violation",
    "load_rollout_summaries",
    "load_run_metadata",
    "verify_backbone_swap",
]


# ---------------------------------------------------------------------------- log loading


def _as_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(p) for p in paths]


def _iter_records(paths: str | Path | Sequence[str | Path]):
    for path in _as_paths(paths):
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{lineno}: invalid JSON line ({exc})") from exc
                if isinstance(record, dict):
                    yield record


def load_rollout_summaries(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """All ``kind == 'rollout_summary'`` records from one jsonl file or a list of files."""
    return [r for r in _iter_records(paths) if r.get("kind") == "rollout_summary"]


def load_run_metadata(paths: str | Path | Sequence[str | Path]) -> dict[str, Any] | None:
    """First ``kind == 'run_metadata'`` record across the given jsonl files, else None."""
    for record in _iter_records(paths):
        if record.get("kind") == "run_metadata":
            return record
    return None


# ---------------------------------------------------------------------------- helpers


def _split_task(task: str) -> tuple[str, bool]:
    """(base task label, is_sim) — sim rollouts carry the ``'sim:'`` task prefix."""
    if task.startswith(SIM_TASK_PREFIX):
        return task[len(SIM_TASK_PREFIX) :], True
    return task, False


def is_safety_violation(summary: Mapping[str, Any]) -> bool:
    """AC-03 violation predicate over one rollout summary (see module docstring)."""
    if bool(summary.get("estopped")):
        return True
    kinds = summary.get("intervention_kinds") or {}
    if any(kind in VIOLATION_KINDS and int(count) > 0 for kind, count in kinds.items()):
        return True
    return summary.get("watchdog_action") == "stop"


def _fault_handled(summary: Mapping[str, Any]) -> tuple[bool, str]:
    """AC-06 predicate: an induced failure must be *handled*, not crashed through.

    Handled = the safety chain visibly reacted (interventions, watchdog timeout, or a
    clean e-stop) and no limit breach was recorded. ``estopped == False`` with a hold is
    the expected outcome; a clean e-stop is also acceptable.
    """
    kinds = summary.get("intervention_kinds") or {}
    if int(kinds.get(LIMIT_BREACH_KIND, 0)) > 0:
        return False, "limit breach recorded"
    reacted = (
        int(summary.get("interventions_total", 0)) > 0
        or int(summary.get("watchdog_timeouts", 0)) > 0
        or bool(summary.get("estopped"))
    )
    if not reacted:
        return False, "no handling evidence (no interventions, watchdog timeouts or e-stop)"
    return True, ""


# ---------------------------------------------------------------------------- models


class AcceptanceCriterion(BaseModel):
    """Outcome of one acceptance criterion (AC-01..AC-07)."""

    model_config = ConfigDict(frozen=True)

    criterion: str
    title: str
    status: CriterionStatus
    detail: str = ""
    n: int = 0
    successes: int = 0
    rate: float | None = None
    required_n: int = 0
    required_rate: float | None = None
    insufficient_n: bool = False
    metrics: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.status == "pass"


class BackboneSwapCheck(BaseModel):
    """AC-05 structural check: >= 2 registry backbones conform to ``BackboneAdapter``."""

    model_config = ConfigDict(frozen=True)

    available: tuple[str, ...]
    checked: tuple[str, ...]
    conformant: tuple[str, ...]
    feature_dims: dict[str, int] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return len(self.available) >= 2 and len(set(self.conformant)) >= 2


class AcceptanceReport(BaseModel):
    """MVP acceptance dashboard: one entry per criterion AC-01..AC-07 (PRD §12.2)."""

    model_config = ConfigDict(frozen=True)

    report_version: str = ACCEPTANCE_VERSION
    known_task: str = DEFAULT_KNOWN_TASK
    generalization_task: str = DEFAULT_GENERALIZATION_TASK
    n_rollouts: int = 0
    n_real: int = 0
    n_sim: int = 0
    run_ids: tuple[str, ...] = ()
    criteria: tuple[AcceptanceCriterion, ...] = ()
    e1_summaries: dict[str, dict[str, float]] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return bool(self.criteria) and all(c.status == "pass" for c in self.criteria)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def overall_status(self) -> str:
        if not self.criteria:
            return "pending"
        if any(c.status == "fail" for c in self.criteria):
            return "fail"
        if all(c.status == "pass" for c in self.criteria):
            return "pass"
        return "pending"

    def criterion(self, criterion_id: str) -> AcceptanceCriterion:
        for entry in self.criteria:
            if entry.criterion == criterion_id:
                return entry
        raise KeyError(criterion_id)

    def failed_criteria(self) -> list[str]:
        return [c.criterion for c in self.criteria if c.status == "fail"]

    def pending_criteria(self) -> list[str]:
        return [c.criterion for c in self.criteria if c.status.startswith("pending")]

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2)

    @classmethod
    def from_json(cls, text: str) -> AcceptanceReport:
        return cls.model_validate_json(text)

    def render_markdown(self) -> str:
        """PASS/FAIL/PENDING dashboard over all seven criteria."""
        counts = {"pass": 0, "fail": 0, "pending": 0}
        for c in self.criteria:
            counts["pending" if c.status.startswith("pending") else c.status] += 1
        lines = [
            "# WAM MVP Acceptance — AC-01..AC-07",
            "",
            f"- Rollout summaries: {self.n_rollouts} (real {self.n_real}, sim {self.n_sim})",
            (
                f"- Known task: `{self.known_task}` · generalization task: "
                f"`{self.generalization_task}`"
            ),
            f"- Run ids: {', '.join(self.run_ids) if self.run_ids else '—'}",
            (
                f"- Overall: **{_STATUS_LABELS.get(self.overall_status, 'PENDING-DATA')}** "
                f"({counts['pass']} pass / {counts['fail']} fail / {counts['pending']} pending)"
            ),
            "",
            "| ID | Criterion | Status | n | Successes | Rate | Detail |",
            "|----|-----------|--------|---|-----------|------|--------|",
        ]
        for c in self.criteria:
            rate = f"{c.rate:.3f}" if c.rate is not None else "—"
            lines.append(
                f"| {c.criterion} | {c.title} | {_STATUS_LABELS[c.status]} "
                f"| {c.n} | {c.successes} | {rate} | {c.detail or '—'} |"
            )
        if self.e1_summaries:
            lines += [
                "",
                "## E1 offline reports",
                "",
                "| Report | MSE | MAE | Gripper accuracy |",
                "|--------|-----|-----|------------------|",
            ]
            for name, metrics in sorted(self.e1_summaries.items()):
                lines.append(
                    f"| {name} | {metrics.get('mse', float('nan')):.6f} "
                    f"| {metrics.get('mae', float('nan')):.6f} "
                    f"| {metrics.get('gripper_accuracy', float('nan')):.3f} |"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------- AC-05


def verify_backbone_swap(names: Sequence[str] | None = None) -> BackboneSwapCheck:
    """AC-05 smoke check via the backbone registry (no schema / robot-adapter change).

    Builds every requested backbone through the same ``get_backbone`` call path and
    verifies protocol conformance (``BackboneAdapter``) plus a usable ``feature_dim``.
    Passing means the fallback backbone can be swapped for a second one behind the
    unchanged adapter interface (FR-09). ``names`` defaults to ``available_backbones()``;
    pass an explicit subset (e.g. the torch-free ``('flux3', 'wan_i2v')``) to control
    which factories are instantiated.
    """
    from wam.backbones import available_backbones, get_backbone
    from wam.interfaces import BackboneAdapter

    available = tuple(available_backbones())
    checked = tuple(names) if names is not None else available
    conformant: list[str] = []
    feature_dims: dict[str, int] = {}
    errors: dict[str, str] = {}
    for name in checked:
        try:
            adapter = get_backbone(name)
        except Exception as exc:  # noqa: BLE001 — a broken factory is a finding, not a crash
            errors[name] = f"{type(exc).__name__}: {exc}"
            continue
        if not isinstance(adapter, BackboneAdapter):
            errors[name] = "does not conform to the BackboneAdapter protocol"
            continue
        dim = int(adapter.feature_dim)
        if dim < 1:
            errors[name] = f"invalid feature_dim {dim}"
            continue
        conformant.append(name)
        feature_dims[name] = dim
    return BackboneSwapCheck(
        available=available,
        checked=checked,
        conformant=tuple(conformant),
        feature_dims=feature_dims,
        errors=errors,
    )


# ---------------------------------------------------------------------------- criteria


def _success_criterion(
    criterion_id: str,
    task: str,
    summaries: Sequence[Mapping[str, Any]],
    *,
    min_rate: float,
    min_n: int,
) -> AcceptanceCriterion:
    """AC-01/AC-02: success rate over real rollouts of one task label."""
    real = [s for s in summaries if _split_task(str(s.get("task", "")))[0] == task]
    sim = [s for s in real if _split_task(str(s.get("task", "")))[1]]
    real = [s for s in real if not _split_task(str(s.get("task", "")))[1]]

    metrics: dict[str, Any] = {"task": task, "sim_n": len(sim)}
    if sim:
        sim_successes = sum(1 for s in sim if bool(s.get("success")))
        metrics["sim_successes"] = sim_successes
        metrics["sim_rate"] = sim_successes / len(sim)

    common = {
        "criterion": criterion_id,
        "title": _TITLES[criterion_id],
        "required_n": min_n,
        "required_rate": min_rate,
        "metrics": metrics,
    }
    if not real:
        if sim:
            return AcceptanceCriterion(
                status="pending_hardware",
                detail=f"only sim rollouts (n={len(sim)}) — real-robot data required",
                insufficient_n=True,
                **common,
            )
        return AcceptanceCriterion(
            status="pending_data",
            detail=f"no rollouts for task '{task}'",
            insufficient_n=True,
            **common,
        )

    n = len(real)
    successes = sum(1 for s in real if bool(s.get("success")))
    rate = successes / n
    stats = {"n": n, "successes": successes, "rate": rate}
    if n < min_n:
        return AcceptanceCriterion(
            status="pending_data",
            detail=f"only {n}/{min_n} real rollouts on task '{task}' (rate so far {rate:.3f})",
            insufficient_n=True,
            **stats,
            **common,
        )
    if rate >= min_rate:
        return AcceptanceCriterion(
            status="pass",
            detail=f"{successes}/{n} succeeded (rate {rate:.3f} >= {min_rate})",
            **stats,
            **common,
        )
    return AcceptanceCriterion(
        status="fail",
        detail=f"rate {rate:.3f} < {min_rate} over {n} rollouts",
        **stats,
        **common,
    )


def _ac03(summaries: Sequence[Mapping[str, Any]]) -> AcceptanceCriterion:
    """AC-03 over non-fault-injection rollouts. Controlled sim rollouts count as evidence
    (the deterministic safety chain is platform-independent, see module docstring); real
    rollouts take precedence as the evaluation population (the n >= AC03_MIN_N sufficiency
    basis) when both exist, but a violation in EITHER population fails the criterion."""
    eligible = [
        s for s in summaries if _split_task(str(s.get("task", "")))[0] != FAULT_INJECTION_TASK
    ]
    real = [s for s in eligible if not _split_task(str(s.get("task", "")))[1]]
    sim = [s for s in eligible if _split_task(str(s.get("task", "")))[1]]
    population, basis = (real, "real") if real else (sim, "sim")

    violations = [s for s in population if is_safety_violation(s)]
    # Sim violations fail AC-03 even when real rollouts define the evaluation basis: the
    # deterministic safety chain runs identically on both platforms (module docstring).
    extra_sim_violations = (
        [s for s in sim if is_safety_violation(s)] if basis == "real" else []
    )
    kinds: dict[str, int] = {}
    for s in (*violations, *extra_sim_violations):
        for kind, count in (s.get("intervention_kinds") or {}).items():
            if kind in VIOLATION_KINDS and int(count) > 0:
                kinds[kind] = kinds.get(kind, 0) + int(count)
    metrics: dict[str, Any] = {
        "population": basis,
        "violations": len(violations),
        "violation_kinds": kinds,
        "violating_rollouts": [
            str(s.get("rollout_id", "?")) for s in (*violations, *extra_sim_violations)[:5]
        ],
        "estopped": sum(1 for s in population if bool(s.get("estopped"))),
        "sim_n": len(sim),
        "sim_violations": sum(1 for s in sim if is_safety_violation(s)),
    }
    common = {
        "criterion": "AC-03",
        "title": _TITLES["AC-03"],
        "required_n": AC03_MIN_N,
        "n": len(population),
        "successes": len(population) - len(violations),
        "insufficient_n": len(population) < AC03_MIN_N,
        "metrics": metrics,
    }
    if violations:
        return AcceptanceCriterion(
            status="fail",
            detail=f"{len(violations)} safety violation(s) in "
            f"{len(population)} {basis} rollouts",
            **common,
        )
    if extra_sim_violations:
        return AcceptanceCriterion(
            status="fail",
            detail=f"{len(extra_sim_violations)} safety violation(s) in {len(sim)} sim "
            f"rollouts despite clean {basis} population",
            **common,
        )
    if not population:
        return AcceptanceCriterion(status="pending_data", detail="no rollouts", **common)
    if len(population) < AC03_MIN_N:
        return AcceptanceCriterion(
            status="pending_data",
            detail=f"0 violations so far, but only "
            f"{len(population)}/{AC03_MIN_N} {basis} rollouts",
            **common,
        )
    return AcceptanceCriterion(
        status="pass",
        detail=f"0 safety violations across {len(population)} {basis} rollouts",
        rate=0.0,
        **common,
    )


def _ac04(metadata_line: Mapping[str, Any] | None) -> AcceptanceCriterion:
    required = ("config_hash", "checkpoint_ref", "dataset_snapshot_ref")
    if metadata_line is None:
        return AcceptanceCriterion(
            criterion="AC-04",
            title=_TITLES["AC-04"],
            status="fail",
            detail="no run_metadata line found in the rollout logs",
        )
    missing = [key for key in required if not str(metadata_line.get(key) or "").strip()]
    metrics = {key: metadata_line.get(key) for key in (*required, "run_id", "git_commit")}
    if missing:
        return AcceptanceCriterion(
            criterion="AC-04",
            title=_TITLES["AC-04"],
            status="fail",
            detail=f"run_metadata missing/empty: {', '.join(missing)}",
            metrics=metrics,
        )
    return AcceptanceCriterion(
        criterion="AC-04",
        title=_TITLES["AC-04"],
        status="pass",
        detail=f"traceable to checkpoint '{metadata_line['checkpoint_ref']}' + "
        f"dataset '{metadata_line['dataset_snapshot_ref']}'",
        metrics=metrics,
    )


def _ac05(check: BackboneSwapCheck) -> AcceptanceCriterion:
    metrics = check.model_dump(mode="json")
    if check.passed:
        return AcceptanceCriterion(
            criterion="AC-05",
            title=_TITLES["AC-05"],
            status="pass",
            detail=f"{len(check.conformant)} protocol-conformant backbones via registry: "
            f"{', '.join(check.conformant)}",
            n=len(check.conformant),
            required_n=2,
            metrics=metrics,
        )
    problems = "; ".join(f"{k}: {v}" for k, v in check.errors.items()) or "fewer than 2 backbones"
    return AcceptanceCriterion(
        criterion="AC-05",
        title=_TITLES["AC-05"],
        status="fail",
        detail=f"backbone swap not verified — {problems}",
        n=len(check.conformant),
        required_n=2,
        insufficient_n=len(check.conformant) < 2,
        metrics=metrics,
    )


def _ac06(summaries: Sequence[Mapping[str, Any]]) -> AcceptanceCriterion:
    faults = [
        s for s in summaries if _split_task(str(s.get("task", "")))[0] == FAULT_INJECTION_TASK
    ]
    real = [s for s in faults if not _split_task(str(s.get("task", "")))[1]]
    sim = [s for s in faults if _split_task(str(s.get("task", "")))[1]]
    population, hardware_pending = (real, False) if real else (sim, bool(sim))

    unhandled: list[str] = []
    for s in population:
        handled, reason = _fault_handled(s)
        if not handled:
            unhandled.append(f"{s.get('rollout_id', '?')}: {reason}")
    metrics: dict[str, Any] = {
        "real_n": len(real),
        "sim_n": len(sim),
        "unhandled": unhandled[:5],
        "estopped_cleanly": sum(1 for s in population if bool(s.get("estopped"))),
    }
    common = {
        "criterion": "AC-06",
        "title": _TITLES["AC-06"],
        "n": len(population),
        "successes": len(population) - len(unhandled),
        "metrics": metrics,
    }
    if not faults:
        return AcceptanceCriterion(
            status="pending_data",
            detail=f"no '{FAULT_INJECTION_TASK}' rollouts recorded",
            **common,
        )
    if unhandled:
        return AcceptanceCriterion(
            status="fail",
            detail=f"{len(unhandled)}/{len(population)} induced failures not handled "
            f"(first: {unhandled[0]})",
            **common,
        )
    if hardware_pending:
        return AcceptanceCriterion(
            status="pending_hardware",
            detail=f"all {len(population)} sim fault injections handled — "
            "real-robot fault injection required",
            **common,
        )
    return AcceptanceCriterion(
        status="pass",
        detail=f"all {len(population)} induced failures handled safely "
        "(hold or clean e-stop, no limit breach)",
        **common,
    )


def _ac07(ablation_json: str | Path | None) -> AcceptanceCriterion:
    if ablation_json is None:
        return AcceptanceCriterion(
            criterion="AC-07",
            title=_TITLES["AC-07"],
            status="pending_data",
            detail="pending real D2 data",
        )
    path = Path(ablation_json)
    try:
        report = AblationReport.from_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return AcceptanceCriterion(
            criterion="AC-07",
            title=_TITLES["AC-07"],
            status="fail",
            detail=f"could not load AblationReport from {path}: {exc}",
        )
    return AcceptanceCriterion(
        criterion="AC-07",
        title=_TITLES["AC-07"],
        status="pass",
        detail=f"ablation documented ({path.name}): {report.verdict}",
        metrics={
            "ablation_json": str(path),
            "baseline": report.baseline_name,
            "candidate": report.candidate_name,
            "verdict": report.verdict,
        },
    )


# ---------------------------------------------------------------------------- entry point


def evaluate_acceptance(
    summaries: Sequence[Mapping[str, Any]],
    *,
    metadata_line: Mapping[str, Any] | None = None,
    e1_reports: Mapping[str, E1Report] | None = None,
    ablation_json: str | Path | None = None,
    known_task: str = DEFAULT_KNOWN_TASK,
    generalization_task: str = DEFAULT_GENERALIZATION_TASK,
    backbone_names: Sequence[str] | None = None,
    backbone_check: BackboneSwapCheck | None = None,
) -> AcceptanceReport:
    """Evaluate AC-01..AC-07 over rollout summaries (shared rollout-log contract).

    ``metadata_line`` is the ``kind == 'run_metadata'`` record (AC-04); ``e1_reports``
    attach offline E1 evidence to the dashboard; ``ablation_json`` points to an
    ``AblationReport`` JSON (AC-07). AC-05 runs :func:`verify_backbone_swap` unless a
    precomputed ``backbone_check`` is given.
    """
    if backbone_check is None:
        backbone_check = verify_backbone_swap(backbone_names)

    criteria = (
        _success_criterion(
            "AC-01", known_task, summaries, min_rate=AC01_MIN_RATE, min_n=AC01_MIN_N
        ),
        _success_criterion(
            "AC-02",
            generalization_task,
            summaries,
            min_rate=AC02_MIN_RATE,
            min_n=AC02_MIN_N,
        ),
        _ac03(summaries),
        _ac04(metadata_line),
        _ac05(backbone_check),
        _ac06(summaries),
        _ac07(ablation_json),
    )

    n_sim = sum(1 for s in summaries if _split_task(str(s.get("task", "")))[1])
    run_ids = tuple(sorted({str(s["run_id"]) for s in summaries if s.get("run_id")}))
    e1_summaries = {
        name: {
            "mse": report.mse,
            "mae": report.mae,
            "gripper_accuracy": report.gripper_accuracy,
        }
        for name, report in (e1_reports or {}).items()
    }
    return AcceptanceReport(
        known_task=known_task,
        generalization_task=generalization_task,
        n_rollouts=len(summaries),
        n_real=len(summaries) - n_sim,
        n_sim=n_sim,
        run_ids=run_ids,
        criteria=criteria,
        e1_summaries=e1_summaries,
    )
