"""Ablation harness (T-18, AC-07): world-action vs. action-only baseline on E1 reports.

Contracts:
- Pure report comparison — no model access. Feed it ``E1Report`` objects produced by
  ``wam.evaluation.offline.e1_metrics`` for runs that differ ONLY in the ablated component
  (same holdout split, same spec) or the deltas are meaningless.
- Verdict rule (documented threshold): decided on the RELATIVE MSE IMPROVEMENT of the
  candidate (world-action) over the baseline (action-only),
  ``improvement_pct = (baseline_mse - candidate_mse) / baseline_mse * 100``:
    improvement_pct >= +threshold  -> 'video branch helps'
    improvement_pct <= -threshold  -> 'hurts'
    otherwise                      -> 'no significant difference'
  Default threshold is 5% — offline action-MSE on small holdout sets fluctuates a few percent
  between reruns/seeds (PRD 10.4: MSE is diagnostic only), so smaller gaps are treated as
  noise. AC-07 asks for a MEASURABLE advantage; tighten/loosen via ``threshold_pct``.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wam.evaluation.offline import EVAL_VERSION, E1Report, _fmt

VERDICT_HELPS = "video branch helps"
VERDICT_NO_DIFF = "no significant difference"
VERDICT_HURTS = "hurts"

DEFAULT_THRESHOLD_PCT = 5.0
"""Minimum |relative MSE improvement| in percent to call the ablation (see module docstring)."""

_BASELINE_HINTS = ("actiononly", "baseline")

# metric name -> higher_is_better
_COMPARED_METRICS: dict[str, bool] = {
    "mse": False,
    "mae": False,
    "gripper_accuracy": True,
    "smoothness_pred": False,
}


class MetricDelta(BaseModel):
    """Baseline vs. candidate values for one metric.

    ``delta`` is candidate - baseline. ``improvement_pct`` is signed so that POSITIVE always
    means the candidate is better, regardless of metric direction; ``None`` when undefined
    (baseline == 0 for a lower-is-better metric with a nonzero candidate, or baseline == 0
    for a higher-is-better metric).
    """

    model_config = ConfigDict(frozen=True)

    higher_is_better: bool
    baseline: float
    candidate: float
    delta: float
    improvement_pct: float | None


class AblationReport(BaseModel):
    """AC-07 comparison of a world-action candidate against the action-only baseline."""

    model_config = ConfigDict(frozen=True)

    report_version: str = EVAL_VERSION
    baseline_name: str
    candidate_name: str
    threshold_pct: float = Field(gt=0.0)
    metrics: dict[str, MetricDelta]
    verdict: Literal["video branch helps", "no significant difference", "hurts"]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> AblationReport:
        return cls.model_validate_json(text)

    def render_markdown(self) -> str:
        lines = [
            "# Ablation: world-action vs. action-only (AC-07)",
            "",
            f"- baseline (action-only): `{self.baseline_name}`",
            f"- candidate (world-action): `{self.candidate_name}`",
            f"- verdict threshold: {_fmt(self.threshold_pct)}% relative MSE improvement",
            "",
            "| metric | direction | baseline | candidate | delta | improvement % |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, m in self.metrics.items():
            direction = "higher is better" if m.higher_is_better else "lower is better"
            imp = "n/a" if m.improvement_pct is None else f"{m.improvement_pct:+.2f}%"
            lines.append(
                f"| {name} | {direction} | {_fmt(m.baseline)} | {_fmt(m.candidate)} | "
                f"{_fmt(m.delta)} | {imp} |"
            )
        mse = self.metrics["mse"]
        basis = (
            "undefined MSE improvement (baseline MSE == 0)"
            if mse.improvement_pct is None
            else f"MSE improvement {mse.improvement_pct:+.2f}% "
            f"vs. threshold {_fmt(self.threshold_pct)}%"
        )
        lines += ["", f"**Verdict: {self.verdict}** ({basis})", ""]
        return "\n".join(lines)


def _improvement_pct(baseline: float, candidate: float, higher_is_better: bool) -> float | None:
    if baseline == 0.0:
        if not higher_is_better and candidate == 0.0:
            return 0.0
        return None
    if higher_is_better:
        return (candidate - baseline) / abs(baseline) * 100.0
    return (baseline - candidate) / abs(baseline) * 100.0


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_pair(
    reports: dict[str, E1Report], baseline: str | None, candidate: str | None
) -> tuple[str, str]:
    names = list(reports)
    if len(names) < 2:
        raise ValueError("compare_runs: need at least two reports")
    for role, name in (("baseline", baseline), ("candidate", candidate)):
        if name is not None and name not in reports:
            raise ValueError(f"compare_runs: {role} {name!r} not in reports {names}")
    if baseline is None:
        hits = [
            n for n in names if n != candidate and any(h in _norm_name(n) for h in _BASELINE_HINTS)
        ]
        if len(hits) == 1:
            baseline = hits[0]
        elif len(names) == 2 and candidate is not None:
            baseline = next(n for n in names if n != candidate)
        else:
            raise ValueError(
                "compare_runs: cannot infer the action-only baseline from "
                f"{names}; pass baseline= explicitly"
            )
    if candidate is None:
        rest = [n for n in names if n != baseline]
        if len(rest) != 1:
            raise ValueError(
                f"compare_runs: cannot infer candidate among {rest}; pass candidate= explicitly"
            )
        candidate = rest[0]
    if baseline == candidate:
        raise ValueError("compare_runs: baseline and candidate must differ")
    return baseline, candidate


def compare_runs(
    reports: dict[str, E1Report],
    *,
    baseline: str | None = None,
    candidate: str | None = None,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> AblationReport:
    """Compare a world-action run against the action-only baseline (AC-07).

    ``reports`` maps run name -> E1Report. When ``baseline``/``candidate`` are omitted, the
    baseline is auto-detected by name (a unique name containing 'action-only'/'action_only'
    or 'baseline'); with exactly two reports the other one becomes the candidate. The verdict
    rule and threshold are documented in the module docstring.
    """
    baseline_name, candidate_name = _resolve_pair(reports, baseline, candidate)
    base, cand = reports[baseline_name], reports[candidate_name]

    metrics: dict[str, MetricDelta] = {}
    for name, higher_is_better in _COMPARED_METRICS.items():
        b = float(getattr(base, name))
        c = float(getattr(cand, name))
        metrics[name] = MetricDelta(
            higher_is_better=higher_is_better,
            baseline=b,
            candidate=c,
            delta=c - b,
            improvement_pct=_improvement_pct(b, c, higher_is_better),
        )

    mse_imp = metrics["mse"].improvement_pct
    if mse_imp is None:
        # Baseline MSE == 0 (perfect baseline) with nonzero candidate error: cannot help.
        verdict = VERDICT_HURTS
    elif mse_imp >= threshold_pct:
        verdict = VERDICT_HELPS
    elif mse_imp <= -threshold_pct:
        verdict = VERDICT_HURTS
    else:
        verdict = VERDICT_NO_DIFF

    return AblationReport(
        baseline_name=baseline_name,
        candidate_name=candidate_name,
        threshold_pct=threshold_pct,
        metrics=metrics,
        verdict=verdict,
    )
