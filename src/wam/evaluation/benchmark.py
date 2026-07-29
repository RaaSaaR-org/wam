"""WAM-Bench (T-27): a laddered offline benchmark scored on real held-out episodes.

Why this exists next to E1 (``offline.py``). E1 reports raw action MSE, and raw MSE is a weak
proxy for the thing we actually want. Two published results drive the design:

- Restricting the same error to task-critical intervals lifts the rank correlation with real
  rollout success from Spearman ~-0.61 to ~-0.87 (CI-MSE, arXiv:2606.29898). Most timesteps in
  a demonstration are quiet, so an average over all of them largely measures how well a policy
  holds still.
- Plausible-looking output does not imply the model responds to the commanded action
  (MiraBench, arXiv:2605.29360). Fidelity and controllability are separate axes and must be
  scored separately.

So the ladder opens with the rung raw MSE never tests: *is the policy better than a trivial
baseline at all?* On our own real-data runs that question was decisive — see ``docs/benchmark.md``.

Contracts:
- Same input as E1: a sequence of ``ChunkPrediction``. No model access, torch-free, float64.
  Anything scored here can therefore be recomputed from an archived ``predictions.jsonl``.
- Every gate threshold is a module constant, fixed BEFORE a run is scored. Reading a number and
  then choosing the threshold that flatters it is how a benchmark stops measuring anything.
- Rungs are ordered and the reported ``level`` is the highest CONTIGUOUS rung passed: clearing
  L2 while failing L1 does not earn L2, because the rungs are premises for each other.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from wam.evaluation.offline import (
    EVAL_VERSION,
    GRIPPER_BINARIZE_THRESHOLD,
    ChunkPrediction,
    _check_pair,
    _fmt,
)

BENCH_VERSION = "0.1.0"

CRITICAL_QUANTILE = 0.8
"""Timesteps at or above this quantile of demonstrated motion energy are 'task-critical'.

CI-MSE annotates critical intervals with a VLM. We have one task and no annotations, so the
proxy is the demonstrator's own commanded motion: the moments they were actively moving are the
moments the task is being performed. Top 20% keeps enough samples to be stable on a 40-episode
holdout while still excluding the quiet majority.
"""

SHIFT_TOLERANCE_STEPS = 1
"""Half-width of the local time-shift relaxation (CI-MSE uses DTW with window w=1)."""

# Pre-registered scoring anchors: value at which a rung earns full marks. Each rung is worth
# RUNG_POINTS; sub-scores are linear from the gate to the anchor and clipped.
RUNG_POINTS = 20.0
ANCHOR_SKILL_VS_ZERO_PCT = 50.0
ANCHOR_SKILL_VS_REPEAT_PCT = 25.0
ANCHOR_CI_SKILL_VS_REPEAT_PCT = 25.0
MAX_HORIZON_RATIO = 4.0
MAX_SMOOTHNESS_RATIO = 2.0

GRIPPER_MIN_DYNAMIC_RANGE = 0.25
"""Below this peak-to-peak range the gripper channel carries no open/close event to score."""

RUNG_NAMES = (
    "L0 beats-doing-nothing",
    "L1 beats-inertia",
    "L2 acts-when-it-counts",
    "L3 holds-the-horizon",
    "L4 moves-like-a-demo",
)


class RungResult(BaseModel):
    """One rung of the ladder: the measured value, the gate it had to clear, and its points."""

    # ser_json_inf_nan: a diverging run legitimately produces inf (see _ratio), and pydantic's
    # default writes that as null — which then fails to parse back, so the artifact of the worst
    # run is the one you cannot re-read. "constants" emits Infinity, which json.loads accepts.
    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    name: str
    question: str
    metric: str
    value: float
    gate: str
    passed: bool
    points: float = Field(ge=0.0, le=RUNG_POINTS)


class BaselineScores(BaseModel):
    """Error of the trivial reference predictors on the identical evaluation set."""

    model_config = ConfigDict(frozen=True)

    zero_mse: float
    repeat_mse: float
    zero_ci_mse: float
    repeat_ci_mse: float


class BenchReport(BaseModel):
    """WAM-Bench result: rung ladder, composite score and the diagnostics behind them.

    - ``level``: highest contiguous rung index passed, -1 when L0 already fails.
    - ``score``: sum of rung points, 0-100. Continuous, so it moves between rung changes.
    - ``mse`` / ``ci_mse``: model error over all chunks / over the task-critical ones.
    - ``skill_vs_*``: percent error reduction against a trivial predictor. Positive = better.
    - ``horizon_ratio``: last-step MSE over first-step MSE — how fast the chunk falls apart.
    - ``smoothness_ratio``: predicted jerk over demonstrated jerk. >1 = jerkier than a human.
    - ``shift_tolerant_mse`` / ``timing_gain_pct``: error when each step may match a neighbour
      within ``SHIFT_TOLERANCE_STEPS``. A large gain means the shape is right and the phase is
      wrong, which is a latency bug, not a capacity one.
    - ``warnings``: conditions that make a headline metric unreadable rather than merely bad.
    """

    model_config = ConfigDict(frozen=True, ser_json_inf_nan="constants")

    report_version: str = BENCH_VERSION
    eval_version: str = EVAL_VERSION
    run_name: str
    num_predictions: int = Field(ge=1)
    num_episodes: int = Field(ge=1)
    num_critical: int = Field(ge=0)
    critical_quantile: float

    level: int
    level_name: str
    score: float
    rungs: tuple[RungResult, ...]

    mse: float
    ci_mse: float
    skill_vs_zero_pct: float
    skill_vs_repeat_pct: float
    ci_skill_vs_zero_pct: float
    ci_skill_vs_repeat_pct: float
    horizon_ratio: float
    smoothness_ratio: float
    shift_tolerant_mse: float
    timing_gain_pct: float
    gripper_accuracy: float
    gripper_dynamic_range: float
    baselines: BaselineScores
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, text: str) -> BenchReport:
        return cls.model_validate_json(text)

    def render_markdown(self) -> str:
        lines = [
            f"# WAM-Bench — `{self.run_name}`",
            "",
            f"**Level {self.level_name} · score {self.score:.1f}/100**",
            "",
            f"- {self.num_predictions} chunks over {self.num_episodes} held-out episodes",
            (
                f"- {self.num_critical} task-critical chunks "
                f"(top {(1.0 - self.critical_quantile) * 100:.0f}% by demonstrated motion energy)"
            ),
            "",
            "## Ladder",
            "",
            "| rung | question | metric | value | gate | result | points |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for rung in self.rungs:
            mark = "PASS" if rung.passed else "FAIL"
            lines.append(
                f"| {rung.name} | {rung.question} | `{rung.metric}` | {_fmt(rung.value)} | "
                f"{rung.gate} | **{mark}** | {rung.points:.1f} |"
            )
        lines += [
            "",
            "## Against trivial baselines",
            "",
            "| predictor | MSE (all) | skill % | CI-MSE (critical) | CI skill % |",
            "| --- | --- | --- | --- | --- |",
            (
                f"| **model** | {_fmt(self.mse)} | {self.skill_vs_zero_pct:+.1f}% | "
                f"{_fmt(self.ci_mse)} | {self.ci_skill_vs_zero_pct:+.1f}% |"
            ),
            (
                f"| zero-delta (hold still) | {_fmt(self.baselines.zero_mse)} | 0.0% | "
                f"{_fmt(self.baselines.zero_ci_mse)} | 0.0% |"
            ),
            (
                f"| repeat-last-action | {_fmt(self.baselines.repeat_mse)} | "
                f"{_skill(self.baselines.zero_mse, self.baselines.repeat_mse):+.1f}% | "
                f"{_fmt(self.baselines.repeat_ci_mse)} | "
                f"{_skill(self.baselines.zero_ci_mse, self.baselines.repeat_ci_mse):+.1f}% |"
            ),
            "",
            (
                "Skill % is the error reduction against zero-delta; the model must also beat "
                "repeat-last-action, which is the harder of the two."
            ),
            "",
            "## Diagnostics",
            "",
            "| metric | value | reads as |",
            "| --- | --- | --- |",
            (
                f"| horizon_ratio | {_fmt(self.horizon_ratio)} | "
                f"last-step error / first-step error |"
            ),
            (
                f"| smoothness_ratio | {_fmt(self.smoothness_ratio)} | "
                f"predicted jerk / demonstrated jerk |"
            ),
            (
                f"| timing_gain_pct | {self.timing_gain_pct:+.1f}% | "
                f"error removed by allowing a +/-{SHIFT_TOLERANCE_STEPS}-step shift |"
            ),
            (
                f"| gripper_accuracy | {_fmt(self.gripper_accuracy)} | "
                f"agreement after binarizing at {GRIPPER_BINARIZE_THRESHOLD} |"
            ),
            (
                f"| gripper_dynamic_range | {_fmt(self.gripper_dynamic_range)} | "
                f"peak-to-peak of the demonstrated gripper signal |"
            ),
        ]
        if self.warnings:
            lines += ["", "## Warnings", ""]
            lines += [f"- {w}" for w in self.warnings]
        return "\n".join(lines) + "\n"


def _ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, with 0/0 read as 1.0 rather than as a blow-up.

    A perfect policy has zero error at both ends of the chunk and zero jerk difference from a
    perfectly smooth demo. Reporting inf there would fail the very rungs it aced; inf is reserved
    for the real pathology, a finite numerator over a zero denominator.
    """
    if denominator > 0.0:
        return numerator / denominator
    return 1.0 if numerator == 0.0 else float("inf")


def _skill(reference: float, value: float) -> float:
    """Percent error reduction of ``value`` against ``reference``; 0.0 when reference is 0."""
    if reference <= 0.0:
        return 0.0
    return (reference - value) / reference * 100.0


def _causal_previous_action(
    chunks: Sequence[ChunkPrediction], index: int, num_steps: int
) -> np.ndarray:
    """The action executed immediately BEFORE ``chunks[index]`` starts, or zeros at episode start.

    This is what makes repeat-last-action a legitimate baseline rather than an oracle. Chunks in
    an episode are emitted every ``stride`` control steps, so the previous chunk's step at index
    ``stride - 1`` lands exactly one control period before the current chunk begins. Taking the
    previous chunk's LAST step instead would reach into the future whenever chunks overlap
    (stride < T), which is precisely the regime the closed-loop runtime uses (FR-05 executes a
    prefix and re-plans), so the general form is the only safe one.
    """
    if index == 0:
        return np.zeros(chunks[0].target.targets.shape[1], dtype=np.float64)
    prev = chunks[index - 1].target
    dt_ns = prev.dt_s * 1e9
    stride = 1 if dt_ns <= 0 else round((chunks[index].t_ns - chunks[index - 1].t_ns) / dt_ns)
    step = int(np.clip(stride - 1, 0, prev.targets.shape[0] - 1))
    return prev.targets[step].astype(np.float64)


def _shift_tolerant_sq(pred: np.ndarray, target: np.ndarray, width: int) -> float:
    """Mean squared error where each predicted step may match any target step within ``width``.

    A per-step relaxation, not a monotone DTW path, so it is an OPTIMISTIC bound on the true
    warped distance — good enough to answer "is this a phase error?", not a distance metric.
    """
    steps = pred.shape[0]
    best = np.full(steps, np.inf, dtype=np.float64)
    for offset in range(-width, width + 1):
        idx = np.clip(np.arange(steps) + offset, 0, steps - 1)
        best = np.minimum(best, ((pred - target[idx]) ** 2).mean(axis=1))
    return float(best.mean())


def bench_metrics(
    predictions: Sequence[ChunkPrediction],
    *,
    run_name: str = "run",
    critical_quantile: float = CRITICAL_QUANTILE,
) -> BenchReport:
    """Score a set of chunk predictions on the WAM-Bench ladder.

    Predictions are grouped by episode and ordered by ``t_ns`` internally, so the caller may pass
    them in any order — but the repeat-last-action baseline is only causal if the predictions are
    the complete, contiguous chunk sequence of each episode.
    """
    if len(predictions) == 0:
        raise ValueError("bench_metrics: no predictions given")
    if not 0.0 <= critical_quantile < 1.0:
        raise ValueError(f"critical_quantile must be in [0, 1), got {critical_quantile}")
    for i, pred in enumerate(predictions):
        _check_pair(i, pred)

    by_episode: dict[str, list[ChunkPrediction]] = {}
    for pred in predictions:
        by_episode.setdefault(pred.episode_id, []).append(pred)
    for chunks in by_episode.values():
        chunks.sort(key=lambda p: p.t_ns)

    # Pass 1: motion energy per chunk, so the critical threshold is a property of the whole
    # evaluation set rather than of whichever episode happens to be scored first.
    energies: list[float] = []
    for chunks in by_episode.values():
        for pred in chunks:
            t = pred.target.targets.astype(np.float64)
            energies.append(float(np.sqrt((t**2).mean())))
    threshold = float(np.quantile(np.asarray(energies), critical_quantile))

    totals = {k: 0.0 for k in ("model", "zero", "repeat", "shift")}
    crit = {k: 0.0 for k in ("model", "zero", "repeat")}
    n_all = 0
    n_crit = 0
    step_sq: np.ndarray | None = None
    step_cnt: np.ndarray | None = None
    jerk = {"pred": 0.0, "target": 0.0}
    jerk_n = {"pred": 0, "target": 0}
    grip_match = 0
    grip_total = 0
    grip_min = np.inf
    grip_max = -np.inf

    for chunks in by_episode.values():
        for i, pred in enumerate(chunks):
            p = pred.predicted.targets.astype(np.float64)
            t = pred.target.targets.astype(np.float64)
            steps = t.shape[0]
            last = _causal_previous_action(chunks, i, steps)
            repeat = np.repeat(last[None, :], steps, axis=0)

            errors = {
                "model": float(((p - t) ** 2).mean()),
                "zero": float((t**2).mean()),
                "repeat": float(((repeat - t) ** 2).mean()),
            }
            for key, value in errors.items():
                totals[key] += value
            totals["shift"] += _shift_tolerant_sq(p, t, SHIFT_TOLERANCE_STEPS)
            n_all += 1

            if float(np.sqrt((t**2).mean())) >= threshold:
                for key, value in errors.items():
                    crit[key] += value
                n_crit += 1

            per_step = ((p - t) ** 2).mean(axis=1)
            if step_sq is None or step_sq.shape[0] < steps:
                grown = np.zeros(steps, dtype=np.float64)
                grown_n = np.zeros(steps, dtype=np.float64)
                if step_sq is not None:
                    grown[: step_sq.shape[0]] = step_sq
                    grown_n[: step_cnt.shape[0]] = step_cnt  # type: ignore[union-attr]
                step_sq, step_cnt = grown, grown_n
            step_sq[:steps] += per_step
            step_cnt[:steps] += 1.0  # type: ignore[index]

            for key, x in (("pred", p), ("target", t)):
                if steps >= 3:
                    d2 = x[2:] - 2.0 * x[1:-1] + x[:-2]
                    jerk[key] += float((d2**2).sum())
                    jerk_n[key] += d2.size

            gp = np.asarray(pred.predicted.gripper_target, dtype=np.float64)
            gt = np.asarray(pred.target.gripper_target, dtype=np.float64)
            grip_match += int(
                ((gp >= GRIPPER_BINARIZE_THRESHOLD) == (gt >= GRIPPER_BINARIZE_THRESHOLD)).sum()
            )
            grip_total += int(gt.shape[0])
            grip_min = min(grip_min, float(gt.min()))
            grip_max = max(grip_max, float(gt.max()))

    assert step_sq is not None and step_cnt is not None  # n_all >= 1 guarantees both are set
    mse = totals["model"] / n_all
    zero_mse = totals["zero"] / n_all
    repeat_mse = totals["repeat"] / n_all
    shift_mse = totals["shift"] / n_all
    # No chunk clears the threshold only when quantile == 1.0, which the guard above rejects;
    # falling back to the all-chunk numbers keeps a degenerate single-chunk input scoreable.
    ci_mse = crit["model"] / n_crit if n_crit else mse
    ci_zero = crit["zero"] / n_crit if n_crit else zero_mse
    ci_repeat = crit["repeat"] / n_crit if n_crit else repeat_mse

    per_step_mse = step_sq / np.maximum(step_cnt, 1.0)
    first, final = float(per_step_mse[0]), float(per_step_mse[-1])
    horizon_ratio = _ratio(final, first)
    jerk_pred = jerk["pred"] / jerk_n["pred"] if jerk_n["pred"] else 0.0
    jerk_target = jerk["target"] / jerk_n["target"] if jerk_n["target"] else 0.0
    smoothness_ratio = _ratio(jerk_pred, jerk_target)

    skill_zero = _skill(zero_mse, mse)
    skill_repeat = _skill(repeat_mse, mse)
    ci_skill_zero = _skill(ci_zero, ci_mse)
    ci_skill_repeat = _skill(ci_repeat, ci_mse)

    rungs = (
        RungResult(
            name=RUNG_NAMES[0],
            question="Better than holding still?",
            metric="skill_vs_zero_pct",
            value=skill_zero,
            gate="> 0%",
            passed=skill_zero > 0.0,
            points=_points(skill_zero, 0.0, ANCHOR_SKILL_VS_ZERO_PCT),
        ),
        RungResult(
            name=RUNG_NAMES[1],
            question="Better than repeating the last action?",
            metric="skill_vs_repeat_pct",
            value=skill_repeat,
            gate="> 0%",
            passed=skill_repeat > 0.0,
            points=_points(skill_repeat, 0.0, ANCHOR_SKILL_VS_REPEAT_PCT),
        ),
        RungResult(
            name=RUNG_NAMES[2],
            question="Still better where the task actually happens?",
            metric="ci_skill_vs_repeat_pct",
            value=ci_skill_repeat,
            gate="> 0%",
            passed=ci_skill_repeat > 0.0,
            points=_points(ci_skill_repeat, 0.0, ANCHOR_CI_SKILL_VS_REPEAT_PCT),
        ),
        RungResult(
            name=RUNG_NAMES[3],
            question="Does the chunk hold together to its last step?",
            metric="horizon_ratio",
            value=horizon_ratio,
            gate=f"<= {MAX_HORIZON_RATIO:g}",
            passed=horizon_ratio <= MAX_HORIZON_RATIO,
            points=_points(-horizon_ratio, -MAX_HORIZON_RATIO, -1.0),
        ),
        RungResult(
            name=RUNG_NAMES[4],
            question="As smooth as the demonstrations?",
            metric="smoothness_ratio",
            value=smoothness_ratio,
            gate=f"<= {MAX_SMOOTHNESS_RATIO:g}",
            passed=smoothness_ratio <= MAX_SMOOTHNESS_RATIO,
            points=_points(-smoothness_ratio, -MAX_SMOOTHNESS_RATIO, -1.0),
        ),
    )

    level = -1
    for i, rung in enumerate(rungs):
        if not rung.passed:
            break
        level = i
    level_name = "none — below L0" if level < 0 else RUNG_NAMES[level]

    warnings: list[str] = []
    if skill_repeat <= 0.0 < skill_zero:
        warnings.append(
            "Beats zero-delta but NOT repeat-last-action: the apparent skill is the "
            "demonstration's own inertia, not learned behaviour. Report skill_vs_repeat_pct "
            "alongside any MSE improvement or the number reads as better than it is."
        )
    dynamic_range = float(grip_max - grip_min) if grip_total else 0.0
    if dynamic_range < GRIPPER_MIN_DYNAMIC_RANGE:
        warnings.append(
            f"Gripper channel is degenerate (peak-to-peak {dynamic_range:.3f} < "
            f"{GRIPPER_MIN_DYNAMIC_RANGE}): it never opens or closes in this data, so "
            f"gripper_accuracy={grip_match / max(grip_total, 1):.3f} is thresholding noise "
            "around the binarization point and is NOT a grasp-success proxy."
        )
    if n_crit < 30:
        warnings.append(
            f"Only {n_crit} task-critical chunks — CI-MSE is noisy below ~30 samples. "
            "Widen the holdout or lower critical_quantile before trusting L2."
        )

    return BenchReport(
        run_name=run_name,
        num_predictions=n_all,
        num_episodes=len(by_episode),
        num_critical=n_crit,
        critical_quantile=critical_quantile,
        level=level,
        level_name=level_name,
        score=float(sum(r.points for r in rungs)),
        rungs=rungs,
        mse=mse,
        ci_mse=ci_mse,
        skill_vs_zero_pct=skill_zero,
        skill_vs_repeat_pct=skill_repeat,
        ci_skill_vs_zero_pct=ci_skill_zero,
        ci_skill_vs_repeat_pct=ci_skill_repeat,
        horizon_ratio=horizon_ratio,
        smoothness_ratio=smoothness_ratio,
        shift_tolerant_mse=shift_mse,
        timing_gain_pct=_skill(mse, shift_mse),
        gripper_accuracy=(grip_match / grip_total) if grip_total else 0.0,
        gripper_dynamic_range=dynamic_range,
        baselines=BaselineScores(
            zero_mse=zero_mse,
            repeat_mse=repeat_mse,
            zero_ci_mse=ci_zero,
            repeat_ci_mse=ci_repeat,
        ),
        warnings=tuple(warnings),
    )


def _points(value: float, gate: float, anchor: float) -> float:
    """Linear rung credit: 0 at the gate, ``RUNG_POINTS`` at the pre-registered anchor.

    Callers negate both value and bounds for lower-is-better metrics, so this stays monotone.
    """
    if not np.isfinite(value):
        return 0.0
    if anchor == gate:
        return RUNG_POINTS if value >= gate else 0.0
    return float(np.clip((value - gate) / (anchor - gate), 0.0, 1.0) * RUNG_POINTS)


def compare_bench(baseline: BenchReport, candidate: BenchReport) -> str:
    """Side-by-side markdown for two runs scored on the identical holdout."""
    rows = [
        ("level", baseline.level_name, candidate.level_name),
        ("score", f"{baseline.score:.1f}", f"{candidate.score:.1f}"),
        ("mse", _fmt(baseline.mse), _fmt(candidate.mse)),
        ("ci_mse", _fmt(baseline.ci_mse), _fmt(candidate.ci_mse)),
        (
            "skill_vs_repeat_pct",
            f"{baseline.skill_vs_repeat_pct:+.1f}%",
            f"{candidate.skill_vs_repeat_pct:+.1f}%",
        ),
        (
            "ci_skill_vs_repeat_pct",
            f"{baseline.ci_skill_vs_repeat_pct:+.1f}%",
            f"{candidate.ci_skill_vs_repeat_pct:+.1f}%",
        ),
        ("horizon_ratio", _fmt(baseline.horizon_ratio), _fmt(candidate.horizon_ratio)),
        ("smoothness_ratio", _fmt(baseline.smoothness_ratio), _fmt(candidate.smoothness_ratio)),
    ]
    lines = [
        "# WAM-Bench comparison",
        "",
        f"| metric | {baseline.run_name} | {candidate.run_name} |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {name} | {a} | {b} |" for name, a, b in rows]
    return "\n".join(lines) + "\n"
