# `wam.evaluation` — E0 → E4 evaluation ladder

**TL;DR** — Everything that decides whether a model is good enough to move on. The ladder is
E0 unit → **E1 offline replay** → **E2 sim** → E3 real robot → E4 generalization, topped by the
AC-01…AC-07 acceptance harness. Torch-free: reports in, verdicts out.

## Files

| File | Level | Contains |
|------|-------|----------|
| `offline.py` | E1 | `e1_metrics`, `evaluate_policy`, `holdout_split`, JSONL prediction I/O |
| `e2_checks.py` | E2 | `e2_static_checks`, `e2_sim_rollout_checks` (T-22) |
| `ablation.py` | AC-07 | `compare_runs` — world-action vs action-only |
| `acceptance.py` | AC-01…07 | `evaluate_acceptance` — the MVP dashboard (T-23) |

## E1 — offline replay (`offline.py`)

Works on **predictions** (`ChunkPrediction` = predicted + demonstrated chunk), fully decoupled
from model internals; any `Policy` can be run through `evaluate_policy`. Metric math is float64.

`E1Report` covers overall MSE/MAE, **per joint** (canonical joint names when a spec is given and
the mode is `JOINT_DELTA`, else `dim_<i>`), **per horizon step** — which shows error growth
along the chunk — gripper accuracy (binarized at 0.5), and smoothness of predicted vs target
(is the policy jerkier than the demos?). Plus per-episode breakdowns and `render_markdown()`.

**These are diagnostic metrics.** They gate offline iteration in M2/M3. The real acceptance
metrics are closed-loop success and safety (E3/E4).

`holdout_split(ids, ratio, seed)` is deterministic: ids are deduplicated and sorted *before*
shuffling, so the split depends only on the id **set** and the seed — never on input order.

## E2 — the deterministic gate before touching a real robot (`e2_checks.py`)

Two entry points, both reporting `GateResult`s (same type as the data-validation gates).

**`e2_static_checks`** probes a live policy against synthetic observations derived from a robot
adapter. **Nothing is executed** — the safety filter runs but its output is discarded; this is
read-only. Observations reuse the current state with seeded joint perturbations clipped to the
adapter's limits, so the whole check is reproducible.

| Gate | Fails when |
|---|---|
| `chunk_valid` | a predicted chunk violates the schema |
| `targets_finite` | NaN/Inf in targets, gripper or `dt_s` |
| `chunk_duration_band` | **never** — warn-only |
| `safety_intervention_rate` | more than `max_intervention_rate` of probes trigger the filter |
| `determinism` | the same observation yields different chunks (bitwise comparison) |
| `policy_latency` | mean latency exceeds the budget implied by ≥ 2 Hz |

The duration gate is warn-only on purpose: the PRD band (0.5–2.0 s) is a **design target, not a
safety limit**, so an out-of-band chunk is surfaced as a warning rather than a failure.

**`e2_sim_rollout_checks`** aggregates rollout summaries: rollouts present, zero e-stops, zero
watchdog timeouts, every rollout's policy rate ≥ 2 Hz (FR-05), and bounded interventions per
control cycle. Missing keys are treated as 0/False.

## AC-07 — the ablation (`ablation.py`)

Pure report comparison, no model access. Feed it two `E1Report`s from runs that differ **only**
in the ablated component (same holdout split, same spec) or the deltas mean nothing.

The verdict is decided on the **relative MSE improvement** of the candidate (world-action) over
the baseline (action-only):

```
improvement_pct >= +threshold  ->  "video branch helps"
improvement_pct <= -threshold  ->  "hurts"
otherwise                      ->  "no significant difference"
```

Default threshold is **5 %**, because offline action-MSE on small holdout sets fluctuates a few
percent between reruns and seeds — smaller gaps are noise, and AC-07 asks for a *measurable*
advantage. `improvement_pct` is signed so positive always means "candidate is better",
regardless of whether the metric is higher- or lower-is-better.

## Acceptance — AC-01…AC-07 (`acceptance.py`)

Consumes rollout logs written per the shared rollout-log contract (`rollout_summary` records,
provenance from the `run_metadata` line) and produces a PASS/FAIL/PENDING dashboard.

| ID | Criterion | Requirement |
|----|-----------|-------------|
| AC-01 | success on known task | ≥ 80 % over ≥ 50 real rollouts |
| AC-02 | generalization | ≥ 50 % over ≥ 30 real rollouts |
| AC-03 | zero safety violations | ≥ 100 clean rollouts |
| AC-04 | reproducibility | run_metadata carries config hash + checkpoint + dataset refs |
| AC-05 | backbone swap | ≥ 2 registry backbones conform to `BackboneAdapter` |
| AC-06 | safe stop on induced failures | every fault-injection rollout handled |
| AC-07 | world-action ablation | an `AblationReport` exists |

Four statuses, not two: `pass`, `fail`, `pending_data` (not enough rollouts yet) and
`pending_hardware` (only sim rollouts exist for a hardware-dependent criterion). Sim rollouts
carry a `sim:` task prefix and their stats are still reported.

**What counts as a safety violation (AC-03).** `estopped == True`, or any `nan_reject` /
`schema_reject` / `state_reject` / `limit_breach` intervention, or a watchdog escalation to
`stop`. `limit_breach` is *external* evidence that a hard limit was actually exceeded — the
deterministic chain never emits it — so it is always a violation. Clamping kinds
(`joint_limit`, `velocity_limit`, `accel_limit`, workspace and gripper clamps) are the safety
layer **doing its job** and are explicitly **not** violations. `fault_injection` rollouts are
excluded here and judged by AC-06 instead.

**Sim counts for AC-03.** The deterministic safety chain (safety layer, watchdog, executor
gates) runs identically in sim and on hardware, so a violation in sim fails AC-03 immediately
and ≥ 100 clean rollouts pass it regardless of platform. Real rollouts take precedence as the
sufficiency basis when both exist — but sim violations still fail the criterion, they are never
discarded.

**AC-06** requires an induced failure to be *handled*, not crashed through: the chain must have
visibly reacted (interventions, a watchdog timeout, or a clean e-stop) with no limit breach.
A hold is the expected outcome; a clean e-stop is also acceptable.

**AC-05** builds every registered backbone through the same `get_backbone` path and checks
protocol conformance plus a usable `feature_dim`. A broken factory is recorded as a finding, not
a crash.
