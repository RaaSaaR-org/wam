# PR-01 — Does the WAM model know anything a linear map on proprioception does not?

**Written 2026-08-02, BEFORE the measurement exists.** Committed on its own so that `git log`
proves the rule predates the number. Per repo convention this file is **annotated, never edited**:
a threshold that must change is versioned as PR-01b, and the original stays readable.

## Why this exists

`scripts/bench_ridge_baseline.py` established that the deployed Wan2.2-5B + LoRA checkpoint loses
**1.76×** on its own holdout to a 7 920-parameter linear map on the 32-dim robot state, and
**1.62×** to a linear map on `dq` alone. Two independent implementations, seven-figure agreement,
archived controls reproduced to the digit.

That result has exactly two readings and nothing measured so far separates them:

- **Reading A — the model is empty.** The visual path contributes no predictive information; the
  fine-tune is a worse proprioceptive regressor than a matrix solve, and its parameters are not
  earning anything.
- **Reading B — the metric is empty.** Chunk MSE on this dataset is dominated by *momentum*. The
  arm is already moving; `targets[k]` are per-step joint deltas integrated onto the current `q`
  (`G1Adapter.execute`), so a constant-velocity rule is nearly correct nearly everywhere. A metric
  a no-parameter extrapolator almost solves cannot rank policies, and was never measuring skill.

Everything downstream is contingent on which it is. Under A, scaling curves (I-8, ~125 GPU-h) are
measuring an axis with no signal on it. Under B, they are measuring the wrong y-axis. Either way
the honest move is to settle this before spending the allocation, and it is settleable on CPU.

## What is measured

Four tests. Two are **leak-free** (no subset chosen using the target); two select on the target and
are therefore admissible in **one direction only**, which is stated per test rather than discovered
afterwards.

All four run on the SAME split the model was scored on — the 40 holdout episodes recovered from
`runs/t16-lora-seed0/eval-t30-regression/predictions.jsonl` via `load_episode_ids` — and against
the SAME scored quantity, mean squared error over the flattened `[16, 15]` target chunk. The ridge
is fitted on train episodes only. Every run must reproduce the archived controls (zero-delta
1.632760e-05, model 1.112983e-05) before any new number is reported; a run that does not reproduce
them is void, not interpreted.

A fifth predictor joins the table as a no-parameter reference: **constant-velocity extrapolation**,
`pred[k, j] = dq[j] * dt_s` for all 16 steps, fitted on nothing at all. It exists to make Reading B
visible directly — if it lands anywhere near the ridge, the metric is a momentum metric.

### T1 — Cross-fitted stacking (leak-free, primary)

Fit two scalars `(α, β)` in `y ≈ α·ŷ_ridge + β·ŷ_model`, by 5-fold cross-fitting **over episodes**
inside the holdout (weights fitted on 4 folds, scored on the 5th, never on their own fold). The
ridge itself never sees a holdout episode in either role.

`β` is the model's incremental weight given the ridge is already there. This is the whole question
reduced to one number, and it cannot be gamed by subset choice because there is no subset.

### T2 — Per-horizon breakdown (leak-free)

MSE at each of the 16 chunk steps separately, for every predictor. Momentum is exactly correct at
short horizon and decays; a model that has learned the *task* rather than the *velocity* should
close ground as the horizon grows. No selection of any kind — every chunk contributes to every
step.

### T3 — Branch-point subset (target-selected; a model LOSS is decisive, a model WIN is not)

Rank holdout chunks by the error of **constant-velocity extrapolation** — a fixed analytic rule
with no fitted parameters, deliberately not either of the two predictors being compared — and keep
the worst quartile. These are the moments the arm is not simply continuing.

This selection removes exactly the chunks the linear map is best at, so it is **biased in the
model's favour**. Therefore: if the model still loses here, that is strong evidence for Reading A,
because the thumb was on its side of the scale. If it wins here, that is weak evidence and must not
be reported as a win without T1 agreeing.

### T4 — Gripper-transition subset (target-selected; same asymmetry as T3)

Keep only chunks whose `gripper_target` crosses the 0.5 binarization threshold used by
`wam.evaluation.gripper`. Opening and closing the hand are the discrete decisions of a pick-and-
place task and are exactly what momentum cannot supply. Same one-directional admissibility as T3,
and additionally reported as gripper-channel MSE alone, since the joint channels dominate the
flattened MSE and would drown the event being asked about.

## The decision rule, fixed in advance

Let `MSE_ridge` and `MSE_model` be the all-state ridge and the model on the full holdout, `β̂_k` the
stacking weight in fold `k`, and `MSE_stack` the cross-fitted stack.

**VERDICT A — the model carries no incremental information.** Declared when ALL of:

- `MSE_stack ≥ 0.95 · MSE_ridge` (stacking the model in buys less than 5 %), AND
- `β̂_k` is not consistently positive — i.e. fewer than 4 of the 5 folds have `β̂_k > 0.05`, AND
- the model does not beat the ridge on T3, AND
- the model does not beat the ridge on T4's gripper-channel MSE.

Consequence, committed to now: **I-8 (~125 GPU-h) is not submitted**, and no further LoRA scaling
work is proposed on this architecture. Offline chunk MSE is retired as a ranking metric and the
next question becomes a task-success question, which this dataset cannot answer and a closed-loop
run can. The velocity-head repairs (D2 step index, D3 `t` embedding) stay unimplemented, because
the branch they repair has been shown to have nothing to transport.

**VERDICT B — the metric was the problem.** Declared when the model beats the ridge on T3 **and**
T1 agrees (`MSE_stack < 0.95 · MSE_ridge` with `β̂_k > 0.05` in ≥ 4 folds).

Consequence: the branch-point subset becomes the headline metric and is added to WAM-Bench as a
gated rung; the full-holdout MSE is demoted to a diagnostic-on-a-diagnostic. Prior ladder results
are re-scored under it before any of them are quoted again. I-8 is reconsidered — on the new axis,
not the old one.

**VERDICT C — mixed.** Anything else. No global claim is made. The exact pattern of which tests
split is reported, and the next step is chosen from that pattern rather than from a headline. In
particular, "T1 says nothing but T4 says the gripper channel carries signal" is a real possible
outcome and would point at the gripper head, not at the video branch.

## What none of this can decide

Whether the model would work on a real robot. Every test here is offline, on demonstrations of one
task, with the apple in the same place. E1 action-MSE is a diagnostic metric (PRD 10.4) and this is
a diagnostic on it. VERDICT A is a statement that *the offline evidence is worthless*, which is
narrower and more actionable than "the model is worthless" — and is precisely why the follow-on
under A is a closed-loop task-success measurement rather than an architecture rewrite.

Nor can it decide anything about generalization. The ridge has no notion of where the apple is and
would not survive it moving; that is the generalization the video branch exists to buy, and no
fixed-scene holdout tests it.
