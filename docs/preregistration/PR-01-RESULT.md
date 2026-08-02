# PR-01 — result

**Measured 2026-08-02.** The pre-registration is `PR-01-incremental-value.md`, committed at
`74aa388` before `scripts/bench_incremental_value.py` existed. Per repo convention that document is
**annotated, never edited**, so the result lives here instead of inside it. Reproduce with:

```bash
.venv/bin/python scripts/bench_incremental_value.py \
    --dataset datasets/gr00t-apple-full \
    --holdout runs/t16-lora-seed0/eval-t30-regression/predictions.jsonl
```

## VERDICT C — mixed, and the split *is* the result

All three archived controls reproduced to the digit before any new number was computed, so the run
is interpretable rather than void.

| predictor | fitted parameters | holdout MSE | vs model |
|---|---:|---:|---:|
| zero-delta (hold still) | 0 | 1.632760e-05 | 0.68× |
| **model** (Wan2.2-5B + LoRA) | 82 519 450 | **1.112983e-05** | — |
| **const-velocity** `dq·dt_s` | **0** | **9.137664e-06** | **1.22× better** |
| ridge (dq only) | 3 840 | 6.869235e-06 | 1.62× better |
| ridge (all state) | 7 920 | 6.330899e-06 | 1.76× better |
| T1 cross-fitted stack | 7 920 + 2 | 6.190756e-06 | 1.80× better |

### Clause by clause

**VERDICT A** (the model carries no incremental information) requires all four; two failed:

| clause | value | holds? |
|---|---|---|
| 1 · `MSE_stack ≥ 0.95·MSE_ridge` | 6.190756e-06 ≥ 6.014354e-06 — stacking the model in buys **2.21 %** against a 5 % bar | ✅ holds |
| 2 · `β̂` not consistently positive | **5 of 5** folds have β̂ > 0.05 (0.1855, 0.1757, 0.1838, 0.1649, 0.1568) | ❌ fails |
| 3 · model does not beat the ridge on T3 | model 2.221910e-05 vs ridge 1.452714e-05 — **1.53× worse** | ✅ holds |
| 4 · model does not beat the ridge on T4 gripper channel | model 3.127760e-04 vs ridge 3.468406e-04 — 1.11× better | ❌ fails |

**VERDICT B** (the metric was the problem) requires the model to beat the ridge on T3. It loses
there, so B falls at its first condition regardless of T1.

Hence C, which the pre-registration anticipated and named.

### What the split means

The model **does** carry a little information the ridge lacks. β̂ ≈ 0.17 in all five
episode-disjoint folds, tight and one-signed — that is not a noise artefact, and it is the one
genuinely positive finding about this checkpoint in the whole investigation.

It is worth **2.2 % of MSE**, less than half the pre-registered bar, and it does not survive being
asked for where it matters. On the worst quartile by constant-velocity error — 260 chunks selected
by a rule that is neither of the two predictors being compared, and which removes exactly the chunks
the linear map is best at, so the thumb is on the *model's* side of the scale — the model is
**1.53× worse than the ridge** and only **1.06× better than a rule with zero fitted parameters**.
PR-01 named a model loss there as the decisive direction, and it lost.

### T2, per-horizon: the shapes differ in kind

The model is beaten by the ridge at every one of the 16 steps, and by const-velocity at 10 of 16.
Const-velocity starts near-exact (2.30e-06 at step 0) and degrades ~6.6× across the chunk; the
model's error is nearly flat (9.77e-06 → 1.29e-05, ~1.3×) and never dips below 9.77e-06. It is not
a short-horizon predictor that decays — it is a roughly constant-error predictor that momentum
eventually catches up to. Momentum wins early; nobody wins late.

### T4: the finding underneath the finding

The clause was scored as written, but the number is not usable and saying so is the point. The
holdout gripper channel **fails its own pre-registered admissibility gate**: peak-to-peak 0.1196
against `GRIPPER_MIN_DYNAMIC_RANGE` = 0.25, and **0 debounced transitions in 1040 chunks**. The 17
raw threshold crossings are a near-constant channel dithering across 0.5, not opens and closes. So
the model's 1.11× "win" is two near-constants disagreeing slightly.

The one channel in a pick-and-place task that is a *discrete decision* rather than smooth trajectory
extrapolation has no event in it to score. PR-01 predicted this shape would point at the gripper
head; it points at the data instead.

## The number that undercuts the framing

**A fixed analytic rule with zero fitted parameters beats the fine-tune by 1.22×** and comes within
1.44× of a 7 920-parameter solve. By PR-01's own stated litmus — *"if it lands anywhere near the
ridge, the metric is a momentum metric"* — that is direct support for Reading B.

But Reading B in its pre-registered form said the model wins once momentum is stripped out, and that
is falsified: on the momentum-free quartile the model is still 1.53× worse than the linear map. So
both halves are true at once — **the metric is substantially a momentum metric, AND the model does
not beat a linear map once momentum is removed.** C is not a hedge; it is the only letter the
clauses allow.

## How much this was attacked

Four independent adversarial lenses (leakage, correctness, test-blindness, independent
reimplementation) audited the first build. **All 8 blocking/major findings reproduced — 11 of 11
targeted mutations survived the original 35-test suite**, including:

- ranking T3 on the *model's own error* instead of const-velocity, which voids T3's entire
  one-directional-admissibility argument, passed everything;
- splitting the stacking folds **row-wise** instead of by episode — the exact leak PR-01 names
  first — passed everything;
- feeding VERDICT A clause 1 the **leaked in-sample** stack instead of the cross-fitted one passed
  everything;
- the whole T4 gripper path had **zero assertions**.

All eight fixed at the root, each with a regression test. 14 of 14 mutations now killed; 50 tests.
The headline was then reproduced a third time, by hand, from `EpisodeReader` upward with nothing
imported from either bench script: zero-delta, model, ridge, const-velocity, the stack and all five
β̂ agree to seven figures, and the β̂ are bit-stable across five repeats (an auditor had seen
β̂ = 0 once and suspected a LAPACK RHS write; not reproducible).

## Consequences, as committed in advance

VERDICT C's pre-registered consequence is that no global claim is made and the next step comes from
the *pattern*. The pattern is: **T1-magnitude and T3 both point at Reading A; T1-β and T4 point away
from it, and T4's pointer aims at a dead channel.**

- **I-8 (~125 GPU-h) stays unsubmitted.** Nothing here supports spending it. A scaling curve
  denominated in a metric that a zero-parameter rule beats measures the wrong axis, and the 2.2 %
  of real signal is far too small a lever to be worth 125 GPU-hours of resolution.
- **The velocity-head repairs (D2 step index, D3 `t` embedding) stay unimplemented.** They were
  already contingent on the flow branch having something to transport; PR-01 does not supply it.
- **Offline chunk MSE is retired as a *ranking* metric** and demoted to what it always was — a
  diagnostic (PRD 10.4). `scripts/bench_incremental_value.py` ships as the permanent check that
  says so, next to `scripts/bench_ridge_baseline.py`.
- **The gripper channel is a data problem before it is a model problem.** No re-score of the
  gripper head means anything on a holdout with 0 debounced transitions.

## What this still cannot decide

Whether the model would work on a real robot, or generalize. Every test here is offline, on
demonstrations of one task with the apple in the same place. VERDICT C is a statement about the
*evidence*, not about the policy — which is exactly why the follow-on is a closed-loop task-success
measurement rather than an architecture rewrite.

> **CORRECTED 2026-08-02 — mine, and left in place rather than edited out.** *"the apple in the
> same place"* was an assumption I never measured, and it is false. The reach target varies by
> **0.6623 rad** between episodes, about 0.35 of the within-episode motion scale, with a grasp
> detected in 402 of 402 episodes. What *is* true is the weaker claim I should have made:
> **61 % of that variation is already predictable from the robot's starting pose** (cross-validated
> R² 0.6136), leaving 0.41 rad out-of-fold that proprioception cannot supply. Grasp *timing*, by contrast, is
> essentially absent from the state (R² 0.0771) — an independent route to `PR-01-GRIPPER.md`'s
> finding, reached without touching the gripper channel's values. Nothing else in this document
> depends on the retracted sentence; the consequences that do are restated in
> `PR-01-TASK-VARIATION.md`.
