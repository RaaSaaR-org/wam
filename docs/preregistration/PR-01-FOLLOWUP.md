# PR-01 follow-up — what the momentum dominance actually is, and what it is not

**Measured 2026-08-02**, CPU only, no allocation spent. Everything here started from one
observation while checking PR-01's output: the "constant-velocity" reference predictor
(`dq·dt_s`, zero fitted parameters) scored 9.137664e-06, matching the archived causal
repeat-last-action baseline 9.13766e-06 to six figures. That is not a coincidence, and chasing it
produced one confirmed fact, one refuted claim of mine, and one actionable defect.

## 1 · The identity is real — and explanatorily inert

`dq[start]·dt_s` **is** the previous chunk's last label. Not approximately; algebraically.

| | |
|---|---|
| comparisons (402 episodes, chunk level) | 151 710 |
| max relative residual | 1.150952e-07 = **0.9655 × float32 eps** |
| within 1 float32 ULP | **100.00 %** (max integer ULP distance: 1) |
| residual vs \|acceleration\| decile | **flat** at ~0.47 × eps across 5 orders of magnitude |
| residual vs joint index | **flat** across a 134× spread in joint motion |
| lag test, k ∈ [−3, +3] | only k = 0 matches, and by 7.1e6× |

The converter derives `dq[t] = (q[t] − q[t−1])/dt_s` and `targets[k] = q[start+k+1] − q[start+k]`
from the same `q`. With non-overlapping 16-step chunks the bench's causal repeat-last-action takes
`prev.targets[15] = q[start] − q[start−1]`, which is `dq[start]·dt_s` exactly. So **15 of the 32
state dims (47 %) are the previous label**, and const-velocity and repeat-last-action are one
number computed two ways, not two independent references.

> **REFUTED, same day — mine.** I wrote that this makes the task *"degenerate by construction"* and
> said so to the user before checking. It does not, and the refutation is decisive: replace the
> exact-identity velocity with a **3-step smoothed** backward velocity `(q[s] − q[s−3])/3`, which
> coincides with no stored label anywhere, and the momentum baseline gets **stronger** —
> 8.859239e-06 against the identity's 9.137663e-06, a 3.0 % improvement. I re-ran this myself:
> k=1 9.137663e-06, k=2 8.890509e-06, **k=3 8.859239e-06**, k=4 8.919538e-06, k=5 9.067497e-06,
> k=8 9.606877e-06, k=16 1.134381e-05. Destroying the construction makes the effect worse.
>
> The momentum dominance comes from **30 Hz arm trajectories being smooth** — physics and control
> rate — not from the converter's relabelling. Two further riders: `dq` is a strictly *backward*
> difference (mean |dq·dt − (q[t]−q[t−1])| = 1.11e-10 vs 6.10e-04 for the forward difference), so
> this is past data a causal policy is entitled to and **not label leakage**; and no element of the
> target is recoverable from the state — even at step 0, momentum leaves 2.30e-06 of 1.65e-05.

## 2 · The metric is a valid falsifier and an invalid certificate

The other thing I got wrong was reaching for "chunk MSE cannot rank policies". It ranks. On an
episode-level bootstrap (5 000 resamples over the 40 holdout episodes) all five predictors separate
with **non-overlapping CIs**, in order of how much of the state each exploits.

The correct statement is about *range*, not about ranking. Against a blind nonlinear ceiling
(random-Fourier ridge on the same 32 proprioceptive dims, hyperparameters selected on an inner
episode-disjoint split of the train episodes and never on the holdout):

| predictor | MSE | share of target energy explained |
|---|---:|---:|
| zero-delta | 1.632760e-05 | 0.0 % |
| **model** (82.5M trainable) | 1.112983e-05 | 31.8 % |
| const-velocity (0 fitted params) | 9.137664e-06 | 44.0 % |
| ridge, linear, 7 920 params | 6.330877e-06 | 61.2 % |
| **blind nonlinear ceiling** | **5.431371e-06** | **66.7 %** |

**66 % of the metric's achievable range is reachable with no vision at all.** So a good score cannot
certify that a policy used its eyes — but a bad score still falsifies, and the model's score is bad.
PR-01 used it only to falsify, so **VERDICT C is not undermined by this; it is strengthened.**

### The degeneracy is a short-horizon phenomenon, not a whole-task one

Fraction of the achievable range (zero-delta → blind nonlinear floor) captured by the zero-parameter
momentum rule, per chunk step:

```
step  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
     .98  .93  .92  .89  .87  .82  .75  .70  .64  .57  .52  .43  .35  .26  .19  .13
```

Steps 0–2 are genuinely near-solved by extrapolation. Steps 11–15 are not — momentum captures
13–43 % there, and **49.9 % of the target energy lives in steps 8–15**. At step 15 momentum buys
7 % over holding still while a blind regressor buys 54 %. "The task reduces to extrapolating a
smooth trajectory" is true of the first two steps of sixteen and false of the second half.

## 3 · The unused action column: real, distinct, and not the fix

`scripts/convert_lerobot_g1.py:55` records that the source's 43-dim `action` column is not used;
actions are relabelled as executed-state deltas (BC). That column is **not** a copy or shift of
`observation.state` — it is a PD setpoint the controller tracks with a large steady offset and a
~3-step (100 ms) lag; best-lag residual 1.473478e-02 rad after removing a per-episode per-joint
constant, which is 9.5× the mean per-step motion. So the relabelling is genuinely our choice.

**Using it does not fix anything.** On the only action-derived JOINT_DELTA target whose cumulative
sum reproduces the vendor command stream, the momentum degeneracy gets *worse*: repeat-last-action
lands at **1.972× the ridge** (2.076× for const-velocity) against 1.443× on the shipped target, and
74.4 % of the target energy collapses into step 0, which is 96.7 % a constant controller-droop
offset. Removing the alleged cause makes the effect stronger — the same verdict as §1, reached by a
second route.

**Recommendation: do not re-label.** It would cost a 402-episode re-conversion with video decode,
a LoRA refit and a re-eval, and it would break the `dataset_snapshot_ref` provenance of every
archived run (AC-04) to fix something it demonstrably does not fix.

## 4 · The gripper is killed by our converter — this one is real

The one channel in a pick-and-place task that is a *discrete decision* rather than smooth
extrapolation is dead in the converted dataset and **fully alive in the source**.

| | debounced transitions/episode | episodes with a grasp cycle | p2p |
|---|---:|---:|---:|
| raw `state.left_hand.max_joint[4]` | **2.015** | **98.8 %** | 1.0000 |
| raw `action.left_hand.max_joint[0]` | **2.037** | 99.8 % | 1.0000 |
| raw right hand (either) | 0.000 | 0.0 % | — |
| converted `action.gripper_target` | **0.000** | **0.0 %** | 0.136956 |

`audit_gripper.py` **PASSes** on the raw source and **FAILs four clauses** on the converted dataset.
The raw right hand is a *single constant across all 171 625 samples*, and `relabel_chunks` takes the
**mean over both hands** — averaging a live hand against a frozen one. Decomposed: the fixed synergy
mapping `clip((x+1)/2, 0, 1)` accounts for 88.8 % of the lost range and the both-hand mean for the
remaining 11.2 %.

Two corrections to how this was first stated:

- It is **not** a property of the T-16 holdout. All 402 converted episodes are dead, so **a
  different split fixes nothing**; a re-conversion does.
- The converter already has `--gripper-mapping active-hand`. A single-hand counterfactual computed
  from the raw parquet restores 2.015 transitions/episode. That counterfactual's p2p of 1.00000 is
  *forced* by a min–max affine fitted over all 402 episodes including the holdout, so quote the
  transition count, not the range.

**Open and load-bearing:** whether a restored gripper channel is a *skill* metric or just another
momentum metric. It is drawn from the same state stream and may inherit the same short-horizon
autocorrelation, and its transition base rate may be too low to carry a metric regardless. That
measurement is cheap and decides whether the re-conversion is worth doing.

## What changed in the recommendations

| | before | after |
|---|---|---|
| I-8 (~125 GPU-h) | held | **still held.** Nothing here supports it |
| velocity-head repairs D2/D3 | held | **still held** |
| re-label from the `action` column | plausible | **rejected on measurement** |
| gripper re-conversion | not considered | **the leading CPU-side candidate**, pending the base-rate check |
| blind baseline | linear ridge | ridge **plus** the blind nonlinear ceiling |
