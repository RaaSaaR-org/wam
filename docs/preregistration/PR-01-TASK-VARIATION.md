# How much does our task actually vary — and is the variation visual?

**Measured 2026-08-02**, CPU only, raw parquet only, no video decoded, no allocation spent.
Scripts: `scripts/measure_task_variation.py` (both measurements, 402 episodes of
`data/raw/gr00t_apple`).

This exists to correct a claim I made without measuring it, and because the corrected version
changes what we should do next.

## The correction

`PR-01-RESULT.md` closes with *"demonstrations of one task with the apple in the same place"*, and
I repeated that to justify the recommendation that no more data of this kind would help. **It was
an assumption, never a measurement, and it is wrong.**

The apple is not in the same place. Reading the reach target off proprioception — the left-arm pose
at the first debounced grasp, detected in the *source* hand channel because our converted one is
dead (T-31) — a grasp is found in **402 of 402 episodes**, and the pose at that instant varies:

| | |
|---|---:|
| between-episode spread of the grasp pose (L2, 7 live joints) | **0.6623 rad** |
| typical within-episode motion (L2, same joints) | 1.9132 rad |
| ratio | **0.3462** |

So the reach target moves by roughly a third of the motion scale. This is not a stereotyped reach
to a fixed point, and "the apple is always in the same place" should not have been said.

Two riders, because the raw numbers are easy to over-read:

- **The right arm's spread is a nuisance, not task variation.** Joints 7–13 show between-episode
  spreads of 0.19–0.61 rad against within-episode ranges of ~0.01 rad (ratios 18–54). That arm does
  not move *during* an episode; it is parked at a different angle in each one. The two headline
  numbers above are therefore over **live** joints only — those with more than 0.05 rad of
  within-episode range, which on this corpus is exactly the 7 left-arm joints. The script derives
  that set mechanically and prints LIVE/PARKED per joint rather than hard-wiring the slice.
- **Grasp timing is tight in relative terms**: mean 0.545 of the episode, std 0.064.

> **Two defects in the first version of this table, found by rebuilding it as a tested script and
> left visible rather than edited out.** (1) It carried a row *"variance explained by a single
> constant pose — 0.7241"* under a rider claiming every number was left-arm only. It was not: 0.7241
> is over all 14 arm joints, parked ones included, and the live-joint value is **0.7925**. Worse,
> the quantity is misnamed — its denominator uses the grand *scalar* mean, so it largely measures
> "which joint am I looking at" rather than "is the reach stereotyped". It is not a meaningful
> answer to the question this document asks, so it has been dropped from the table; the script still
> emits both scopes, labelled, and gates on the archived 0.7241. (2) The residual spread below was
> quoted in-sample beside cross-validated numbers without saying so — see the correction there.

## But most of that variation is already in the state

Whether the variation is *visual* is a different question from whether it exists, and it is the one
that decides whether a video model can help. Cross-validated (5 folds over episodes) R² of the
grasp pose:

| predictor | R² |
|---|---:|
| the t=0 state, all 43 dims | **+0.6136** |
| the t=0 left arm only, 7 dims | +0.4178 |
| episode length alone, 1 dim | +0.0539 |
| a constant (the mean pose) | 0.0000 |

**61 % of the reach-target variation is predictable from where the robot starts.** Residual spread
after the state is used: **0.4117 rad out-of-fold**, against a raw 0.6623 — so roughly 0.19 of the
motion scale is left that proprioception cannot supply.

> **Corrected.** The first version of this line quoted **0.3594 rad**, which is the *in-sample*
> residual, sitting unlabelled next to cross-validated R² values. The honest number is the
> out-of-fold one, 0.4117, and it is the one every conclusion below now uses. The script prints
> both, marks which is which, and gates on 0.3594 so the archived figure stays reproducible.

That is the honest shape of it. Vision has real work to do here — 0.41 rad of reach target that
proprioception cannot supply — but it is a minority of the variation, and our headline metric
averages it across 16 steps and 15 joints together with the momentum that dominates them (PR-01).

## The exception, and it is the same exception as before

**When** the grasp happens is essentially not in the state at all:

| target | R² from the t=0 state |
|---|---:|
| grasp *pose* | +0.6136 |
| grasp *instant* | **+0.0771** |

This is an independent route to `PR-01-GRIPPER.md`'s finding. That document measured the flip
instant on the restored gripper channel and found every blind predictor at a coin toss (52.98 % at
k…k+3) with a blind ceiling of only 70.91 %. This measurement never touches the gripper channel's
values — only the timestamp of the transition — and lands in the same place: the *timing* of the
grasp is the visual decision in this task, and the trajectory to it largely is not.

## What changes

- The recommendation that **more episodes of this task buy little** survives, but its reason
  changes. Not "the apple never moves" — it does — but "61 % of where it moves is already in the
  proprioception, and our metric buries the remaining 39 % under momentum."
- **Randomized object placement is the cheap lever** for D1/D2 recording. It is free at recording
  time and it is the difference between a corpus where vision must matter and one where it need
  not. This is now a measured recommendation rather than an intuition.
- The screening criterion for any candidate dataset is the one PR-02 formalises: **does a blind
  baseline fail on it?** Not "does it match our hands and cameras".

## What this does not establish

That a video model *would* recover the residual 0.36 rad, or the grasp timing. Both measurements
are ceilings on what proprioception can do, not evidence about what pixels can do — a linear/ridge
readout of the starting state is a weak model, and a stronger blind model would close some of the
gap (`scripts/bench_ridge_baseline.py`'s nonlinear ceiling exists for exactly that reason and is
not applied here).

Both numbers are also computed at a single instant per episode — the grasp — so they say nothing
about the other ~99 % of the frames, which is where the metric spends its weight.
