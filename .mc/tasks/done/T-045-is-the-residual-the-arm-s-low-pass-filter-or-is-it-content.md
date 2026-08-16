---
id: T-045
aliases:
- T-45
- T-045
title: "Is the residual the arm's low-pass filter, or is it content?"
slug: is-the-residual-the-arm-s-low-pass-filter-or-is-it-content
status: done
priority: 1
owner: ''
projects: []
customers: []
tags:
- m3
- data
- eval
- prereg
sprint: ''
depends_on:
- T-39
- T-044
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "RAN 2026-08-16, verdict R. Zero GPU-hours, 29 evaluations, ~2 min. All three G0 gates passed: the four no-op cells reproduced T-44 to <=0.003 pp drift, oracle_state +100.00 %, and the filter provably reached the array (RMS change 1.55e-03 at 12 Hz rising monotonically to 2.14e-02 at 1 Hz). NO cutoff clears L1 at either anchor. The best cell is 2 Hz at d=-2, worth +4.63 pp against a 224.89 pp deficit -- about 2 % of it, less than half the borrowed 10 pp floor, and PR-10's re-anchoring was already the small answer at ~11 %. THE NUMBER THAT REFRAMES THE PROBLEM: a 1 Hz cutoff on a 30 Hz signal moves smoothness_ratio only 5.675 -> 5.381, a 5.2 % change, which should be impossible if the excess jerk lived in high frequencies; horizon_ratio meanwhile sits at 0.006-0.008 at EVERY cutoff. Reading (marked as interpretation, not established here): the deficit is a level offset between command and state at the chunk's ANCHOR, so a filter reshapes steps 1..15 while the error sits in step 0 -- and smoothness_ratio may have been reporting that same first-step discontinuity all along rather than high-frequency content, which would bear on docs/benchmark.md's L4 gate. Between PR-10 and PR-11 both obvious cheap repairs to the label space are now measured and priced. Next is NOT another repair: PR-04's collection spec. Two follow-ups named and deliberately NOT run after seeing the verdict: the per-step error profile, and whether smoothness_ratio measures what its name claims. Result: docs/preregistration/PR-11-RESULT.md, artifact runs/t45-lowpass-sweep/sweep.json."
---

# Is the residual the arm's low-pass filter, or is it content?

## Description

**Both PR-10 runs ended at the same sentence and this is the experiment it names.** They agree that
the commanded column leads the executed state by ~2 control steps, that the offset is real and
replicates out-of-sample, and that it recovers only ~9–11 % of the deficit — and that **what
survives every anchoring either run tried is the jerk**, 3–5× the executed trajectory's against an
L4 gate of 2.0. A shift re-indexes a signal; it cannot smooth one. **The hypothesis:** a
position-controlled arm *is* a low-pass filter — inertia, gearing and the servo loop mean the joint
does not follow the command's high-frequency content — so if that is the whole residual, low-pass
filtering the commanded column before it becomes a chunk should recover most of what is left,
because it reconstructs in software the filter the arm applies in hardware. If it does not, the two
streams differ in something no post-processing of these labels reaches, and **collecting different
data (PR-04) is ahead of processing this data better** — a materially different project state,
which is why this is worth running rather than assuming. **The filter is defined in the
pre-registration, not chosen during the run:** a zero-phase symmetric Hann-windowed sinc, numpy
only, applied per channel over the whole episode with edge-clamped padding. Zero phase because a
causal filter's lag would be indistinguishable from the delay PR-10 just measured — the experiment
would confound its own manipulation with the previous one. numpy because `scipy` is absent from the
WAM venv and installing it would change the dependency set every number in `docs/benchmark.md` was
produced under, the same argument `72_build_t39_env.sbatch` makes for the trainer's venv. Whole
episode because filtering inside a chunk puts a discontinuity at every boundary, exactly where
`horizon_ratio` looks. **Grid:** `fc ∈ {1,2,3,5,8,12} Hz` plus an explicit no-op control, at two
anchors — `d = −2` primary (does filtering fix what is left *after* the known fix) and `d = 0`
secondary — with `d` **taken from T-44 and held fixed**, because searching anchor and cutoff at
once is a 2-D garden of forking paths. Chunk set and the A/B holdout split are inherited unchanged
from PR-10; `fc*` is fitted on A and read on B. **THE TRAP, and it is the whole reason for the
guard: a low-pass filter shrinks magnitude, and shrinking a noisy prediction toward its mean
improves an MSE ratio whether or not the removed part was noise.** So F additionally requires
`skill_vs_zero_pct` to improve by the material floor — shrinkage toward zero cannot improve the
comparison against predicting no motion — and a cell clearing L1 without it is verdict **S**,
recorded as shrinkage rather than as a finding. **G0 can stop everything:** the no-op cell must
reproduce T-44's `−224.89`/`−379.68` (at `d=−2`) and `−253.70`/`−410.03` (at `d=0`) to ±0.5 pp;
`oracle_state` must still reach 90 %; and **the filter must provably reach the array** — every
filtered cell records a non-zero RMS change that grows monotonically as `fc` falls, because a
filter threaded through but never applied yields a flat grid and a confident verdict that the jerk
is irreducible, which is the same shape as a real **R** and indistinguishable from it in the
output. Verdicts, precedence fixed in advance as **E → R → S → F → I** (PR-10 left precedence open
and its driver had to decide it in a docstring): **F** the residual is the arm's low-pass;
**S** the gain is shrinkage; **R** no cutoff clears L1 at either anchor, so post-processing will
not reconcile the streams; **E** the lowest cutoff wins monotonically, nothing concluded, one
extension to 0.5 Hz only; **I** otherwise. **F is the expensive verdict** and carries the margin,
the held-out half and the guard. **What no outcome licenses:** any statement about GR00T or any
policy (no model is trained, loaded or consulted — this scores oracles against oracles), relabelling
the corpus, retro-validating the fourteen negatives, unblocking training after T-39's `VOID`, or
attributing the physics — a cutoff that helps is equally consistent with arm dynamics, a
controller-side filter and the recording chain.

## Notes / Report

**Verdict `R`** — `docs/preregistration/PR-11-RESULT.md` has the full grid at both anchors, both
halves, both bench specs, and the three gate readings.

The line worth carrying forward is not the verdict, it is the diagnostic underneath it: **a 1 Hz
cutoff on a 30 Hz signal moves `smoothness_ratio` by 5.2 %.** That should be impossible for a signal
whose excess jerk is high-frequency. Together with `horizon_ratio` sitting at 0.006-0.008 at every
cutoff, it points at the error being a **level offset at the chunk's anchor** rather than a spectral
property — and therefore at `smoothness_ratio` having been read wrongly in every document that has
cited it, this project's L4 *moves-like-a-demo* gate included.

Marked as interpretation, not measurement, and the experiment that would settle it (the per-step
error profile) was deliberately not run after the verdict was known.

**Both cheap repairs to the label space are now priced: anchor ~11 %, filter ~2 %.** The next thing
is not a third repair.
