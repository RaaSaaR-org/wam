---
id: T-045
aliases:
- T-45
- T-045
title: "Is the residual the arm's low-pass filter, or is it content?"
slug: is-the-residual-the-arm-s-low-pass-filter-or-is-it-content
status: todo
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
status_note: "Pre-registered 2026-08-16 as PR-11, rule T45_RULE_V1, BEFORE anything is filtered. Zero GPU-hours, CPU only, artifacts already on disk. The number PR-11 was CLAIMED ACROSS SESSIONS before the file was written and confirmed unclaimed by the author of PR-10-anchor-delay-sweep.md — that is the fix for the PR-10 duplication, which is still unresolved and is the user's to settle. Not yet implemented: PR-11 §8 lists four things that must exist first, three of them tests, including one that FAILS when the filter is removed."
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

_Empty until the task runs._
