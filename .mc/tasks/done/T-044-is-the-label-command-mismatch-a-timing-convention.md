---
id: T-044
aliases:
- T-44
- T-044
title: "Is the label/command mismatch a timing convention, or is it the space?"
slug: is-the-label-command-mismatch-a-timing-convention
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
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "RAN 2026-08-16, verdict J. Zero GPU-hours, CPU only, 40 holdout episodes. Both G0 gates passed: oracle_state +100.00 % on the trimmed set, and the d=0 bridge reproduced PR-07's -359.41 to -359.4078 (drift +0.002 pp) with smoothness_ratio 8.5175 and horizon_ratio 0.004413 against PR-07's 8.52 / 0.0044 -- the same measurement through a different driver. NO delay in the -4..+4 window clears L1 on half A, so the commanded and executed spaces are not a shifted copy of one another and the anchor is not the defect. THE INTERESTING PART IS THE SUB-FINDING: a real timing offset does exist, d* = -2 on BOTH halves independently, worth +28.81 pp (A) / +30.35 pp (B) -- material by the borrowed 10 pp floor and replicated out-of-sample -- but it closes only ~11 % of a ~254 pp gap and leaves the arm 3.2x worse than repeat-last-action. So the constant-lag reading PR-07-RESULT declined to establish is partially confirmed and decisively insufficient. smoothness_ratio moves 6.21 -> 5.67 only, i.e. re-anchoring removes ~9 % of the excess jerk; a shift re-indexes a signal and cannot smooth one, which is the substance of J. Curve is smooth, unimodal and interior (E did not apply), and both bench specs agree to nine decimals. Licenses nothing about GR00T or any policy, relabels nothing, retro-validates none of the fourteen negatives, and does not unblock training after T-39's VOID. Result: docs/preregistration/PR-10-RESULT-T-44.md, artifact runs/t44-anchoring-sweep/sweep.json."
---

# Is the label/command mismatch a timing convention, or is it the space?

## Description

**T-39 returned `VOID (labels)` and named exactly one piece of follow-up work; this is it**
(`docs/preregistration/PR-10-label-anchoring-delay-sweep.md`, pre-registered 2026-08-16 before any
sweep is run; rule `T44_RULE_V1` in git first). PR-07-RESULT measured that the corpus's own
commanded action column fails L1 by **−359.41 pp** while `oracle_state` scores a bit-exact
`mse 0.0` — so the label *space*, not our adapter, is what disagrees with the corpus. It then
measured the shape of that failure (`mse 4.20e-05` absolute, `horizon_ratio 0.0044`,
`smoothness_ratio 8.52`) and **refused to explain it**, recording the constant-lag reading as *"an
interpretation consistent with them and not established here"*. A pure one-step lag predicts
exactly that first-step concentration of error; so the reading is tempting, which is precisely why
the gate is written before the numbers exist. **The manipulation is one variable:** the commanded
chunk is built exactly as `eval_t39_baseline.commanded_to_chunk` builds it today, with the source
index shifted to `commanded[s+t+d]` and the chaining, chunk length and anchor untouched, swept over
`d ∈ {−4 … +4}` — nine values, ±133 ms at 30 fps, symmetric because a lead and a lag are equally
admissible and a one-sided window would smuggle in the conclusion. **This does not re-open a settled
test, and the distinction is the whole design:** `tests/test_t39_baseline.py` already kills
`action[t+1] - q[t]` and two other mis-anchorings as mutants, and those tests answer *what our
convention means*; this asks whether **this corpus satisfies the convention's premise**, which is
the parenthesis in the adapter's own docstring — `action[i] == q[i+1]`, perfect tracking within one
step. A controller that tracks with a lag leaves the convention correctly implemented and its
premise violated, and reading a non-zero `d*` as "the mutation tests were wrong" is a misreading
that PR-10 §2 forbids in advance. **The chunk set is the intersection over the whole sweep**, not
per-`d`: a shifted index runs off the episode at both ends, and nine scores over nine different
chunk sets are not a comparison — so the first and last chunk of every episode are dropped for
every arm and every `d`, which means no number here is directly comparable to PR-07's 1 040-chunk
table and `d = 0` is the only bridge between them. **Nine values and take-the-best is a garden of
forking paths**, so `d*` is fitted on holdout half **A** (even index in the committed
`t18_holdout_episodes.txt`) and every verdict-bearing number is read on half **B** alone at that one
`d*`, with no further search — a deterministic split of a file already in git rather than a fresh
seeded draw, so it cannot be re-rolled. **Symmetry, because here the POSITIVE is the expensive
conclusion** — this is the reverse of PR-07: a confirmed timing defect licenses re-labelling the
whole corpus and re-reading `docs/benchmark.md` end to end, so **T** carries the borrowed
`MATERIAL_FLOOR_PP = 10.0` margin *and* the held-out confirmation, while **J** ("not a delay")
changes nothing and starts no work. Verdicts: **T** `d* ≠ 0`, B clears L1 at `d*` and beats its own
`d = 0` by ≥ 10 pp → the anchor is off by `d*` steps; **J** no `d` clears L1 on A → the spaces are
not a shifted copy of one another and the 8.5× jerk is the object of study, with PR-04's collection
spec becoming the live question; **E** `d*` on a range endpoint → nothing concluded, the window
extends once to `±8` and there is no second extension; **I** anything else, including L1 on A but
not on B → nothing licensed. `d* = 0` cannot produce **T** by construction. **G0 runs first and can
stop everything:** `oracle_state` at `d = 0` must still reach 90 %, and `oracle_action` at `d = 0`
must land within ±0.5 pp of PR-07's −359.41 **on the full 1 040 chunks** before the trimmed set is
adopted. **What no outcome licenses:** any statement about GR00T or any policy (PR-07 §6 forbids it
because the policy arm never ran, and nothing here runs one — this scores two oracles against each
other), re-labelling the corpus inside this task, retro-validating any of the fourteen negatives,
or unblocking training after a `VOID` gate, which stays the project owner's call.

## Notes / Report

**Verdict `J`** — see `docs/preregistration/PR-10-RESULT-T-44.md` for the full curve, both halves, both
bench specs and the diagnostics.

The one line worth carrying forward: **the anchor is real and it is not the problem.** `d* = -2`
replicated on a held-out half and cleared the material floor, which is exactly what a genuine
timing offset looks like — and it bought 29 pp against a 254 pp deficit. What survives every
anchoring in the window is that the command carries high-frequency content the executed trajectory
does not (`smoothness_ratio` 6.21 -> 5.67 against an L4 gate of 2.0), and a shift cannot smooth a
signal. **`smoothness_ratio` is the object of study, and PR-04's collection spec — what *kind* of
data — is the live question**, not another anchoring fix and not another model.

Named as follow-up rather than answered: a per-joint or velocity-dependent lag is not ruled out.
PR-10 §9 predicted it would show as "a partial, unsatisfying improvement", and that is what
appeared — with L1 and L2 disagreeing about where the optimum sits (-2 vs -1). Designing that
experiment after seeing this curve is what the next pre-registration is for.
