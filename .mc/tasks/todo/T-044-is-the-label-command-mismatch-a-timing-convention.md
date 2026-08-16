---
id: T-044
aliases:
- T-44
- T-044
title: "Is the label/command mismatch a timing convention, or is it the space?"
slug: is-the-label-command-mismatch-a-timing-convention
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
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "Pre-registered 2026-08-16 as PR-10, rule T44_RULE_V1, BEFORE any sweep is run. Zero GPU-hours: an offline re-score of artifacts already on disk, no cluster, no allocation, no download. Named by PR-07-RESULT as the follow-up that distinguishes its one un-established reading. Not yet implemented: PR-10 §8 lists the three things that must exist first — the driver importing commanded_to_chunk rather than re-implementing it, the two mutants, and the d=0 bridge against PR-07's -359.41."
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

_Empty until the task runs._
