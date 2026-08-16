---
id: T-046
aliases:
- T-46
- T-046
title: "Step 0 is the whole defect. Does making it homogeneous repair the labels?"
slug: step-0-is-the-whole-defect-does-homogenising-it-repair-the-labels
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
- T-045
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "PRE-REGISTERED 2026-08-16 as docs/preregistration/PR-12-step-zero-anchor-heterogeneity.md, rule T46_RULE_V1, claimed across live sessions before it was written. Registers exactly ONE prediction -- P2, that V-chain clears L1 -- because the peer session had already measured the diagnostic half and said so before this file was committed. Not yet run."
---

# Step 0 is the whole defect. Does making it homogeneous repair the labels?

## Description

**This was read off the code, not hypothesised about the data**, which is what separates it from
PR-10's re-indexing and PR-11's re-filtering. `convert_lerobot_g1.py:365` builds every one of the
target's sixteen steps as `q[s+t+1] - q[s+t]`, a homogeneous state-to-state first difference.
`eval_t39_baseline.py:254-255` builds fifteen of the prediction's steps as `q_cmd[t] - q_cmd[t-1]`,
command-to-command — and **step 0 as `q_cmd[0] - q_state[s]`, the only element in either chunk that
subtracts a quantity of one kind from a quantity of another.** That anchoring is correct under the
premise it states (`action[i] == q[i+1]`, perfect tracking) and `tests/test_t39_baseline.py` kills
three plausible alternatives to it. The premise is the exposure: a steady-state tracking offset `c`
cancels in every command-to-command difference, never enters any state-to-state difference, and
**survives at full magnitude in step 0 alone**.

**The diagnosis is measured, not predicted, and it is prior work by the peer session.**
`scripts/audit_smoothness_ratio.py` decomposes the jerk sum by within-chunk index over the PR-10
prediction files already on disk, reproducing `bench.json`'s `smoothness_ratio` to `1.8e-14`. Index
0 carries **96.7–96.8 %** of the *predicted* jerk sum and **6.6 %** (= 1/14, flat) of the *target's*
— the target's profile is flat across all fourteen indices, exactly as a homogeneous first
difference must be. `smoothness_ratio` with index 0 dropped from **both** arms is **0.2747** at
`k = −2`, against a published 7.7011. Three consequences: `benchmark.py:538` is arithmetically
right — `targets` are already first differences, so a second difference of them is a true third
derivative, and PR-11-RESULT's reasoning toward a wrong formula is corrected here; the metric is
correct and **the vector it is fed has one corrupted element**, which survives every unit test the
metric has; and the published direction is **inverted** — over steps 1–15 the command is ~3.6×
*smoother* than the executed trajectory, not 8× jerkier.

**The one at-risk prediction is P2: `targets[0] = q_cmd[0] - q_cmd[-1]` (V-chain) clears L1.**
V-chain is identical to the current anchoring under perfect tracking, so it is not a change of
premise but the same premise made robust to the premise failing; it is available at inference, since
a policy knows its own last command; and it has a stated cost — the chunk loses its only tie to the
measured state, so the label never corrects accumulated drift and trusts FR-05's re-observe loop to
do it. P2 fails if `c` is not constant, in which case it does not cancel between consecutive
commands either. A second cell, **V-mask** (score steps 1–15 only, change no label), runs as an
*instrument* to size step 0's share of the MSE sum — a different sum from the jerk sum §2 measured.

**Three traps, all registered.** A V-chain that silently reduces to the current anchoring produces a
flat grid that reads as a confident negative, so **G0.3 is two-sided**: non-zero RMS change on row 0
**and bit-identical rows 1–15**. V-mask removes the largest element of a sum and reports the sum got
smaller, which is arithmetic — so it is only ever read against baselines masked the same way. And
because the command is ~3.6× *smoother* over steps 1–15, a V-chain label space is exposed at spec
0.2.0's **two-sided** L4 band (`0.5 ≤ r ≤ 2.0`) on the *bland* side, the opposite gate from the one
this project has been worrying about — registered as expected, not explained afterwards.

**Verdicts** (`T46_RULE_V1`, read on holdout half B at `d = −2`, precedence X→C→D→I): **C** V-chain
clears L1 with a `≥ 10 pp` gain — the deficit was an anchoring heterogeneity and the repair is one
adapter line; **D** it does not — the diagnosis holds and the cheap fix does not reach it, licensing
one measurement of the offset's structure; **X** V-mask's step-0 MSE share is `< 50 %` — coherence
failure, nothing concluded whatever V-chain did, because that would mean this document misunderstood
its own instrument. **C is the expensive verdict** and carries the material margin and the held-out
reading.

Zero GPU-hours, CPU only, minutes. **Licenses nothing about GR00T or any policy** (PR-07 §6
untouched), **does not unblock training** after T-39's `VOID`, and **touches neither
`src/wam/evaluation/benchmark.py` nor `docs/benchmark.md`** — what the L4 gate should do about a
metric dominated by an anchor discontinuity is a separate, separately pre-registered decision that
is not one session's to take.

## Notes / Report
