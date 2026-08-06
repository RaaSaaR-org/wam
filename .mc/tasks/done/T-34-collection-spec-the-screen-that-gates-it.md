---
id: T-34
aliases:
- T-34
title: "Collection spec + the screen that gates it"
slug: collection-spec-the-screen-that-gates-it
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- interfaces
- data
- hardware
- prereg
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-02
updated: 2026-08-02
---

# Collection spec + the screen that gates it

## Description

Collection spec + the screen that gates it (`docs/preregistration/PR-04-collection-spec.md`,
`scripts/screen_corpus.py`, 10 tests) — *PR-03's STOP routed to "collect demonstrations", which is
the most expensive conclusion in the project (a G1 EDU4, a teleop rig, months of recording). This is
that conclusion turned into measured requirements plus a gate that fires after **30 episodes instead
of 400**. **Eight requirements, each a measurement rather than an impression:** M1 0.660 / M2 0.333
(a blind predictor gets two thirds of the metric); R²(grasp pose | t=0 state) **+0.6136**, so 61 %
of where the apple moves is already in proprioception — randomized placement is the cheap lever and
it must be randomized *after* the arm is home; grasp timing **0.545 ± 0.064** of the episode, which
is why a clock-only model scores 56.04 % post-flip; the right hand frozen at 0.0007 rad across all
402 episodes and 171 625 samples; **0 of 402 failure episodes**, so optimism bias is not computable
rather than unimplemented; lag-1 autocorrelation 0.927 with 80.2 % of chunk energy in the chunk mean
and only 8.2 % above 2.8 Hz; one camera at 160×120 from a 480×640 source; no IMU. **Volume is
derived from PR-03's own power ladder, not guessed:** 150 episodes gave 2 682 post-flip steps at
2.01 grasps/episode = **17.88 steps/episode**, and `h ∝ 1/√n`, so h ≤ 3.5 needs 3 680 steps — **207
episodes at 2 grasps each, 103 at 4, 52 at 8**. The 207 reproduces PR-03's independently-derived
"roughly 205", which is the arithmetic checking itself. Plan: **D1-pilot 30 / D1 120 / D2 300**, all
at ≥ 4 grasps per episode, ~5 h of robot time for D2 against the whole GR00T corpus's 95.1 min.
**The caveat is stated rather than buried:** the bootstrap resamples *episodes*, so extra grasps
inside one episode reduce that episode's noise without adding an independent unit — the
grasps/episode columns are optimistic by an unknown factor, `h` is floored by episode count, and the
pilot measures the real exchange rate rather than the table defending itself. **What shipped is the
gate, not just the document.** `scripts/screen_corpus.py` is PR-02's M1/M2/M3 screen as committed
code, and it **reproduces the archived GR00T values**: zero-delta **1.632760e-05** and
const-velocity **9.137664e-06** to every digit, M1 0.6557 vs 0.660, M2 0.3284 vs 0.333, M3 2.0149 vs
2.01. The ~0.005 gap is a slightly stronger ceiling (5.361517e-06 vs 5.431371e-06) and moves M2 in
the **conservative** direction. That is the like-for-like check PR-03's gate 1 could not perform,
because the code it needed had never been committed — though it also means the archived numbers were
themselves scratch output, so this is two implementations agreeing, not one being canonical. Four
gates (M1 ≤ 0.45, M2 ≥ 0.45, M3 ≥ 2.0, `ceiling_dominates`), verdicts A–D with ties to the cheap
branch (revise protocol, re-pilot 30 episodes, ~30 min), and **verdict D is the interesting one**:
three failed protocol revisions makes the finding about the task family rather than the dataset, and
redirects to task selection instead of more episodes. M3's bar is a floor that catches pipeline bugs
(T-31's converter scored 0.00 on data physically containing 2.015/episode), not the ≥ 4 the plan
targets. **Adversarial review 2026-08-02 found four defects, all fixed:** (a) **`ACTIVE_HAND` was
hardcoded to 0**, so a rig whose live hand is channel 1 — a perfectly healthy rig — reported M3
**0.000** and routed to verdict C, "the recording or conversion killed the channel": the exact
inversion of the failure G3 exists to catch. Demonstrated on a transposed copy of our own corpus.
The live channel is now **detected** (`m3_active_hand`, `m3_transitions_by_hand`), and on
`gr00t-apple-grip` it measures `[810, 0]` — T-31's finding reproduced rather than assumed. (b)
**`load_episode` was executed by no test at all**, and three mutants inside it each flipped a gate
verdict on real data — most seriously a **future-reading `_lagged`**, the precise leak this screen
exists to detect, which took M1 from 0.5989 to 0.3845 and turned an M1 **FAIL into a PASS**. Now
covered by tests over a synthetic on-disk corpus. (c) `--out` wrote a bare `NaN` token, which Python
and `jq` accept but standards-conformant readers reject — and a collapsed ceiling is exactly when M1
goes NaN, so the artifact most needing inspection was the one that could not be parsed. (d)
**verdict C misrouted a G4 failure** to the gripper channel, leaving a collapsed ceiling with no
branch at all; **verdict E** added. The review also refuted a claim I had made in the document: "a
stronger ceiling is the conservative direction" is true for M2 and **false for M1** — a stronger
ceiling grows M1's denominator and makes G1 *easier*, and a hypothetical perfect blind ceiling
scores M1 **0.4404**, which **passes** G1 on the very corpus PR-04 cites as its canonical failure
(while failing G2 at M2 = 0.000). The document now says plainly that the two gates move in opposite
directions under ceiling strength, which is what makes requiring **both** ungameable*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
