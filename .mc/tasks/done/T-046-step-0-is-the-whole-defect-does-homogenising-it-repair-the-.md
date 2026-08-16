---
id: T-046
aliases:
- T-46
- T-046
title: "Step 0 is the whole defect. Does making it homogeneous repair the labels?"
slug: step-0-is-the-whole-defect-does-homogenising-it-repair-the-labels
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
- T-045
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "RAN 2026-08-16, verdict C. Zero GPU-hours, 26 evaluations, ~3 min. THE WHOLE FINDING IS ONE LINE -- per-step MSE, half A, d=-2: unmodified [4.79e-04, 3.34e-06, 3.26e-06, ...], v_chain [3.51e-06, 3.34e-06, 3.26e-06, ...]. Step 0 carried 143x the error of its neighbours; V-chain changes that one number and G0.3 measured the difference on rows 1..15 at exactly 0.000e+00. That element was 90.10 % of the summed per-step MSE (91.26 % at d=0), against the ~92 % PR-12 4 derived in advance from horizon_ratio. HELD-OUT HALF B: -379.68 -> +69.15 at d=-2 (gain 448.82 pp), -410.03 -> +69.41 at d=0 (gain 479.44 pp), against a 10 pp floor. L2 moves with it (+76.30) and skill_vs_zero is +82.91, so it is NOT shrinkage -- and the repeat/zero baselines are numerically IDENTICAL between cells because the targets never changed, so the entire movement is the model arm's step-0 error falling. All gates passed: bridge drift <=0.0026 pp, oracle_state +100.00 %, retained counts identical at 474/486 across all twelve cells, per-step profile reproduces the scorer's horizon_ratio to 0.00e+00. CORROBORATION NOBODY AIMED AT: PR-10 registered that a correct anchoring would drive horizon_ratio toward 1.0 and recorded that the delay shift FAILED it; V-chain at d=0 gives 1.0021. THIS RETIRES ONE OF OUR OWN RESULTS: under the unmodified anchoring d=-2 beat d=0, which both PR-10 runs read as a 67 ms controller lag; under V-chain the preference FLIPS and d=0 is at least as good on both halves. So PR-10's ~11 % and PR-11's ~2 % were fractions of a deficit that was ~90 % a single subtraction. Registered in advance and duly arrived (PR-12 5C): smoothness_ratio 0.276 is L4 top-rung under spec 0.1.0 but BELOW spec 0.2.0's two-sided floor of 0.5 -- the command is ~3.6x SMOOTHER than the executed trajectory over steps 1-15, reproducing the peer's independent 0.2747. TWO DRIVER DEFECTS, both found by gates firing and both recorded in the result rather than quietly fixed: run 1 returned INVALID because G0.3 compared untrimmed chunk dicts (chunks that enter no number), run 2 returned X because _verdict read step_zero_share off the V-mask cell, a quantity that cannot exist. Cell values are bit-identical across all three runs; only the plumbing changed, and both corrected checks are strictly tighter. ALSO a design defect in PR-12 found by its own tests before any cell existed: at d!=0 V-chain moves the anchor too, so the unconfounded test is the d=0 cell -- pinned by a test, recorded, NOT back-fitted into the pre-registration. Licenses a defect report against commanded_to_chunk's step-0 anchoring carrying the stated cost (the chunk loses its only tie to measured state, leaning on FR-05's re-observe loop). Licenses NOTHING about GR00T or any policy, does NOT discharge T-39's VOID, does not retro-validate the fourteen negatives, and touches neither benchmark.py nor docs/benchmark.md. Result: docs/preregistration/PR-12-RESULT.md, artifact runs/t46-step-zero/probe.json."
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

**Reported 2026-08-16 — `C`.** Full record `docs/preregistration/PR-12-RESULT.md`, artifact
`runs/t46-step-zero/probe.json`. Zero GPU-hours, nothing submitted.

### The finding, in one line

Per-step MSE, half A, primary anchor `d = −2`, 474 chunks:

```
unmodified   4.79e-04  3.34e-06  3.26e-06  3.37e-06 ... 3.50e-06  3.70e-06
v_chain      3.51e-06  3.34e-06  3.26e-06  3.37e-06 ... 3.50e-06  3.70e-06
```

Step 0 carried **143×** the error of its neighbours. V-chain changes that one number; G0.3 measured
rows 1…15 at exactly **`0.000e+00`**. That element was **90.10 %** of the summed per-step MSE.

**Held-out half B: −379.68 → +69.15** at `d = −2` (gain **448.82 pp**), **−410.03 → +69.41** at
`d = 0` (gain **479.44 pp**), against a borrowed floor of 10 pp. L2 moves with it (+76.30) and
`skill_vs_zero` is **+82.91**, so it is not shrinkage — and the repeat and zero baselines are
**numerically identical** between the cells because the targets never changed, so the entire
movement is the model arm's step-0 error falling.

### Corroboration nobody aimed at

PR-10 registered that a correct anchoring would drive `horizon_ratio` toward 1.0, and both PR-10
result documents recorded that the delay shift **failed** it — it moved the wrong way. V-chain at
`d = 0` gives **1.0021**.

### It retires one of our own results

Under the unmodified anchoring `d = −2` beat `d = 0`, which both PR-10 runs read as a real ~67 ms
controller lead. Under V-chain the preference **flips**: `d = 0` is at least as good on both halves.
The delay sweep was optimising the one contaminated element. **PR-10's ~11 % and PR-11's ~2 % were
fractions of a deficit that was ~90 % a single subtraction.**

### Registered in advance and duly arrived

PR-12 §5C predicted exposure on the *bland* side. `smoothness_ratio` **0.276** is **L4
moves-like-a-demo** under spec 0.1.0 and **below spec 0.2.0's two-sided floor of 0.5** — the command
is ~3.6× *smoother* than the executed trajectory over steps 1–15, reproducing the peer session's
independent 0.2747 from different code on a different chunk set.

### Three defects, all recorded rather than quietly fixed

1. **Run 1 → `INVALID`**: G0.3 compared *untrimmed* chunk dictionaries — chunks that enter no
   number. Corrected to the registered scored set, plus a **stricter** new gate asserting equal
   scored counts across all three cells (474 / 486 everywhere).
2. **Run 2 → `X`**: `_verdict` read `step_zero_share_pct` off the **V-mask** cell, a quantity that
   cannot exist because dropping step 0 is what V-mask is. §4 names it as a property of the
   unmodified profile.
3. **A design defect in PR-12 itself**, found by its own tests **before any cell existed**: at
   `d ≠ 0` V-chain moves the anchor as well, so the unconfounded test of P2 is the `d = 0` cell.
   Pinned by a test that bounds the confound at exactly `d` steps, recorded in the driver and the
   result, and **not** back-fitted into the pre-registration.

**Cell values are bit-identical across all three runs** — only the plumbing changed, and both
corrected gates are strictly tighter than what they replaced.

### Licences

Licenses a defect report against `commanded_to_chunk`'s step-0 anchoring, naming
`q_cmd[0] − q_cmd[−1]`, and carrying its stated cost: the repaired chunk loses its only tie to the
measured state and leans on FR-05's re-observe-and-re-plan loop to correct drift.

Licenses **nothing** about GR00T or any policy (PR-07 §6 untouched — this scored oracles against
oracles), does **not** discharge T-39's `VOID`, does **not** retro-validate the fourteen negatives,
and touches neither `src/wam/evaluation/benchmark.py` nor `docs/benchmark.md`.

**Next**, and it is a code question rather than another label experiment: `commanded_to_chunk` is
the *evaluation* adapter. The same step-0 question has to be asked of the **training** path and the
**runtime executor** before "one line of the adapter" is true of anything that ships.
