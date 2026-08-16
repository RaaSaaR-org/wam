---
id: T-047
aliases:
- T-47
- T-047
title: "Does T-39's VOID survive its instrument being repaired?"
slug: does-t-39-s-void-survive-its-instrument-being-repaired
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
- T-046
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "RAN 2026-08-16, verdict W, all four gates passed on the FIRST run. Zero GPU-hours, 5 cells, ~2 min. THE BRIDGE REPRODUCED PR-07-RESULT's -359.41 TO A DRIFT OF +0.002 pp on the full 1040 chunks, so one number in the artifact is the archive's number produced by this driver and the other two are comparable to it. On the anchorable set (1000 = 1040 - 40 episodes, both compared cells identical): unmodified -344.54, REPAIRED +68.10 L1 / +75.40 L2 / +82.04 vs-zero, horizon_ratio 0.9717, smoothness_ratio 0.280, step-0 share 6.03 %, level L4 MOVES-LIKE-A-DEMO. oracle_state +100.00 % on 1040. V-chain row0 RMS 2.3607e-02 with rows 1..15 at exactly 0.000e+00. THE REGISTERED MAGNITUDE HELD: 4a predicted +55 to +75 and slightly below PR-12's half-A, on the reasoning that the anchorable set re-adds each episode's LAST chunk (end-of-task, low-motion, where repeat-last-action is strongest); measured +68.10, below the pooled v_chain d=0 figures of +67.14/+69.41. One self-correction: 4a named +67.30/+68.57 as the comparators and those are the v_mask cells, not v_chain -- the prediction is unaffected but the numbers cited were the wrong two. NEWLY MEASURED AND NOBODY HAD IT: dropping each episode's FIRST chunk moves the unmodified arm -359.41 -> -344.54, i.e. 14.87 pp from 40 of 1040 chunks -- ~4 % of the set carrying a disproportionate share, which is what an anchor defect at episode start predicts. The trim adopted in PR-10 for an unrelated reason silently removed the worst-affected chunks; no earlier conclusion turns on it, since all compared within one set. W IS NOT A VERDICT ON T-39 AND CANNOT BE -- P/N/M/I all require the policy arm, which never ran. It licenses CORRECTING every document asserting no policy trained on this corpus's action column can clear the bar (root CLAUDE.md, subprojects/README.md, subprojects/edge-wam/CLAUDE.md -- all three corrected in this task, gate text preserved verbatim). CORRECTING A CLAIM IS NOT LIFTING A GATE: the training gate is untouched and remains the project owner's call, no statement about GR00T or any policy is licensed (PR-07 6 stands), the fourteen negatives are not retro-validated, and docs/benchmark.md's L4 gate stays an open decision -- the repaired cell is L4 under spec 0.1.0 and below spec 0.2.0's two-sided floor, so the specs disagree about it. STILL UNANSWERED: whether a policy can LEARN these labels. This scores oracles; testing it is a training run. Result: docs/preregistration/PR-13-RESULT.md, artifact runs/t47-g0-rederivation/rederivation.json."
---

# Does T-39's VOID survive its instrument being repaired?

## Description

**T-39's `VOID` was decided by one clause, not by the policy arm — which never ran.**
`T39_RULE_V1`'s G0 (`PR-07-positive-control.md:137-141`) says `oracle_action` must reach L1, "below
that […] T-39 is VOID and the finding is recorded against the label pipeline". It came in at
**−359.41 %**, and the whole project has run on that clause's consequence: *no policy trained on
this corpus's action column can clear our bar.*

**PR-12 showed that number came through a defective instrument.** `commanded_to_chunk` builds the
chunk's step 0 as `command − STATE` while every other step of both arms is a homogeneous first
difference; that one element carried **~90 % of the summed per-step MSE and 143× its neighbours**,
and homogenising it (`targets[0] = q_cmd[0] − q_cmd[−1]`) took a held-out half from **−379.68 to
+69.15**. T-47 asks the single remaining question: **does G0's blocking clause still fire when the
instrument is repaired?**

**Three cells, `d = 0` throughout** — PR-12 retired the delay, so none is applied and none fitted.
A **bridge** (unmodified, the full 1 040, directly comparable to `PR-07-RESULT.md`), and a
**control** and **repaired** pair scored on the identical *anchorable* set. That set is registered
rather than discovered: V-chain needs `start ≥ 1` and so drops each episode's first chunk, so the
count must be `full_count − num_episodes` and **both compared cells must score exactly it** —
quoting the repaired cell against the 1 040-chunk bridge would compare two sets.

**Registered as confirmatory by design, and §4 says so before it runs.** After +69.15 on a held-out
half the outcome is not seriously in doubt, and pretending otherwise would repeat the error PR-12
nearly made. The value is that PR-12 measured a *different* set through a *different* driver in
halves, while G0 read the full untrimmed set in one pass — and a gate's condition should be
re-derived on the gate's own set before anyone declares its premise withdrawn. **What is genuinely
open** is whether the effect survives the set change: the full set includes each episode's first
chunks and its tail, which every cell in PR-10, PR-11 and PR-12 dropped uniformly and none ever
scored.

**Verdicts** (`T47_RULE_V1`, precedence S→W→I): **W** — repaired cell clears L1 with
`skill_vs_repeat_pct ≥ 10.0` **and** clears L2, so G0's blocking clause does not hold under a
corrected instrument and the premise of `VOID (labels)` is withdrawn by measurement; **S** — it
fails L1, so the VOID **survives** its instrument being repaired, which makes it a far stronger
result than the project currently holds and means PR-12's gain was a property of the trimmed set;
**I** — anything else. **W is the expensive conclusion** and therefore carries the material margin
and L2, not L1 alone.

**W IS NOT A VERDICT ON T-39 AND CANNOT BE.** `T39_RULE_V1`'s P/N/M/I all require the policy arm.
The most W says is that the clause which decided `VOID` no longer fires. It licenses **correcting**
every document asserting that no policy trained on these labels can clear the bar — `CLAUDE.md`,
`subprojects/README.md`, `subprojects/edge-wam/CLAUDE.md` — and **correcting a claim is not lifting
a gate**. It explicitly does **not** discharge the training gate, does **not** license any statement
about GR00T or any policy (PR-07 §6 untouched, oracles against oracles), does **not** retro-validate
the fourteen negatives (PR-12-RESULT's blast radius already excluded them), and does **not** amend
`PR-07-positive-control.md` — rules here are versioned, which is why PR-13 is a new document.

Zero GPU-hours, CPU only, minutes.

## Notes / Report

**Reported 2026-08-16 — `W`.** `docs/preregistration/PR-13-RESULT.md`, artifact
`runs/t47-g0-rederivation/rederivation.json`. Zero GPU-hours, nothing submitted. All four gates
passed on the **first** run.

| cell | anchoring | chunks | L1 | L2 | vs-zero | `horizon` | `smooth` | step-0 | level |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **bridge** | unmodified | **1040** | **−359.41** | −102.54 | −157.11 | 0.0044 | 8.518 | 93.39 % | below L0 |
| control | unmodified | 1000 | −344.54 | −93.81 | −150.36 | 0.0045 | 8.328 | 93.26 % | below L0 |
| **repaired** | V-chain | **1000** | **+68.10** | **+75.40** | **+82.04** | **0.9717** | 0.280 | 6.03 % | **L4** |

**The bridge reproduced `PR-07-RESULT.md`'s `−359.41` to `+0.002 pp`**, so one number here is the
archive's, produced by this driver, and the other two are comparable to it.

### The registered magnitude held

§4a predicted **+55 to +75**, slightly below PR-12's half-A, because the anchorable set re-adds each
episode's **last** chunk — end-of-task, low-motion, where repeat-last-action is strongest.
**Measured +68.10**, below the pooled `v_chain` `d = 0` figures (+67.14 / +69.41).

*Self-correction:* §4a named "+67.30 / +68.57" as the comparators; those are the **`v_mask`** cells,
not `v_chain`. The prediction is unaffected — +68.10 is in band against either — but the two numbers
cited were the wrong ones.

### Newly measured, and nobody had it

Dropping each episode's **first** chunk moves the unmodified arm from **−359.41 to −344.54** —
**14.87 pp out of 40 chunks in 1040.** Four percent of the set carries a disproportionate share of
the damage, which is what an anchor defect at episode start predicts: the robot starts at rest and
the standing command-versus-state gap is widest there. PR-10 adopted the trim for an unrelated
reason and it silently removed the worst-affected chunks. **No earlier conclusion turns on this** —
every one compared cells within a single set — but it belongs on record.

### What `W` is not

**Not a verdict on T-39, and `T47_RULE_V1` registered that before the run.** `P`/`N`/`M`/`I` all
require the policy arm, which never ran (job 187813, dead at 108 s). T-39 becomes a gate whose
blocking condition was measured through a broken instrument.

**Corrections made under this licence** (gate text preserved verbatim in all three): root
`CLAUDE.md`, `subprojects/README.md`, `subprojects/edge-wam/CLAUDE.md`. **Correcting a claim is not
lifting a gate** — the training decision is the project owner's and no session may edit those files
to lift it.

**Still unanswered, and it is the whole remaining question:** whether a policy can *learn* these
labels. This scores oracles. Testing it is a training run.
