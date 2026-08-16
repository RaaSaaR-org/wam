---
id: T-047
aliases:
- T-47
- T-047
title: "Does T-39's VOID survive its instrument being repaired?"
slug: does-t-39-s-void-survive-its-instrument-being-repaired
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
- T-046
due_date: ''
created: 2026-08-16
updated: 2026-08-16
status_note: "PRE-REGISTERED 2026-08-16 as docs/preregistration/PR-13-t39-g0-rederivation.md, rule T47_RULE_V1, claimed across live sessions before it was written. Scope is the project owner's narrowing: re-derive T-39's G0 oracle_action clause on T-39's OWN full untrimmed chunk set under PR-12's repaired anchoring. Registered as CONFIRMATORY BY DESIGN -- after +69.15 on a held-out half the answer is not in doubt, and the value is that the gate's own condition gets re-derived on the gate's own set rather than inferred across a set and driver boundary. Not yet run."
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
