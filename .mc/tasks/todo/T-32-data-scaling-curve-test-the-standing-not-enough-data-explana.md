---
id: T-32
aliases:
- T-32
title: "Data-scaling curve — test the standing \"not enough data\" explanation"
slug: data-scaling-curve-test-the-standing-not-enough-data-explana
status: backlog
priority: 2
owner: ''
projects: []
customers: []
tags:
- m3
- data
- eval
- cluster
- hardware
sprint: ''
depends_on:
- "[[T-39]]"
due_date: ''
created: 2026-08-01
updated: 2026-08-01
status_note: "Staged on Discoverer+ (~109 GPU-h) and deliberately BLOCKED behind T-39. Fitting a data-scaling curve on a method no positive control has validated measures the scaling of brokenness if the method is broken; that ordering is pre-registered, not a preference. UPDATE 2026-08-16: T-39 reported VOID (labels), and PR-12 (C) / PR-13 (W) traced that VOID to our own evaluation adapter rather than the corpus — repaired, the corpus's own action column scores +68.10 L1, level L4. But T-39's POLICY arm never ran (G0 fires before it), so the thing this task is blocked on — a positive control that validates the METHOD — still does not exist. The block stands on its original reason."
---

# Data-scaling curve — test the standing "not enough data" explanation

## Description

Data-scaling curve — test the standing "not enough data" explanation before buying it with months
(I-8, `docs/improvements.md`). **Staged 2026-08-01: `sbatch
cluster/discoverer/55_train_i8_rung.sbatch` per rung, then `62_eval_i8_curve.sbatch`.** Every
negative we have — T-18, T-15/T-24/T-26, T-16 — came from the *same* 402 success-only episodes of
one task, and "not enough data" has explained all of them without once being tested. It is also the
most expensive conclusion in the project: it implies a G1 EDU4, a teleop rig and months of
recording. Retrain at **40 / 120 / 362** episodes, identical everything else, score on the same
40-episode holdout. Splits are committed and nested — `configs/splits/i8_train_{040,120,362}.txt`,
40 ⊂ 120 ⊂ 362, zero overlap with the holdout, seeded shuffle rather than a sorted prefix so a rung
is a random sample and not the corpus's alphabetical head. Rung 362 is `runs/t16-lora-seed0`,
already scored; its `bench.json` is consumed, not recomputed. **Rule `I8_RULE_V3`, in git before the
first rung is submitted**, and three things it had to be taught: (a) the equal-STEPS confound is
**symmetric** — 147.5 epochs at N=40 against 16.9 at N=362 manufactures a flat curve as readily as a
steep one, so gating only the expensive verdict on it let the cheap one through free; all three
verdicts now require the equal-EPOCH control. (b) N\* is an **OLS extrapolation** 2.8 doublings past
the largest measured N with one residual degree of freedom, so verdict A now needs the whole
bootstrapped interval — propagated from the *measured* seed spread via the `i8-rung040-seed1`
replicate — inside what a campaign could deliver, not just the point estimate. (c) "Not
data-limited" was **two worlds sharing one sentence**: it fires on `not (MONOTONE and SPAN ≥
MATERIAL)`, so three noisy rungs out of order while spanning materially printed "the headline did
not move" when it had moved by more than a seed does; split out as **C-NOISY**, which claims no
readable curve and routes to a seed replicate rather than to I-9. **Runs after T-30**, because a
readout swap moves every rung's headline at once and fitting a scaling curve through numbers a
pending decode change may move is fitting to a moving target

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
