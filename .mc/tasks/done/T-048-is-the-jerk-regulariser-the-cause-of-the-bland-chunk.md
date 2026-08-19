---
id: T-048
aliases:
- T-48
- T-048
title: "Is the jerk regulariser the cause of the bland chunk?"
slug: is-the-jerk-regulariser-the-cause-of-the-bland-chunk
status: done
priority: 2
owner: ''
projects: []
customers: []
tags:
- m3
- eval
- prereg
- train
sprint: ''
depends_on:
- T-16
- T-29
due_date: ''
created: 2026-08-16
updated: 2026-08-19
status_note: "RAN 2026-08-17/18 on the local RTX 5090, verdict S (shrinkage), zero Discoverer+ GPU-hours. Rule T48_RULE_V1, docs/preregistration/PR-14-smoothness-ablation.md, registered 2026-08-16 before the run produced a single step and unedited since; result docs/preregistration/PR-14-RESULT.md, commit 19f4eee. shape_moved is a CONJUNCTION and it split: chunk_rms 0.00226 -> 0.00293 CLEARED its 1.25x threshold (0.002825) while smoothness_ratio 0.3198 -> 0.3548 fell far short of its 2x one (0.6396), an 11 percent move against the 100 percent the rule required. The chunks got bigger without getting jerkier, so the jerk regulariser is NOT a material cause of the bland output and L2 shrinkage / mean-seeking survives as the explanation. This licenses DROPPING the regulariser hypothesis, not ADOPTING the mean-seeking one. Note the conjunction earned its keep: on chunk_rms alone the rule would have returned a positive for a model whose smoothness is still 0.71 of spec 0.2.0's admissible floor. l1_cleared FALSE at -3.4288 so no L verdict fires; l1_material TRUE at +18.371 pp against A's archived -21.80 but licenses nothing on its own, and PR-14 section 3 makes the 5090-vs-H200 confound live in exactly that positive direction. Run pr14-nosmooth-seed0 completed 20000/20000 as ONE CONTINUOUS CHAIN across an external SIGTERM at step 7385: the resume re-entered at step 7385 AND sampler position epoch 6 batch 1112, which is what section 6 requires for the run to be scoreable rather than I. THIS TASK ID IS CONTESTED -- see the Notes section; the number T-48 was claimed twice on 2026-08-16."
---

# Is the jerk regulariser the cause of the bland chunk?

## Description

T-16/T-29's LoRA reaches WAM-Bench **L0** with `skill_vs_repeat_pct` **−21.80 %**, and the shape of
its failure is specific rather than generic: the predicted chunks are both **too small** (`chunk_rms`
0.00226 against the demonstrations' 0.00404, a 44 % shortfall) and **too smooth** (`smoothness_ratio`
0.3198 against an admissible floor of 0.5). Two explanations fit that pair, and they imply opposite
next moves:

1. **The jerk regulariser.** `JointTrainingConfig` carries a smoothness penalty. If it is doing the
   work, the fix is a config line and costs nothing.
2. **L2 shrinkage / mean-seeking.** An L2-trained head hedges toward the conditional mean, which is
   smaller and smoother than any individual demonstration. If that is the cause, the fix is a loss,
   not a flag.

The ablation separates them: **train B identically to A with `smoothness = 0.0` and change nothing
else.** Registered as `T48_RULE_V1` in
[`PR-14-smoothness-ablation.md`](../../../docs/preregistration/PR-14-smoothness-ablation.md) before
the run, with both thresholds **derived rather than chosen** — `2×` on `smoothness_ratio` because
that lands the arm inside spec 0.2.0's admissible band, `1.25×` on `chunk_rms` because it closes half
the measured 44 % magnitude shortfall.

**Verdicts** (`T48_RULE_V1`, precedence `I → S → R-PENDING/R`, with `L` orthogonal):

- **`I`** — the run cannot complete 20 000 steps and the resumes do not reconstruct one continuous
  chain. **No shape claim is made at all**: a partially-trained B against a fully-trained A is not a
  weaker version of this experiment, it is a different one.
- **`S`** — `not shape_moved`. The regulariser is not a material cause; shrinkage survives.
- **`R` / `R-PENDING`** — `shape_moved`. Because the arms differ in hardware and batch shape, a
  positive is `R-PENDING` until a local A′ re-run separates the two.
- **`L` / `L-MATERIAL`** — orthogonal, and both require `l1_cleared`.

**What this does NOT do.** It does not test the mean-seeking hypothesis — ruling one explanation out
is not ruling the other one in. It is not a WAM-vs-anything comparison, it is not a robot result, and
it says nothing about the corpus, the label space or convergence at 20 000 steps. No vendored model
is loaded, trained or consulted in either arm.

**Cost: zero Discoverer+ GPU-hours.** Local RTX 5090 only, ~6 h wall.

## Notes / Report

**Reported 2026-08-19 — `S`.**
[`docs/preregistration/PR-14-RESULT.md`](../../../docs/preregistration/PR-14-RESULT.md), commit
`19f4eee`, artifacts in `runs/pr14-nosmooth-seed0/`. Zero GPU-hours on the allocation, nothing
submitted.

| term | threshold | B measured | |
|---|---|---:|---|
| `B.smoothness_ratio ≥ 2 × A_SMOOTHNESS_RATIO` | ≥ 0.6396 | **0.35478** | **FALSE** |
| `B.chunk_rms ≥ 1.25 × A_CHUNK_RMS` | ≥ 0.002825 | **0.002931** | TRUE |
| **`shape_moved`** | both | | **FALSE** |
| `l1_cleared` = `skill_vs_repeat_pct > 0` | > 0 | **−3.4288** | **FALSE** |
| `l1_material` = `B − A_SKILL_VS_REPEAT ≥ 10.0` | ≥ 10.0 pp | **+18.371 pp** | TRUE |

`shape_moved` FALSE ⇒ **`S`**. `l1_cleared` FALSE ⇒ not `L`; `L-MATERIAL` needs both, so it does not
fire either.

### The conjunction is what carried this

Removing the regulariser moved the **magnitude** and left the **shape** alone — `chunk_rms` closed
40 % of the 44 % shortfall and cleared its own threshold, while `smoothness_ratio` moved 11 % against
the 100 % required. **The chunks got bigger without getting jerkier.** A regulariser that were the
material cause would have moved both. Had `T48_RULE_V1` been written on `chunk_rms` alone it would
have returned a positive for a model whose smoothness is still **0.71 of spec 0.2.0's floor** — which
is the argument for writing conjunctions into a rule before seeing the numbers, not after.

### The validity chain, because it nearly broke

§6 voids the experiment if the run cannot reach 20 000 steps as one continuous chain. An external
`SIGTERM` stopped it at step 7385. It was **resumed, not restarted**, and the log proves continuity
rather than asserting it: the checkpoint recorded step 7385 *and* sampler position epoch 6 batch
1112, and the resume re-entered at both, so the two legs are one pass over the data rather than a
replay of the first epochs. `config_hash`
`69aba5309f911a450ebae8aa3395e9eb8bbdf255413a1478361725ce0e62e93b` is identical across both legs and
in `DONE`; the evaluator proved the split disjoint (362/40) against `dataset_snapshot_ref` rather
than trusting it.

### Newly measured, and nobody had it

- **`chunk_rms` is not in `bench.json`** and had to be computed from `predictions.jsonl`. The same
  computation over the `target` column returns **0.004041**, reproducing the registered
  `DEMO_RMS = 0.00404` — so the instrument reading B is the instrument that produced the constants B
  is compared against. That check is the reason the number is quotable.
- Re-scored under both bench specs from stored predictions (2026-08-19, CPU, `scripts/run_bench.py`):
  **56.0/100 under spec 0.1.0, 36.0/100 under 0.2.0**, level **L0** under both. The 20-point gap is
  L4, exactly as for `t16-lora-seed0`. Row added to
  [`docs/benchmark.md`](../../../docs/benchmark.md).
- `runs/t16-lora-seed0/` is **empty on this box**. The verdict is unaffected because `T48_RULE_V1`
  pinned A's constants as literals — which is precisely why pre-registrations pin literals — but any
  future A′ has to re-materialise that run first.

### What `S` is not

`S` rules the regulariser **out**. It does not rule shrinkage **in**; that needs its own test. And
the +18.371 pp on L1 must not be read as an effect of removing the regulariser: it is a *positive*
direction, which is exactly where PR-14 §3 registers the 5090-vs-H200 and batch-shape confound as
live, and no A′ has been run. Everything the rule licenses on that axis is `l1_cleared = FALSE`.

*Self-correction, recorded rather than quietly fixed:* the draft result copied `T48_RULE_V1`'s
citation of `src/wam/evaluation/benchmark.py:71` for `MIN_SMOOTHNESS_RATIO`. That line is
`MAX_SMOOTHNESS_RATIO = 2.0`; the floor is derived at `:72` and deliberately never written as a
literal. The pre-registration's **value** (0.5) is right and its line number is off by one. The
pre-registration was **not** edited — rules here are versioned, never amended in place — and the
correction lives in the result.

### This task ID is contested, and that is recorded rather than resolved

**The number `T-48` was claimed twice on 2026-08-16 by two sessions, and neither minted a file.**

1. **This task.** `PR-14-smoothness-ablation.md:3` and `PR-14-RESULT.md:4` both say "Task T-48",
   inside a pre-registration that is versioned and may not be amended in place.
2. **A documentation correction.** `TASKS.md:58` — "**Correction, 2026-08-16 (T-48).**" — about two
   of the four `tests/test_runtime.py` runtime failures not being about the missing file, landed as
   commit `b8da4e3 docs(t-48): …`.

This file mints `T-048` for **(1)**, on the narrow ground that it is the only claimant named inside a
document that cannot be edited to point elsewhere; (2) lives in prose and a commit subject, both of
which read fine as an informal tag. **That is a tie-break, not a ruling, and it is the second
duplicate-number collision in this repo after PR-10** — which is still unresolved and is the
project owner's to settle. Renaming this file costs one index line; PR-10 has 163 references across
34 files, which is what the difference in urgency is made of.
