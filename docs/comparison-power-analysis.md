# The planned comparison cannot see its own effect — 9 % power against a quadrupling

Computed 2026-08-16, exactly (no simulation, no seed). Tool: `scripts/power_closed_loop_eval.py`.
**Zero GPU-hours.** This is a property of the measuring device, not a result about augmentation.

**Deliberately not a pre-registration and deliberately unnumbered.** It scores nothing and has no
verdict; it is a decision input, like `smoothness-ratio-audit.md`. The PR numbers are contended
enough already.

## The question

PR-08's generation leg proposes **~10 050 restyled clips** and **two Recipe B retrains** so that an
augmented arm can be compared against an unaugmented one. Before spending that, one thing should be
known: **what effect could the instrument at the end have detected?**

The instrument is `../vla-training/eval/run_apple_eval.py` — closed-loop success over a fixed paired
seed set, analysed with **McNemar's exact test**. That is that repo's own choice, recorded in
`eu-hub/RUNS.md`, not one invented here. Its protocol freezes the seed count at **20**.

## Why 20 seeds is far less information than it sounds

McNemar counts only the seeds where the two arms **disagree**. Concordant seeds — both succeed, both
fail — carry no information and do not enter the test at all. With `b` seeds where only the baseline
won and `c` where only the treatment did, the null is a fair coin on each discordant seed, so the
two-sided exact p is `2 · P(X ≤ min(b,c))` for `X ~ Binomial(b+c, ½)`.

**20 seeds are not 20 observations. They are however many disagreements the two arms happen to
produce — and against a 1/20 baseline that is a very small number.**

## The floor

Even in the best possible case — *every* discordant seed favouring the treatment, not one going the
other way — significance needs **6 discordant seeds**:

```
2 · 0.5^5 = 0.0625   does not reach 0.05
2 · 0.5^6 = 0.0312   does
```

So against the measured **1/20** baseline the restyled arm must reach at least **7/20**, and only if
it keeps the success the baseline already had. That is the floor, not the requirement.

## Power at n = 20, baseline 1/20

| true treatment rate | | power |
|---|---:|---:|
| 2/20 | 0.10 | **0.5 %** |
| 4/20 | 0.20 | **9.2 %** |
| 6/20 | 0.30 | 34.0 % |
| 8/20 | 0.40 | 64.1 % |
| 10/20 | 0.50 | 85.8 % |

**Read the second row.** If photoreal restyling *quadrupled* the success rate — 1/20 → 4/20 — this
experiment would report it as null **91 times out of 100**. A null result from the planned run would
therefore carry almost no information: it is the outcome whether or not the augmentation works.

Independence between the two arms is assumed throughout, which is **optimistic**. Two policies on
the same task and the same seeds fail on the same hard seeds; that correlation makes seeds
concordant, and concordant seeds are invisible to this test. The real numbers are at or below these.

## What the existing runs already show

Every in-house result sits in the region where the test cannot work, and the repo's own analysis
says so without drawing the conclusion:

| run | score | source |
|---|---|---|
| GR00T Recipe A, IsaacLab-Arena | 1/20 | `../vla-training/docs/isaaclab-arena-eval.md` |
| GR00T Recipe B, IsaacLab-Arena | 1/20 | same |
| vendor checkpoint, Arena | 5/20 | same |
| GR00T Recipe B, MuJoCo | 2/20 | `../vla-training/eu-hub/RUNS.md` |

**McNemar exact p = 0.500 on every pairing ever scored there.** Not "no effect found" — 0.500 is
what this test returns when there is essentially nothing to work with.

## The fix is cheap, and it is not more clips

| target effect | paired seeds for 80 % power |
|---|---:|
| 1/20 → 2/20 | > 400 |
| 1/20 → 4/20 | **83** |
| 1/20 → 6/20 | 41 |
| 1/20 → 8/20 | 26 |
| 1/20 → 10/20 | 19 |

**Detecting a quadrupling needs 83 paired seeds, not 20.** Closed-loop sim seeds are the cheapest
thing in this entire pipeline — they run in `../vla-training`'s harness on dz-226, not on
Discoverer+, and consume none of the EuroHPC allocation. Raising the seed count is an afternoon.
Generating 10 050 clips is not.

**The ordering this implies:** fix the instrument before feeding it. A restyle run scored at n = 20
produces a number that cannot be interpreted in either direction, and the ~10 050 clips are
unrecoverable once spent.

## What this does not say

- **Nothing about whether restyling works.** It has never been measured here and this document does
  not estimate it. Feeding a hoped-for effect size into a power calculation does not make the effect
  exist.
- **Nothing about the baseline being 1/20.** That is `../vla-training`'s measured number under its
  own protocol, taken as given. If the baseline moves the whole table moves with it — a higher
  baseline needs *more* seeds, not fewer, because the floor is a count of disagreements.
- **It does not license or block the generation gate.** PR-08 §1 gates T-040 on T-39 and that is the
  project owner's call. This says only what the far end of the pipeline could resolve.
- **It is not an argument against the comparison.** The pixels-only axis — WAM emits restyled
  frames, `../vla-training` trains its own recipe and scores both arms closed-loop — is the one
  design where the label spaces never have to be reconciled, and it remains the right design. It
  just needs an instrument that can read the result.

## A separate trap, recorded here because it is adjacent

Any number computed from an existing `../vla-training` checkpoint against WAM's 40-episode holdout
is a **training-set** number. Their corpus declares `splits: {"train": "0:402"}` with
`val_dataset_path: null`, so every one of those checkpoints has seen all 402 episodes — including
all 40 of `configs/splits/t18_holdout_episodes.txt`. Such a number would look excellent for reasons
that have nothing to do with the model, and the caveat has to travel **with** it, at the moment it
is produced, not after someone quotes it.
