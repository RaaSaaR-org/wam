# PR-03 — gate result: the power gate FAILS. The refit is not submitted.

**Measured 2026-08-02**, CPU only, no GPU, no allocation spent. Reproduce with:

```bash
.venv/bin/python scripts/bench_grasp_anticipation.py \
    --holdout configs/splits/pr03_holdout_150.txt --out runs/pr03/gate2-power.json
.venv/bin/python scripts/bench_grasp_anticipation.py \
    --holdout configs/splits/t18_holdout_episodes.txt \
    --restrict configs/splits/t18_holdout_episodes.txt --out runs/pr03/gate1-archive-check.json
```

The pre-registration is `PR-03-grasp-anticipation.md`, committed at `4bf36a3` before
`scripts/bench_grasp_anticipation.py` existed. Per repo convention that document is **annotated,
never edited**, so the result lives here.

## The verdict

**Gate 2 fails on its CI clause. Per PR-03, the refit is not submitted and no GPU-hours are
requested.** The pre-registered consequence is `PR-01-GRIPPER.md` §4 — collect demonstrations,
not more analysis.

| clause | measured | bar | |
|---|---:|---:|---|
| `n_postflip` | **2 682** | ≥ 2 000 | ✅ |
| `ci_halfwidth` | **4.10 pts** | ≤ 3.5 | ❌ |
| implied minimum detectable effect | **~8.2 pts** | — | |

150 holdout episodes, 252 training episodes, 302 transition chunks over 148 of 150 episodes.
Widening from 40 → 150 did what it was supposed to: post-flip steps went 675 → 2 682 (4.0×) and
the half-width 6.84 → 4.10. It landed **0.60 points short of the bar**.

PR-03 fixed 3.5 before any of this was measured, and the repo convention is that a threshold which
has to change is **versioned, never edited in place**. So this is a fail, not a near-pass, and
re-arguing 3.5 after seeing 4.10 is precisely the move pre-registration exists to prevent.

## The blind suite on the 150, since it is the useful part

Post-flip accuracy on transition chunks, primary (`episode-latch`) definition:

| predictor | fitted params | post-flip | pre-flip | k…k+3 |
|---|---:|---:|---:|---:|
| repeat-last-gripper | 0 | 24.76 % | 76.09 % | 17.13 % |
| ridge, 32-dim state | 32 | 32.92 % | 76.93 % | 21.52 % |
| time-in-episode only | — | 56.04 % | 62.37 % | 46.28 % |
| gripper const-velocity | 0 | 66.59 % | 83.53 % | 47.98 % |
| **blind nonlinear ceiling** | 2 048 | **75.35 %** | 82.47 % | **59.01 %** |

The shape `PR-01-GRIPPER.md` described survives on 3.75× more data: a **V, not a decay**. Pre-flip
is easy for everything (76–84 %), the two structurally-stale rules collapse **below chance**
post-flip, and the worst place for every blind predictor is the four steps at the flip.

**The headroom is real but smaller than archived.** Blind-unreachable share is `100 − ceiling`:
**24.65 points** post-flip and **40.99** at k…k+3, against the ~29/47 implied by
`PR-01-GRIPPER.md`'s 70.91/52.98. The difference is not the data — it is that this ceiling is
stronger, for a reason worth recording (below). A stronger ceiling is the conservative direction:
it raises the bar a model would have to clear.

## Gate 1 — MISS, and the miss is ambiguous by construction

Same 40 episodes, same 362-episode training pool, i.e. `PR-01-GRIPPER.md`'s exact setup:

| predictor | re-measured | archived | Δ | tol | |
|---|---:|---:|---:|---:|---|
| transition chunks | **78** | 78 | 0 | — | ✅ exact |
| post-flip steps | 675 | 660 | +15 | — | +2.3 % |
| repeat-last | 21.48 % | 19.70 % | +1.78 | ±0.5 | ❌ |
| const-velocity | 63.70 % | 62.88 % | +0.82 | ±0.5 | ❌ |
| ridge-32 | 25.93 % | 24.09 % | +1.84 | ±2.0 | ✅ |
| time-only | 53.19 % | 39.09 % | **+14.10** | ±2.0 | ❌ |
| ceiling | 73.19 % | 70.91 % | +2.28 | ±2.0 | ❌ |
| ceiling CI half-width | 6.84 | 7.56 | −0.72 | — | |

PR-03 pre-declared how to read this: `PR-01-GRIPPER.md`'s numbers came from a scratch script that
was never committed, so a miss outside tolerance is **ambiguous between "the channel differs" and
"this re-implementation differs"** and may not be reported as a channel finding. It is ambiguous,
and here is what points which way:

- **The channel is almost certainly fine.** The transition-chunk count reproduces *exactly*
  (78/78) and episodes-with-a-transition is 39/40 as archived. Those are properties of the channel
  and the latch, and they are the numbers a botched conversion would break first.
- **The two zero-parameter rules miss a ±0.5 tolerance by 0.8–1.8 points while scoring 15 more
  post-flip steps (675 vs 660).** Those rules are deterministic given the channel and the flip
  index, so a discrepancy in them is a discrepancy in *which steps are post-flip* — an indexing
  convention, not a value.
- **`time-only` at +14.1 is a different predictor, not a different channel.** PR-03 fixed the
  metric but never the control's functional form; 20 phase bins plus a linear term is evidently
  richer than whatever produced 39.09 %. That is a defect in the pre-registration, mine, and the
  fix is that the form is now in committed code rather than in prose.

So gate 1 is recorded as **MISS (ambiguous)**. It does not license a claim that the conversion
damaged the channel, and it does not license treating the archived table as reproduced.

> **One thing I got wrong on the way, left visible.** The first run's "ceiling" scored **59.11 %**
> post-flip — *below* the zero-parameter const-velocity rule at 63.70 %. A ceiling a free rule
> beats is not a ceiling, and had I written it up as one, every subsequent comparison would have
> been against a bar that was too low, making a mediocre model look like a win. Two causes, both
> mine: the RBF bandwidth grid was carried over from a 32-dim problem and applied to a 256-dim
> input, where `exp(−0.02 · 512)` underflows and the kernel degenerates; and the ceiling was fitted
> by least squares on the **continuous** channel while being scored on **accuracy at 0.5**, so it
> inherited exactly the mean-seeking bias that makes the linear controls score below chance. Fixed
> by scaling γ as 1/D and fitting the ceiling on the binarized label — 59.11 → 73.19 %. The tell
> was structural, not statistical: a 2 048-parameter model cannot legitimately lose to a rule whose
> solution lies inside its own linear span. `bench_grasp_anticipation.py` now reports
> `ceiling_dominates` on every run so this cannot pass silently again.

## A second, independent reason not to have run it

Under the **`self-contained`** definition the ceiling is **50.93 %** and a clock-only model scores
**54.48 %** — `ceiling_dominates: false`. On that definition there is no valid ceiling to score a
model against, so PR-03's binding robustness clause ("if the verdict differs between definitions,
there is no verdict") could not have been evaluated soundly even had gate 2 passed. Any future
attempt needs a ceiling family that dominates under all three definitions, not just two.

## What would change the answer, as a separate decision

`h` scales as `1/√n`, so reaching 3.5 needs `(4.10/3.5)² ≈ 1.37×` the post-flip steps — roughly
**205 holdout episodes, leaving ~197 to train on**. That is a **new pre-registration**, not an edit
to this one, and it is not obviously a good trade:

- `PR-01-GRIPPER.md` asserts the fine-tune "is not starved at 250 episodes". That is untested at
  197, and PR-03 would be spending its training data to buy resolution.
- The corpus is finite. 402 episodes cannot be split to satisfy both sides indefinitely, which is
  the same wall every negative in this project has hit.

The cheaper reading is the one PR-03 already committed to: **the bottleneck is demonstrations.**
This gate cost one CPU pass and answered the question that would otherwise have cost ~20–40
GPU-hours to answer worse.

## What this does not say

Not that the model would have failed — it was never run, and nothing here is evidence about the
checkpoint. Not that grasp anticipation is unmeasurable — it is measurable, at ~8.2 points of
resolution, on a corpus where the effect is plausibly smaller than that. And not that the
re-conversion was wasted: `datasets/gr00t-apple-grip` is the first dataset in this project with a
live grasp channel (T-31: 2.01 debounced transitions/episode, 99.0 % of episodes with a full
cycle), and it is the reason this question could be asked at all.
