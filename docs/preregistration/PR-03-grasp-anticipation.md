# PR-03 — can the fine-tune anticipate a grasp flip that no blind predictor can?

**Pre-registered 2026-08-02, before any predictor was fitted on `datasets/gr00t-apple-grip` and
before any number was computed on `configs/splits/pr03_holdout_150.txt`.** What existed first: the
re-converted dataset (T-31, commit `d379c61`) and its admissibility audit, which is a property of
the *channel* and not of any predictor; and the widened holdout file, which is a seeded permutation
that no measurement informed. No accuracy, MSE or R² has been computed on the 110 newly held-out
episodes.

This is the metric `PR-01-GRIPPER.md` §3 requires to exist before the refit it recommends, written
so the refit cannot be scored on a rule invented after seeing its result.

## Why this exists, and why it is narrow

Four negatives (T-15, T-24, T-26, T-16/T-29) and PR-01 all measured arm-trajectory MSE on the same
402 success-only episodes. PR-01 established that **66.0 % of that metric's achievable range is
reachable with no vision at all**, and that a zero-parameter rule beats the 82.5 M-parameter
fine-tune by 1.22×. The metric was retired as a ranking metric for exactly that reason.

`PR-01-GRIPPER.md` then found the one place in this corpus where a blind predictor demonstrably
cannot go: the ~4 steps around a grasp flip. Everything else about the gripper channel is
momentum — a zero-parameter repeat-last rule scores 97.82 % on the full holdout, covering 90 % of
the accuracy range. **Full-holdout gripper accuracy is a momentum metric and is pre-registered here
as forbidden.** Only the flip discriminates, and only barely: at the flip every blind predictor is
at a coin toss, and **44 % of the accuracy range is not blind-reachable**.

That gap is the entire hypothesis. It is small, it is local, and this document exists to stop it
being reported as more than it is.

## What is scored

| | |
|---|---|
| dataset | `datasets/gr00t-apple-grip` (T-31, `--gripper-mapping active-hand`, affine `active=left offset=-0.438654 span=0.466748` fitted over all 402 episodes) |
| holdout | `configs/splits/pr03_holdout_150.txt` — 150 episodes, a strict superset of the 40-episode T-18 holdout, seed 0 |
| training pool | the remaining 252 episodes |
| channel | `action.gripper_target` (the active/left hand alone), binarized at 0.5 |
| chunks | 16 steps, non-overlapping — as converted |

`runs/t16-lora-seed0` **cannot** be scored here: it trained on 110 of these 150 episodes.
`scripts/eval_t16.py` proves the split from the checkpoint's recorded training-set hash and will
refuse. PR-03 therefore scores a **refit**, on the 252-episode pool, and nothing else.

### The metric

**`postflip_accuracy` — binarized gripper accuracy on post-flip target steps of transition
chunks.** A *transition chunk* is a holdout chunk whose 16 target steps contain at least one
debounced transition; *post-flip* steps are those at or after the first such transition index `k`.

Debounce is `wam.evaluation.gripper.debounced_transitions(threshold=0.5, margin=0.1)` — the
shipped hysteresis latch, whose state carries across the whole episode. That definition is chosen
because it is the one the admissibility gate already uses, so the metric and the gate cannot
disagree about what a grasp is. It is also the **most generous** of the three definitions
`PR-01-GRIPPER.md` names (episode-latch 78 chunks, self-contained 60, label-steps-only 53), and
quoting only the most generous one is a mistake that document already had to correct. So:

> **Robustness clause, binding.** All three definitions are reported. If the verdict differs
> between them, there is no verdict — the result is reported as definition-dependent, which is a
> negative for the purposes of every decision below.

Reported alongside, never instead: `k…k+3` accuracy (the coin-toss window) and pre-flip accuracy
(which must stay high — a model that wins post-flip by getting worse before the flip has learned to
predict late, not to anticipate).

## Gate 1 — the archive gate. Nothing proceeds until this reproduces.

The blind control suite is re-measured on the new dataset and the new holdout **before** any model
number exists. Because the 150 is a strict superset of the 40, restricting the new run to the
original 40 episodes must reproduce `PR-01-GRIPPER.md`'s table:

| predictor | fitted params | post-flip, 40 eps (archived) |
|---|---:|---:|
| repeat-last-gripper | 0 | 19.70 % |
| gripper const-velocity | 0 | 62.88 % |
| ridge, 32-dim state | 32 | 24.09 % |
| time-in-episode only | — | 39.09 % |
| **blind nonlinear ceiling** | 2 048 | **70.91 %** |

Tolerance: the two **zero-parameter** rules are deterministic given the channel and must reproduce
to **±0.5 points**. The three fitted predictors depend on a random-feature draw and a
hyperparameter search and must reproduce to **±2.0 points**. A miss outside tolerance means the
converted channel is not the channel PR-01-GRIPPER restored analytically, and everything stops
until that is explained — it is not rounded away and it is not re-tuned into agreement.

## Gate 2 — the power gate, and it is checked BEFORE any GPU is spent

`PR-01-GRIPPER.md` §4: *"If a widened holdout still cannot resolve 7 points, the answer is more
demonstrations, not more analysis."* That is a pre-condition, not a post-hoc excuse, so it is
measured on CPU from the blind ceiling alone — no checkpoint required:

- **`n_postflip`** — post-flip target steps in the holdout. Archived value at 40 episodes: **660**.
- **`ci_halfwidth`** — episode-level bootstrap (5 000 resamples, seed 0) half-width on the blind
  ceiling's post-flip accuracy. Archived value at 40 episodes: **7.56 points**.

**The refit is submitted only if `n_postflip ≥ 2000` and `ci_halfwidth ≤ 3.5` points.**

Both thresholds are fixed here, before measurement. 3.5 points is chosen because the minimum
detectable effect runs at roughly twice the half-width (7.56 → ~15 points at 40 episodes), so
≤ 3.5 is what buys the ~7-point resolution §2 projects. 2000 is roughly the ~2 500 steps that
document projects, less a margin for the projection being a linear extrapolation of a count that
depends on where flips fall.

**If either fails, the refit does not happen and no GPU-hours are requested.** The recorded
consequence is `PR-01-GRIPPER.md` §4 — collect demonstrations. That branch is a legitimate outcome
of this document, not a failure of it, and it costs nothing to reach.

## The decision rule for the refit, if gate 2 passes

Let `ceiling` be the blind nonlinear ceiling's post-flip accuracy on the same 150 episodes and
`h` the bootstrap half-width from gate 2.

- **VERDICT A — vision anticipates the grasp.** `postflip_accuracy > ceiling + h`, **and** the
  robustness clause holds across all three debounce definitions, **and** pre-flip accuracy is not
  below the ceiling's pre-flip accuracy by more than `h`. Consequence: the first positive result in
  this project. AC-07 gets a second look on this metric, and the closed-loop measurement
  (`PR-01-RESULT.md`'s stated follow-on) becomes the next step rather than an architecture change.
- **VERDICT B — no incremental value.** `postflip_accuracy ≤ ceiling + h`. Consequence: the fifth
  negative, and the one measured on a channel chosen specifically because a blind rule *cannot*
  solve it. The data verdict stands and the bottleneck is demonstrations, not the model. **No
  further offline metric on this corpus is pre-registered after B** — the corpus will have been
  asked its one remaining question.
- **VERDICT C — the model wins, but so does something blind.** `postflip_accuracy > ceiling + h`
  while a *zero-parameter* rule also exceeds `ceiling`. That is a statement about the ceiling being
  mis-estimated, not about the model; the ceiling is re-derived and PR-03 is re-run before anything
  is claimed.

Ties (`|postflip_accuracy − ceiling| ≤ h`) are B. The burden is on the model.

## What this cannot decide, stated before it is tempting

- **Not grasp success.** The channel is the demonstrator's commanded aperture. Beating the ceiling
  means predicting *when the operator closed the hand*, from pixels, ~0.3–0.5 s ahead.
- **Not a robot claim.** Offline, one task, 402 success-only episodes, no failures anywhere in the
  corpus. E3 is the only thing that speaks to hardware.
- **Not "vision works".** The gripper channel is read off the same proprioceptive stream a
  closed-loop policy observes, so a policy that merely tracks the operator's hand gets the first
  ~6 steps of every chunk free. Only the flip is evidence, which is why only the flip is scored.
- **The ceiling is a lower bound**, not a proven optimum. 70.91 % means "at least this much is
  blind" — a better blind model would move the bar up, and two independent blind families
  (8-lag RFF, 11-lag RFF over 1.67 s) already landed in the same place, which is evidence but not
  proof.

## Consequences fixed in advance

Whatever the outcome, the I-8 rung files (`configs/splits/i8_train_*.txt`) are **stale against this
holdout** — `scripts/widen_holdout.py` prints which and by how much (40 / 110 / 110 episodes of
overlap). They were generated against the 40-episode holdout and must be regenerated before T-32
runs against PR-03's split. Neither experiment may silently adopt the other's.
