# The gripper channel — recoverable, and worth recovering, but only under one metric

**Measured 2026-08-02**, CPU only, raw parquet only, no video decoded, nothing re-converted, no
allocation spent. Follow-on from `PR-01-FOLLOWUP.md` §4, which established that the gripper is dead
in all 402 converted episodes and alive in the source. The open question there was the one that
decides whether a re-conversion is worth doing: **is a restored gripper channel a skill metric, or
another momentum metric?**

**The answer is both, depending on where you score it — and that is the whole result.**

## Scored on the full holdout it is another momentum metric. Do not use it.

Restored `active-hand` channel, 1040 holdout chunks, accuracy at the 0.5 threshold:

| predictor | fitted params | MSE | accuracy |
|---|---:|---:|---:|
| constant (train mean 0.7114) | 0 | 7.184765e-02 | 78.12 % |
| repeat-last-gripper | **0** | 9.851521e-03 | 95.87 % |
| **gripper const-velocity, lag 1** | **0** | **6.718439e-03** | **97.82 %** |
| ridge, 32-dim state | 32 | 8.515384e-03 | 96.02 % |
| blind nonlinear, state + history | 2048 | 4.427746e-03 | 98.00 % |

A rule with **zero fitted parameters covers 90.0 % of the accuracy range and 90.6 % of the MSE
range**, and the full blind ceiling adds 0.9 accuracy points on top. The clock alone — a
time-in-episode-only model, no state at all — covers ~51 % of the accuracy range.

That is PR-01's failure re-run in a new channel. **Full-holdout gripper accuracy and gripper chunk
MSE must never be pre-registered as a metric.**

## Scored at the flip it is not. This is where the signal is.

78 of 1040 holdout chunks contain a debounced transition (7.50 %, over 39 of 40 episodes). Aligning
to the flip index rather than the step index — which is what makes the shape legible:

| predictor | fitted params | pre-flip | **post-flip** | k…k+3 | ≥ k+4 |
|---|---:|---:|---:|---:|---:|
| repeat-last-gripper | 0 | 77.55 % | **19.70 %** | — | — |
| gripper const-velocity | 0 | 84.35 % | **62.88 %** | — | — |
| ridge, 32-dim state | 32 | 77.72 % | **24.09 %** | — | — |
| time-in-episode only | — | — | **39.09 %** | 33.68 % | — |
| **blind nonlinear ceiling** | 2048 | 85.20 % | **70.91 %** | **52.98 %** | 84.53 % |

The shape is a **V, not a decay**: everything is easy except the ~4 steps around the flip, where
*every* blind predictor is at a coin toss (52.98 %). The two zero-parameter rules and the linear
ridge are all *below chance* post-flip (19.70 %, 24.09 %) because they are structurally committed to
the stale class. **44 % of the accuracy range at the flip is not blind-reachable.** That is real
headroom, and it is the only real headroom found anywhere in this investigation.

Two controls, both of which came out in favour of the metric:

- **Not stereotypy.** Train P(transition | decile) peaks at 23.5 % and is ≤ 1.1 % in five deciles,
  so grasp timing is loosely localized — but on transition chunks a time-only model gets 52.88 %
  and post-flip 39.09 %, and adding time to the blind ceiling moves it +0.48 points. The metric is
  not measuring demonstration uniformity.
- **Not proprioception left on the table.** A tuned RFF over 11 lags out to 1.67 s of `q`, `dq` and
  gripper (341 extra dims) reaches 65.00 % post-flip — *worse* than the 8-lag model. Two independent
  blind families land in the same place, so the gap at the flip is not something a better blind
  model closes.

## The power problem, which is the reason not to rush

The evidentiary base is **660 post-flip target steps of 16 640** (3.97 %), across 78 chunks and 39
episodes. Episode-bootstrap CI half-width on the ceiling is **7.56 points** at post-flip and 8.33 at
k…k+3 — so a vision model must beat 70.9 % by roughly **15 points** to be distinguishable. Against a
checkpoint PR-01 measured at 2.2 % total incremental value, that is the bet already lost.

Two corrections to how the numbers were first reported, both mine to pass on:

- The base rate depends on the definition and the most generous one was quoted without saying so:
  **53–78 chunks** across three debounce definitions (episode-latch 78, self-contained 60, label-
  steps-only 53). Quote the range.
- The first CI (±4.0) was for a blended all-16-steps number, not for the post-flip sub-metric being
  proposed. The real half-width is ±7.6 — optimistic by about 2×.

## Recommendation

1. **Re-convert** — `--out datasets/gr00t-apple-grip --gripper-mapping active-hand`. One CPU pass,
   887 MB of mp4 in, ~81 MB out. It writes a **new directory**; `datasets/gr00t-apple-full` is
   untouched and every archived `dataset_snapshot_ref` keeps verifying (AC-04). The provenance cost
   only exists if someone overwrites in place — don't.
2. **Widen the holdout in the same pass.** The metric is bottlenecked entirely by holdout size, not
   by training data: the fine-tune is not starved at 250 episodes, but 40 holdout episodes cannot
   resolve the effect. Holding out **150** yields ~290 transition chunks and ~2 500 post-flip steps,
   halving the CI to ~±3.7 and bringing the minimum detectable effect from ~15 points to ~7.
   Convert once, re-split, refit once.
3. **Pre-register the metric before the refit**, as accuracy on post-flip steps of transition
   chunks, carrying all five blind controls (repeat-last 19.70 %, const-velocity 62.88 %, ridge
   24.09 %, time-only 39.09 %, ceiling 70.91 %) and stating n and the CI half-width on its face.
4. If a widened holdout still cannot resolve 7 points, **the answer is more demonstrations, not more
   analysis.**

## What this does not establish

That the model would win. The metric measures anticipating the demonstrator's grasp timing ~0.3–0.5 s
ahead from pixels — not grasp success, and not anything about a real robot. The channel is read off
the same proprioceptive stream a closed-loop policy observes, so a policy that merely tracks the
operator's hand gets the first ~6 steps free; only the flip discriminates.

The blind ceiling is a **lower bound** on what no-vision can do, not a proven optimum. 70.91 % means
"at least this much is blind".
