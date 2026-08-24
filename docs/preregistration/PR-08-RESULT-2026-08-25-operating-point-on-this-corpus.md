# PR-08 §4 — what the committed operating point actually does on AppleToPlate

**Measured 2026-08-25 from artifacts already on disk. No new job was run, no model was loaded, and
no parameter was changed.** This document reports the **detection-score distribution over a full
pass** and the **retry counters** that `GATE_QUALIFICATION_BLOCKERS`'s second entry names as part of
its own discharge condition.

**It discharges nothing.** See §5. Producing evidence and discharging a blocker are two different
acts, and only the second is a judgement.

---

## 0. What this is and is not

| | |
|---|---|
| answers | the *full-pass* half of blocker 2: "the recorded detection-score distribution and retry counts (`n_frames_retry_fired` / `n_frames_retry_recovered`) from a full pass, so the retry's contribution is visible rather than assumed" |
| does not answer | blocker 2's other half — "discharged by the same evidence as blocker 1", i.e. the human look. Blocker 1 is untouched by this document |
| does not answer | blocker 3 (per-frame vs propagation). `mask_method.version` records `prop=per_frame` on all 16 shards; nothing here compares the two |
| changes | no code, no config, no gate, no committed artifact |
| licenses | **no clip, no training run.** `T40_RULE_V1` §1 binds in full |

---

## 1. The question blocker 2 asks

`scripts/estimators/apple_sam2.py` sets `GATE_QUALIFIED = False`. Its second blocker accepts that
`box_threshold=0.15` / `text_threshold=0.25` is **correct** — it is Cosmos-Transfer2.5's own
operating point, read off its `sam2_model.py`, which is what §4 step 2 asks for — and says the
choice-defect half is therefore discharged and inverted. What survives is not a choice, it is an
unknown, in the blocker's own words:

> *"nothing has measured what this operating point does on THIS corpus, and the retry at (0.10,
> 0.10) buys detections by accepting weak ones, which on an occluded frame can replace an honest
> all-False mask with a confident box on the wrong object. That inflates coverage while degrading
> the mask, i.e. it hides itself in the one number the harness gates on."*

That is a **falsifiable prediction about a rate**, and the GEOM_TOL full pass recorded the numbers
that test it. They had simply never been read.

---

## 2. What was measured

Source: the 16 shard artifacts of the GEOM_TOL corpus pass,
`runs/pr08-geom-tol/shards/shard-{0..15}.json`. Every shard carries a `detection_scores` array of
one score per frame per episode, and an `estimator_stats.this_run` counter block.

- 16/16 shards present, 402/402 episodes, **171 625 frames**, no episode missing scores.
- All 16 shards report an **identical** `mask_method.version` string, so this is one operating point
  and not a mixture:
  `det=grounding-dino-base@12bdfa31; seg=sam2-hiera-large@e6a8e880; depth=Depth-Anything-V2-Metric-Indoor-Large@d2fc6a93; prompt='apple.'; box_thr=0.15; text_thr=0.25; retry_box_thr=0.1; retry_text_thr=0.1; box_sel=highest_score; prop=per_frame; mask_val_min_iou=0.1`

### 2.1 The retry

Summed over all 16 shards:

| counter | value |
|---|---|
| `n_segment_calls` | 171 625 |
| `n_detection_scores` | 171 625 |
| **`n_frames_retry_fired`** | **0** |
| **`n_frames_retry_recovered`** | **0** |
| `n_frames_without_detection` | 0 |
| `n_frames_with_empty_mask` | 0 |
| `n_frames_mask_refused` | 36 |
| `n_frames_mask_refused_no_reference` | 0 |

**The (0.10, 0.10) retry never fired, on any frame of the corpus.** The retry branch is reached only
when the primary pass returns no box at all; the primary pass returned a box on all 171 625 frames.

So the specific mechanism blocker 2 describes — *the retry* buying weak detections — **did not
occur, and cannot have contributed to any number in this project.** That is a stronger statement
than "we measured it and it was small".

### 2.2 The score distribution

| statistic | score |
|---|---|
| p1 | 0.5489 |
| p5 | 0.7332 |
| p25 | 0.8204 |
| **median** | **0.8497** |
| p75 | 0.8742 |
| p95 | 0.9002 |
| p99 | 0.9117 |

Partitioned at the two operating points:

| band | frames | share | reading |
|---|---|---|---|
| score ≥ 0.35 | 171 533 | **99.946 %** | would have been detected at the OLD threshold too |
| **0.15 ≤ score < 0.35** | **92** | **0.054 %** | **exactly the frames the operating-point change bought** |
| score < 0.15 | **0** | 0.000 % | rejected at both operating points |

**The operating-point change from 0.35 to 0.15 altered the outcome on 92 frames out of 171 625.**

### 2.3 The 92 frames are not spread over the corpus

| episode | frames | in-band | share of episode |
|---|---|---|---|
| **`episode_000094`** | 509 | **52** | **10.2 %** |
| `episode_000112` | 415 | 8 | 1.9 % |
| `episode_000244` | 469 | 5 | 1.1 % |
| `episode_000264` | 427 | 4 | 0.9 % |
| `episode_000093` | 448 | 4 | 0.9 % |
| …27 further episodes | | 1–3 each | |

**57 % of the weak detections in the entire corpus are in one episode**, and within it they form a
contiguous run — frames ~101 to ~155 score 0.16–0.31 while the episode's own median is normal.

### 2.4 Only one episode of 402 lost a frame

`n_frames != n_frames_with_centroid` holds for **exactly one** episode:

| episode | `n_frames` | `n_frames_with_centroid` | gap |
|---|---|---|---|
| `episode_000094` | 509 | 473 | **36** |

`n_frames_mask_refused` summed over all shards is **36**. The two numbers are equal and the gap
occurs in exactly one episode, so **the 36 mask refusals of the entire corpus pass are that
episode's 36 frames.** This is arithmetic over the artifacts, not an inference from coincidence.

---

## 3. What this converges with, and the limit on that convergence

`runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json` (2026-08-22) independently reports that in
`episode_000094`, frames in the f109–f144 region carry *"a confident, well-formed mask of THE
PLATE, at scores 0.155–0.259, ~31 000 px, plate overlap 0.985–0.992, zero IoU with the colour
heuristic"*, on frames where the hand has hidden the fruit almost completely. The contact sheet
`runs/pr08-mask-audit-local-cpu/sheets/occluded-00.png` shows f00129 and f00130 as full-plate masks
at scores 0.213 and 0.155.

So three instruments — a visual audit, the corpus score distribution, and the refusal counter —
point at the same contiguous run of frames in the same one episode of 402.

**Three limits on that, stated rather than buried:**

1. **Different corpora.** The mask audit ran on the H.264-lossless tree; these shards ran on
   `data/pr08-apple-640x480` (the AV1 tree). The decodes differ, so the per-frame scores differ
   slightly — f129 reads 0.213 in the audit and 0.232 here. **The phenomenon replicates; the
   numbers are not the same numbers** and must not be quoted interchangeably.
2. **The exact 36 are not recorded.** The shard artifacts carry per-frame *scores* but no per-frame
   centroid-present flag, so "the 36 refusals are f109–f144" is **not** established here. What is
   established is that all 36 are in `episode_000094` and that the low-score run in that episode
   spans ~f101–f155.
3. **A correlated observer.** `MODEL_OBSERVATIONS.json` says so in its own header, and this document
   is written by a model of the same family. It is a finding, not the check.

---

## 4. What the evidence says about blocker 2's prediction

Blocker 2 predicted a mechanism: the retry accepts weak detections, coverage inflates, the mask
degrades, and the degradation hides in the coverage number.

On this corpus, **the mechanism did not operate and a neighbouring one did.**

- The retry fired **zero** times. It bought nothing.
- The **primary** 0.15 threshold bought 92 frames.
- On at least some of those frames the mask *is* a confident mask of the wrong object — §3's plate
  masks are exactly the failure blocker 2 describes, arriving by the primary threshold rather than
  by the retry.
- The failure is **not distributed**. It is one episode, one occlusion event, 0.054 % of the corpus.

**Both directions of that matter and neither should be dropped.** The rate is small and
concentrated, which is reassuring about the corpus. The failure mode is real and produces a
plausible measurement rather than an error, which is exactly why blocker 1 exists and is exactly
why a rate this small is not self-evidently safe: 36 refusals is what the instrument *caught*, and
nothing here establishes what it missed.

---

## 5. Why this does not discharge blocker 2, let alone blocker 1

Blocker 2's discharge condition is conjunctive:

> *"Discharged by the same evidence as blocker 1, **plus** the recorded detection-score distribution
> and retry counts … from a full pass."*

This document supplies the second conjunct and **not** the first. Blocker 1's condition is *"a human
looking at a sample of overlaid masks spanning the corpus … and/or a mask-vs-ground-truth IoU
distribution"*, and neither exists: the overlays exist and have been read only by models, and there
is no ground-truth IoU on this corpus at all.

**Two further reasons this cannot be treated as a discharge**, both about provenance:

- **`git_commit` is `null` on all 16 shards.** These artifacts were produced before the measurement
  path learned to stamp its commit, so the code that produced these numbers cannot be pinned. The
  `mask_method.version` string is identical across shards, which establishes that they agree with
  each other; it does not establish *which* revision they agree at.
- **The shards are dated 2026-08-23 (10) and 2026-08-24 (6).** They are a mixed-date series on the
  AV1 tree, which is the series whose staleness is recorded in the 2026-08-24 result document.

A discharge is an edit to `GATE_QUALIFICATION_BLOCKERS` moving wording into
`GATE_QUALIFICATION_DISCHARGED` with the evidence beside it. **No such edit is made here, and this
session does not make it.**

---

## 6. Provenance

| | |
|---|---|
| kind | measurement report over existing artifacts. **Registers no rule** |
| date | 2026-08-25 |
| measured from | `runs/pr08-geom-tol/shards/shard-{0..15}.json` (16/16, 402 episodes, 171 625 frames) |
| corroborated by | `runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json`, `sheets/occluded-00.png` — **on a different tree**, see §3 |
| new jobs run | **none** |
| code changed | **none** |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
