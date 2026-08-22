# PR-08 V6 — the mask-validity filter, and which frames §4 and §6 are measured on

**Rule `T40_RULE_V6`. Registered 2026-08-22, before the corrected estimator is used to produce any
gate number — `GEOM_TOL` and `EST_DRIFT_P95` are both still `null` in
`configs/transfer25/pr08_geom_tol.json` as this is written.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), which is registered as
`T40_RULE_V1` and **has not been edited and must not be**. The repo's discipline is
`docs/handoff.md` §3 — *"Rules are versioned, never edited in place. A gate rewritten after seeing
its output is not a gate."* V6 is that versioning, not a revision. `T40_RULE_V2` (arm C frame
matching), `T40_RULE_V3` (seed schedule), `T40_RULE_V4` (the T-39 gate premise) and `T40_RULE_V5`
(the MuJoCo ground-truth route, registered separately the same day) stand unchanged; **V6 depends on
none of them and changes nothing in any of them.**

Task: [[T-040]]. Generator: **Cosmos-Transfer2.5, frozen**. Estimator adapter:
`scripts/estimators/apple_sam2.py`.

**Nothing in this document licenses generation, training, or any statement of a result.**
`T40_RULE_V1` §1's prohibition is untouched and still binds in full.

---

## 0. What V6 does not change

Stated first and exhaustively, because a V6 that quietly moves a threshold is the failure the
versioning discipline exists to prevent.

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, not coined. V6 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` | still **derived** — median per-step object-centroid displacement in the source clips, computed and committed before generation. V6 changes no part of that definition |
| `EST_DRIFT_P95` | still the **p95 of the object-centroid displacement in pixels** between the estimated and the true segmentation (§4 step 4), still subtracted from G0b's budget, still recorded as a **lower bound** on the real error, and a G0b margin that only clears under a lower bound is still not a pass |
| **§4's five steps** | unchanged, all of them. Step 0 (annotators), step 1 (Isaac renders), step 2 (the same estimator and the same segmenter on RGB only), step 3 (absolute depth error and centroid displacement), step 4 (the p95 is the budget) |
| **§4's stated weakness** | unchanged — Isaac frames are not real frames and the estimator's error on synthetic renders is plausibly optimistic. V6 adds a *second* threat to validity (§5) and removes none of the first |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still VOID |
| **G0b** geometry invariance | unchanged — the generator is held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, generation does not start. V6 moves neither term and changes neither the subtraction nor the ≤ 0 rule |
| **G0c** embodiment | unchanged — real robot pixels unconditionally composited back; robot-mask IoU recorded as diagnostic, never as a gate. `scripts/robot_composite.py` reaches this adapter's detector and predictor directly and **does not call `segment()`**, so the filter does not touch G0c at all |
| **Ladder** | unchanged — L1 `skill_vs_repeat_pct > 0`, L2 `ci_skill_vs_repeat_pct > 0` (`ci_` = task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (§6) | unchanged in every cell, including that P requires *both* B − A ≥ floor *and* B − C ≥ floor, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P |
| **The headline** | unchanged — `skill_vs_repeat_pct` on `EVAL_STYLES`, arm B against arm A, with arm C deciding attributability |
| **The four arms** (§5) | unchanged, including V2's frame-matched arm C and V3's seed schedule |
| **§1's prohibition** | unchanged and still binding — nothing is generated, no weight is trained on generated frames, and no number from PR-08 is quoted as a result |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a P is a claim about held-out *generated* appearance and licenses exactly one thing |
| **§8's items** | unchanged. Item 4 (GEOM_TOL and EST_DRIFT_P95 measured and committed) is still **open**; V6 does not close it and produces no number toward it |
| **The committed style partition** | `configs/transfer25/pr08_style_partition.json`, rule `T40_STYLES_V1`, `source_content_sha256 = 4da3875d0c76e9b23821c1ca9fe20f965f9fc0867edcd085fb792b587a680da8`. V6 changes **no style, no id, no slug and no prompt string**, and therefore does not change that hash |
| **The detection operating point** | unchanged and **must remain** unchanged: `box_threshold = 0.15`, `text_threshold = 0.25`, one retry at `(0.10, 0.10)`, highest-scoring box, prompt `"apple."`. These are Cosmos-Transfer2.5's own numbers, read off its `sam2_model.py`, which is what §4 step 2 asks for |
| **The three checkpoints and their pinned revisions** | unchanged — `IDEA-Research/grounding-dino-base@12bdfa31…`, `facebook/sam2-hiera-large@e6a8e880…`, `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf@d2fc6a93…` |
| **`GATE_QUALIFIED`** | still `False`. V6 flips nothing, and §7 below says why producing a fix is not accepting one |
| **`GATE_QUALIFICATION_BLOCKERS`** | **not edited by V6.** All three entries stand verbatim, including blocker 3 (per-frame segmentation vs upstream's `SAM2VideoPredictor` propagation), which is untouched by everything here |
| **`GATE_QUALIFICATION_DISCHARGED`** | **not edited by V6.** Nothing was moved into it |

V6 changes exactly one thing: **which frames `GEOM_TOL` and `EST_DRIFT_P95` are measured on.**
Nothing else.

---

## 1. The finding

### 1.1 Two independent runs, the same defect

| | local CPU audit | cluster job **189637** |
|---|---|---|
| frames segmented | 169 | 382 |
| episodes | — | 24 of 402 |
| artifact | — | `/valhalla/…/runs/pr08-mask-audit/MASK_AUDIT.json`, sheets pulled to `runs/pr08-mask-audit/sheets/` |
| defect found | yes | yes, **12 frames** |

**Twelve of 382 frames carry a confident, well-formed mask of the PLATE.** They are not ragged, not
empty and not obviously broken; they produce a centroid, a displacement and a p95 that all look
exactly like measurements. All twelve are in `episode_000094`.

| | correct masks (370) | plate masks (12) |
|---|---|---|
| plate overlap fraction | 0.00 | **0.97 – 0.98** |
| IoU vs the colour heuristic | 0.7492 – 0.9838 | **0.0000, all twelve** |
| detection score | p05 0.362, median 0.829, max 0.917 | 0.167 – 0.309 |
| mask area (px) | median 6 185, p95 8 194 | **30 892 – 31 151** |

A person has looked at the contact sheets `runs/pr08-mask-audit/sheets/flagged-00.png` and
`flagged-01.png`. The green mask is the plate.

### 1.2 Three facts that shape the fix, and each rules something out

**(a) The retry is not the cause.** `n_frames_retry_fired = 0` on **both** runs. The adapter's
second gate-qualification blocker names the `(0.10, 0.10)` retry as the mechanism by which a weak
detection could be bought on an occluded frame; on this evidence it never fired. **These masks were
bought by the PRIMARY `BOX_THRESHOLD = 0.15`.** Removing or tightening the retry would therefore fix
nothing, and would make this a different segmenter from the generator's (§0, last-but-two row).

**(b) The segmenter oscillates between the two objects.** In `episode_000094`:

```
f00149  plate  (30 913 px, IoU 0.00)
f00150  apple  (   471 px, IoU 0.78)
f00151  apple  (   670 px, IoU 0.85)
f00152  plate  (30 892 px, IoU 0.00)   <- recorded adjacent-frame step: 245.87 px
```

So the corruption hits **both tails at once**: near-zero displacements for as long as the mask is
locked on the stationary plate, and a spurious ~246 px jump at every switch. `EST_DRIFT_P95` is a
p95 — a tail statistic — and `GEOM_TOL` is a median of per-step displacements, so one defect
contaminates both terms of §6's subtraction in different directions.

**(c) It is not only an occlusion problem.** In `f00152` the apple is plainly visible: 877 warm
pixels, outlined by the colour heuristic in the contact sheet, and the mask is still on the plate.
The refusal below therefore cannot be justified as "we only drop frames where the object is hidden",
and §5 does not try to.

### 1.3 What the defect does to a gate number, if nothing is done

Nothing crashes. Coverage stays at 1.0 — a box was returned on every frame. The plate's centroid is
a perfectly good centroid, the displacements are perfectly good floats, and the p95 they produce is
subtracted from `GEOM_TOL` to form the tolerance the restyled corpus is held to. This is the failure
mode the adapter's first gate-qualification blocker names in as many words: *"a plausible mask on
the wrong object (the plate, the hand, the whole tabletop) which produces a centroid, a displacement
and a p95 that all look like measurements."*

---

## 2. What V6 changes, precisely

> **A mask that contains essentially none of the object it claims to be is not that object, and is
> refused.**

Concretely, in `scripts/estimators/apple_sam2.py`:

1. `segment(rgb)` computes a second, non-learned opinion about where the fruit is —
   `object_color_reference(rgb)`, the warm-and-saturated predicate `r > 90 ∧ r − b > 50 ∧
   saturation > 0.35`, which is `build_identity_calibration.apple_mask`'s own discriminator and the
   one `probe-scan` measured all 154 447 frames of this corpus with.
2. If the mask SAM 2 drew is non-empty and its **IoU against that reference is below
   `MASK_VALIDITY_MIN_IOU = 0.10`**, `segment` returns an **all-False mask** and counts the frame in
   `MASK_REFUSED_FRAMES`.
3. Both PR-08 §4 harnesses already treat an all-False mask as a frame with no measurable centroid:
   `measure_geom_tol.centroid_of_mask` returns `None` and `measure_est_drift` drops the step and
   counts it into `coverage`. **Neither harness is changed by V6** — verified against both call
   sites before relying on it.
4. A refused frame therefore counts against each harness's coverage floor
   (`DEFAULT_MIN_COVERAGE = 0.90`, the same number `run_g0_gates` borrows for G0b). **That is the
   fail-closed property this fix depends on:** if the filter ever fires at scale, the run is
   disqualified loudly rather than producing a quietly narrower — or quietly wider — number.

**The scope of the change is therefore exactly this: which frames enter the population that
`GEOM_TOL` and `EST_DRIFT_P95` are computed over.** It is applied **identically to both**, by living
in the one module both harnesses call, which is what keeps §4 step 2's "the same segmenter" true of
the filtered measurement as well as of the unfiltered one.

### 2.1 Measured effect on the audited frames

The number the audit already records as `warm_apple_iou` **is** the number the filter computes — the
same predicate, the same frame, the same IoU — so the before/after is arithmetic on the committed
artifact rather than a prediction:

| | before | after |
|---|---|---|
| frames measured | 382 | 370 (**12 refused**, exactly the twelve plate frames) |
| refusals where the fruit was not visible at all | — | **0 of 12** (fact (c) above) |
| mask area, max | 31 151 px | 8 902 px |
| adjacent-frame centroid step, max | **245.87 px** | 58.91 px |
| adjacent-frame centroid step, mean | 5.22 px | 1.88 px |
| adjacent-frame centroid step, p95 | 5.08 px | 4.90 px |
| measurable adjacent-frame steps | 143 | 136 (7 dropped, both endpoints required) |

The surviving 58.91 px step (`episode_000073` f00104, mask 2 328 px, IoU 0.90 — a *correct* mask
that moved) is **still flagged** by the audit rig. That is the intended behaviour: the filter
removes a defect, it does not quieten the instrument that finds defects.

---

## 3. Why this is not a segmenter change, which §4 step 2 would forbid

§4 step 2 requires the estimator to be *the same segmenter*, and this project takes the strong
reading: the same one **the generator** will use, so that the drift we budget for is the drift the
generator actually commits. A number of ours in the detection path is exactly what that forbids.

The filter is not in the detection path.

| | before V6 | after V6 |
|---|---|---|
| detector, checkpoint, revision | GroundingDINO `@12bdfa31…` | identical |
| segmenter, checkpoint, revision | SAM 2 hiera-large `@e6a8e880…` | identical |
| text prompt | `"apple."` | identical |
| `box_threshold` / `text_threshold` | 0.15 / 0.25 | identical |
| retry | exactly one, at `(0.10, 0.10)`, only on "no box at all" | identical |
| box selection | highest score | identical |
| the box SAM 2 is prompted with, on every frame | upstream's | identical |
| **the mask drawn, pixel for pixel** | — | **identical: no mask is altered, ever** |
| what changes | — | **whether we are willing to measure on this frame** |

Two consequences worth stating rather than leaving to be inferred:

- **The generator still draws the plate mask.** V6 does not improve Cosmos-Transfer2.5's
  conditioning; it has no access to it. What it fixes is our *measurement* of the generator's mask
  error, which was being computed partly from frames where our own copy of the segmenter was
  looking at the wrong object. Whether the generator's own propagation-based segmenter makes the
  same mistake is a separate, open question — it is blocker 3, and V6 does not touch it.
- **The fix is deliberately NOT "raise `BOX_THRESHOLD` above 0.309".** That would exclude the twelve
  plate masks by making our detector stricter than the generator's, which does not improve the
  budget; it makes it a budget for an error nobody commits. The rejected fix is recorded here so it
  is not proposed again as an obvious simplification.

---

## 4. The threshold, and why its exact value does not matter

A threshold introduced into a gate path is a number somebody chose, and this repository's rule is
that the choice of a number must not be able to become the finding. The defence here is not that
0.10 was chosen carefully. It is that **the value is irrelevant over a range two-thirds of the unit
interval wide**, and that this is checkable.

Over the 382 audited frames the two populations do not overlap and are nowhere near overlapping:

```
plate masks   :  IoU = 0.0000   (all 12; max = 0.0000)
correct masks :  IoU ≥ 0.7492   (all 370; min = 0.7492)
```

**Every cut in the open interval (0.0000, 0.7492) produces the identical partition of those
frames.** 0.10 is a value read off a gap, not a value tuned against an outcome.

That claim is **checked, not asserted**:
`tests/test_apple_sam2_estimator.py::test_every_threshold_in_the_gap_partitions_the_audited_frames_identically`
sweeps the threshold from 0.01 to 0.74 in 1 pp steps and asserts, at every one of the 74 values,
that the set of refused frames is exactly the twelve a person flagged as the plate in the contact
sheets — an identity fixed by the flag, not recomputed from the threshold. A companion test asserts
that the value the module ships sits strictly inside that range. The 382 audited IoUs are embedded
in the test file, because `runs/` is not tracked and a test that skips when an artifact is missing
is not a test.

**What would make the threshold matter, and what happens then.** A corpus, a restyle or a render on
which the two populations were *not* separated by a gap. That is a measurable condition, not a
hypothetical: the per-frame IoUs are recorded (§6), so a distribution with mass between 0.1 and 0.7
is visible in any artifact this adapter writes. **If that ever occurs, the correct response is a
further version alongside this one, not a threshold moved inside this one.** `MASK_VALIDITY_MIN_IOU`
has deliberately **no environment override** — every other knob in that module has one — so it
cannot be changed for a single run without editing the file, the committed contract and a
pre-registration together.

---

## 5. Threat to validity — this errs in the generator's favour, and that is recorded

**Refusing a frame where the object cannot be found removes a HARD frame from the population.** That
is the honest cost of this fix and it is not glossed here.

### 5.1 The direction of the bias

`EST_DRIFT_P95` is a **p95** — a tail statistic — and it is **subtracted**: G0b holds the generator
to `GEOM_TOL − EST_DRIFT_P95`. The hard frames (heavy occlusion, the fruit at the frame edge, the
instant of the grasp) are precisely the frames where a per-frame estimator's centroid is furthest
from the truth, i.e. they are over-represented in that tail. Removing them plausibly makes
`EST_DRIFT_P95` **smaller**, which makes the tolerance **wider**, which makes G0b **easier for the
generator to pass**.

That is the unsafe direction. It is stated in advance so that it cannot be discovered afterwards and
read the convenient way.

Two things bound it, neither of which is a defence:

- **On the audited sample this effect is zero by measurement, not by argument.** All twelve refusals
  had a visible fruit (54–877 warm px); **none** was a "the object cannot be found" refusal. On this
  evidence the filter removed only wrong masks. The sample is 382 frames of a 171 600-frame corpus
  and deliberately over-weights the hard cases, so it is *stronger* evidence about hard frames than
  a uniform sample would be — and it is still not the corpus.
- **The effect is counted separately, everywhere.** `n_frames_mask_refused_no_reference` is the
  sub-case where the colour reference found no fruit anywhere in the frame. It is in `stats()`, in
  both harnesses' `estimator_stats` blocks, and per frame in the audit artifact. Its being non-zero
  is not by itself a defect; its being non-zero and **unread** would be.

### 5.2 What would measure it

Stated concretely, so this is a claim someone can settle rather than a caveat:

1. **Measure `EST_DRIFT_P95` both ways on the same Isaac capture** — once with the filter and once
   with it disabled — and record the two p95s, the two coverages and the count of refused frames.
   The difference between them **is** the size of this bias on the frames that set the budget. This
   costs one extra pass over a few hundred calibration frames and needs no new code beyond a
   temporary constant, and it should be done **before** either number is committed.
2. **For the refused frames specifically, compare against Isaac's ground-truth mask.** On the
   `EST_DRIFT_P95` side there *is* ground truth, so "the filter refused a frame whose mask was
   actually correct" is directly measurable rather than inferable. A refusal rate that is high while
   the refused frames' true displacement is small would mean the filter is discarding easy frames,
   which is the opposite failure and would show up here.
3. Both numbers belong in the artifact that commits `EST_DRIFT_P95`, beside the existing
   `is_lower_bound: true`.

### 5.3 A second exposure: the reference is a property of the SOURCE corpus's appearance

The colour predicate describes AppleToPlate's real apple under AppleToPlate's real lighting. Two
places drive this adapter over pixels that are **not** that:

- **The Isaac calibration renders** (§4 steps 1–2), where the apple is synthetic.
- **The restyled clips** — G0b runs the same segmenter over the generator's output, and PR-08 is a
  *photoreal augmentation* pre-registration whose entire point is to change how the scene looks.

If the reference does not fire on those pixels, the filter refuses everything and `coverage`
collapses. **That fails CLOSED** — a coverage-floor breach is a loud refusal in both harnesses and
in `run_g0_gates`, not a quietly wider tolerance — and the reason it is legible rather than merely
loud is `n_frames_mask_refused_no_reference`, which distinguishes "the segmenter is wrong on this
corpus" from "the reference does not fit this corpus". Before either number is committed, a short
pass over Isaac renders and over a restyle pilot clip must record that counter; a high value there
is a finding about the reference and calls for a further version, not for a lowered threshold.

---

## 6. What is recorded, and where

Because a filter whose firing is not recorded is indistinguishable from a corpus that never
triggered it.

| | where |
|---|---|
| `n_frames_mask_refused`, `n_frames_mask_refused_no_reference` | `apple_sam2.stats()`; differenced per run into `estimator_stats.this_run` by `measure_geom_tol.EstimatorStatsProbe`, which `measure_est_drift` imports and uses unchanged |
| the per-frame validity IoU | `apple_sam2.MASK_VALIDITY_IOU`, raw values in call order, one per frame the check ran on — the same design as `DETECTION_SCORES`, for the same reason: raw values pool exactly through JSON where two histograms only pool if they were binned identically |
| the refusal, **per frame** | `scripts/audit_apple_masks.py` records `mask_refused`, `mask_refused_no_reference` and `mask_validity_iou` on every frame's record, and flags refused frames. So a future audit can show the filter fired **on exactly the frames a person flagged**, rather than reporting a count that has to be taken on trust |
| the run's totals | `MASK_AUDIT.json` → `mask_validity_filter`, including the validity-IoU distribution and a `present` flag that is `false` against an adapter predating the filter (zeros that mean "no such mechanism" must not read as "it never fired") |
| the threshold and the reference | `apple_sam2.SEGMENTER_CONTRACT` **and** the committed `configs/transfer25/pr08_geom_tol.json` (`segmenter.mask_validity_min_iou`, `segmenter.mask_validity_reference`), sha256 sidecar updated |
| the filter's presence, in one string | `ESTIMATOR_VERSION` gains `mask_val_min_iou=0.1` |

**Why the committed contract had to move too.** `measure_est_drift.cross_check_geom_tol` compares
the adapter's contract against the committed one **field for field**, and
`contract_disagreements()` counts a field present on one side and absent on the other as a
disagreement. Recording the filter there is what makes the dangerous case refusable: a `GEOM_TOL`
measured **with** the filter minus an `EST_DRIFT_P95` measured **without** it is a subtraction across
two different frame populations, and it would otherwise still look like arithmetic. The
pre-commitment was amended before either number existed — both `geom_tol_px` and `est_drift_p95_px`
are `null` in that document as V6 is registered — which is the only point at which amending it is
legitimate at all.

**The audit's flagging is not weakened.** `mask_refused` is additive: every pre-existing flag still
applies to a refused frame, and `disagrees_with_warm_apple` in particular — the fruit is plainly
visible and the returned mask is nowhere near it — is exactly what a *wrong* refusal would look like
from the audit's side. A new consistency check makes an all-False mask with **no** recorded reason a
defect in the audit rig: there are exactly three reasons (no detection, empty mask, validity
refusal) and a fourth would be a step silently dropped from every coverage number downstream.

---

## 7. What V6 does not discharge

**`GATE_QUALIFIED` is still `False`, and V6 does not touch it.**

- **Blocker 1 (NOBODY HAS LOOKED AT A MASK) is not discharged here.** Producing a fix and accepting
  it are different acts. Someone has now looked, and what they saw is the finding in §1 — but the
  discharge is a person's reviewable edit to `GATE_QUALIFICATION_BLOCKERS` citing the audit
  artifact, moving the retired wording into `GATE_QUALIFICATION_DISCHARGED` rather than deleting it.
  No session may make that edit on the strength of having written the fix.
- **Blocker 2 is not discharged.** It asks for the detection-score distribution and retry counts
  **from a full pass**. §1 is a 382-frame sample that deliberately over-weights the hard frames, and
  its rates are not corpus rates.
- **Blocker 3 is untouched, in both directions.** Per-frame segmentation versus upstream's
  `SAM2VideoPredictor` propagation is unaffected by anything here, and nothing here may be read as
  reducing it. Note in particular that the filter does *not* address propagation's characteristic
  failure — drifting off the object and staying off — because a per-frame estimator does not exhibit
  it; the two-sided bias argument in that blocker stands exactly as written.
- **§8 item 4 is still open**, and V6 produces no number toward it.
- **`T40_RULE_V1` §1's prohibition still binds in full**: nothing is generated, no weight is trained
  on generated frames, and no number from PR-08 is quoted as a result.

**V6 licenses nothing.**

---

## 8. Provenance

| | |
|---|---|
| rule | `T40_RULE_V6` |
| registered | 2026-08-22, before the corrected estimator produced any gate number |
| supersedes | nothing. It **supplements** `T40_RULE_V1`, which stands and is unedited |
| changes | which frames `GEOM_TOL` and `EST_DRIFT_P95` are measured on (§2), applied identically to both |
| decided by | the project owner, 2026-08-22 — a mask-validity filter, approved as a PR-08 amendment |
| evidence | cluster job **189637** (382 frames, 24 episodes, `MASK_AUDIT.json`, contact sheets `flagged-00.png` / `flagged-01.png`) and an independent 169-frame local CPU audit; both found the same 12-frame defect |
| threshold | `MASK_VALIDITY_MIN_IOU = 0.10`, inside the measured gap (0.0000, 0.7492); insensitivity swept and asserted in `tests/test_apple_sam2_estimator.py` |
| touched | `scripts/estimators/apple_sam2.py`, `scripts/audit_apple_masks.py`, `scripts/measure_geom_tol.py` (`ADAPTER_RUN_COUNTERS` only), `configs/transfer25/pr08_geom_tol.json` (+ sha256 sidecar), the two test files |
| not touched | `GATE_QUALIFIED`, `GATE_QUALIFICATION_BLOCKERS`, `GATE_QUALIFICATION_DISCHARGED`, the detection operating point, the checkpoints, the style partition, any gate, any verdict |
| partition | `configs/transfer25/pr08_style_partition.json`, `T40_STYLES_V1`, sha256 `4da3875d…a680da8` — **unchanged** |
| generation licensed | **no** |
| training licensed | **no** |
