# PR-08 V10 — the mask-validity reference is defined for one label and one appearance, and now says so

**Rule `T40_RULE_V10`. Drafted 2026-08-23. UNSIGNED — see §8. Nothing here is in force until the
project owner signs it, and no number produced under it may be quoted before that.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), which is registered as
`T40_RULE_V1` and **has not been edited and must not be**. The repo's discipline is
`docs/handoff.md` §3 — *"Rules are versioned, never edited in place. A gate rewritten after seeing
its output is not a gate."* V10 is that versioning, not a revision. `T40_RULE_V2` (arm C frame
matching), `V3` (seed schedule), `V4` (the T-39 gate premise), `V5` (the MuJoCo ground-truth route),
`V6` (the estimator's mask-validity filter), `V7` (the consumer contract), `V8` (the hallucination
probe) and `V9` (the robot mask's object grounding) all stand unchanged. **V4 and V8 are signed and
are not touched here. V6 is not edited, weakened or superseded** — §0 lists every one of its terms
that survives, which is all of them.

Task: [[T-040]]. Generator: **Cosmos-Transfer2.5, frozen**. Adapter:
`scripts/estimators/apple_sam2.py`.

**Nothing in this document licenses generation, training, or any statement of a result.**
`T40_RULE_V1` §1's prohibition is untouched and still binds in full.

---

## 0. What V10 does not change

Stated first and exhaustively, because a version that quietly moves a threshold is the failure the
versioning discipline exists to prevent.

| | unchanged |
|---|---|
| **`MASK_VALIDITY_MIN_IOU = 0.10`** | V6's threshold. Not moved, not rescaled, not made per-label, not made per-style, and still with no environment override |
| **`MASK_VALIDITY_REFERENCE`** | the string `warm_saturated_rgb(r>90, r-b>50, saturation>0.35)`, byte for byte. V10 does not change the predicate, its three numbers, or `object_color_reference`'s implementation |
| **`SEGMENTER_CONTRACT`** | **not one field added, removed or changed**, and that is load-bearing rather than tidy: `measure_geom_tol.contract_disagreements` counts a field present on one side and absent on the other as a disagreement, so a new key here would disqualify every `GEOM_TOL` run against the committed `configs/transfer25/pr08_geom_tol.json`. That file is **not edited by V10** and its sha256 sidecar does not move |
| **The detection operating point** | `box_threshold = 0.15`, `text_threshold = 0.25`, one retry at `(0.10, 0.10)`, highest-scoring box, prompt from `$WAM_PR08_OBJECT_PROMPT`. Cosmos-Transfer2.5's own numbers, unchanged |
| **The three checkpoints and their pinned revisions** | unchanged — `IDEA-Research/grounding-dino-base@12bdfa31…`, `facebook/sam2-hiera-large@e6a8e880…`, `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf@d2fc6a93…` |
| **The mask drawn, pixel for pixel** | unchanged. Nothing is re-detected, re-prompted or re-drawn, and no returned mask is altered. V10's only possible effect on a frame is a **refusal**, and its only possible effect on a run is a **refusal** |
| **V6's three events and their counters** | `n_frames_without_detection`, `n_frames_with_empty_mask`, `n_frames_mask_refused` and its sub-case `n_frames_mask_refused_no_reference` all keep their meanings. The new counter is a **sub-case inside** `n_frames_mask_refused`, so `n_frames_without_detection + n_frames_with_empty_mask + n_frames_mask_refused` still spans the whole coverage shortfall this module is responsible for |
| **`MASK_VALIDITY_IOU`** | still one entry per frame the check ran on, still in call order, still satisfying `len == n_segment_calls − n_frames_without_detection − n_frames_with_empty_mask`. `scripts/audit_apple_masks.py` reads it unchanged |
| `MATERIAL_FLOOR_PP = 10.0` | still borrowed from `I8_RULE_V3`, not coined, not moved, not made per-arm |
| `GEOM_TOL`, `EST_DRIFT_P95` | still derived, still subtracted, still `null` in the committed document. V10 produces **no number toward either** |
| **G0a**, **G0b**, **G0c** | unchanged in every term. V10 adds no gate, moves no budget, and compares nothing to a pass/fail threshold |
| **`T40_RULE_V9`** | not edited and not superseded. `scripts/robot_composite.py` is **not touched by V10**; `ROBOT_MASK_OBJECT_MAX_IOU`, `ROBOT_TEXT_PROMPT`, `check_mask` and `max_frame_fraction` are all untouched. §5.4 below records that V9's masker is exposed to the same defect and that V10 does not fix it there |
| **`T40_RULE_V8`'s verdict** | job 189926's `H` stands exactly as measured. V10 licenses no re-reading of it and produces no number toward `human_review.looked_at`, which is still `false` |
| **The committed style partition** | `configs/transfer25/pr08_style_partition.json`, `T40_STYLES_V1`, sha256 `4da3875d…a680da8`. V10 changes no style, no id, no slug and no prompt string, and therefore does not change that hash |
| **The verdict table**, **the ladder**, **the four arms**, **the headline** | unchanged in every cell |
| **`GATE_QUALIFIED`**, **`GATE_QUALIFICATION_BLOCKERS`**, **`GATE_QUALIFICATION_DISCHARGED`** | not edited by V10. Nothing was moved into or out of any of them |

V10 changes exactly two things, and both are the same thing said twice: **which labels the
mask-validity filter will run for, and which frames it will claim to have decided.** Nothing else.

---

## 1. The finding

Both defects were recorded as **open** in `.mc/tasks/todo/T-040-*.md`'s 2026-08-23 entry, findings
(2) and (3), and as V9 §5.4 — *"a defect in `T40_RULE_V6`'s blast radius, found while writing this
and not fixed here"*. **Both were reproduced from scratch before anything was changed**, on this
workstation's own GPU against the local corpus and the local hub cache, because this project has
withdrawn a whole verdict's premise for trusting an instrument it did not re-derive (PR-12/PR-13).
Where a re-derivation disagrees with the prose it is corrected here rather than repeated.

### 1.1 Defect 1 — the `plate.` pass refuses 100 % of frames, on the SOURCE corpus

`scripts/run_g0_gates.py` documents §6's plate half in its own module docstring: the adapter takes
one text prompt per process, so *"object AND plate"* is two passes per side, the second with
`WAM_PR08_OBJECT_PROMPT="plate."`. That reaches `measure_geom_tol` and `module.segment(rgb)`.

Run on 20 source frames of `episode_000000`
(`~/wam-t041/pr08-apple-640x480/videos/episode_000000.mp4`, pyav, BGR→RGB as `sam2_mask_via` does),
with the adapter **unmodified**:

| | `plate.` | `apple.` — the same twenty frames |
|---|---|---|
| `n_segment_calls` | 20 | 20 |
| `n_frames_without_detection` | 0 | 0 |
| `n_frames_with_empty_mask` | 0 | 0 |
| **`n_frames_mask_refused`** | **20** | **0** |
| `n_frames_mask_refused_no_reference` | 0 | 0 |
| `n_frames_retry_fired` | 0 | 0 |
| `segment()` returned a non-empty mask on | **0 of 20** | 20 of 20 |
| validity IoU | **0.0000 on every frame** | 0.9686 – 0.9744 |
| winning detection score | **0.7524 – 0.7773** | — |
| mask area | 0 px on every frame | 8 519 – 8 525 px |

**The detector was doing its job perfectly.** It found the plate on every frame at scores an order
of magnitude above the ones V6's audit recorded for its twelve false-positive plate masks
(0.167–0.309). What refused all twenty was V6's filter, scoring a correct plate mask against a
warm-**fruit** predicate — and V6's own audit already records exactly that number from the other
side: `0.0000` on all twelve plate masks it caught.

`apple_sam2.object_color_reference` is the apple predicate unconditionally. It never reads
`OBJECT_TEXT_PROMPT`. So §6's plate half could not be measured at all, and — this is the part that
matters — **the failure presented as `coverage: 0.0`, a fact about the corpus.** Downstream,
`run_g0_gates.run_g0b` raises `REFUSED: … coverage 0.000 < --min-coverage 0.9`. The run already
failed; it failed with the wrong reason, and a reader chasing it would go looking at the corpus and
the segmenter.

**Difference from V9 §5.4, recorded rather than smoothed over:** V9 reports validity IoUs
`0.0000–0.0036`; this re-derivation measures `0.0000` on all twenty. The two runs did not name the
same copy of the corpus (this one used the PR-08 640×480 build). Nothing in either reading moves:
both are "essentially none of the object", which is what the filter tests for.

### 1.2 Defect 2 — on a restyle the reference does not go quiet, it moves to the table

V6 §5.3 anticipated the reference not firing on generated pixels, argued correctly that this fails
**closed**, and named the counter that makes the failure legible:
`n_frames_mask_refused_no_reference`, *"which distinguishes 'the segmenter is wrong on this corpus'
from 'the reference does not fit this corpus'"*.

That counter answers the question only when the reference is **empty**. Five of the committed styles
put a fruit in the frame that is not warm (green Granny Smith, pale-green waxy, Golden Delicious,
russet, Pink Lady), and at least one of them also puts a **warm table** there.

Measured on job 189926's committed contact sheets — the only restyled pixels that exist on this
workstation — with the same adapter, the same pins and the same operating point:

| twelve generated panels of `episode_000000` | `train-01-oak-tungsten` | `train-02-linen-overcast` |
|---|---|---|
| detector's box | `[188,127,236,176]` on all 12 | `[188,127,236,175/176]` on all 12 |
| mask SAM 2 drew | 1 808 – 1 876 px | 1 779 – 1 812 px |
| colour reference | **31 112 – 43 319 px = 40.5 – 56.4 % of the panel** | 3 278 – 3 372 px = 4.27 – 4.39 % |
| validity IoU | **0.0250 – 0.0300** | 0.4640 – 0.4861 |
| V6's decision | **REFUSED, all 12** | kept, all 12 |
| `n_frames_mask_refused_no_reference` | **0** | 0 |

**The same detector box, on the same object, producing the same-sized mask, is kept in one style and
refused in the other. Nothing about the mask changed. The reference moved.** An overlay of the two
regions on `train-01-oak-tungsten` shows it directly: the mask outline is tight around a bright
green apple on a white plate, and the reference outlines the warm-lit half of the oak table.

**V9 §5.4's number is reproduced.** V9 records `34 632 px of warm oak table and about a thousand of
green apple`, measured on this same sheet; 34 632 sits inside the 31 112 – 43 319 range these twelve
panels span on the same instrument.

So the exact shape V6 §5.3 relied on being impossible: the reference is emphatically **non-empty**,
so the sub-case counter stays 0, and twelve refusals of a **correct** mask are recorded as *"the
segmenter is wrong here"* when the true statement is *"the reference does not fit here"*.

### 1.3 The half of it nobody had named: the reference does not only reject, it ACCEPTS

Refusing a correct mask is the visible half. The other half is that a predicate covering half the
scene will **agree** with a mask of the scene: a mask of the warm oak region scores IoU 1.0 against
the reference by construction, clears `MASK_VALIDITY_MIN_IOU` comfortably, and hands a **table**
centroid to a geometry gate that has no way to see the difference.

That is the failure mode `GATE_QUALIFICATION_BLOCKERS`'s first entry names in as many words — *"a
plausible mask on the wrong object … which produces a centroid, a displacement and a p95 that all
look like measurements"* — arriving through the filter that was installed to catch it. A fix that
only tightened the refusal branch would have closed the visible half and left this one armed, which
is why §2's check gates the whole decision.

---

## 2. What V10 changes, precisely

> **A predicate that describes the scene is not a reference to the object, and a label it was never
> a predicate for has no reference at all. In both cases the filter cannot decide, and a filter that
> cannot decide refuses — loudly, never quietly, and never by accepting.**

Concretely, in `scripts/estimators/apple_sam2.py`:

1. **`MASK_VALIDITY_REFERENCE_LABELS = frozenset({"apple."})`** — the object labels
   `object_color_reference` is a reference **for**. Exactly one, because exactly one has ever been
   measured. `segment()` calls `_require_mask_validity_reference()` **before any counter moves and
   before any weight loads**, and raises `MaskValidityReferenceUndefined` when the process's
   `OBJECT_TEXT_PROMPT` is not in it. The message carries §1.1's measurement, names the label the
   filter *is* defined for, and says in capitals that this is not a fact about the corpus.
2. **`MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION = 0.10`** — the largest fraction of a frame the
   reference may cover and still be a reference to an object rather than to a scene. `segment()`
   computes it on every frame the validity check runs on, records it, and **refuses the frame before
   the IoU is allowed to decide anything** when it is exceeded.
3. The refusal is counted in **`MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES`**, which is a
   sub-case **inside** `MASK_REFUSED_FRAMES` — the same shape as V6's `no_reference` sub-case, so
   every coverage number downstream is arithmetically unchanged and the *attribution* is the only
   thing that is new.
4. **The empty-reference path is untouched.** An empty reference is object-scale (0.0), falls
   through to the IoU exactly as before, is refused there, and is counted in
   `MASK_REFUSED_NO_REFERENCE_FRAMES` exactly as before. V10 closes the **other end** of the
   distribution. It does not claim to have closed this one — see §5.3.

### 2.1 Measured effect

On the artifacts rather than as a prediction. Every row below is a run of the shipped module.

| | before V10 | after V10 |
|---|---|---|
| `plate.` on 20 source frames | 20 frames refused, `coverage 0.0`, run fails at the coverage floor | the run refuses at the first frame, by name, having loaded no weight and moved no counter |
| `apple.` on the same 20 source frames | 0 refused, IoU 0.9686–0.9744, masks 8 519–8 525 px | **identical, value for value** |
| `train-01-oak-tungsten`, 12 generated panels | 12 refused; `no_reference` 0; attribution: "the mask was wrong" | 12 refused; `no_reference` 0; **`reference_not_object_scale` 12** — "the reference does not fit here" |
| `train-02-linen-overcast`, 12 generated panels | 12 kept | **12 kept, unchanged** — the fix does not fire where the filter works |
| the existing 93 tests of `tests/test_apple_sam2_estimator.py` | pass | pass |

**The source corpus is unaffected, and that is measured rather than argued.** The largest colour
reference on any of 17 307 source frames is 3.00 % of the frame, against a bound of 10 %. §4 is the
whole distribution.

---

## 3. The two fixes that were considered and rejected, and why each is worse

Recorded so they are not proposed again as obvious simplifications — and because one of them is the
direction this workstream was handed, which makes saying why it is worse the more useful half of the
work.

### 3.1 Rejected for defect 1 — make the reference a function of the prompt

A `plate.` predicate would be a colour discriminator for a **neutral-white object on a neutral
cloth**. `object_color_reference`'s own docstring records the corpus observation that makes this
hopeless: *"the cloth and the plate are neutral to within two counts"*. Any such predicate is
several numbers coined by us, sitting in the gate path, with **no measured gap to read them off** —
which is precisely what PR-08 §4 step 2, V6 §4 and V9 §4 all exist to forbid. It would also be
undefined again on the first restyle, since the committed prompts vary the table.

**A third option was also rejected: silently skipping the filter for an unregistered label.** It
fails open in the way that is hardest to see afterwards — the committed contract states
`mask_validity_min_iou`, `measure_est_drift.cross_check_geom_tol` compares it field for field, and
an artifact from a run whose filter never ran would still claim it did. Whether §6's plate half is
measured **unfiltered** may well be the right answer; it is a decision that has to be visible in the
committed document, and this module cannot make it alone.

### 3.2 Rejected for defect 2 — the paired SOURCE frame's own mask as the reference

The proposal, from V9 §5.4 and T-040's 2026-08-23 entry, is to stop asking a colour predicate to
identify an object whose colour the pre-registration deliberately varies, and to use the paired
source frame's own mask instead: *"G0b already carries `--restyled-source-map`, and geometry
invariance is G0b's premise, so if the source mask is not a valid reference for the generated frame,
THAT IS THE G0b FINDING."*

**The diagnosis is right and the mechanism cannot express it.** V6's filter has exactly one action —
return an all-False mask — and in G0b an all-False mask is not a finding, it is a **dropped frame**:

- `run_g0_gates.paired_displacements` (`scripts/run_g0_gates.py:1486`) measures a frame *"only when
  BOTH sides found the object"* and drops the rest into `n_dropped_object_not_visible`.
- `--g0b-percentile` **defaults to 100** (`scripts/run_g0_gates.py:2468`) — *"every measured frame"*,
  i.e. the gate statistic is the **maximum** per-frame displacement.
- The coverage floor is `MIN_COVERAGE_DEFAULT = 0.90`, applied to the **pooled** run
  (`scripts/run_g0_gates.py:2079`).

Put those together. A validity filter that refuses a generated mask when it disagrees with its
paired source mask refuses **exactly the frames whose displacement is largest** — and under
`p100` the largest displacement *is* the verdict. Refuse a handful of them and the gate statistic
falls, while the run stays far inside a 10 % pooled coverage floor. **The proposal converts G0b
failures into dropped frames, in the gate's own most sensitive statistic.** That is the fail-open
direction, and it arrives through the repair.

Three further objections, each sufficient on its own:

1. **It is not colour-free.** The source frame's mask is drawn by the same SAM 2 and validated by the
   same colour predicate one step earlier. What the proposal removes is the *restyle's* colour
   dependence, not the reference's.
2. **`segment(rgb)` cannot see a pairing.** It is handed one frame and no identity — it is the
   contract both PR-08 §4 harnesses call, and V6 §2 leans on that. Wiring the paired source frame in
   means changing that contract or moving the filter into the harnesses, which is a larger change
   than the defect.
3. **There is no paired source for a source frame.** G0b's source side runs the same `segment()`
   over the source clips to produce the source centroids in the first place, so the reference would
   have to be one thing on one side of the subtraction and another on the other — the exact shape V6
   §6 says must be refusable.

**If geometry invariance is to be checked against the source mask, that check belongs in G0b, where
a disagreement is reported as a displacement and read as a verdict — which is what G0b already
does.** It does not belong in a filter whose only verb is "drop this frame". Nothing in V10 forecloses
it; V10 is about what the estimator can honestly claim to have decided.

### 3.3 Why the fix that was taken is not a segmenter change

§4 step 2 requires the estimator to be *the same segmenter* the generator will use, so a number of
ours in the detection path is exactly what it forbids. The filter is not in the detection path, and
V10 does not move it there.

| | before V10 | after V10 |
|---|---|---|
| detector / segmenter / depth, checkpoints and revisions | pinned | identical |
| text prompt, `box_threshold`, `text_threshold`, the single `(0.10, 0.10)` retry, box selection | upstream's | identical |
| the box SAM 2 is prompted with, on every frame | upstream's rule | identical |
| **the mask drawn, pixel for pixel** | — | **identical: no mask is altered, ever** |
| what changes | — | **whether this module is willing to claim it decided this frame** |

---

## 4. The bound, and why its exact value does not matter

A number admitted into a gate path is a number somebody chose. The defence is not that 0.10 was
chosen carefully. It is that **every value over a range wider than an order of magnitude produces the
identical partition of every frame anyone has measured**, and that this is checkable.

### 4.1 The two populations

| | frames | reference, worst case | fraction of frame |
|---|---|---|---|
| **The reference IS the object** | | | |
| source corpus, every frame of 40 episodes (local, 2026-08-23) | 17 307 | 9 220 px / 307 200 | **3.00 %** |
| source corpus, committed 382-frame audit, job 189637 (`warm_apple_px`) | 382 | 8 922 px | 2.90 % |
| source corpus, local CPU audit | 169 | 8 608 px | 2.80 % |
| source corpus, census over 362 episodes (largest per-episode **median**) | 154 447 | 7 731.5 px | 2.52 % |
| the probe's own robot-free run, full resolution | 96 | 7 182 px | 2.34 % |
| source panels of job 189926's sheet (half resolution) | 12 | 3 466 px / 76 800 | 4.51 % |
| `train-02-linen-overcast` generated panels — filter working correctly | 12 | 3 372 px / 76 800 | 4.39 % |
| **The reference is the SCENE** | | | |
| `train-01-oak-tungsten` generated panels — smallest of the twelve | 12 | 31 112 px / 76 800 | **40.5 %** |
| `train-01-oak-tungsten` generated panels — largest | | 43 319 px | 56.4 % |

**The gap is (4.51 %, 40.5 %), a factor of nine, and nothing lies inside it.**

### 4.2 The two scales, and why the choice does not depend on reconciling them

The restyled frames exist only on the cluster; the only restyled pixels on this workstation are job
189926's contact sheets, which are **half resolution with a green mask outline drawn on**. That path
inflates this predicate by a **measured 1.93–1.94×**, established by matching each of the twelve
source panels to the frame it came from decoded locally at full resolution (MSE ~275, ratio steady to
two decimals across all twelve). Two controls say the inflation is the sheet's own encode chain and
not the downsample: a source frame resized locally by the same call reproduces its full-resolution
count to 0.6 %, and the local half-resolution counts are exactly ¼ of the full-resolution ones.

Deflating the sheet rows by 1.94 puts `train-02-linen-overcast` at **2.20–2.26 %** — inside the
source corpus's own range, which is what a correctly-working reference should look like — and
`train-01-oak-tungsten` at **20.9–29.1 %**.

So the gap is **(4.51 %, 40.5 %) on the sheet scale** and **(3.00 %, 20.9 %) deflated**, and
**0.10 is inside both**, with margins 2.2×/4.1× and 3.3×/2.1×. The deflation factor was measured on
warm-**red** fruit pixels and its transfer to warm-**oak** pixels is not established; the bound is
chosen so that it does not have to be.

**That claim is checked, not asserted.**
`tests/test_apple_sam2_estimator.py::test_every_bound_in_the_gap_partitions_the_measured_frames_identically`
sweeps the bound from 0.05 to 0.40 in 1 pp steps and asserts, at every one of the 36 values, that the
observations it calls *inapplicable* are exactly the ones a person identified as the table from the
contact sheets — an identity fixed by what the frame **is**, not recomputed from the bound. A
companion test asserts the shipped value lies strictly inside the range on **both** scales. The
measured numbers are embedded in the test file, because `runs/` is not tracked and a test that skips
when an artifact is missing is not a test; a third test re-checks the embedded copies against the
committed artifacts on any machine that still has them.

### 4.3 What would make the bound matter, and what happens then

A corpus, a restyle or a render on which the two populations were **not** separated by a gap. That is
a measurable condition, not a hypothetical: `MASK_VALIDITY_REFERENCE_FRACTION` records the value on
every frame the check ran on, so a distribution with mass between 0.03 and 0.21 is visible in any
artifact this adapter writes. **If that ever occurs the correct response is a further version
alongside this one, not a bound moved inside it.** `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION` and
`MASK_VALIDITY_REFERENCE_LABELS` deliberately have **no environment override** — for
`MASK_VALIDITY_MIN_IOU`'s reason exactly, and asserted by a test that sets plausible variables and
checks that neither moved.

---

## 5. Threats to validity

### 5.1 The restyle side has been measured only through a contact-sheet proxy

Stated first because it is the weakest thing here. The generated clips live on the cluster at
`/valhalla/…/runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/`, every one suffixed
`.mp4.quarantined`. Every restyle number in §1.2 and §4 comes from a half-resolution PNG with an
outline drawn on it, deflated by a factor measured on the same artifact.

**What would settle it, and it is already asked for twice.** One `segment()` pass over a few hundred
frames of **each committed style's** restyle, recording `n_frames_mask_refused`,
`n_frames_mask_refused_no_reference`, `n_frames_mask_refused_reference_not_object_scale` and the
distributions of `MASK_VALIDITY_IOU` and `MASK_VALIDITY_REFERENCE_FRACTION` **per style**. V6 §5.3
asks for it, V9 §5.4 asks for it, and the frames now exist. Until it is run, the claim this document
supports is *"on the twelve panels of one style, on one episode"*, and it is not a corpus rate.

### 5.2 The direction of the bias

**V10 can only refuse, so it can only remove frames**, and removing frames is the bias V6 §5.1
already records: for `EST_DRIFT_P95` — a p95 that is **subtracted** from `GEOM_TOL` — a smaller
population plausibly means a smaller number, a wider tolerance, and a G0b that is easier for the
generator to pass. Named in advance so it cannot be discovered afterwards and read the convenient
way. Two things bound it, neither a defence:

- **On the source corpus the effect is zero by measurement.** The bound is 10 %; the worst of 17 307
  source frames is 3.00 %, of the 382 audited frames 2.90 %, of 154 447 census frames 2.52 %. The
  `apple.` pass over 20 source frames is value-for-value identical before and after.
- **On anything else, the effect is a loud refusal rather than a quiet number.** A style the
  reference does not fit refuses every frame, coverage collapses, and both harnesses and
  `run_g0_gates` refuse at the coverage floor — and now the counter beside it says which of the two
  reasons it was.

The Isaac calibration renders (§4 steps 1–2) are **untested under V10** for the same reason they were
untested under V6: nobody has run this adapter over them. If the synthetic apple's colour does not
satisfy the predicate, V10 changes nothing (an empty reference is V6's path); if the synthetic
*background* does, V10 refuses loudly where V6 would have quietly measured the background. That is
the intended direction and it is still not a measurement.

### 5.3 The empty-reference case is still ambiguous, and V10 does not pretend otherwise

An empty reference means *"the fruit is occluded or out of frame"* on the source corpus — a hard
frame, V6 §5.1's recorded bias — and *"the reference does not fit"* on a restyle whose apple is not
warm. **One frame cannot tell those apart**, and this module is handed one frame. V10 closes the
other end of the distribution, which V6 §5.3 assumed could not open; it does not close this one.

What would close it is a **declaration**, not a measurement: the caller knows whether it is feeding
source pixels, an Isaac render or a generated clip, and the estimator does not. A required
declaration with no default would refuse every existing caller — including the `GEOM_TOL` job running
as this is written — so it is not proposed here as a same-day change. It is the shape of the next
version if the ambiguity ever costs a reading.

### 5.4 V9's robot masker is exposed to the same defect and V10 does not fix it

`scripts/robot_composite.py` drives `object_color_reference` over **generated** pixels in two places,
and V9's own §5.1 says so: the composite's robot-mask IoU diagnostic, and
`scripts/probe_hallucination.py`, which is `T40_RULE_V8`'s instrument. On a warm-table restyle V9's
per-candidate filter will score robot candidates against a table, not against a fruit.

**That file belongs to V9 and to whoever signs it, and this workstream did not touch it.** What V10
does is export `reference_is_object_scale()` and `reference_frame_fraction()` beside the two
functions V9 already reaches for, so that if the exposure is ever closed there is **exactly one
definition of "this reference is applicable here"** in the repository — which is V9's own stated
reason for reaching into this module rather than restating anything. `apple_sam2` keeps exporting
`object_color_reference`, `mask_validity_iou` and `MASK_VALIDITY_REFERENCE`, so V9's refusal-if-absent
check is unaffected.

### 5.5 The bound catches a scene, not a second small object

A restyle that added a warm object of *apple scale* — a second fruit, a terracotta bowl — would keep
the reference under 10 %, and the filter would arbitrate against the union of two objects and keep a
mask that covered either at ~0.5, or refuse one that covered one at ~0.33. This is V9 §5.3's
one-fruit assumption, inherited rather than introduced: it is `object_color_reference`'s own, and
`configs/transfer25/styles.toml` does not currently permit a second object. Written down because the
failure would be silent.

---

## 6. What is recorded, and where

Because a filter whose firing is not recorded is indistinguishable from a corpus that never
triggered it.

| | where |
|---|---|
| `n_frames_mask_refused_reference_not_object_scale` | `apple_sam2.stats()`, and the module attribute `MASK_REFUSED_REFERENCE_NOT_OBJECT_SCALE_FRAMES` |
| the per-frame reference fraction | `apple_sam2.MASK_VALIDITY_REFERENCE_FRACTION`, raw values in call order, index-aligned to `MASK_VALIDITY_IOU` — same design as `DETECTION_SCORES`, for the same reason: raw values pool exactly through JSON where two histograms only pool if they were binned identically |
| the bound, the registered labels, and whether this process has a reference at all | `stats()['mask_validity_reference_max_frame_fraction']`, `['mask_validity_reference_labels']`, `['mask_validity_reference_is_defined_for_this_prompt']` |
| the whole finding, in one string a reader cannot skim past | `stats()['mask_validity_reference_scope']` — both defects, with their measured numbers, and what is still not separated |
| the filter's presence in the instrument's IDENTITY | `ESTIMATOR_VERSION` gains `mask_val_ref_max_frac=0.1` |
| the refusal for an unregistered label | `MaskValidityReferenceUndefined`, raised out of `segment()`, carrying §1.1's twenty-frame measurement in the message |

**Why the version string had to move, and what it costs.** `run_g0_gates.instrument_disagreements`
compares `ESTIMATOR_VERSION` **between the two sides of G0b** and refuses when they differ, because
*"different weights are a different segmenter wearing the same name"*. A source centroid record
dumped before this bound existed and a restyled one dumped after it are two instruments, and they
must not be compared. The cost is real and belongs to whoever sequences the landing: **`GEOM_TOL`
shards from job 189935 carry the pre-V10 string, and pooling them with post-V10 shards is exactly
what this refusal is for.** The behaviour on the source corpus is identical value for value (§2.1),
so the honest sequence is to land after that job completes, or to re-measure — not to leave the
string alone so the two look the same.

**Why the committed contract did NOT move**, unlike under V6: `measure_geom_tol.contract_disagreements`
counts a field present on one side and absent on the other as a disagreement, so adding
`mask_validity_reference_max_frame_fraction` to `SEGMENTER_CONTRACT` would disqualify every run
against `configs/transfer25/pr08_geom_tol.json` until that file was edited too. **Editing it is
outside this workstream and is part of what signing V10 would authorise.** Until then the bound is
recorded in `stats()` and in `ESTIMATOR_VERSION`, both of which reach every artifact.

**One gap, named rather than left to be found in an artifact.**
`measure_geom_tol.ADAPTER_RUN_COUNTERS` is a tuple in that module, which this workstream did not
edit. It lists `n_frames_mask_refused`, so the **coverage arithmetic** in `estimator_stats.this_run`
is correct and complete. It does not list the new sub-case, so the new counter's **attribution**
reaches an artifact as a lifetime total of the process rather than as this run's number. One line
closes it, in a file V10 does not touch; a test asserts the current state so the note cannot go
stale silently.

---

## 7. What V10 does not discharge

- **`GATE_QUALIFIED` is still `False`** and V10 does not touch it. All three of `apple_sam2`'s
  gate-qualification blockers stand verbatim. Blocker 1 in particular — *nobody has looked at a
  mask* — is not discharged by a session having looked at twelve contact-sheet panels while fixing
  something else. Producing a fix and accepting it are different acts.
- **§8 item 4 is still open.** V10 produces no number toward `GEOM_TOL` or `EST_DRIFT_P95`.
- **The plate half of §6 is still not measurable.** V10 makes the failure legible; it does not
  supply a reference for `plate.` and does not decide whether that half should be measured
  unfiltered. That is the project owner's call and it needs the committed contract to say so.
- **`T40_RULE_V6` stands unedited**, including the counter whose ambiguity §5.3 records.
- **`T40_RULE_V9`'s §5.4 is answered, not adopted.** §3.2 rejects its candidate fix on measured
  grounds; V9 itself says the choice is the project owner's and licenses nothing.
- **`T40_RULE_V8`'s `human_review.looked_at` is still `false`.**
- **`T40_RULE_V1` §1's prohibition still binds in full**: nothing is generated, no weight is trained
  on generated frames, and no number from PR-08 is quoted as a result.

**V10 licenses nothing.**

---

## 8. Provenance

| | |
|---|---|
| rule | `T40_RULE_V10` |
| status | **UNSIGNED DRAFT.** Not in force |
| drafted | 2026-08-23, before the corrected filter has produced any gate number |
| supersedes | nothing. It **supplements** `T40_RULE_V1`, which stands and is unedited |
| changes | which labels the mask-validity filter will run for, and which frames it will claim to have decided (§2). In the estimator only |
| decided by | **nobody yet.** Signing is the project owner's, and no session may sign it or act as though it were signed |
| evidence | both defects reproduced from scratch before any change: the `plate.` pass and its matched `apple.` control on 20 source frames of `episode_000000`; the twelve generated and twelve source panels of job 189926's `episode_000000__train-01-oak-tungsten` and `…__train-02-linen-overcast` contact sheets, with overlays; a 17 307-frame scan of the source corpus; and the committed `runs/pr08-mask-audit/MASK_AUDIT.json`, `runs/pr08-mask-audit-local-cpu/MASK_AUDIT.json` and `runs/t040-identity-prompt/calibration-2/probe_census.json` |
| bound | `MASK_VALIDITY_REFERENCE_MAX_FRAME_FRACTION = 0.10`, inside the measured gap on both scales — (4.51 %, 40.5 %) raw and (3.00 %, 20.9 %) deflated; insensitivity swept and asserted in `tests/test_apple_sam2_estimator.py` |
| touched | `scripts/estimators/apple_sam2.py`, `tests/test_apple_sam2_estimator.py`, this document |
| not touched | `scripts/robot_composite.py`, `scripts/run_g0_gates.py`, `scripts/measure_geom_tol.py`, `scripts/measure_est_drift.py`, `configs/transfer25/pr08_geom_tol.json` and its sidecar, `SEGMENTER_CONTRACT`, `MASK_VALIDITY_MIN_IOU`, `MASK_VALIDITY_REFERENCE`, `MATERIAL_FLOOR_PP`, `max_frame_fraction`, every detection threshold, every cap, every gate, every verdict, `GATE_QUALIFIED`, either blocker tuple, and every signed rule document |
| jobs submitted | **none.** No Slurm job and no ssh; every measurement here ran locally on the workstation's own GPU against the local corpus, the local hub cache and committed artifacts |
| generation licensed | **no** |
| training licensed | **no** |
