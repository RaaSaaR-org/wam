# PR-08 V9 — the robot mask may not be the apple

**Rule `T40_RULE_V9`. Drafted 2026-08-23. UNSIGNED — see §8. Nothing here is in force until the
project owner signs it, and no number produced under it may be quoted before that.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), which is registered as
`T40_RULE_V1` and **has not been edited and must not be**. The repo's discipline is
`docs/handoff.md` §3 — *"Rules are versioned, never edited in place. A gate rewritten after seeing
its output is not a gate."* V9 is that versioning, not a revision. `T40_RULE_V2` (arm C frame
matching), `V3` (seed schedule), `V4` (the T-39 gate premise), `V5` (the MuJoCo ground-truth route),
`V6` (the estimator's mask-validity filter), `V7` (the consumer contract) and `V8` (the
hallucination probe) all stand unchanged; **V9 depends on none of them and changes nothing in any of
them.** V4 and V8 are signed and are not touched here.

Task: [[T-040]]. Generator: **Cosmos-Transfer2.5, frozen**. Adapter: `scripts/robot_composite.py`
(the G0c compositor), driving `scripts/estimators/apple_sam2.py`'s pinned detector and predictor.

**Nothing in this document licenses generation, training, or any statement of a result.**
`T40_RULE_V1` §1's prohibition is untouched and still binds in full.

---

## 0. What V9 does not change

Stated first and exhaustively, because a version that quietly moves a threshold is the failure the
versioning discipline exists to prevent.

| | unchanged |
|---|---|
| `ROBOT_TEXT_PROMPT` | `"robot arm. robotic hand. robotic gripper."`, still a committed constant with no environment override and no flag. V9 does not narrow it, widen it, or make it per-run |
| **The detection operating point** | `box_threshold = 0.15`, `text_threshold = 0.25`, read live off `apple_sam2` so it tracks the generator's own numbers. Unchanged |
| **Upstream's `(0.10, 0.10)` retry** | still deliberately NOT run here, still recorded in `upstream_retry_not_run`, and still argued the same way. V9 adds no retry, no second pass and no fallback |
| **The box rule** | still the union of every detection above threshold. V9 removes candidates from that union; it does not change how boxes are selected, scored or ordered, and it never adds one |
| **`check_mask`** | untouched, both refusals. The empty-mask refusal still has no threshold — "zero is zero" — and the area refusal still reads `max_frame_fraction` out of the committed artifact |
| `MATERIAL_FLOOR_PP = 10.0` | still borrowed from `I8_RULE_V3`, not coined, not moved, not made per-arm |
| `GEOM_TOL`, `EST_DRIFT_P95` | still derived, still subtracted, still `null` in `configs/transfer25/pr08_geom_tol.json`. V9 produces no number toward either and touches neither harness |
| **G0a**, **G0b** | unchanged in every term. V9 is entirely inside G0c's compositor and does not reach `measure_geom_tol`, `measure_est_drift` or `run_g0_gates` |
| **G0c's sentence** | unchanged — real robot pixels unconditionally composited back, robot-mask IoU recorded as a diagnostic and never as a gate. V9 does not add a gate, does not compare anything to a pass/fail threshold, and its only possible effect on a clip is a REFUSAL |
| **The hard edge** | unchanged — no feather, no dilation, no blend. The mask is used exactly as the segmenter returned it, minus whole candidates |
| **The area bound** | `max_frame_fraction` is still not coined anywhere, the `measure` mode still writes `null`, and V9 writes no bound and suggests none. `configs/transfer25/pr08_robot_mask_area.json` does not exist in the tree as this is written |
| **The verdict table**, **the ladder**, **the four arms**, **the headline** | unchanged in every cell |
| **The committed style partition** | `configs/transfer25/pr08_style_partition.json`, `T40_STYLES_V1`, sha256 `4da3875d…a680da8`. V9 changes no style, no id, no slug and no prompt string, and therefore does not change that hash |
| **The three checkpoints and their pinned revisions** | unchanged |
| **`GATE_QUALIFIED`**, **`GATE_QUALIFICATION_BLOCKERS`**, **`GATE_QUALIFICATION_DISCHARGED`** | not edited by V9. Nothing was moved into or out of any of them |
| **`T40_RULE_V6`** | not edited, not superseded, not weakened. V9 does not touch `apple_sam2.MASK_VALIDITY_MIN_IOU`, `segment()`, or which frames `GEOM_TOL` and `EST_DRIFT_P95` are measured on. §5.4 below reports a defect FOUND in V6's blast radius; finding it here is not fixing it here |
| **`T40_RULE_V8`'s verdict** | job 189926's `H` stands exactly as measured, by the pre-V9 masker. V9 licenses no re-reading of it and produces no number toward `human_review.looked_at` |

V9 changes exactly one thing: **which detections are allowed into the robot mask.** Nothing else.

---

## 1. The finding

### 1.1 The prompt grounds the apple, and it was measured rather than supposed

`runs/pr08-robot-mask-empty/DIAGNOSIS.json` (verdict **ABSENT**, `d739a87`) settled that the empty
robot masks are correct answers: the robot is genuinely out of frame on ~36 % of this corpus. **That
is not what V9 is about, and V9 does not revisit it.** V9 is about the opposite error, recorded in
the same artifact as a secondary finding and never acted on.

GroundingDINO grounds phrases; it does not decide that a phrase is absent. Handed
`"robot arm. robotic hand. robotic gripper."` and a frame containing a tablecloth, a plate and a
piece of fruit, it returns its best-scoring box above 0.15 — and here that box lands on the fruit.

Re-derived for this document from the committed artifacts, and the repository's own prose corrected
where it is loose:

| | |
|---|---|
| detect plan | `runs/pr08-robot-mask-empty/plan_corpus.json` — 40 episodes, 710 frames, equal quotas per visibility bucket |
| robot-absent frames sampled | 240 |
| of those, masker returned a NON-EMPTY mask | **98 — 40.83 %** |
| their mask areas | median **6 368 px**, and 66 of the 98 fall in 3 000–10 000 px, i.e. apple-scale at 640×480 |

**The "41 %" in `d739a87`'s commit body and in `DIAGNOSIS.json`'s headline is arithmetically
confirmed (98/240 = 40.83 %) and its wording is corrected: it is not a corpus rate.** The plan
samples equal quotas per bucket and `stratified_plan`'s own docstring says so — *"THE RESULTING
RATES ARE NOT CORPUS RATES and nothing may read them as such."* The defensible sentence is: **on
robot-absent frames of the 40-episode sample, 40.8 % returned a non-empty mask.** The corpus-wide
rate of the composite defect is not known and this document does not claim one.

### 1.2 What those masks are, per detection rather than per frame

The committed artifact records `mask_px` but no per-detection geometry, so the question "how many of
these are the apple" cannot be answered from it. It was therefore measured, locally, on the same 710
frames, with the same pins, the same prompt and the same operating point, scoring each box's SAM 2
mask against `apple_sam2.object_color_reference` — the warm-and-saturated predicate `T40_RULE_V6`
already runs on every `segment()` call. 2 845 detections:

```
apple detections :  IoU in [0.9364, 0.9847]   (146; the mask IS the colour region)
everything else  :  IoU <= 0.5131             (2 699; robot, plate, tablecloth, gripper-over-apple)
```

A person has looked. `runs/pr08-robot-mask-apple/sheet_absent_now_empty.png` is nine of the 146:
in every one the frame contains no robot at all and the box is drawn tightly around the fruit.

### 1.3 Why this is the worst shape a PR-08 defect can have

Under G0c the robot mask is the region composited back **from the source**. An apple inside it means
the generated apple is overwritten by the real one — the object the task is about silently stops
being restyled, and arms B, C and D become arm A for that object while still costing their GPU
hours.

**No gate in the pre-registration can see it.** G0a measures labels. G0b measures geometry, and a
pixel-identical apple has moved zero pixels: it does not merely pass G0b, it passes perfectly. §6
says the robot-mask IoU is "a diagnostic on the generator, never a gate", in those words, twice.
An apple-sized mask is ~0.02 of the frame, far below any plausible area bound. And the mask is not
empty, so the one refusal that would have fired never does.

The defect does not evade the gates. **It manufactures a pass.**

---

## 2. What V9 changes, precisely

> **A detection that is the object is not the robot, and is dropped before the union.**

Concretely, in `scripts/robot_composite.py`:

1. `Sam2RobotMasker.mask(rgb)` scores every candidate mask SAM 2 drew against
   `apple_sam2.object_color_reference(frame)`, using `apple_sam2.mask_validity_iou` — both reached
   from the adapter, neither restated here, so there is exactly one definition of "this region is
   the apple" in the repository.
2. Any candidate whose IoU **exceeds `ROBOT_MASK_OBJECT_MAX_IOU = 0.70`** is dropped. The union is
   taken over the survivors.
3. If that leaves nothing, the frame's mask is **all-False**, and `check_mask`'s existing
   empty-mask refusal takes the clip — by name, with no threshold and no number to loosen.

**The filter is per candidate and not on the finished union**, and that is load-bearing rather than
incidental. On a grasp frame the detector returns real robot boxes *and* a box on the fruit
(`runs/pr08-robot-mask-apple/DETECTIONS.json`, `episode_000392` f102: seven detections, one at IoU
0.959 and three robot boxes at 0.12 that also swallow the fruit). Filtering the union would have to
choose between discarding the robot and admitting the apple. Filtering candidates has to do neither:
measured on that frame, dropping the apple box removes **140 px of a 31 710 px mask**, of which 43
are not apple-coloured, because the gripper's own pixels are covered by the gripper's own
detections.

### 2.1 Measured effect, on the artifact rather than as a prediction

Re-scored offline over all 710 planned frames:

| | before | after |
|---|---|---|
| detections in the union | 2 845 | 2 699 (**146 dropped**) |
| frames whose mask becomes EMPTY, and therefore refuse | — | **70** |
| frames whose mask shrinks but survives | — | **76** |
| frames unchanged | — | 327 (plus 237 that were already empty) |
| robot-absent frames returning a non-empty mask | 98 of 240 | **50 of 240** (48 go empty and refuse; 27 shrink and survive; 23 were never touched) |
| robot-**present** frames the filter touches at all | — | 8, of which 2 go empty |

Those 8 were rendered and looked at (`sheet_present_now_empty.png`, `sheet_present_shrunk.png`), as
were nine of the twenty band frames the filter empties (`sheet_ambiguous_now_empty.png`) and nine of
the absent ones (`sheet_absent_now_empty.png`). **The filter removed no true robot detection in any
frame that was inspected** — every emptied frame that was looked at contains no robot and its sole
detection is a box drawn tightly around the fruit. The 2 that go empty contain no robot —
the visibility reference had over-called, which `DIAGNOSIS.json` §q4 already established it does
(all 19 of its apparent detector failures were inspected and none contained a robot). On the other
6, the removed region is the apple to within tens of pixels: 8 228 px removed against an 8 213 px
colour reference on `episode_000088` f126, 5 114 against 5 154 on `episode_000232` f106.

### 2.2 It makes G0c refuse MORE, and cannot manufacture a pass

`DIAGNOSIS.json` already concluded that with a median 152 robot-absent frames per episode, *"every
clip refuses. G0c as written cannot produce a single composited clip on this corpus."* **V9 does not
change that and does not pretend to.** Whether G0c's per-frame refusal is the right rule for a
corpus the robot is absent from is a separate, open decision and belongs to the project owner.

What V9 removes is the **other** outcome: the clips that would have been composited with the apple
frozen and would have passed. There is deliberately no fallback that keeps the best-scoring reject
to avoid a refusal — that is the same trade upstream's `(0.10, 0.10)` retry offers, which this
module already refuses in its docstring for the same reason: here a weak detection does not recover
a frame, it suppresses a refusal.

---

## 3. Why this is not a segmenter change, and not the IoU threshold §6 refuses

Two separate objections, answered separately because they are separate.

### 3.1 It is not a change to the segmenter

| | before V9 | after V9 |
|---|---|---|
| detector, checkpoint, revision | GroundingDINO `@12bdfa31…` | identical |
| segmenter, checkpoint, revision | SAM 2 hiera-large `@e6a8e880…` | identical |
| text prompt | `"robot arm. robotic hand. robotic gripper."` | identical |
| `box_threshold` / `text_threshold` | 0.15 / 0.25 | identical |
| retry | none (upstream's is deliberately not run) | identical |
| box rule | union of every detection above threshold | identical |
| the boxes SAM 2 is prompted with, on every frame | every box above threshold | identical |
| **the mask drawn, pixel for pixel** | — | **identical: no mask is altered, ever** |
| what changes | — | **whether a mask SAM 2 already drew is admitted to the union** |

It is a check on the OUTPUT. Nothing is re-detected, re-prompted or re-drawn. Same shape, same
argument, as `T40_RULE_V6` §3 makes for `apple_sam2.segment`.

**The rejected fix is recorded so it is not proposed again as an obvious simplification.** It is
*not* "raise `box_threshold` above the apple detections' scores". Those score 0.150–0.264 and true
robot detections score 0.150–0.623: the ranges overlap at the bottom, `DIAGNOSIS.json` says so, and
a threshold that separated them would make this detector stricter than the generator's while
discarding real robot pixels — under-coverage, which is the defect G0c exists to exclude.

### 3.2 It is not the IoU threshold G0c refuses

§6's refused number is a **gate**: a pass/fail cut on the robot-mask IoU *between source and
generated*, i.e. a verdict on the generator, which §6 rejects because `video_fidelity` cannot see
the defect and any such cut would be coined. V9's number is a different object in every respect that
matters:

| | §6's refused threshold | `ROBOT_MASK_OBJECT_MAX_IOU` |
|---|---|---|
| compares | source mask vs generated mask | one candidate mask vs a non-learned colour predicate |
| computed on | source **and** generated frames | the source frame alone, before any generated pixel exists |
| what it decides | whether the generator passes | whether a detection is the robot at all |
| possible outcomes | pass, fail | drop a candidate — and therefore, at most, a REFUSAL |
| derived from | nothing | a measured gap 0.42 wide (§4) |

It gates nothing and licenses nothing. **It is still a number in a gate's path**, which is why it is
pre-registered rather than merely commented, and why this document exists at all.

---

## 4. The threshold, and why its exact value does not matter

The defence is not that 0.70 was chosen carefully. It is that **the value is irrelevant over an
interval 0.42 wide**, and that this is checkable.

Over the 2 845 measured detections the two populations do not overlap and are nowhere near
overlapping:

```
non-apple detections :  IoU <= 0.5131   (2 699; max = 0.5131)
apple detections     :  IoU >= 0.9364   (146;   min = 0.9364)
```

**Every cut in the open interval (0.5131, 0.9364) produces the identical partition of those
detections.** 0.70 is a value read off a gap, with 0.187 of margin below it and 0.236 above.

That claim is **checked, not asserted**:
`tests/test_robot_composite_object_filter.py::test_every_threshold_in_the_gap_partitions_the_measured_detections_identically`
sweeps from 0.52 to 0.93 in 1 pp steps and asserts, at every one of the 42 values, that exactly the
146 detections a person confirmed from the contact sheets are dropped — an identity fixed by the
flag, not recomputed from the threshold. A companion test asserts the shipped value lies strictly
inside the interval and not on either edge. The measured IoUs are embedded in the test file, because
`runs/` is not tracked and a test that skips when an artifact is missing is not a test.

**The gap is not an accident of this sample, it is what the two shapes are.** A mask of the apple
and the colour predicate are two outlines of one object, so they agree at ~0.95. Anything else on
this corpus contains essentially none of those pixels — the robot is black and bare metal, the cloth
and the plate are neutral to within two counts, which are `object_color_reference`'s own recorded
seed observations. The only in-between shape is a real robot detection whose box also swallows the
fruit during a grasp, and those are the 0.19–0.51 tail: kept, correctly.

**Symmetric IoU rather than one-sided containment, deliberately.** "How much of the candidate is
apple-coloured" would also drop the whole-tablecloth masks, since they contain the fruit — and those
are `check_mask`'s area bound's business. A filter that quietly took over another check's failure
mode would turn an over-large-mask refusal into an empty-mask refusal and tell the operator the
wrong thing.

**What would make the threshold matter, and what happens then.** A corpus, a restyle or a render on
which the two populations were not separated by a gap. That is a measurable condition: the counters
in §6 make a filter that fires on everything, or on nothing, visible in any artifact this module
writes. **If that occurs, the correct response is a further version alongside this one, not a
threshold moved inside it.** `ROBOT_MASK_OBJECT_MAX_IOU` has deliberately no environment override
and no flag, for `ROBOT_TEXT_PROMPT`'s reason exactly: a per-run value here is a per-run decision
about which pixels the generator may touch, taken on a submit line, recorded nowhere a reader would
look, and invisible in the output.

---

## 5. Threats to validity

### 5.1 The reference is a property of the SOURCE corpus's appearance, and G0c is unexposed to that BY PLACEMENT

`object_color_reference` justifies itself with *"the only saturated warm thing in any of these
frames is the fruit"*, which is a claim about AppleToPlate's real pixels under AppleToPlate's real
lighting. **It is false on a restyle by construction.** `configs/transfer25/styles.toml` makes apple
colour and variety an ALLOWED axis and the committed prompts produce a bright green Granny Smith, a
pale-green waxy, a pale-yellow Golden Delicious, a brown russet and a mottled Pink Lady.

Measured, not supposed, on job 189926's contact sheet — the first restyled frames this project has:
on `train-01-oak-tungsten` the warm-saturated predicate returns **34 632 px of warm oak table** and
about a thousand of green apple. It does not merely go quiet on a restyle; **it moves to a different
object.**

**G0c's composite is unexposed, and by placement rather than by luck.** The mask that decides which
pixels are composited is made from the SOURCE frame — `composite_clip` masks `src`, never `gen` —
and that is where the predicate's claim is true. This is pinned by
`test_the_composite_takes_its_mask_from_the_source_frame_and_only_from_there`, which fails if a
future edit ever masks a generated frame and composites that.

**Two paths DO run this masker over generated pixels, and neither is a gate:**

1. **`composite_clip`'s robot-mask IoU diagnostic** (`masker.mask(gen[index])`). §6 calls it "a
   diagnostic on the generator, never a gate", twice. On a restyle the filter will under-fire (a
   green apple is not warm-saturated, so an apple-grounded candidate is kept) or mis-fire (a robot
   candidate overlapping a warm oak table could score high). Its counts are differenced out of the
   per-clip record before this loop runs, so the two populations are never pooled.
2. **`scripts/probe_hallucination.py`**, which is `T40_RULE_V8`'s instrument and masks generated
   frames. **V9 therefore changes V8's instrument for any FUTURE probe run.** Job 189926's `H` was
   produced by the pre-V9 masker and stands exactly as measured; V9 licenses no re-reading of it.
   On the styles whose apple stays warm-red (`source-red-yellow-glossy`, `deep-red-matte`,
   `striped-gala`, `dark-crimson-glossy`) the filter will remove apple-grounded candidates from the
   generated mask and the candidate-invention count will drop; on the five non-warm styles it will
   not. **A probe run under V9 must therefore report the filter's counters per style**, or its
   count is a mixture of two instruments.

An exemption for the probe was considered and rejected: two maskers is the drift failure PR-13 is
about, and a masker that behaves differently depending on who called it cannot be described in a
record.

### 5.2 The direction of the bias, stated in advance

**A wrongly dropped candidate leaves generated manipulator in the frame** — the defect G0c exists to
exclude, arriving through G0c's own repair. That is the unsafe direction and it is named here so it
cannot be discovered afterwards and read the convenient way.

Two things bound it, neither of which is a defence:

- **On the measured sample the effect is zero by measurement, not by argument.** 8 robot-present
  frames are touched, 6 of them lose the fruit and keep the robot to within tens of pixels, and the
  2 that go empty contain no robot. But 710 frames of a 171 625-frame corpus is not the corpus, and
  the sample deliberately over-weights the hard cases.
- **The effect is counted, per clip and per run** (§6). `frames_emptied_by_the_filter` being
  non-zero is not by itself a defect — it is the intended behaviour on a robot-absent frame. Its
  being non-zero and **unread** would be.

### 5.3 The predicate describes ONE fruit, and the IoU is against the whole frame's warm region

If a frame contained two warm-saturated objects, a candidate covering one of them would score ~0.5
against the union of both and would be kept. AppleToPlate has exactly one apple, so this does not
arise on this corpus; it would arise on any corpus with more, and it is written down because the
failure would be silent. The corpus assumption is `object_color_reference`'s own and is not
introduced here.

### 5.4 A DEFECT IN `T40_RULE_V6`'s BLAST RADIUS, FOUND WHILE WRITING THIS AND NOT FIXED HERE

Reported because it was found, and separated from V9 because it is not V9's to fix. **V6 is not
edited, weakened or superseded by this section.**

`apple_sam2.segment()` applies V6's validity filter against `object_color_reference` on **every**
call, and `object_color_reference` is the APPLE predicate unconditionally — it does not read
`OBJECT_TEXT_PROMPT`. Two consequences, both on paths `run_g0_gates.py` documents in its own
docstring:

1. **The `plate.` pass refuses 100 % of frames, on the SOURCE corpus.** `run_g0_gates`'s documented
   invocation is `WAM_PR08_OBJECT_PROMPT="plate." … --gates g0b --source-clips …`, which reaches
   `measure_geom_tol` and `module.segment(rgb)`. Executed on 20 source frames of
   `episode_000000`: `n_segment_calls 20`, `n_frames_mask_refused` **20**,
   `n_frames_mask_refused_no_reference` **0**, validity IoUs 0.0000–0.0036, `segment()` non-empty on
   **0 of 20**. A correct plate mask scores ~0 against a warm-fruit reference — V6's own audit
   records exactly that, `0.0000` on all twelve plate masks — so with V6 in place §6's plate half
   cannot be measured at all, and the failure presents as `coverage: 0.0`, i.e. as a fact about the
   corpus rather than as a filter that is not defined for this label.
2. **On a restyle the counter that was meant to make the failure legible reports the wrong one.**
   V6 §5.3 anticipates the reference not firing on generated pixels, says it fails CLOSED, and
   relies on `n_frames_mask_refused_no_reference` to distinguish *"the segmenter is wrong on this
   corpus"* from *"the reference does not fit this corpus"*. On `train-01-oak-tungsten` the
   reference **does** fire — on 34 632 px of oak table — so a refused green-apple mask is counted as
   the first case when it is the second.

Neither is repaired by V9 and neither may be repaired by editing V6 in place. What would settle it
is stated so it can be settled rather than caveated:

- **Measure, do not argue.** One pass of `segment()` over a few hundred frames of each committed
  style's restyle, recording `n_frames_mask_refused`, `n_frames_mask_refused_no_reference` and the
  validity-IoU distribution **per style**. V6 §5.3 already asks for this and it is still open; the
  frames to do it on now exist.
- **The candidate fix, if one is wanted, is not a style-aware colour predicate.** It is to stop
  asking a colour predicate to identify the object on frames whose colours the pre-registration
  deliberately varies. The reference with no colour assumption in it is **the paired SOURCE frame's
  own mask**: G0b already carries `--restyled-source-map`, and geometry invariance is G0b's own
  premise, so if the source mask is not a valid reference for the generated frame then that IS the
  G0b finding. Per-style references committed in `styles.toml` are the fallback and cost 25 more
  coined numbers. For the `plate.` pass the smaller fix is that the reference must be a function of
  the prompt, or the filter must refuse to run when it is not defined for the label instead of
  refusing every frame.
- Either way it is a **further version alongside V6**, decided by the project owner, and none of it
  is licensed by this document.

---

## 6. What is recorded, and where

Because a filter whose firing is not recorded is indistinguishable from a corpus that never
triggered it.

| | where |
|---|---|
| `frames_masked`, `detections_segmented`, `detections_dropped_as_object`, `frames_with_a_dropped_detection`, `frames_emptied_by_the_filter`, `frames_with_no_object_reference` | `Sam2RobotMasker.filter_counters`, cumulative; read by `filter_record()` |
| per clip | `composite_clip`'s record → `robot_mask_object_filter`, differenced across the SOURCE pass only, with `masks_from_cache` beside it so that all-zero counts on a cache hit cannot read as "it never fired" |
| the rule and the threshold, in one string a reader cannot skim past | `Sam2RobotMasker.provenance()['object_grounding_filter']` — in every clip's record |
| the filter's presence in the segmenter's IDENTITY | `SEGMENTER_IDENTITY_FIELDS` gains `object_grounding_filter`, which is read by `MaskCache.key` and by `load_area_bound`'s cross-check |

**Why the identity had to move too.** `SEGMENTER_IDENTITY_FIELDS` is this file's one definition of
"everything that changes which pixels come back for a given frame", and the filter changes exactly
that. Both consequences are intended: a mask cached before V9 is a different mask and must not be
reused, and a robot-mask area distribution measured before V9 is a distribution of a different
masker and must not be sat above. No committed artifact is invalidated today —
`configs/transfer25/pr08_robot_mask_area.json` is not in the tree — so this is a pre-commitment
rather than a repair, which is the only point at which making it is legitimate at all.

**The adapter is required to keep declaring what this depends on.** If `apple_sam2` stops exporting
`object_color_reference`, `mask_validity_iou` or `MASK_VALIDITY_REFERENCE`, this module **refuses**
rather than silently compositing an unfiltered mask or writing an unverifiable sentence into 10 050
records. Same treatment as `upstream_retry_not_run`, and for the same reason.

---

## 7. What V9 does not discharge

- **`GATE_QUALIFIED` is still `False`** and V9 does not touch it. All three of
  `apple_sam2`'s gate-qualification blockers stand verbatim.
- **The G0c refusal rate is not addressed.** `DIAGNOSIS.json`'s conclusion that G0c as written
  cannot composite a single clip on this corpus is unchanged, and V9 moves it in the stricter
  direction. Whether G0c's per-frame empty-mask refusal is the right rule for a corpus the robot is
  absent from ~36 % of the time is the project owner's open decision, and **nothing here may be read
  as answering it.**
- **The area bound is still not measured and still not coined.** §8 item 4's siblings stay open.
- **`T40_RULE_V8`'s `human_review.looked_at` is still `false`.** §5.1 explains why V9 is direct
  evidence about job 189926's `H` and why that is still not a person having looked at the sheets.
  Producing a fix and accepting it are different acts.
- **§5.4 is a finding, not a repair.** V6 stands unedited and the plate pass is still broken.
- **`T40_RULE_V1` §1's prohibition still binds in full**: nothing is generated, no weight is trained
  on generated frames, and no number from PR-08 is quoted as a result.

**V9 licenses nothing.**

---

## 8. Provenance

| | |
|---|---|
| rule | `T40_RULE_V9` |
| status | **UNSIGNED DRAFT.** Not in force |
| drafted | 2026-08-23, before the filter has produced any gate number and before any clip has been composited |
| supersedes | nothing. It **supplements** `T40_RULE_V1`, which stands and is unedited |
| changes | which detections enter the robot mask (§2), in the G0c compositor only |
| decided by | **nobody yet.** Signing is the project owner's, and no session may sign it or act as though it were signed |
| evidence | `runs/pr08-robot-mask-empty/` (committed; verdict ABSENT, the 98/240 re-derivation), a local re-segmentation of the same 710-frame plan recording 2 845 per-detection IoUs, contact sheets under `runs/pr08-robot-mask-apple/`, and job **189926**'s sheets for §5.1 |
| threshold | `ROBOT_MASK_OBJECT_MAX_IOU = 0.70`, inside the measured gap (0.5131, 0.9364); insensitivity swept and asserted in `tests/test_robot_composite_object_filter.py` |
| touched | `scripts/robot_composite.py`, `tests/test_robot_composite_object_filter.py` (new), `tests/test_restyle_transfer25.py` (two stub adapters now declare `MASK_VALIDITY_REFERENCE`, as the real one does) |
| not touched | `ROBOT_TEXT_PROMPT`, `check_mask`, `max_frame_fraction`, every detection threshold, `MATERIAL_FLOOR_PP`, `GEOM_TOL`, `EST_DRIFT_P95`, any gate, any verdict, `GATE_QUALIFIED`, either blocker tuple, `apple_sam2.MASK_VALIDITY_MIN_IOU`, and every signed rule document |
| jobs submitted | **none.** No Slurm job and no ssh; every measurement here ran locally on the workstation's own GPU against the local corpus and the local hub cache |
| generation licensed | **no** |
| training licensed | **no** |
