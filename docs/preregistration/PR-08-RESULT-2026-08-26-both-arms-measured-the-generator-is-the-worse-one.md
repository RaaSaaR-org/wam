# PR-08 — both arms measured on one capture, and at p95 the generator's topology is the worse one

**Written 2026-08-26 from a measurement made 2026-08-25. No pixel was re-measured, no code was
edited, no gate, config or blocker was touched, and no bound is written or implied.
`GATE_QUALIFIED` is still `False` and all three blockers are still in the tuple.**

`scripts/estimators/apple_sam2.py`'s **third** gate-qualification blocker names its own discharge
condition — the same capture segmented **both ways**, per frame and by propagation from frame 0,
with the two p95s recorded side by side — and predicts that the bias between the two arms is
**two-sided**, one limb safe and one limb unsafe. That experiment has now been run.
`runs/pr08-est-drift/EST_DRIFT-ARMS-mujoco-trajectory-f480.json` carries it.

The short version: **limb (a) is refuted in direction at the percentile that matters**, and **limb
(b) was not observed on this capture**. Neither of those is a discharge, for two reasons stated at
length in §8 — the capture is MuJoCo where the blocker says Isaac, and 480 frames of one
trajectory is not the corpus.

---

## 1. What blocker 3 says, verbatim

The third entry of `GATE_QUALIFICATION_BLOCKERS` (`scripts/estimators/apple_sam2.py`, ~line 661,
read-only for this document) opens by conceding that everything else in §4 step 2's "the same
segmenter" now matches Cosmos-Transfer2.5's `sam2_model.py` exactly, and then states the one
difference left:

> But upstream drives SAM2VideoPredictor.init_state(video_path=...) and PROPAGATES one mask across
> the clip, while this adapter re-detects and re-segments every frame independently, because
> segment(rgb) is the contract both harnesses call. The bias is TWO-SIDED, which is why this cannot
> be waved through as conservative:

> **(a)** independent re-detection jitters frame to frame where propagation is temporally smooth,
> so our tail — and EST_DRIFT_P95 is a p95, i.e. the tail — is INFLATED relative to the
> generator's, which subtracts more from GEOM_TOL and tightens G0b (safe);

> **(b)** propagation's own characteristic failure, drifting off the object and staying off for a
> run of frames, is invisible to a per-frame estimator that recovers on the next frame, so the
> generator commits an error our budget never sees (unsafe).

and it names what would settle them:

> Discharged by: measuring the same Isaac capture BOTH ways — this adapter per frame, and the video
> predictor propagating from frame 0 — and recording the two p95s, so the direction and size of the
> difference are a measurement rather than the argument above.

Note the structure. Limb (a) is a claim about **the tail of a displacement distribution**, and two
p95s answer it. Limb (b) is a claim about **contiguity** — a run of frames — and a p95 cannot
answer it at all, which is why the artifact carries a separate `low_iou_runs` statistic. §3 and §5
take them in that order.

## 2. The measurement

`runs/pr08-est-drift/EST_DRIFT-ARMS-mujoco-trajectory-f480.json`, `measured_utc`
`2026-08-25T17:43:26+00:00`, `arm_comparison` block.

| | per-frame arm | propagation arm |
|---|---|---|
| topology | GroundingDINO + SAM 2 re-run independently on every frame — `estimator.segment(rgb)` | one GroundingDINO detection on frame 0, one `SAM2VideoPredictor` seed, mask propagated forward |
| module | `scripts/estimators/apple_sam2.py` | `scripts/estimators/apple_sam2_video.py` |
| `est_drift_p95_px` | **0.29077062684224225** | **0.47006167975525187** |
| `n_frames` | 480 | 480 |
| `n_measured` | 479 | 480 |
| `n_dropped` | 1 | 0 |
| `coverage` | 0.9979166666666667 | 1.0 |
| mask-vs-ground-truth IoU, p95 | 0.9944258607198748 | 0.9890373616731923 |
| mask-vs-ground-truth IoU, mean | 0.9820856795761793 | 0.9763142844390104 |
| mask-vs-ground-truth IoU, min | 0.0 | 0.879245283018868 |
| `low_iou_runs.n_runs` (threshold 0.5) | **1**, `[[345, 345]]` | **0**, `[]` |
| `low_iou_runs.longest_run` | 1 | 0 |

`delta.propagation_minus_per_frame_px` = **0.17929105291300962**, and the artifact states the
reading rule itself: *"POSITIVE means the propagated mask's centroid sits FURTHER from ground truth
at the 95th percentile than the per-frame adapter's, i.e. blocker 3's limb (a) … does not hold on
this capture in that direction."*

Both arms are held to the same operating point — the same SAM 2 and GroundingDINO checkpoints at
the same pinned revisions, the `apple.` prompt, `threshold=0.15` / `text_threshold=0.25`, the
single `(0.10, 0.10)` retry, highest-score box selection — and to the same pixels; §6 is about the
"same pixels" half, which is the one that could have been silently false.

The capture is `runs/pr08-est-drift/capture-mujoco-trajectory-f480`: 480 scene states on the
`trajectory` schedule, `configs/sim/g1_scene.xml` through the `head` camera at 480x640, median
interframe ground-truth centroid motion **1.317 px** and max **43.65 px** (the max is an occlusion
transition behind the Dex3 fingers, not a jump cut). That temporal coherence is what makes the clip
propagatable from frame 0 at all; the committed lattice capture is not, because its neighbours
teleport the object.

## 3. Limb (a) is refuted in direction at p95 — and is *correct* further out

The full percentile comparison, from both arms' `centroid_displacement.percentiles_px`:

| percentile | per-frame (px) | propagation (px) | worse arm |
|---|---:|---:|---|
| p0 | 0.0017 | 0.0187 | propagation |
| p5 | 0.0197 | 0.1046 | propagation |
| p25 | 0.0458 | 0.1783 | propagation |
| p50 | 0.0714 | 0.2543 | propagation |
| p75 | 0.1224 | 0.3370 | propagation |
| p90 | 0.2117 | 0.4218 | propagation |
| **p95** | **0.2908** | **0.4701** | **propagation** |
| p99 | 1.0679 | 0.5627 | per-frame |
| p100 | 2.1837 | 0.6487 | per-frame |

Two things are true at once here and it is worth being exact about both, because a summary that
kept only the headline would be misleading.

**The mechanism blocker 3 describes is real.** The per-frame arm's error is near zero almost all of
the time (median 0.071 px) with occasional spikes; the propagated arm's error is a small, steady,
systematic offset (min 0.019 px, mean 0.265 px, std 0.113 px, max 0.649 px). That is exactly
"independent re-detection jitters where propagation is temporally smooth", and it shows: the
per-frame arm is the worse arm at p99 (1.068 vs 0.563) and at p100 (2.184 vs 0.649).

**But the ordering reverses below p99, and PR-08 §6 subtracts a p95.** Across the whole body of the
distribution up to and including p95 the propagated centroid sits further from ground truth than
ours does — 1.62x further at p95 — because a steady 0.25 px offset dominates jitter that is usually
0.07 px. Limb (a)'s conclusion, *"our tail … is INFLATED relative to the generator's"*, is a claim
about the statistic PR-08 actually uses, and at that statistic it is false on this capture. Our
p95 **understates** the propagated arm's by 0.179 px.

One asymmetry runs the same way and makes the delta a conservative estimate of the understatement:
the per-frame arm's p95 is computed over 479 frames because frame 345 was dropped, and frame 345 is
that arm's single total failure (§7). Its own worst frame is not in its own tail. The propagation
arm's p95 is over all 480.

## 4. What a too-small `EST_DRIFT_P95` does to §6's subtraction

PR-08 §6 (`docs/preregistration/PR-08-photoreal-augmentation.md:175`) holds the generator to
`GEOM_TOL − EST_DRIFT_P95`, "if that is ≤ 0, the estimator is not good enough and generation does
not start". I checked how that subtraction is actually consumed rather than assuming it.

**`scripts/run_g0_gates.py`.** `gate_budget()` (line 809) reads both terms from the committed
`configs/transfer25/pr08_geom_tol.json`, computes `margin = tol_px - drift_px`, and returns it as
`gate_margin_px`. G0b then uses that number as a hard per-clip threshold: the criterion block
records `"rule": "displacement <= {budget} px (GEOM_TOL - EST_DRIFT_P95)"`, applied to a percentile
of per-frame centroid displacement taken **per clip** (default p100), and a clip whose statistic
exceeds the budget is a VOID row. Three refusals guard it: a null `geom_tol_px` refuses, a null
`est_drift_p95_px` refuses, and a margin `<= 0` refuses quoting §6. The runner also refuses an
`est_drift_p95_px` whose segmenter the document does not name, and verifies both halves came from
the same segmenter on the same pixel grid.

**`scripts/measure_geom_tol.py`.** It measures only the `GEOM_TOL` half and says so
(lines 147, 169-172): it writes `est_drift_p95_px: null` together with `est_drift_p95_blocked_by`,
and `refuse_est_drift_without_estimator()` (line 1983) refuses to write an `est_drift_p95_px`
without the segmenter that produced it. The two halves are measured by two scripts into two files
and merged into one document.

**The direction, therefore, is not in doubt.** `est_drift_p95_px` enters with a minus sign and
nowhere else. A smaller value produces a **larger** margin, a **wider** G0b tolerance, and more
restyled clips that pass. `run_g0_gates.py`'s own docstring makes precisely this argument for the
degenerate case, in its own words:

> A drift budget assumed to be zero WIDENS the tolerance, which looks conservative and is
> backwards, and PR-08 §4 records `EST_DRIFT_P95` as a LOWER bound on the real error in the first
> place

The measurement in §2 is the same argument at 0.179 px instead of 0.291 px. If the number that goes
into the subtraction is the per-frame arm's p95 while the segmenter the generator actually runs is
the propagation arm, then on this capture the gate hands the generator **0.179 px of tolerance it
should not have**, and a G0b margin that clears only under the per-frame number is not a pass. That
is blocker 3's own closing sentence, now with a measured sign attached to it.

**Nothing is currently mis-gated by this, because nothing is gated at all.**
`configs/transfer25/pr08_geom_tol.json` today records `geom_tol_px: null`, `est_drift_p95_px:
null` and `gate_margin_px: null` (`spec_version` 1.1.0, `measurement_fields` all empty), so
`gate_budget()` refuses before it can subtract anything. This document changes that file in no way
and proposes no value for either term. It records which direction the error goes for whoever
eventually fills them in.

Two smaller notes belong here rather than buried. The artifact stamps `is_lower_bound: false` with
a reason arguing that MuJoCo's non-photoreal rasteriser makes a detector do *worse*, so the p95 is
*larger* and the error lands against the generator — and the same field records
`error_direction_measured: false`, i.e. that is an argument about the **photorealism** axis and
not a measurement. It says nothing about the **topology** axis, and on the topology axis this
comparison is a measurement, and it points the other way. Separately, the propagation arm
deliberately does **not** apply `apple_sam2`'s mask-validity colour filter, because
Cosmos-Transfer2.5 has no such filter and applying a per-frame refusal to a propagated mask would
erase the very failure limb (b) is about; the artifact records
`n_frames_the_colour_filter_would_have_refused: 0`, so on this capture the choice made no
difference to the numbers.

## 5. Limb (b) was not observed — and what "not observed" is worth

`low_iou_runs` exists for exactly one reason, and the artifact states it:

> 'Propagation drifts off the object and STAYS off for a run of frames' does not show up in a mean,
> does not reliably show up in a p95, and cannot show up at all in a per-frame estimator that
> recovers on the next frame — a per-frame arm scattering the same count of bad frames across the
> capture and a propagation arm losing the object for a contiguous stretch produce the SAME IoU
> distribution. They produce different runs.

At threshold 0.5, over all 480 frames, with `n_unscored_frames: 0` and
`unscored_frames_break_a_run: true`:

| | per-frame | propagation |
|---|---|---|
| `n_runs` | 1 | **0** |
| `longest_run` | 1 | **0** |
| `n_frames_in_runs` | 1 | **0** |
| `runs` | `[[345, 345]]` | `[]` |

**The propagation arm never lost the object.** Its worst frame of 480 scored IoU 0.879 against the
renderer's exact geom-id ground truth (`n_frames_zero_iou: 0`, `n_frames_below_half: 0`); its p1 is
0.916. The mask it seeded on frame 0 from box `[295.90, 262.52, 343.80, 313.11]` tracked the apple
through 480 frames including the occlusion transitions, with `n_frames_empty_propagated_mask: 0`.
The single sub-0.5 frame in the whole comparison belongs to **our** arm, not to the generator's.

That is genuine evidence and it should be read as exactly what it is: **evidence of absence on this
capture**, and on nothing else. The statistic that could have shown limb (b) was built, it was
computed, and it reported nothing. It did not report "the failure mode does not exist"; it reported
that this 480-frame MuJoCo trajectory contains no instance of it. §8 is why that distance matters
more than usual here.

It is also worth recording that propagation is *consistently slightly worse* in mask IoU without
ever being *badly* worse: 34 of 480 frames below IoU 0.95 against the per-frame arm's 19, mean
0.9763 against 0.9821, but a floor of 0.879 against the per-frame arm's floor of 0.0. The two arms
fail differently, which is the whole content of blocker 3 and is now visible in numbers rather
than in prose.

## 6. Why the delta is attributable to propagation and not to a codec

This is the part of the experiment that could have been void without looking wrong, and the
propagation module was built around it.

`SAM2VideoPredictor.init_state` conventionally ingests **a directory of JPEG files** —
`sam2.utils.misc.load_video_frames` accepts a JPEG folder or an MP4 and raises `NotImplementedError`
on anything else. Our captures are lossless `rgb.npy`. Had the propagation arm been driven the
conventional way while the per-frame arm read the raw arrays, the difference between the two p95s
would have been **the codec plus propagation, in unknown proportions, reported as propagation**.
The number would have been plausible, gate-shaped, and meaningless.

`scripts/estimators/apple_sam2_video.py` therefore encodes nothing. `frames_to_normalized_tensor`
performs upstream's own `_load_img_as_tensor` arithmetic — PIL resize to the model grid, `/255`,
then the ImageNet mean/std — directly on the `uint8` arrays the harness already holds, and
`propagate()` installs it in place of `load_video_frames` for the duration of **exactly one**
`init_state` call. Everything else `init_state` does, and the whole of `propagate_in_video`, is the
installed `sam2` package's unmodified code.

`tests/test_apple_sam2_video_propagation.py` is the demonstration, and it is worth saying what it
actually asserts rather than paraphrasing "it tests the ingest". On a 4-frame synthetic clip built
with hard edges and saturated colour — structure a codec visibly disagrees with:

- `test_the_in_memory_ingest_is_bitwise_upstreams_ingest_of_a_lossless_file` writes the same frames
  as **PNG**, runs upstream's own `misc._load_img_as_tensor` over them, applies the ImageNet
  normalisation, and asserts `torch.equal(ours, theirs)` — bitwise equality of the float tensors,
  not closeness — plus that both routes report the source height and width.
- `test_a_jpeg_route_would_have_changed_the_pixels_which_is_why_there_is_no_jpeg_route` writes the
  same frames as **JPEG at quality 95**, reads them back, ingests them through the same function,
  and asserts `not torch.equal(lossless, through_jpeg)`. The confound is demonstrated to be real,
  rather than assumed away. Its docstring notes that if this test ever went green the other way,
  the argument above would be wrong and worth re-reading.
- `test_the_source_contains_no_encode_path_at_all` greps the module's own source for encode calls.
- `test_this_module_discharges_nothing` asserts `apple_sam2.GATE_QUALIFIED is False` and
  `len(apple_sam2.GATE_QUALIFICATION_BLOCKERS) == 3`.

Per-run evidence that it happened is in the artifact as well, not only in the tests:
`arm_comparison.identical_input_pixels` records 480 common frames, 480 shown to each arm,
`equal: true`, `frames_that_differ: []`, and the same chained SHA-256 for both arms —
`7cea8916f3c5db93ccb875be1dfbe775037fac570757eebb99b29134d2ca4f41` — re-derivable from the capture
with six lines and no other input.

**So the 0.179 px is propagation.** It is not JPEG.

## 7. Frame 345, the one bad frame, belongs to us

`runs/pr08-est-drift/FRAME345_DIAGNOSIS.json` re-ran `apple_sam2.segment()` **unmodified** on
`frames/000345/rgb.npy` and compared it against `seg_ids.npy`. The verdict:

> FALSE NEGATIVE, not a wrong-object mask. The estimator returned an EMPTY mask on a frame where
> the renderer's exact ground truth carries 1100 px of visible apple.

`ground_truth_apple_px: 1100`, `estimated_mask_px: 0`, IoU 0.0. The obvious hypothesis — that the
`apple.` prompt had latched onto the scene's `cube` geom, 1688 px visible on that frame, which is
precisely blocker 1's failure mode — is **refuted by the same file**:
`fraction_of_estimated_mask_on_cube: 0.0`. The estimator did not pick the wrong object; it picked
nothing. The apple is partially occluded by the Dex3 fingers there, and neighbours 343/344/346/347
carry 1243/1172/1035/976 ground-truth pixels and score normally, so it is a single-frame miss
inside a run of comparable frames rather than a visibility cliff.

Two consequences for this document. First, the per-frame arm's one low-IoU run is a **silent honest
failure** — a dropped frame, not a confident false measurement — which is the benign direction of
the two. Second, and the diagnosis file says this itself, that is a fact about a *simulator*
capture: `runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json` records the same adapter producing
confident full-**plate** masks at scores 0.155-0.259 on occluded frames of episode_000094, i.e. on
the real corpus the observed failure **is** the wrong-object mask that this sim frame is not. The
two failure modes do not match.

## 8. The two caveats, stated where they cannot be skipped

### 8.1 The capture is MuJoCo. The blocker says Isaac.

Blocker 3's discharge condition reads *"measuring the same **Isaac** capture BOTH ways"*. This
capture is `backend: mujoco`, `ground_truth_route: mujoco`, `MuJoCoGroundTruthBinding` over
`configs/sim/g1_scene.xml`.

**This measurement therefore does not satisfy the condition as written.** PR-08-V5 (T40_RULE_V5)
rerouted §4 step 1's ground truth from Isaac to any simulator with exact per-pixel segmentation —
but it did so for a different purpose and without addressing this blocker, which names Isaac in its
own words and was written after. Whether a MuJoCo capture may stand where the blocker says Isaac is
a **rule question for the project owner**. The artifact registers it as `open_rule_question` and
adds the sentence this document repeats rather than dilutes: *"No session may answer it by writing
a capture and pointing at it."*

The capture's own `object_limitations` block travels with every number above and is not softened
here: the apple is an untextured 14-group convex-decomposition proxy mesh, a static prop teleported
between scene states, drawn by a rasteriser that is neither ray-traced nor photoreal, with a `cube`
distractor left in the scene and `distance_to_image_plane` depth semantics. The real AppleToPlate
corpus is a real apple on a real tablecloth through a D435.

### 8.2 480 frames of one trajectory is not a corpus.

The real corpus is **402 episodes, 171 625 frames**
(`runs/pr08-geom-tol/pr08_geom_tol.json`, `--merge`, 2026-08-24, 16 shards). This comparison is
**480 frames of one trajectory**, one seed, one camera, one lighting condition, one object pose
schedule.

That matters most for §5. A single trajectory can trivially contain no drift event while the corpus
contains many: limb (b) is a claim about a *failure that happens sometimes*, and the rate at which
it happens is not estimable from one clip that had none. Zero runs out of one trajectory is
consistent with a corpus-wide rate that would matter a great deal. Reading `n_runs: 0` as "SAM 2
video propagation does not drift" would be exactly the inference this document declines to make.

It matters for §3 too, though less. The 0.179 px delta is one capture's number, and the propagated
arm's advantage in the extreme tail (p99, p100) versus its disadvantage through the body is a shape
that could plausibly move with clip length, seed frame quality, or how much of the clip is occluded.

## 9. What this does NOT establish

- **Not a discharge of blocker 3.** The evidence its discharge condition names now exists in the
  shape it names — two arms, one capture, two p95s recorded side by side — but the condition says
  *Isaac* (§8.1), and producing the evidence a blocker asks for is a different act from accepting
  it. Accepting it is a reviewable edit to `GATE_QUALIFICATION_BLOCKERS` made by a person, moving
  the retired wording into `GATE_QUALIFICATION_DISCHARGED` with the evidence attached. No such edit
  is made here or licensed by here.
- **Not a discharge of blockers 1 or 2.** Nothing measured here touches "nobody has looked at a
  mask" or the retry's contribution. The artifact says so in its own `discharges` field.
- **Not that limb (b) is false.** §5. Zero runs on one 480-frame MuJoCo trajectory is evidence
  about that trajectory.
- **Not that our per-frame estimator is good.** It has the only sub-0.5 IoU frame in the comparison
  (§7), and its mask being close to ground truth *in a rasterised simulator against a proxy mesh*
  says nothing about its masks on the real corpus, where the recorded failure mode is different.
- **Not a value for `EST_DRIFT_P95`, and not a proposal for one.** `configs/transfer25/
  pr08_geom_tol.json` is untouched and both its measurement terms remain `null`. §4 records the
  *sign* of an error, not a number to write anywhere.
- **Not a licence to generate.** G0b's budget cannot even be computed today, and if it could, §4 is
  a reason it would be too wide rather than a reason to trust it.
- **Not a bound.** No `max_frame_fraction` is written, proposed or implied.
- **Not a statement about GR00T.** PR-07 §6 forbids it, and nothing here bears on it.

---

## 10. Provenance

| | |
|---|---|
| kind | reading of an existing measurement. **Registers no rule, measures no new pixels, edits no code** |
| date | 2026-08-26 |
| primary artifact | `runs/pr08-est-drift/EST_DRIFT-ARMS-mujoco-trajectory-f480.json` (`measured_utc` 2026-08-25T17:43:26+00:00, `arm_comparison`) |
| capture | `runs/pr08-est-drift/capture-mujoco-trajectory-f480` — 480 frames, `trajectory` schedule, MuJoCo, 480x640, `head` camera, `configs/sim/g1_scene.xml` |
| arms | per-frame `scripts/estimators/apple_sam2.py`; propagation `scripts/estimators/apple_sam2_video.py` |
| headline | per-frame p95 **0.29077062684224225 px**, propagation p95 **0.47006167975525187 px**, delta **+0.17929105291300962 px** |
| limb (b) statistic | `low_iou_runs` @ 0.5 — per-frame 1 run `[[345, 345]]`; propagation **0 runs** |
| same-pixels evidence | `identical_input_pixels`, both arms SHA-256 `7cea8916…4f41`, `equal: true`; `tests/test_apple_sam2_video_propagation.py` |
| supporting artifacts | `runs/pr08-est-drift/FRAME345_DIAGNOSIS.json`, `runs/pr08-geom-tol/pr08_geom_tol.json` (corpus size), `runs/pr08-mask-audit-local-cpu/MODEL_OBSERVATIONS.json` |
| consumer checked | `scripts/run_g0_gates.py:809` `gate_budget()`, `scripts/measure_geom_tol.py:147,169-172,1983` |
| open rule question | MuJoCo where blocker 3 says Isaac — **open, the project owner's** (§8.1) |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched — the artifact itself records `gate_qualified: false` and `estimator_not_gate_qualified` |
| generation licensed | **no** |
| training licensed | **no** |
