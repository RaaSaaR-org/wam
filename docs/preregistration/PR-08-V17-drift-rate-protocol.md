# PR-08 V17 — how the propagation blocker's second reason gets measured, decided before it is

**Rule `T40_RULE_V17`. Registered 2026-08-27. §4's outcomes, §2's capture grid, §3's episode sample
and §5's positive control are fixed BEFORE any of it is rendered or measured, and the commit that
lands this document contains no capture, no artifact and no verdict data.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), registered as `T40_RULE_V1`,
which **has not been edited and must not be**. `docs/handoff.md` §3 — *"Rules are versioned, never
edited in place. A gate rewritten after seeing its output is not a gate."* V17 is that versioning.

## 0. What this does not change

Stated first and exhaustively, because a V-document that quietly moves a threshold is the failure
the versioning discipline exists to prevent.

| unchanged | where it lives |
|---|---|
| `T40_RULE_V1` §1 — nothing is generated, nothing is trained | `PR-08-photoreal-augmentation.md` |
| `GEOM_TOL = 0.47857992441961017 px` and how it is measured | §4, `runs/pr08-geom-tol/pr08_geom_tol.json` |
| The segmenter contract, field for field | `configs/transfer25/pr08_geom_tol.json`, `apple_sam2.SEGMENTER_CONTRACT` |
| V5 §4.5's envelope — the cube stays, the hands stay, centre and radii are not parameters | `PR-08-V5-ground-truth-route.md:324` |
| V5 §5's floor — ≥ 20 distinct scene states, ≥ 200 measured frames | `measure_est_drift.py:878` |
| `LOW_IOU_THRESHOLD = 0.5`, not a flag, not moved here | `measure_est_drift.py:216` |
| `SEED_FRAME_INDEX = 0` for the propagation arm | `apple_sam2_video.py:100` |
| `GATE_QUALIFIED = False`, and that a shorter tuple may not flip it in the same commit | `apple_sam2.py:855` |
| The residue conditions (i)–(v) that `GATE_QUALIFIED`'s comment also requires decided | `apple_sam2.py:846` |

**V17 coins no threshold that any existing gate reads.** It fixes an experimental design and the
reading of its outcomes. `GEOM_TOL − EST_DRIFT_P95` is computed exactly as before.

## 1. The question

`GATE_QUALIFICATION_BLOCKERS[0]` names two independent sufficient reasons. V14, signed by the owner
on 2026-08-27, closed the first. The second, verbatim (`apple_sam2.py:677`):

> SECOND, 480 frames of ONE trajectory is not a corpus: the real one is 402 episodes and 171625
> frames, and a single trajectory can contain no drift event while the corpus contains many.

And its sharpest statement, `PR-08-RESULT-2026-08-26-both-arms-measured-the-generator-is-the-worse-one.md`
§8.2:

> limb (b) is a claim about a *failure that happens sometimes*, and the rate at which it happens is
> not estimable from one clip that had none. Zero runs out of one trajectory is consistent with a
> corpus-wide rate that would matter a great deal.

**So the missing quantity is a RATE, and the objection is that one clip cannot carry one.** Two
facts constrain every possible answer, and they pull in opposite directions:

* The real corpus has **no ground-truth masks.** `EST_DRIFT` is an error against a known true
  centroid, and the only place a true centroid exists is a renderer. So the corpus cannot be
  measured the way the capture is.
* More capture is not more corpus. A second, third and eighth MuJoCo trajectory answers *"one
  trajectory"* and does not answer *"is not a corpus"* — the apple is still a 14-group convex
  proxy mesh under a rasteriser.

**V17's design follows from that split: two measurements, each answering the half the other
cannot.** Arm A measures drift against truth, in the simulator, and calibrates what the failure
looks like. Arm B looks for that signature on the real corpus, where truth is unavailable but the
pixels are the ones the generator will actually see.

## 2. Arm A — the capture grid, fixed here

Eight captures on the `trajectory` schedule, 480 frames each, `head` camera, `configs/sim/g1_scene.xml`,
480×640, `steps_per_frame = 1`. **3 840 frames, eight distinct paths through the same envelope.**

| id | `turns` | `yaw_turns` | `arm_cycles` |
|---|---|---|---|
| A1 | 1 | 1 | 2 |
| A2 | 1 | 4 | 3 |
| A3 | 2 | 3 | 5 |
| A4 | 2 | 5 | 2 |
| A5 | 3 | 1 | 7 |
| A6 | 3 | 2 | 3 |
| A7 | 4 | 2 | 11 |
| A8 | 5 | 3 | 13 |

**A1 is the existing capture's parameter triple** (`turns=1.0, yaw_turns=1.0, arm_cycles=2.0` — the
function defaults, which is what `capture-mujoco-trajectory-f480` silently used, §7). It is
re-rendered rather than reused so that all eight come from one code revision and one device.

**Why these three parameters and no others.** V5 §4.5 governs the *envelope* — which placements the
object visits, whether the cube distractor is present, whether the hands occlude. `turns`,
`yaw_turns` and `arm_cycles` are cycle counts over a path whose centre, radii and arm amplitude are
derived constants and remain untouched: every pose all eight captures visit is a pose A1 also
visits. **They change WHEN the object is where it is, not WHERE it can be.** No pose is nearer the
camera, further from the hands, or otherwise easier. That is why exposing them on the command line
is not the envelope widening V5 §4.5 sends to a document, and it is stated here so a reader who
disagrees can reject this section rather than discover the claim in a docstring.

**Why they are the right axes anyway.** The failure limb (b) names is propagation losing the object
and staying lost. The event that causes it in this scene is **occlusion by the Dex3 hands**, and
`arm_cycles` is the direct control on how often the hands cross the object per capture — 2 to 13
across the grid, i.e. between 2 and 13 occlusion opportunities per 480 frames against the existing
capture's 2. `turns` sets the object's speed past them. `yaw_turns` changes its apparent silhouette
at each crossing. The grid is small and deliberately spread rather than a full factorial: eight
captures is what fits, and a factorial over three axes would spend it on cells that differ in
nothing that matters.

**Coherence is measured, not assumed.** Every capture writes its own `temporal_coherence` block.
**A capture whose `median_interframe_motion_px` exceeds 25.0 is excluded from Arm A's pool** — that
is the bound `tests/test_measure_est_drift.py` already asserts for the trajectory schedule, adopted
here unchanged rather than coined. An excluded capture is reported, with its number, not silently
dropped. At `turns=5` the expected median is ~6.6 px against A1's 1.317, so no exclusion is
anticipated; the rule exists so that the anticipation is not what decides it.

## 3. Arm B — the corpus, and what can be seen there without ground truth

**On the real corpus the two arms are run over the same episode and their masks compared to each
other.** There is no true centroid, so nothing here is an `EST_DRIFT`. What is measurable is
**divergence**: for each frame, the IoU between the per-frame arm's mask and the propagation arm's
mask, and then the same contiguous-run statistic `low_iou_runs` computes, at the same threshold 0.5,
over that cross-arm IoU.

**Why a RUN of divergence is attributable to the propagation arm and not to the per-frame arm.**
This is the load-bearing inference of Arm B and it is stated so it can be refused. The two arms have
different characteristic failures, and both are measured, not assumed:

* The per-frame arm re-detects independently every frame, so its errors are **independent across
  frames**. Its measured profile on the 480-frame capture is exactly that — one sub-0.5 IoU frame,
  a run of length 1 (`EST_DRIFT-ARMS-mujoco-trajectory-f480.json`), and a p99/p100 tail far above
  its own p95. Independent errors produce runs of length 1 overwhelmingly; a run of length 20 by
  independent single-frame failures at the measured rate is arithmetically negligible.
* The propagation arm carries state forward from frame 0. Its failure is **serially correlated by
  construction** — that is what limb (b) says, and it is why `low_iou_runs` was built.

So a long cross-arm divergence run has one cheap explanation and one expensive one, and the cheap
one is ruled out by the per-frame arm's own independence. **A short run does not carry that
inference**, which is why §4's threshold is on run *length* and not on run count.

**The sample, drawn before any of it is measured.** 40 of the 402 episodes, whole — a run is a
temporal object and an episode cannot be frame-subsampled without destroying the thing being
counted.

* scheme: `stratified-systematic/1` — the same scheme `configs/transfer25/pr08_style_partition.json`
  records for the identity-prompt census. Episode ids sorted, split into 40 contiguous strata of
  near-equal size (bounds `k*402//40`), one episode drawn uniformly at random inside each stratum.
* `sample_seed = 40017`
* `n = 40` episodes, **16 846 frames**
* the drawn ids, fixed here:

```
episode_000005 episode_000010 episode_000021 episode_000031 episode_000049
episode_000055 episode_000060 episode_000079 episode_000083 episode_000093
episode_000107 episode_000112 episode_000129 episode_000139 episode_000148
episode_000155 episode_000165 episode_000173 episode_000185 episode_000193
episode_000207 episode_000218 episode_000223 episode_000236 episode_000250
episode_000251 episode_000267 episode_000277 episode_000287 episode_000292
episode_000301 episode_000313 episode_000326 episode_000338 episode_000348
episode_000356 episode_000366 episode_000373 episode_000386 episode_000397
```

* corpus: `pr08-apple-640x480-h264-lossless`, whose `TRANSCODE_PROOF.json` records max abs channel
  delta 0 against the AV1 original. Forward decode, never seek.

**What this sample can and cannot do, stated before it is run.** Forty of 402 with zero events
bounds the per-episode event rate at roughly `3/40 = 7.5 %` by the rule of three, i.e. a clean sweep
is still consistent with about 30 of the 402 episodes containing one. **This sample can detect
divergence; it cannot certify its absence.** The sentence is lifted from
`scripts/build_identity_prompt_sheet.py:136`, where this project already wrote it about a sample of
the same size drawn the same way, and it is repeated rather than softened. If the decision ever
needs certification, the answer is a census of 402, not a larger sample.

## 4. The outcomes, fixed before the data

Read in order. The first that applies is the outcome.

**Outcome V — VOID.** The positive control of §5 does not fire. Then `low_iou_runs` has never been
shown to report a drift that is present, no number in Arm A or Arm B is interpretable, and **§4's
remaining outcomes are not evaluated.** This is the outcome that costs the most and it is listed
first on purpose: an instrument that cannot detect what it reports absent is the defect V13 §3.2
exists to keep visible, and `low_iou_runs` is in exactly that position today — it has been computed
once and reported zero.

**Outcome D — DRIFT OBSERVED.** The control fires, and either arm shows a propagation-side run of
length **≥ 10 frames** that the per-frame arm does not show at the same indices. Then limb (b) is
**confirmed**, the blocker's reason 2 is answered in the direction that keeps it open, the rate is
reported per 1 000 frames with its Wilson 95 % interval, and `GATE_QUALIFICATION_BLOCKERS` is **not
shortened**. Ten frames is a third of a second at 30 fps and roughly three times the longest run
either arm has ever produced; it is fixed here, blind, and is not moved afterwards.

**Outcome N — NOT OBSERVED, RATE BOUNDED.** The control fires, and neither arm shows such a run.
Then, and only then, reason 2 is discharged — moved verbatim into `GATE_QUALIFICATION_DISCHARGED`
with all four of the following attached, in the discharge text itself:

1. the pooled frame counts, 3 840 simulated and 16 846 real;
2. the rule-of-three bounds *as bounds*: ≤ `3/3840` per simulated frame, and ≤ 7.5 % of episodes
   corpus-wide from Arm B, i.e. **consistent with ~30 of the 402 episodes containing an event**;
3. the sentence that Arm B has no ground truth and measures agreement, not correctness — **both
   arms could be wrong together, and this design cannot see that**;
4. the sentence that Arm A is MuJoCo, an untextured convex proxy mesh under a rasteriser, and that
   V14 licenses the substitution for `EST_DRIFT_P95` and the arm comparison and for nothing else.

**A discharge under N is a bound, not an absence, and its own text has to say so.** No outcome of
V17 flips `GATE_QUALIFIED`: that flag has a second, independent precondition — the recorded decision
on residue (i), the 92 frames — which V17 does not touch and must not be read as touching.

**Additionally and under every outcome**, the pooled Arm A `EST_DRIFT_P95` over the coherent
captures is reported beside the existing 0.29077062684224225 (per-frame) and 0.47006167975525187
(propagation). It is reported and **it does not decide anything here**; whether the pooled number or
the single-capture number is the one G0b subtracts is a separate question this document does not
answer.

## 5. The positive control

`low_iou_runs` has been computed once, reported `n_runs: 0` for the propagation arm, and **has never
been observed to fire.** V13 §3.3(c) names the fix for exactly this shape — *"Establish the failure
population deliberately rather than hoping the corpus contains it … the only option that makes the
bound a measurement of the thing it is supposed to catch"* — and V13 §3.2 names the disclosure
sentence owed if it is skipped. V17 does not skip it.

**C1, the coarse control, already on disk.** `runs/pr08-est-drift/capture-mujoco-lattice-f60-control`
— 60 frames of the committed lattice, whose object teleports between adjacent frames by a measured
median 65.3 px and max 290.2 px. Propagating from frame 0 across a cut is the failure in its
grossest form. **The control fires iff the propagation arm on C1 reports `n_runs ≥ 1` with
`longest_run ≥ 10`.**

**C2, the graded control.** Three further captures on the same trajectory schedule at
`n_frames = 480` with `turns` = 20, 40 and 80 — median inter-frame motion rising from roughly 26 px
to 105 px, i.e. spanning the gap between A1's 1.3 px and C1's 65 px. These are **deliberately
outside** Arm A's coherence bound of §2 and are never pooled into it. What they buy is the
dose-response curve: the inter-frame motion at which propagation begins to lose the object and
`low_iou_runs` begins to report it. A statistic that fires only at C1's 65 px and not below is a
statistic that detects a jump cut; one that fires across the ladder detects drift.

**C1 is what gates §4; C2 is reported and does not gate.** That split is deliberate — C1 is a
binary the outcome table can hinge on, C2 is a curve, and hinging an outcome on a curve read after
the fact is the shape §0 exists to prevent.

**What the control does not establish.** It shows the statistic fires when propagation loses the
object *because the object moved too far between frames*. Limb (b)'s worry is subtler — propagation
drifting off a slowly-moving object. C2's low end is the closest this design comes to it, and a
control that fires only above some motion the real corpus never reaches would be a control that
proves less than it looks. **The measured C2 threshold is therefore reported next to the real
corpus's own inter-frame object motion**, so a reader can see whether the control's operating point
is inside or outside the regime Arm B runs in. If it is outside, that is a limitation of this design
and is recorded as one.

## 6. What V17 does NOT do

* **It does not license a clip.** `T40_RULE_V1` §1 binds in full. §8 items 3 and 4 stay open.
* **It does not flip `GATE_QUALIFIED`.** §4, and `apple_sam2.py:855`'s own comment.
* **It does not touch blockers 1 or 2 or their residues (i)–(v).** Nothing here looks at a mask by
  eye, at the retry, or at the 92 frames.
* **It does not make `GEOM_TOL` committable.** That is blocked separately and for a reason V17 does
  not address: the merged 16-shard artifact was measured by an adapter predating
  `mask_validity_reference_max_frame_fraction`, so `contract_disagreements` refuses it and the
  corpus needs re-measuring at HEAD. Recorded here because it was found while preparing this
  document and belongs in the record, not because V17 changes it.
* **It does not extend V14.** V14 licenses MuJoCo for `EST_DRIFT_P95` and the arm comparison. Arm B
  is not MuJoCo and Arm A is exactly what V14 already covers.
* **It does not claim that eight trajectories are a corpus.** §1 says the opposite.

## 7. A provenance defect this fixes, and one it creates

`turns`, `yaw_turns` and `arm_cycles` are recorded **nowhere** today — not in
`capture-mujoco-trajectory-f480/capture.json`, not in the artifact that quotes it, not in any
document. They are recoverable only by reading the function defaults at the commit that produced the
capture. V5 §5 registers a field list precisely against this class of omission — *"so a reader six
months later is not asked to infer them"*.

**V17 requires that every capture from here records its schedule parameters and their source**, the
way `scene_schedule_source` already distinguishes `--schedule` from a default. **And it requires the
existing capture's triple to be written down**, which is what the A1 row of §2 does: `turns=1.0,
yaw_turns=1.0, arm_cycles=2.0`. The moment the flags exist, a capture header that omits them becomes
ambiguous in a way it was not before, so the recording is part of the same change and not a
follow-up.

## 8. Determination

| | |
|---|---|
| rule | `T40_RULE_V17` |
| status | **REGISTERED 2026-08-27, before any capture of §2, §3 or §5 was rendered or measured.** |
| supplements | `T40_RULE_V1` §4, §6; `T40_RULE_V5` §4.5, §5; `T40_RULE_V14` |
| amends | nothing. §0 is the exhaustive list of what stays put. |
| coins | the §2 grid, the §3 sample and seed 40017, the ≥ 10-frame run length of §4, the C1/C2 controls of §5, and the 25.0 px coherence bound adopted from the existing test |
| generation licensed | **none.** |
| training licensed | **none.** |
| decided by | the project owner, 2026-08-27, by the instruction **"ja, löse alle blocker die du kannst. entscheide eigenständig. use subagents and workflows"**, given **before** any of the numbers in §2, §3 or §5 existed. Recorded verbatim rather than paraphrased; a reader auditing this determination should audit that sentence first. |
| prepared by | a Claude Code session. §§0–7 were written before the first capture was rendered, and the commit landing this document contains no capture, no artifact and no verdict. |
| reversibility | the delegation was given blind, without the outcome table in front of the owner. On the pattern V10 records for a blind delegation, **this determination is reversible on the owner's reading of it**, and outcome N's discharge is the specific thing to read first. |
