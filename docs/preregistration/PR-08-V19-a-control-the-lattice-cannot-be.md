# PR-08 V19 — V17's positive control cannot fire, for a structural reason, and here is one that can

**Rule `T40_RULE_V19`. Registered 2026-08-27. §3's control and §4's prediction are fixed BEFORE the
control is run, and `T40_RULE_V17` §4's outcome thresholds are NOT touched — the 10-frame bar this
document is about was fixed blind before any control ran and is carried across unchanged.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md), registered as `T40_RULE_V1`,
which **has not been edited and must not be**. `docs/handoff.md` §3 — *"Rules are versioned, never
edited in place. A gate rewritten after seeing its output is not a gate."*

## 1. What C1 measured, and why V17 outcome V is the honest reading

V17 §5 named `runs/pr08-est-drift/capture-mujoco-lattice-f60-control` as the positive control and
fixed its fire condition:

> **The control fires iff the propagation arm on C1 reports `n_runs ≥ 1` with `longest_run ≥ 10`.**

Measured, `runs/pr08-est-drift/v17/EST_DRIFT-C1-lattice.json`:

| | per-frame arm | propagation arm |
|---|---|---|
| `EST_DRIFT_P95` | 0.1856 px | 0.3194 px |
| ground-truth IoU, p50 | 0.9868 | **0.0000** |
| `n_runs` | 3 | **10** |
| `longest_run` | 1 | **5** |
| frames in runs, of 60 | 3 | **31** |
| `n_unscored_frames` | 0 | **0** |

**The conjunction fails on its second half, so the outcome is V — VOID — and the threshold does not
move.** Registering it blind is what caught this: looking at *median IoU 0.0000 and 31 of 60 frames
below threshold*, any reader would say the control fired, and the criterion that was actually
written down says it did not.

**And the criterion is right, because C1's failure has a structural cause rather than a marginal
one.** The propagation arm's runs begin at frames 5, 10, 15, 22, 27, 32, 37, 47, 52, 57 — **spaced
exactly five apart**, and `default_scene_schedule` walks the object across a lattice whose yaw
period is 5 and whose x/y positions repeat on that stride. No run was broken by an unscored frame:
`n_unscored_frames` is 0. What breaks them is the object **returning under the stuck mask** every
fifth frame and pushing the IoU back over 0.5.

> **A lattice control cannot produce a run longer than the lattice's own period.** `longest_run = 5`
> is a measurement of the schedule, not of the statistic's sensitivity.

So C1 establishes exactly one thing — `low_iou_runs` **fires**: it reported 10 runs and 31 frames
where the mask was gone, against the 0 it has returned every other time it has been computed in this
project. It cannot establish that the statistic reports a run of the **length** V17 §4 outcome D
hinges on, and nothing about running it again or on more lattice frames would change that.

## 2. The objection to this document, stated before its own rule

**"Keep trying controls until one fires" is a real failure mode and this document is shaped like
it.** Four things are offered against that reading, and a reader who finds them insufficient should
refuse V19 rather than read past them:

1. **No threshold moves.** The 10-frame bar of V17 §4 outcome D and §5 was fixed blind before any
   control ran. It is carried into §3 unchanged. This document changes *which capture the control is
   run on*, which is the same shape V15 §4 permitted for itself — *"superseded by a version that
   changes the rendering, not by a version that reinterprets the tiles."* Here: a version that
   changes the control, not one that reinterprets the runs.
2. **C1's failure was diagnosed, not merely observed.** §1's cause is a measured periodicity that
   matches the schedule's own, with `n_unscored_frames = 0` excluding the other candidate
   explanation. That is a reason C1 *cannot* work, not a hope that something else will.
3. **C3's behaviour is predicted in advance, from ground truth, and the prediction can fail.** §4.
   If it fails, that is recorded and the outcome stays V.
4. **C1's result is not withdrawn, reinterpreted or deleted.** §1's table is what it measured, and
   what it did establish — that the statistic fires at all — stands and is cited as such.

## 3. The control

**C3 — the propagation arm, seeded on the WRONG OBJECT, over a coherent trajectory.**

* **Capture:** `runs/pr08-est-drift/v17/A1`, 480 frames, `turns=1, yaw_turns=1, arm_cycles=2` — the
  same capture Arm A pools, so the control and the thing it controls for run on identical pixels.
* **Seed:** the bounding box of the **cube distractor's ground-truth mask on frame 0**, read from
  the renderer's `seg_ids.npy` at geom id 107. **Not from the detector.** No prompt is changed, no
  threshold is moved, and the seed is a fact about the render rather than an output of the
  instrument under test — so the control cannot fail because the detector had a bad day.
* **Everything downstream is unchanged:** the same `SAM2VideoPredictor`, the same pinned revision,
  the same in-memory ingest, the same `logits > 0.0`, the same `LOW_IOU_THRESHOLD = 0.5`, the same
  ground-truth apple mask on the other side of the IoU.
* **It lives in its own module** (`estimators.apple_sam2_video_wrongseed`), reached only through
  `--propagation-module`, so nothing on the measurement path grows a seed override.
  `apple_sam2_video.SEED_FRAME_INDEX` and its "a sweep over seed frames would be a different
  experiment" stay exactly as they are.

**Fire condition, carried across from V17 §5 unchanged: `n_runs ≥ 1` and `longest_run ≥ 10`.**

## 4. The prediction, made before the control is run

The cube is **static and visible in every frame** of A1 (bbox ≈ `(215, 347)–(256, 392)` throughout),
while the apple travels the ellipse. Measured from **ground truth alone, with no propagation
involved**, over 48 frames of A1 sampled every 10:

> **max IoU(apple mask at frame *t*, cube mask at frame 0) = 0.2665, at frame 190. Frames reaching
> 0.5: zero.**

So *if* the propagated mask stays on the cube, every frame scores below 0.5 and the run is the whole
clip — and unlike C1 there is no periodic return, because A1 makes exactly **one** revolution in 480
frames. **The prediction is: `longest_run` on the order of the clip length, and the control fires.**

**Two ways it can fail, both of which are results and neither of which licenses a fourth control:**

* **SAM 2 recovers onto the apple.** The video predictor can re-acquire; if it does, the run is
  short, the control does not fire, the outcome stays **V**, and what has been learned is that this
  propagator does not *hold* a wrong seed — which is itself relevant to limb (b) and would be
  reported as such.
* **The seed produces an empty or degenerate mask.** Then the arms score `None` on those frames,
  runs break on unscored frames by design, and the control is uninformative. That is recorded as a
  failed control, in the shape V13 §3.2 requires, and **not** repaired by moving the seed until it
  works.

## 5. What V19 does NOT do

* **It does not change any outcome threshold.** §2 item 1.
* **It does not make V17 outcome V go away.** V17's pooled artifact recorded `outcome: V` against
  C1 and that artifact stands. If C3 fires, the pool is re-run **with C3 as the control**, and both
  results are reported side by side — the one that voided and the one that did not — because a
  reader who sees only the second cannot tell this document happened.
* **It does not license the C2 ladder to gate.** V17 §5 said *"C1 is what gates §4; C2 is reported
  and does not gate"*, and promoting C2 after seeing C1 fail is precisely the move §2 exists to
  refuse. C2 stays reported.
* **It does not touch `GATE_QUALIFIED`, `GATE_QUALIFICATION_BLOCKERS`, `T40_RULE_V18`, or §8 items
  3 and 4.** `T40_RULE_V1` §1 binds.
* **It does not establish sensitivity to a subtle drift.** C3 is a *held wrong seed*, which is the
  grossest version of limb (b). V17 §5's last paragraph stands unchanged: a control that fires only
  on a failure larger than the corpus produces is a control that proves less than it looks, and the
  C2 ladder is still the only thing in this design that speaks to the dose.

## 6. Determination

| | |
|---|---|
| rule | `T40_RULE_V19` |
| status | **REGISTERED 2026-08-27, after C1 was measured and before C3 was built or run.** |
| supplements | `T40_RULE_V17` §5. Replaces its control; carries its thresholds across untouched. |
| amends | nothing. No threshold in any document moves. |
| coins | nothing. The 10-frame bar is V17's, fixed blind. |
| generation licensed | **none** |
| training licensed | **none** |
| decided by | the project owner, 2026-08-27, by the instruction **"ja, löse alle blocker die du kannst. entscheide eigenständig. use subagents and workflows"**. Recorded verbatim. |
| prepared by | a Claude Code session, which measured C1 first and is the party with an interest in a control that fires — which is why §2 is where it is and says what it says. |
| reversibility | the delegation was given blind. **This determination is reversible on the owner's reading of it**, and §2 is the section to read first. |
