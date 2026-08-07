# PR-08 — Photoreal augmentation of the real AppleToPlate corpus

**Pre-registered 2026-08-06, before any clip is generated, before any weight is trained, and
before any job is submitted.** Rule `T40_RULE_V1` is fixed in §6 of this document. If it turns out
wrong the fix is a `V2` alongside it, never an edit (`docs/handoff.md` §3).

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen** —
fine-tuning a generator is [[T-041]] and is a different document.

---

## 1. What this licenses, and what it forbids

**Licenses:** writing the pipeline, committing the style partition, measuring the estimator error
budget (§4), and timing one episode on an H200 (§8 item 3).

**Forbids, until every item in §8 is closed and T-39 has reported:** generating a corpus, training
any weight on generated frames, and quoting any number from this document as a result.

Two standing decisions this document sets out to overturn, named so that neither is stepped over
quietly:

- **"Sim frames are NOT training data"** (`docs/sim.md`, T-25). §3 chooses the path that does *not*
  require overturning it. If §3's path is later abandoned for the Isaac path, this document is void
  and the sim decision has to be argued directly.
- **T-36 / PR-06 priced generated video as supervision and it lost** — the anchored dream scored
  **16.656** from the truth where holding the conditioning frame scored **12.020**, i.e. 39 % worse
  than standing still. §2 states why that result does not transfer, rather than assuming it.

**Relation to the PR-07 §7 freeze.** The freeze names T-32, *any Cosmos3-Super generation*, and
*any Cosmos3-Edge work*. Transfer2.5 is not named and this document is not covered by its letter.
It is covered by its reason: until T-39 reports whether **any** method clears the bar on this
corpus, "the data is wrong" and "the method is wrong" are not separable, and generating data is a
bet on the first. So the document is written now — it costs nothing — and **nothing is generated
until T-39 reports.** No amendment to PR-07 is required or made.

## 2. Why PR-06's negative does not settle this

PR-06 measured a **prediction**: given a conditioning frame, produce the *unseen* next frames, and
they were worse than not moving. This measures a **restyle**: given a frame that is already true,
change its appearance and keep its geometry. The difference is load-bearing for exactly one
reason, and it is not "different model":

**The labels do not come from the generator.** In PR-06 the generated pixels were the thing being
scored against truth. Here the actions are the recorded teleop trajectory, carried over unchanged,
and the generated pixels are only an input perturbation. A generator error in PR-06 was a wrong
answer; a generator error here is a corrupted training input whose label is still correct — unless
it moves geometry, which is what §6's G0b exists to catch.

That is the whole argument, and it is narrow. It does **not** imply the restyle will be good. It
implies PR-06's number does not predict this one.

## 3. The path: real teleop frames, Isaac used for calibration and not as corpus

T-040 leaves two paths open. This document picks the **real-teleop path** and states the cost.

| | Isaac path | **real-teleop path (chosen)** |
|---|---|---|
| frames restyled | sim renders | the 402 real demos |
| depth + segmentation | exact, but **not wired** — see the correction below | **estimated** from RGB |
| actions | sim teleop — a different corpus | the recorded trajectories, unchanged |
| collides with T-25 | directly | not at all |

**Correction to T-040's premise, checked against the code on `main` 2026-08-06.** T-040 states that
`isaac_transport.py` "already emits exactly those three [depth + segmentation + Canny]". **It does
not.** `isaac_binding.py` makes exactly one `AnnotatorRegistry.get_annotator` call and it is
`"rgb"`; `distance_to_camera`, `distance_to_image_plane`, `semantic_segmentation` and
`instance_segmentation` appear nowhere in `src/` or `tests/`. Replicator can produce all of them —
this is a small, well-understood wiring change, not a missing capability — but it is a code change
that has to be made and tested, and it is now §8 item 5. The correction does not change the path
decision (the Isaac path was rejected on the trajectories, not on the conditioning), but it does
mean **the calibration rig of §4 does not exist yet either**, and a plan that assumed it was free
would have been wrong about the ordering.

The Isaac path is rejected as the corpus because it changes the trajectories, which forfeits the
one property that makes this experiment cheap and readable: the labels are already correct and
already scored. An experiment on sim teleop episodes is not an augmentation of AppleToPlate, it is
a different dataset, and it would have to clear T-25 first.

**Isaac is still used — as the calibration rig, not the corpus.** Once the two annotators above are
attached, Isaac renders ground-truth depth and segmentation for the same G1 + Dex3. That is where
the monocular estimator's error budget is measured (§4). This is a deliberate substitution for
Humanoid Everyday's measured
depth, which T-040 proposed — and the substitution **survives OD-09**, which now permits training
on HE (2026-08-07). The reason is no longer the licence:

- Isaac's depth is **exact ground truth**; HE's is a RealSense measurement with its own error, so
  calibrating against HE means measuring one estimator against another sensor.
- Isaac renders **the AppleToPlate scene**; HE is 247 other tasks, many of them locomanipulation on
  a mobile base, where AppleToPlate is a gantry-mounted static G1. Estimator error is scene- and
  distance-dependent, so HE's budget would be transferred across a domain gap anyway.

HE remains the **confirmatory** measurement — it is the only real-camera ground truth on this
embodiment, and §4's stated weakness (synthetic renders make `EST_DRIFT_P95` a lower bound) is
exactly what it would settle. It is now available to do that; it is still not required.

*(Amended 2026-08-07, before this document's first commit and before any measurement was taken
under it. That is a draft edit, not an amendment to a registered rule — `T40_RULE_V1` in §6 is
unchanged and has never been in git under a different form.)*

**Resolution.** The converted corpus is 120×160 and a photoreal restyle at that size is worthless
to a VLA training at 640×480. This path re-derives from the HF source at full resolution and does
**not** reuse `datasets/gr00t-apple-full/`.

## 4. The conditioning signals do not exist, and the error budget is a gate input

Transfer2.5 consumes depth + segmentation + Canny. AppleToPlate ships one RGB camera (`ego`, from
one head RealSense D435 colour topic), so only Canny is computable. Depth and segmentation are
estimated, and **estimation error lands as geometry drift — exactly what G0b forbids.** Therefore
the estimator is characterised *before* generation, never after:

0. Attach the `distance_to_camera` and `semantic_segmentation` annotators in `isaac_binding.py`
   (§8 item 5) — they are not wired today, so step 1 cannot run yet.
1. Render N Isaac episodes with ground-truth depth + segmentation.
2. Run the same monocular depth estimator and the same segmenter on the Isaac **RGB only**.
3. Record the error distribution: absolute depth error, and object-centroid displacement in pixels
   between the estimated and the true segmentation.
4. The **95th percentile of that centroid displacement** is `EST_DRIFT_P95`, and it enters G0b's
   tolerance as a budget rather than being assumed to be zero.

**Stated weakness:** Isaac frames are not real frames, and a monocular estimator's error on
synthetic renders is not its error on RealSense footage — plausibly optimistic. So `EST_DRIFT_P95`
is a **lower bound** on the real error, it is recorded as such, and a G0b margin that only clears
under a lower bound is not a pass. This is the single biggest soft spot in the design and it is why
the HE licence request is worth sending even though it is off the critical path.

## 5. Arms

All four train the **same** policy under the same recipe on the same trajectories; only the pixels
differ. Recipe is fixed in §8 item 1.

| arm | training frames | what it isolates |
|---|---|---|
| **A `real-only`** | the 402 unmodified episodes | the control. Not trivial — it is the same policy, same budget |
| **B `real+restyled`** | A + restyles drawn from `TRAIN_STYLES` | the intervention |
| **C `real+identity`** | A + restyles whose style prompt *is the source's own appearance* | **generator-fingerprint control.** Same generator, same pipeline, same frame count, no added diversity. Separates "visual diversity helped" from "passing frames through a diffusion model helped" |
| **D `restyled-only`** | `TRAIN_STYLES` restyles alone | diagnostic. Never the headline |

Arm C is the one this design adds over T-040's sketch, and it is not optional: a denoised,
re-rendered frame differs from its source in ways that have nothing to do with domain
randomisation, and without C a gain from B is uninterpretable.

**The style partition is committed before generation.** The style pool is split into disjoint
`TRAIN_STYLES` and `EVAL_STYLES` — as a committed file, in git, before the first clip — so the
evaluation domain cannot be chosen after seeing which restyles came out well.

## 6. Gates — `T40_RULE_V1`

No threshold is coined. The ladder is WAM-Bench's own, the margin is borrowed, and the two
geometry numbers are derived from the corpus itself.

**Borrowed:** `MATERIAL_FLOOR_PP = 10.0`, from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, the
repo's existing floor for "material on this headline" — the same constant PR-07 borrows, from the
same place, for the same reason: so the choice of floor cannot become the finding.

**Ladder:** **L1** `skill_vs_repeat_pct > 0`; **L2** `ci_skill_vs_repeat_pct > 0` (`ci_` is the
task-**critical** chunk subset, not a confidence interval).

### G0 · VOID gates — all three run before any training, all on CPU

**G0a · Label integrity — and why `screen_corpus.py` is an identity check here, not a screen.**
T-040 requires `screen_corpus.py` (T-34) on the generated corpus. Taken at face value that gate is
**vacuous**: M1, M2 and M3 are computed from proprioception, the clock and the gripper channel, and
a restyle changes no action, so the restyled corpus must return the source's numbers by
construction. It is kept, with its job restated: it must **reproduce the source's M1/M2/M3 within
`EXPECT_TOL`** (0.02, 0.02, 0.05 — the script's own archived tolerances). A deviation is not a
finding about the corpus, it is proof that the generation pipeline **corrupted or reordered the
action labels**, and it is VOID. Recorded as `screen_corpus --expect` against the source's values.

**G0b · Geometry invariance.** Object and plate centroids in the restyled clip must agree with the
source. Tolerance is **derived, not coined**: `GEOM_TOL = median per-step object-centroid
displacement in the source clips`, computed and committed before generation. Rationale — a drift
larger than what one action step actually moves the scene makes the carried-over label describe a
different scene than the pixels. `EST_DRIFT_P95` (§4) is subtracted from the budget, so the
generator is held to `GEOM_TOL − EST_DRIFT_P95`; if that is ≤ 0, the estimator is not good enough
and generation does not start.

**G0c · Embodiment — solved by construction rather than by a threshold.** `video_fidelity`
provably cannot see the generic-manipulator defect (`runs/backbone_eval/video/embodiment_grid.png`),
and any IoU threshold on the robot mask would be a coined number. So the real robot's pixels are
**unconditionally composited back** over every generated frame, using the robot segmentation mask.
The defect cannot enter, and no threshold is needed. Robot-mask IoU between source and generated is
still recorded, as a diagnostic on the generator, never as a gate.

### The headline, and the verdicts

Headline: `skill_vs_repeat_pct` on the **visually shifted** eval set (`EVAL_STYLES`, disjoint from
`TRAIN_STYLES` by the committed partition), arm **B** against arm **A**, with arm **C** deciding
whether the gain is attributable.

| | condition | reading |
|---|---|---|
| **P** | B − A ≥ `MATERIAL_FLOOR_PP` **and** B − C ≥ `MATERIAL_FLOOR_PP` **and** B reaches L1 | visual domain randomisation materially helps, and the gain is diversity rather than the generator |
| **F** | B − A ≥ `MATERIAL_FLOOR_PP` but B − C < `MATERIAL_FLOOR_PP` | the gain is the **generator**, not the diversity. Restyling is a denoiser here. Records against the pipeline, licenses no corpus |
| **N** | B − A ≤ 0 | augmentation does not help on this corpus. Closes the augmentation direction for AppleToPlate |
| **I** | anything else — in particular `0 < B − A < MATERIAL_FLOOR_PP` | indeterminate. Run a second seed before recording anything |

**Recorded regardless of verdict:** all four arms under both bench specs, L1/L2 reached, the style
partition hash, `GEOM_TOL`, `EST_DRIFT_P95`, measured throughput, GPU-h spent, and the generator
checkpoint id **and revision**.

## 7. The threat to validity that no arm removes

**`EVAL_STYLES` is generated, so a P is a claim about generalising to held-out *generated*
appearance, not to real appearance.** Arm C removes the fingerprint shared between training and
eval frames; it does not make the eval real. There is no real visually-shifted AppleToPlate footage
in existence, and HE — the only real diversity available on this embodiment — is unlicensed and a
different task set.

Pre-registered consequence, so it cannot be softened later: **a P under this document licenses
exactly one thing — recording a small real shifted eval set (different tablecloth, apple, lighting)
and re-running arms A and B against it.** It does **not** license adding restyled data to any
training corpus, and it must never be reported as "augmentation works", only as "augmentation
survives a held-out generated shift, real shift untested."

## 8. What must exist before a single clip is generated

1. **The recipe is `--tune-visual` (Recipe B, lr 5e-5, `submit_chain.sh visual`)** — fixed here.
   Recipe A freezes the vision tower, and varied pixels into a frozen tower is a strictly weaker
   experiment whose null would say nothing. Recorded because T-040 requires it stated: unstated,
   the result is not readable.
2. **The consumer contract with `emai/vla-training`**, written down: LeRobot v3.0, 28-dim
   arms+hands, right hand index-before-middle, and the action labels come from the *source*
   recording, never from the generator.
3. **A measured throughput number** — one timed episode on an H200 at 640×480 — and a GPU-h ceiling
   derived from it, enforced in the sbatch as `MAX_RESTARTS` enforces T-39's. The multiplier is
   ~172 000 frames per variant (402 episodes × ~427 mean frames, range 249–749 ≈ 95 min of 30 fps
   video). **No budget line exists until that measurement does.** Cluster constraints: 4 h
   `MaxWall`, `MaxJobsPU=4`, billing `GPUs×1.0 + MemGB×0.25 + Threads×0.036` per minute, so `--mem`
   is not free; generation must be chunked and resumable in the shape `submit_chain.sh` already
   uses. The login node is off limits for anything that computes.
4. **`GEOM_TOL` and `EST_DRIFT_P95` measured and committed** (§4, §6).
5. **Depth and segmentation annotators wired into `isaac_binding.py`** — `distance_to_camera` and
   `semantic_segmentation` alongside the existing `rgb`, with tests. Blocks §4 entirely. Discovered
   while writing this document; T-040 had recorded the capability as already present.
6. **The `TRAIN_STYLES` / `EVAL_STYLES` partition committed** (§5).
7. **T-39 has reported** (§1).

## 9. What this cannot answer

- **It is not T-32.** Ten restyles of 402 demos is still 402 trajectories. It cannot speak to data
  volume and must not be reported as if it could.
- **It is not the fix for the Isaac 0/10.** `vla-training/docs/apple-pnp-ursachen.md` ranks that as
  U1 — saturated raw radians against a Dex3 drive at its 5 N·m limit, contact 10/10 and lift
  identically 0.000 mm in 10/10 — an actuation defect, with the visual hypotheses refuted 3-to-0.
  Claiming this task addresses 0/10 would be wrong.
- **It says nothing about the embodiment defect in the generators**, because G0c composites around
  it rather than measuring it. Fixing it at the source is [[T-041]].
- **It is one task, one scene, one camera.** A P generalises to nothing beyond AppleToPlate without
  a second corpus.
