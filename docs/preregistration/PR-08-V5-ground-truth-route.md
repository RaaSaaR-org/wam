# PR-08 V5 — §4's ground-truth source: from Isaac specifically to a simulator with exact segmentation

**Rule `T40_RULE_V5`. Registered 2026-08-22, BEFORE ANY CAPTURE IS RUN, before any clip is
generated, before any weight is trained, and before any job is submitted.**

*"Before any capture is run" is checkable and was checked while writing this:
`configs/transfer25/pr08_est_drift.json` — the tracked path `measure_est_drift.py` writes its
artifact to — **does not exist in this tree**, and `runs/` carries no `pr08-est-drift/` directory.
**No `EST_DRIFT_P95` has been measured by any route, by anybody.** That is the point of registering
this now: an amendment written after seeing the number it licenses is not an amendment, it is a
result being made admissible.*

*Said exactly, because "no capture" would be too strong and the difference is the whole discipline:
while building the shim, frames **were** rendered — into a session scratch directory, never into
`runs/` and never into a tracked path — and `measure` **was** run over them, with a red-threshold
**stub** standing in for the estimator, to prove the arithmetic runs end to end. The real estimator
pair has never been run on a MuJoCo frame; `apple_sam2` is not gate-qualified and could not produce
an admissible number today in any case. **Nothing in this document was written after seeing an
`EST_DRIFT_P95`, because there is not one to have seen.**

This document sits **alongside**
[`PR-08-photoreal-augmentation.md`](PR-08-photoreal-augmentation.md) (`T40_RULE_V1`),
[`PR-08-V2-arm-c-frame-matching.md`](PR-08-V2-arm-c-frame-matching.md) (`T40_RULE_V2`) and
[`PR-08-V3-seed-schedule.md`](PR-08-V3-seed-schedule.md) (`T40_RULE_V3`). **None of the three has
been edited and none may be.** The discipline is `docs/handoff.md` §3 — *"Rules are versioned,
never edited in place. A gate rewritten after seeing its output is not a gate."* V5 is that
versioning, not a revision.

**On the number.** `docs/isaac-est-drift-runbook.md` §6 anticipates this amendment and calls it
*"a `PR-08-V4`"*. `PR-08-V4-t39-gate-premise.md` is already drafted, on an unrelated question (the
T-39 premise behind §8 item 7), and is **UNSIGNED and not in force**. V5 therefore takes the next
free number. A rule id is an identifier, not a count; renumbering a drafted rule to close a gap
would be worse than the gap. **V5 neither depends on V4, cites V4 as authority, nor signs it.**

**And on `T40_RULE_V6`**, registered separately the same day
([`PR-08-V6-mask-validity.md`](PR-08-V6-mask-validity.md)): it adds two fields
(`mask_validity_min_iou`, `mask_validity_reference`) to the committed segmenter contract, which
decide *which frames* each side of §6's subtraction is measured on. **V5 and V6 are independent
and neither depends on the other.** V5 changes which *simulator* renders the ground truth and
touches no field of that contract; V6 changes the estimator's output validity check and touches no
renderer. They compose without interaction: a MuJoCo capture measured under V6's filter is still a
MuJoCo capture, and V5's §0 row saying the segmenter contract is "untouched" means *untouched by
V5*, not frozen against V6.

Task: [[T-040]]. Consumer: `emai/vla-training`. Generator: **Cosmos-Transfer2.5, frozen**.

---

## 0. What V5 does not change

Stated first and exhaustively, because a V-document that quietly moves a threshold is the failure
the versioning discipline exists to prevent. **V5 moves no gate, no threshold, no verdict, no arm,
no clip count, no style, no seed and no ceiling.** It changes one sentence of `T40_RULE_V1` §4:
*which simulator* renders the ground truth that `EST_DRIFT_P95` is measured against.

| | unchanged |
|---|---|
| `MATERIAL_FLOOR_PP = 10.0` | still **borrowed** from `I8_RULE_V3` in `62_eval_i8_curve.sbatch`, not coined. V5 does not move it, rescale it, or make it per-arm |
| `GEOM_TOL` — the definition | unchanged and still **derived, not coined**: the median per-step object-centroid displacement in the **source** clips, measured on the real corpus, computed and committed before generation. V5 touches neither its definition, its corpus, its method, nor its value, and **supplies no value for it** |
| `GEOM_TOL` — the step | unchanged: `GEOM_STEP_FRAMES = 1`, one source frame at `fps = 30`, overlapping offsets, as `T40_RULE_V3` §4.3 registers |
| `GEOM_TOL` — the grid | unchanged: **pixels at 640×480**, and `EST_DRIFT_P95` is still measured on that same grid because §6 subtracts one from the other. The grid is still read from `configs/transfer25/pr08_geom_tol.json` → `segmenter.pixel_grid_hw` at run time and is still **never** a literal in a script |
| `EST_DRIFT_P95` — the definition | unchanged: the **95th percentile of the object-centroid displacement, in pixels, between the estimated segmentation and the true segmentation of the same frame** (§4 step 4). V5 changes what produces "the true segmentation", not what the number is |
| `EST_DRIFT_P95` — its role in the gate | unchanged: still **subtracted** from G0b's budget, still *"enters G0b's tolerance as a budget rather than being assumed to be zero"*, and **assuming it is zero is still forbidden** |
| `EST_DRIFT_P95` — as a bound | unchanged in kind: it is **still a bound whose direction must be recorded**, and a G0b margin that only clears under a bound pointing the wrong way is **still not a pass**. §3 below registers what "recorded" means for a route that is not Isaac. **V5 does not weaken this clause; it makes it apply to more than one direction** |
| §4 step 2 — *"the same segmenter"* | unchanged, in the strong reading `scripts/estimators/apple_sam2.py` already takes: the same detector, segmenter, prompt, both thresholds, the single retry pair, the box-selection rule and the propagation mode as the generator's own, pinned by 40-hex revision. `cross_check_geom_tol` still compares that block **field for field** and still disqualifies on any disagreement |
| The committed segmenter contract | `configs/transfer25/pr08_geom_tol.json`, its `contract_fields`, its `segmenter` block and `object_text_prompt: "apple."` are **untouched**. V5 changes no field in it and its sha256 sidecar is unaffected |
| §4 step 3 — the depth error | unchanged in content, and unchanged in **status**: it was already **recorded, not gated** (`apple_sam2`'s own docstring: *"It is recorded, not gated"*). §2 below states that explicitly rather than moving it |
| **G0a** label integrity | unchanged — `screen_corpus --expect` against the source's M1/M2/M3 within `EXPECT_TOL` (0.02, 0.02, 0.05); a deviation is still VOID |
| **G0b** geometry invariance | unchanged — the generator is held to `GEOM_TOL − EST_DRIFT_P95`; **if that is ≤ 0, generation does not start** |
| **G0c** embodiment | unchanged — the real robot's pixels are unconditionally composited back over every generated frame; robot-mask IoU is recorded as a diagnostic, **never** as a gate |
| **The ladder** | unchanged — **L1** `skill_vs_repeat_pct > 0`, **L2** `ci_skill_vs_repeat_pct > 0` (`ci_` = the task-**critical** chunk subset, not a confidence interval) |
| **The P / F / N / I verdict table** (V1 §6) | unchanged in every cell, including that **P** requires *both* B − A ≥ floor *and* B − C ≥ floor, that **F** is the generator-attributable case, that **N** is B − A ≤ 0, and that `0 < B − A < MATERIAL_FLOOR_PP` is **I**, not a weak P |
| **Arms A / B / C / D** | unchanged. B is the intervention, C is the generator-fingerprint control, D is diagnostic and never the headline |
| **Arm C's size** (`T40_RULE_V2`) | unchanged — R2, frame-matched on added training volume |
| **Clip totals and the seed schedule** (`T40_RULE_V3` §1) | unchanged. V5 registers no seed and no clip count |
| **The GPU-h ceiling reading** (`T40_RULE_V3` §3) | unchanged. V5 supplies no ceiling value and exempts nothing from one |
| **The committed style partition** | `configs/transfer25/pr08_style_partition.json`. V5 changes **no style, no id, no slug and no prompt string**, and therefore changes no partition hash |
| **§1's prohibition** | unchanged and still binding in full — nothing is generated, no weight is trained on generated frames, and no number from PR-08 is quoted as a result, until **every** §8 item is closed **and** T-39 has reported |
| **§7's threat to validity** | unchanged — `EVAL_STYLES` is generated, so a **P** is a claim about held-out *generated* appearance and licenses **exactly one thing**: recording a small real shifted eval set and re-running arms A and B against it |
| **§8's seven items** | unchanged as a list. Item 4 (`GEOM_TOL` **and** `EST_DRIFT_P95` measured **and committed**) is still **open**, and V5 closes neither half of it |
| **`T40_RULE_V3` §5.3's refusal of a VOID** | unchanged. A T-39 **VOID** still closes PR-08 rather than opening it |
| **`PR08_OVERRIDE_T39_VOID`** | unchanged, ungranted, **not exercised**, and its value is written nowhere here |
| **The estimator's gate qualification** | unchanged. `scripts/estimators/apple_sam2.py`'s `GATE_QUALIFIED` is still `False` and its `GATE_QUALIFICATION_BLOCKERS` tuple is **untouched by this document** — in particular *"nobody has looked at a mask"* and *"per-frame segmentation is not upstream's propagation"* are still open, and `estimator_not_gate_qualified` will still be stamped on any capture measured today, by either route |
| **`measure_est_drift.py`'s exit contract** | unchanged — **0** gate-qualified, **2** nothing measured, **3** measured but must not be subtracted from `GEOM_TOL`, artifact written on 3 |
| **Every other refusal in the rig** | unchanged: the render grid against the committed contract, the object named once (`--object-class` defaulting to the estimator's own prompt, an explicit disagreement fatal), mixed frame geometries fatal, `--limit` forcing disqualification, an unmeasurable frame **dropped and counted rather than folded in as a zero** |

**And, said loudly because it is the failure mode this project keeps having to name:**

- **V5 does not license generating a corpus.** `T40_RULE_V1` §1 forbids it and V5 does not touch
  that clause. Measuring the estimator's error budget is what §1 *already* licenses, and measuring
  it is the only activity this document is about.
- **V5 does not license training any weight**, on generated frames or otherwise. Whether training
  may start, and against which label space, is the project owner's call (`CLAUDE.md`).
- **V5 does not gate-qualify the estimator.** It moves no entry of `GATE_QUALIFICATION_BLOCKERS`,
  and no capture made under it can exit 0 while that tuple is non-empty.
- **V5 supplies no number.** It contains no value for `GEOM_TOL` and none for `EST_DRIFT_P95`.
- **V5 says nothing about `docs/benchmark.md`'s L4 gate**, which `CLAUDE.md` records as a separate
  open decision for the owner. V5 must not be cited in it.

---

## 1. The one sentence that changes

`T40_RULE_V1` §4, step 1, verbatim:

> 1. Render N Isaac episodes with ground-truth depth + segmentation.

**`T40_RULE_V5` registers, in its place:**

> 1. Render N episodes in a **simulator that emits exact per-pixel object segmentation**, with
>    ground-truth depth where that simulator provides it. The simulator and the scene are recorded
>    in the capture artifact. *(§4 step 3's depth error is recorded, not gated — see §2 — so a
>    simulator that provides no ground-truth depth, or provides a different depth quantity, still
>    produces the gated number in full and loses only a recorded diagnostic, which it must say.)*

And, consequentially, `T40_RULE_V1` §4 step 0 and §8 item 5, which name `isaac_binding.py`
specifically:

> 0. Attach the ground-truth depth and segmentation channels in the binding for whichever
>    simulator step 1 uses. *(`distance_to_camera` and `semantic_segmentation` in
>    `src/wam/robot/isaac_binding.py` for Isaac — landed 2026-08-21, commit `5ef3535`. The
>    equivalent for MuJoCo is `src/wam/robot/mujoco_binding.py`.)*

**Nothing else in §4 is replaced.** Steps 2, 3 and 4 stand word for word, including *"the same
monocular depth estimator and the same segmenter"*, *"on the RGB **only**"*, and *"the 95th
percentile of that centroid displacement is `EST_DRIFT_P95`"*.

**The route registered for the first measurement is MuJoCo**, on the project owner's decision of
2026-08-22, taken on the ranked comparison in `docs/isaac-est-drift-runbook.md` §6. V5 does not
close the Isaac route: it remains available, unchanged in every flag and refusal, and a later Isaac
capture is a second measurement of the same quantity rather than a different one.

---

## 2. Why this is not a weakening: `EST_DRIFT_P95` is a segmentation quantity

The whole amendment rests on one reading of §4, and the reading is checkable rather than
rhetorical.

**§4 step 4 defines the gated number on segmentation alone.** *"The **95th percentile of that
centroid displacement**"* — the displacement between the estimated **mask** and the true **mask**.
Depth appears in step 3 and only in step 3, as *"absolute depth error"*, listed **beside** the
centroid displacement in a sentence about *"the error distribution"*.

**That depth half is already recorded and not gated, and three independent places say so:**

- `scripts/estimators/apple_sam2.py`'s module docstring, verbatim: `segment` produces *"the only
  number here that enters a gate: §6 G0b holds the restyled corpus to `GEOM_TOL - EST_DRIFT_P95`.
  So the gate rides entirely on `segment`."* And of `estimate_depth`: *"It is **recorded, not
  gated**."*
- `scripts/measure_est_drift.py` computes `depth_absolute_error_over_object` into the artifact and
  **no disqualification reason depends on it**. Every entry of `gate_disqualified_reasons` is about
  the estimator, the segmenter contract, the object name, the pixel grid, the coverage or the
  capture's provenance.
- `scripts/run_g0_gates.py` — the consumer — reads `est_drift_p95_px` and never a depth field.

**Registered consequence.** A ground-truth route that supplies exact **masks** and no ground-truth
**depth** satisfies §4 in full for the gated quantity. A route whose depth is a *different physical
quantity* from Isaac's must **say which quantity it is in the artifact**, rather than being read as
the same number. This is not hypothetical and it is the one place the two routes are genuinely not
interchangeable:

| | Isaac (`isaac_binding.GROUND_TRUTH_ANNOTATORS`) | MuJoCo (`mujoco_binding.render_depth`) |
|---|---|---|
| what the channel is | `distance_to_camera` — **euclidean ray length** from the camera origin | the depth buffer, reconstructed to metres — **distance to the image plane** |
| relation | differ by `1 / cos(angle off the optical axis)` — **1.41 at 45°** | |
| which one a metric monocular estimator predicts | image-plane depth | image-plane depth |
| who owes a conversion | **the Isaac route** (its own binding's comment says so: *"comparing the two directly inflates the error by 1/cos… an inflated budget is not a conservative one — it is a different number"*) | nobody |
| gated? | **no**, either way | **no**, either way |

**V5 registers that the capture artifact must carry the depth semantics by name** (§5). It does not
convert one into the other, and it does not claim the two routes' recorded depth errors are
comparable.

---

## 3. The direction of the bound is a REGISTERED field, and MuJoCo's is conservative

### 3.1 Why direction is the property that matters here

`T40_RULE_V1` §6 **subtracts**: the generator is held to `GEOM_TOL − EST_DRIFT_P95`. So an
`EST_DRIFT_P95` that is too **small** leaves the tolerance too **wide**, and the error lands in the
generator's favour — a restyled clip that moved the geometry more than the rule intended passes
anyway. An `EST_DRIFT_P95` that is too **large** leaves the tolerance too **tight**, and the error
lands against the generator — a clip that would have been acceptable is rejected.

Those two failures are not symmetric for this experiment. G0b is a **VOID gate**: its job is to
refuse a corpus whose pixels no longer match their carried-over labels. A gate that is too strict
costs clips; a gate that is too loose costs the finding. §4 already concedes that Isaac's number
errs the second way — *"plausibly optimistic… so `EST_DRIFT_P95` is a **lower bound** on the real
error"* — and calls it *"the single biggest soft spot in the design"*.

**`T40_RULE_V5` therefore registers, for every route: the artifact must record which way the
error points, and whether that direction was argued or measured.** `T40_RULE_V1`'s "it is recorded
as such" is preserved and made explicit rather than replaced.

### 3.2 The argument that MuJoCo's direction is conservative — stated as an argument

Not asserted. The premises, so that each can be attacked separately:

**P1. `EST_DRIFT_P95` is the error of a photograph-trained pipeline on a rendered frame.** The
segmenter is GroundingDINO-base plus SAM2-hiera-large at Cosmos-Transfer2.5's own operating point.
Both were trained on photographs and web images, not on renders.

**P2. Detection and segmentation accuracy degrade with distance from the training distribution.**
This is the same premise §4 uses — it is *why* §4 calls Isaac's number optimistic. §4 applies it in
one direction (Isaac renders are *closer* to photographs than the model's worst case, so the error
is small); V5 applies the identical premise in the other.

**P3. MuJoCo's rasteriser is further from a photograph than Isaac's RTX path.** No path tracing, no
global illumination on the object, three static lights, one flat untextured material on the object,
hard shadows. Nothing subtle about this: it is a rasterised scene and Isaac's is a ray-traced one.

**Therefore**: the same pipeline does **worse** on MuJoCo frames than on Isaac frames, so the
measured p95 is **larger**, so `GEOM_TOL − EST_DRIFT_P95` is **smaller**, so **G0b is stricter**.
The error lands against the generator, which is the safe direction and the opposite of the failure
§4 warns about.

### 3.3 The limb of this that is NOT argued, and is registered as unmeasured

**The MuJoCo scene is also SIMPLER than a real D435 frame, and that pushes the other way.** One
object and one distractor on a clean tabletop, no clutter, no motion blur, no rolling shutter, no
sensor noise, no depth-of-field, perfectly static lighting. On the *scene-complexity* axis the
frames are **easier** than reality, and an easier frame lowers the p95.

So the honest registered statement is two-limbed:

> **`EST_DRIFT_P95` measured under this route is argued conservative on the PHOTOREALISM axis and
> is UNMEASURED on the SCENE-COMPLEXITY axis. It is not registered as an upper bound, and it must
> not be quoted as one.**

`T40_RULE_V1` §4's clause survives intact in the form that matters: **a G0b margin that clears only
because `EST_DRIFT_P95` came out small must be re-examined rather than accepted**, and the
re-examination is not discharged by this document.

The runbook's §4.7 note sharpens the same point from a second direction and is not softened here:
the estimator's own open blocker — per-frame re-detection where upstream propagates one mask —
biases the number **both** ways, so at the estimator level the direction is unknown for a reason
that has nothing to do with the renderer. **V5 registers a direction for the RENDERER only.**

### 3.4 What would settle it, registered so the claim is falsifiable

Two measurements, neither of which V5 performs and neither of which it requires before a capture:

1. **The same-frame comparison** (`docs/isaac-est-drift-runbook.md` §6c, adapted): run the same
   estimator on a MuJoCo render and on a real `pr08-apple-640x480` frame of comparable object
   distance and scale, and compare the detection scores and mask quality. If the estimator does
   *better* on the MuJoCo frame, P3 is refuted for this scene and the direction registered in §3.2
   is withdrawn.
2. **The two-route bracket**: measure `EST_DRIFT_P95` again on an Isaac capture of an equivalent
   scene. Two estimates whose errors are argued to point in opposite directions bracket the truth
   far better than either alone. V5 keeps the Isaac route open precisely so this stays possible.

---

## 4. The threat to validity: the object is a stand-in, and this is stated at §4's own volume

`T40_RULE_V1` §4 states its weakness in four sentences and calls it *"the single biggest soft spot
in the design"*. This route has a second one, and it is stated here rather than in a docstring.

### 4.1 The objection

**An apple in a simulator is not an apple in a RealSense frame.** `EST_DRIFT_P95` is a budget for
the estimator's error on **the apple in the AppleToPlate corpus** — a real fruit, photographed
through a D435, on a real tablecloth. Whatever object stands in the simulated scene, the number
measured on it is transferred across that gap. The runbook put the objection in its sharpest form:
`configs/sim/g1_scene.xml`'s only graspable object is a **45 mm orange cube**, and *"a budget for
finding a cube in a MuJoCo render"* is not a budget for finding an apple.

### 4.2 What was checked, and what was found

**An apple mesh was searched for offline and one was found.** Nothing was downloaded. Searched: the
repository, `~/models`, every MuJoCo/robosuite/Menagerie-shaped asset directory on this box, and the
Isaac and IsaacLab-Arena trees. Findings, measured 2026-08-22:

| candidate | verdict |
|---|---|
| `~/.cache/isaac-sim-docker-tmp/.../objects/objaverse/apple_01.usd` | **not usable.** It is a USD, which MuJoCo does not import, and it is root-owned mode `600` — not readable without escalation. Its sibling `gltf/` directory is empty |
| ManiSkill2-real2sim `.../models/apple/collision.obj`, vendored in SimplerEnv inside `~/IsaacLab-Arena/submodules/Isaac-GR00T/external_dependencies/` | **used.** User-readable, Apache-2.0 (`ManiSkill2_real2sim/LICENSE`) inside an MIT repo (`SimplerEnv/LICENSE`), 17 723 vertices / 35 390 triangles, axis-aligned extent **80.8 × 80.7 × 80.7 mm** — a real apple's diameter |
| LIBERO `stable_hope_objects` / `stable_scanned_objects` / `turbosquid_objects` | no apple. Nearest are `orange_juice`, `tomato_sauce`, bowls and plates |
| MuJoCo Menagerie, robosuite, dm_control | **not on this box at all.** The only MuJoCo asset tree here is the Menagerie `unitree_g1` description `scripts/fetch_g1_model.py` fetches |

**So the cube substitution is not taken.** Registered as a fact of this route, not as a permission.

### 4.3 The cube was not merely undesirable — it was unavailable, and the mechanism matters

Had no mesh been found, the cube would still not have been an option, for a reason stronger than
taste. The estimator's prompt is part of the **committed segmenter contract**: `apple_sam2`'s
`SEGMENTER_CONTRACT` carries `object_text_prompt`, `configs/transfer25/pr08_geom_tol.json` commits
`"apple."`, and `measure_est_drift.cross_check_geom_tol` compares that block **field for field**. A
cube capture would require prompting `"orange cube."`, which disagrees with the committed contract,
which stamps `segmenter_params_disagree_with_geom_tol` and **disqualifies the run**. There is no
version of the cube route that produces a gate input.

**`T40_RULE_V5` registers that this is not to be worked around.** Changing the committed prompt to
match a stand-in object would make `GEOM_TOL` and `EST_DRIFT_P95` two measurements of two different
objects, and §6 subtracts them. If a future route must use a different object, that is a further
V-document with the subtraction argued, not a field edit.

### 4.4 What remains true even with the apple mesh, and travels with the number

The substitution is narrowed, not eliminated. All four of these are **named fields in the capture
artifact** (`capture.object_limitations`), not prose:

1. **`object_mesh_is_convex_decomposition_proxy: true`.** The mesh is the *collision* proxy of a
   scanned apple — 14 convex parts, merged — not the scan itself. Its silhouette is an apple's; its
   surface is a union of hulls.
2. **`object_is_untextured: true`.** One flat material, shaded by the scene's three static lights.
   The real apple has specular highlights, a stem, colour variation and a shadow terminator that a
   detector can use — and that a detector can also be confused by.
3. **`renderer: "mujoco rasteriser (not ray-traced, not photoreal)"`.** This is §3.2's premise P3
   restated as a field. It is the basis of the conservative-direction argument and is therefore
   **not a defect to be fixed** — a more photoreal render would move the number the unsafe way.
4. **`object_is_static_prop: true`.** The object's pose varies **across** scene states and not
   within one; it is teleported, not dropped or grasped. A calibration capture measures the
   estimator, not a manipulation.

**Registered consequence, and it is the operative clause of this section: no `EST_DRIFT_P95`
produced under this route may be quoted, carried, or subtracted without `object_limitations`
travelling with it.** `measure_geom_tol.py --carry-est-drift` already refuses a number whose
segmenter is unnamed, for the same class of reason; this is the object's half of that discipline
and it is enforced by the field being in the artifact rather than by anybody remembering.

### 4.5 One choice made deliberately against the number's favour, recorded so it cannot be reversed quietly

**The orange cube is left in the scene as a distractor, and the hands are left where they occlude.**
An orange cube beside a red apple is a real opportunity for a text-prompted detector to return the
wrong box, and the schedule's table-edge placements put the object partly behind the Dex3 hands.
Removing either would lower the p95, widen `GEOM_TOL − EST_DRIFT_P95` and land the error in the
generator's favour. **V5 registers that neither may be removed to improve the number**, and any
change to the capture scene that raises the object's visibility is a change that must be argued in
a further V-document, not made in a commit.

---

## 5. What the capture and the artifact must record — registered before any capture

Registered so that a run cannot omit them and so a reader six months later is not asked to infer
them. All are implemented in `scripts/measure_est_drift.py` and
`src/wam/robot/mujoco_binding.py` as of 2026-08-22, with tests.

| field | why it is registered |
|---|---|
| `capture.binding`, `capture.backend`, `capture.ground_truth_route` | §1 makes the simulator a variable. An artifact that does not say which one produced it is not auditable |
| `is_simulated_binding` | unchanged in meaning: **true** for anything that is not a ground-truth simulator, and still the thing that keeps a laptop capture out of a gate |
| `capture.asset`, `capture.asset_source`, `capture.camera`, `capture.camera_prim` | which scene, named once. A capture of a stage with no object in it and a capture that measured something look identical in the frames if the segmentation happens to be empty in both |
| `capture.object_limitations` (all fields of §4.4) | §4.4's operative clause |
| `capture.n_scene_states_scheduled` / `n_scene_states_visited` | **PR-08 §4.6's "N counted in distinct configurations, not frames"**, which had no field and which the runbook could only suggest putting in a commit message |
| `label_vocabulary_seen` | what the *scene* called its objects. `object_ids` matches `strip().lower()` equality and forgives nothing else; a scene calling the fruit `apple_01` produces a full run, zero coverage and no crash |
| `object_class`, `object_class_source`, `object_class_requested`, `estimator_object_text_prompt` | **what the estimator was actually prompted with**, recorded beside what the ground truth was matched on, so the two cannot silently be different objects |
| `resolution_hw`, `capture.render_hw_source` | the grid, and the document it was read from |
| `is_lower_bound`, `is_lower_bound_reason`, `error_direction`, `error_direction_measured` | §3. The last two are new fields; the first two are unchanged for the Isaac route, down to the wording |
| `estimator_stats`, `geom_tol_cross_check`, `coverage`, `n_dropped` | unchanged, and still additive/read-only |

**One registered minimum, and it is a TIGHTENING.** `T40_RULE_V1` §4 says "N episodes" and names no
floor; the runbook's §4.6 recommends one and a runbook is not a rule. **`T40_RULE_V5` registers, for
this route: ≥ 20 distinct scene states and ≥ 200 *measured* frames** before a p95 from it may be
carried into `configs/transfer25/pr08_geom_tol.json`. Below ~100 measured frames a p95 is
essentially the fifth-largest sample; over one configuration it is a percentile over one viewpoint.
**This adds a condition and removes none**, which is the only kind of change to a live rule that is
safe to make in a document like this.

---

## 6. The one code change that carries a large meaning, and exactly how far it goes

Two edits in `scripts/measure_est_drift.py` are the entire mechanical content of V5, and both are
the kind that could let something that is not ground truth become G0b's budget. They are written
out here so the review is of the change and not of the diff.

**(a) `capture_is_not_from_isaac_sim` becomes an allow-list.** It was
`type(binding).__name__ != "IsaacSimBinding"`, one hard-coded comparison. It is now a lookup in
`GROUND_TRUTH_BINDINGS`, a table in one place with the reason for each entry beside it.
**`FakeIsaacBinding` is deliberately absent and must stay absent** — its "ground truth" is a moving
square, and every capture anyone has run to date came from it. The *reason string* is unchanged, on
purpose: it is quoted in the runbook's §4.7 table and in test fixtures, it is still literally true
of everything it fires on, and renaming a committed disqualifier vocabulary buys nothing.

**(b) `is_lower_bound` becomes per-route.** It was stamped `true` unconditionally with a reason
string quoting §4. It is now looked up from the same table, and **the fallback for anything not in
that table is the old stamp, verbatim** — so a capture from the fake, or from a stub, produces
byte-identical output to before. What such a capture gets instead is the new `error_direction`
field saying `"unknown — not a ground-truth capture"`, which did not previously exist to be changed.

**What (b) explicitly does NOT do.** The Isaac reason string still carries the sentence about
Humanoid Everyday's licence, which `T40_RULE_V1` §3 itself withdrew on 2026-08-07 (*"The reason is
no longer the licence"*) and which the runbook records as its §7 defect 4. **V5 does not correct
it.** Rewriting a justification that quotes a pre-registration verbatim is a judgement for whoever
owns PR-08 — the replacement has to say something true about two different things where the current
string says one false thing about both — and folding it into an amendment about a different
question would be exactly the quiet edit this discipline exists to prevent. **It is named here so
that it is a known open defect and not an oversight.**

**And the Isaac path is otherwise untouched**, deliberately and testably: `--backend` defaults to
`isaac`, the render-grid check against the committed contract, the camera validation at parse time
against `DEFAULT_CAMERA_PRIMS`, the `--asset`/`--scene` one-knob refusal, the `--camera-prim`
mechanism and every message they emit are the ones they were.

---

## 7. What V5 does not license

- **Not generation.** `T40_RULE_V1` §1 stands in full.
- **Not training**, on generated frames or on anything else.
- **Not a gate-qualified `EST_DRIFT_P95` today.** `apple_sam2.GATE_QUALIFIED` is `False`,
  `GEOM_TOL` is unmeasured, and a capture made under V5 today still exits **3**.
- **Not the cube**, and not any other stand-in object, and not a change to the committed prompt to
  accommodate one (§4.3).
- **Not the removal of the distractor or the occluding hands** (§4.5).
- **Not a claim that this number is an upper bound** (§3.3).
- **Not a comparison between the two routes' recorded depth errors** (§2).
- **Not a statement about GR00T.** `PR-07-positive-control.md` §6's prohibition is untouched.
- **Not a determination about `T40_RULE_V1` §8 item 7.** That is `PR-08-V4`'s question and V4 is
  unsigned.

---

## 8. Provenance

| | |
|---|---|
| rule | `T40_RULE_V5` |
| registered | **2026-08-22, before any capture is run** — `configs/transfer25/pr08_est_drift.json` does not exist and no `EST_DRIFT_P95` has been measured by any route |
| decided by | the project owner, 2026-08-22, on `docs/isaac-est-drift-runbook.md` §6's ranked comparison |
| supplements | `T40_RULE_V1`, `T40_RULE_V2`, `T40_RULE_V3` — all three stand and all three are **unedited** |
| relationship to `T40_RULE_V4` | none. V4 is **drafted and UNSIGNED**, on a different question. V5 does not depend on it, cite it as authority, or sign it |
| replaces | **one sentence**: `T40_RULE_V1` §4 step 1's *"Render N Isaac episodes…"*, plus step 0's and §8 item 5's naming of `isaac_binding.py` specifically (§1) |
| makes explicit without moving | §4 step 3's depth error is **recorded, not gated** (§2) |
| adds | a registered requirement that the bound's **direction** be recorded, and a registered **minimum** of ≥ 20 scene states / ≥ 200 measured frames (§3, §5) — both tightenings |
| changes | **no gate, no threshold, no verdict, no arm, no clip count, no style, no seed, no ceiling, and no committed artifact** |
| generation licensed | **no** |
| training licensed | **no** |
| a T-39 VOID | **still stops PR-08** |
| `PR08_OVERRIDE_T39_VOID` | **not granted, not exercised, and its value is written nowhere here** |
| implemented by | `src/wam/robot/mujoco_binding.py`, `scripts/measure_est_drift.py` (`--backend`, `GROUND_TRUTH_BINDINGS`), `tests/test_mujoco_binding.py`, `tests/test_measure_est_drift.py` |
| measurements taken here | **none.** V5 computes no `EST_DRIFT_P95` and submits no job. The only numbers in it are properties of a mesh file and of MuJoCo's renderer, measured on this workstation on 2026-08-22 and reproducible in seconds |

## See also

- `docs/preregistration/PR-08-photoreal-augmentation.md` — `T40_RULE_V1`; §4 defines `EST_DRIFT_P95`, §6 subtracts it, §8 item 4 requires it committed
- `docs/isaac-est-drift-runbook.md` — §6 is the ranked comparison this decision was taken on; §4.2 is the missing-Isaac-scene blocker that made the Isaac route the slowest; §4.7 is the disqualifier table
- `src/wam/robot/mujoco_binding.py` — the capture shim, and the authoritative record of what it refuses
- `scripts/measure_est_drift.py` — the rig, `GROUND_TRUTH_BINDINGS`, and every refusal it makes
- `scripts/estimators/apple_sam2.py` — `SEGMENTER_CONTRACT`, and `GATE_QUALIFICATION_BLOCKERS`, which V5 does not touch
