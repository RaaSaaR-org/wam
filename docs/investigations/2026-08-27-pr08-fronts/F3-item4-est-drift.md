# FRONT 3 — PR-08 §8 item 4, second half: `EST_DRIFT_P95`, the carry, and the margin

Written 2026-08-27 by a Claude Code session on the user's workstation. **Nothing under
`/home/humanoid/develop/wam` was modified.** Every command I ran wrote only into
`docs/investigations/2026-08-27-pr08-fronts/`.
No cluster was touched, no git-write command was run.

Claim labels used throughout: **[M]** = I measured it in this session, on this workstation, and
the command is in §7; **[A]** = recorded in a committed artifact or a tracked file, path + line
cited; **[I]** = my inference from (M) and (A), and labelled as such.

---

## 0. The headline, in four sentences

1. **`EST_DRIFT_P95` is already measured to the standard the protocol registered**, on this
   workstation, on 2026-08-27: 3 840 simulated frames over eight trajectories (Arm A) plus 16 846
   real corpus frames (Arm B). **It does not need a cluster run — not now, and not after the flip.**
   `docs/SPRINT-2026-08-27-cosmos3-dataset-generation.md:129` labels §3.3 *"**cluster**, after 3.1"*;
   the "cluster" half of that label is wrong and describes a job that does not need to exist.
2. **What is not measured is a *committable* artifact**, and the two things blocking it are a
   signature and a `GEOM_TOL` re-measure, not compute for `EST_DRIFT` itself.
3. **The authoritative arm is the propagation arm**, on this project's own settled reading of
   §4 step 2 — the one it already applied to the thresholds, the retry and the box rule. The margin
   that follows is **0.02997017900403076 px, 6.26 % of `GEOM_TOL`**.
4. **The plumbing carries the other arm.** `measure_est_drift.py:2334-2337` writes the per-frame
   number into `est_drift_p95_px` *"by definition"*, and `--carry-est-drift` reads only that field.
   Left as is, the first successful carry hands G0b a **5.56× wider tolerance** than the
   authoritative arm supports. That is defect **D1** and it is the one thing on this front that
   would corrupt a gate rather than waste a job.

---

## 1. THE CENTRAL QUESTION — which arm is authoritative

### 1.1 The actual text of §4 step 2

`docs/preregistration/PR-08-photoreal-augmentation.md`, lines 105-119, verbatim **[A]**:

> ## 4. The conditioning signals do not exist, and the error budget is a gate input
>
> Transfer2.5 consumes depth + segmentation + Canny. AppleToPlate ships one RGB camera (`ego`, from
> one head RealSense D435 colour topic), so only Canny is computable. Depth and segmentation are
> estimated, and **estimation error lands as geometry drift — exactly what G0b forbids.** Therefore
> the estimator is characterised *before* generation, never after:
>
> 0. Attach the `distance_to_camera` and `semantic_segmentation` annotators in `isaac_binding.py`
>    (§8 item 5) — they are not wired today, so step 1 cannot run yet.
> 1. Render N Isaac episodes with ground-truth depth + segmentation.
> **2. Run the same monocular depth estimator and the same segmenter on the Isaac RGB only.**
> 3. Record the error distribution: absolute depth error, and object-centroid displacement in pixels
>    between the estimated and the true segmentation.
> 4. The **95th percentile of that centroid displacement** is `EST_DRIFT_P95`, and it enters G0b's
>    tolerance as a budget rather than being assumed to be zero.

The load-bearing word is **"the same"**, and §4 step 2 never says *same as what*. It has exactly two
candidate referents, and the project has already chosen between them **in writing, twice, on a
different axis of the same sentence.**

### 1.2 The referent the project already chose

`scripts/measure_geom_tol.py:478-482` **[A]**:

> ```
> #: Checkpoints Cosmos-Transfer2.5 itself names for this pipeline
> #: (``cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py``), so the generator's own
> #: segmenter is the one the tolerance is measured with.
> ```

`scripts/estimators/apple_sam2.py:851-857`, inside the *discharged* blocker 2 wording **[A]**:

> They are no longer 'upstream demo defaults we happened to copy' (0.35/0.25): they are
> Cosmos-Transfer2.5's own operating point, read off its sam2_model.py, **which is precisely what §4
> step 2 asks for.** The choice-defect half of this blocker is therefore DISCHARGED and inverted —
> measuring these on our corpus and moving them to whatever reads best **would MAKE this a different
> segmenter from the generator's, and the budget would then be a budget for an error nobody
> commits.**

That is the rule, stated as a rule, and applied: **"the same segmenter" means the generator's.** It
is why `BOX_THRESHOLD` is 0.15 and not a number chosen on our corpus, why the retry is (0.10, 0.10),
and why `box_selection` is `highest_score`. The blocker that was retired on 2026-08-27 says the same
thing about the one axis left (`apple_sam2.py:696-706`, verbatim in
`GATE_QUALIFICATION_DISCHARGED`) **[A]**:

> PER-FRAME SEGMENTATION IS NOT UPSTREAM'S PROPAGATION, and it is the one difference left.
> Everything else in §4 step 2's 'the same segmenter' now matches Cosmos-Transfer2.5's sam2_model.py
> exactly … But upstream drives SAM2VideoPredictor.init_state(video_path=...) and PROPAGATES one
> mask across the clip, while this adapter re-detects and re-segments every frame independently.

And the propagation module itself, in the field it stamps on every artifact
(`runs/pr08-est-drift/v17/EST_DRIFT-A1.json`, `arm_comparison.propagator.contract.mask_validity_filter_reason`) **[A/M]**:

> Cosmos-Transfer2.5 has no such filter and **this arm exists to measure what the generator's
> segmenter does.**

**Answer: the propagation arm is authoritative.** §4 step 2's "the same segmenter" is the
generator's segmenter, the generator's segmenter propagates, and a budget measured on the per-frame
topology is — in the discharged blocker's own words — *"a budget for an error nobody commits."*

### 1.3 The margin that follows

Re-derived **[M]** in this session from the artifacts on disk (§7, command C1), against
`GEOM_TOL = 0.47857992441961017 px` (`runs/pr08-geom-tol/pr08_geom_tol.json`, 402 episodes,
171 625 frames) **[A]**:

| arm | pooled `EST_DRIFT_P95` (3 840 frames) | margin | % of `GEOM_TOL` |
|---|---|---|---|
| per-frame (this adapter) | 0.3120786214328541 px | 0.16650130298675608 px | 34.790699419470585 % |
| **propagation (the generator's)** | **0.4486097454155794 px** | **0.02997017900403076 px** | **6.262314291677947 %** |

Both figures reproduce the numbers on record to the last digit. **The authoritative margin is
0.02997 px — 6.26 %, i.e. about three hundredths of a pixel.**

Two things must travel with that number and are not softened here:

* **`GEOM_TOL` is not committable.** The merged 16-shard artifact predates
  `mask_validity_reference_max_frame_fraction` in `SEGMENTER_CONTRACT`, and
  `contract_disagreements` counts an absent field as a disagreement, so the corpus must be
  re-measured at HEAD (sprint §3.2 **[A]**). 0.478579… is a **magnitude for sizing**, not a
  committed tolerance. So `6.26 %` is indicative arithmetic, not the gate's own.
* **The subtraction mixes routes.** `EST_DRIFT_P95` is MuJoCo (V14 licenses that, *"for
  `EST_DRIFT_P95` and the arm comparison and for nothing else"*), `GEOM_TOL` is the real corpus.
  V14 §3.2 already says the honest reading of G0b is *"passes by almost nothing, or does not pass"*,
  not *"has room"* **[A]**.

### 1.4 The one fact that could flip the answer, and why it does not today

The referent of "the same segmenter" is not a pure matter of reading — it depends on **who computes
the conditioning maps at generation time**, and the repo currently answers that by omission.

* `scripts/build_pr08_source.py:30-35` **[A]**: *"`depth` and `segmentation` are absent from every
  episode entry, and that is a decision rather than an oversight."*
* `scripts/restyle_transfer25.py:339-343` **[A]**: *"control_path omitted ON PURPOSE ONLY WHERE THE
  MANIFEST HAS NOTHING TO OFFER: omitting it makes Transfer2.5 estimate the map itself, with its OWN
  depth/segmentation models."*

**So as the pipeline stands today, the segmentation conditioning is produced by Transfer2.5's own
`sam2_model.py` — propagation — and the propagation arm is the one that measures the error the run
will actually commit. [I, from two [A]s]**

If the project instead decides to **pre-compute and supply** the maps from `apple_sam2` per-frame
(the `control_path` branch), then per-frame becomes authoritative and 34.79 % is the legitimate
margin. **That decision has not been made, no producer for those maps exists in the repo, and
`restyle_transfer25.py:339-343`'s own comment about *"a different estimator from the isaac_binding.py
annotators GEOM_TOL was measured against"* is stale and factually wrong twice** — `GEOM_TOL` was
measured with `apple_sam2` on the real corpus, not against `isaac_binding` annotators. That comment
is the only place in the repo that reads §4 step 2 the other way, and it reads it off a false
premise. Flagged as defect **D6**.

---

## 2. Is `EST_DRIFT_P95` already measured to a committable standard, or does it need a cluster run?

### 2.1 The measurement exists and it ran here

**[A]** `runs/pr08-est-drift/v17/` holds thirteen `wam.est_drift/1` artifacts plus two pooled ones.
Their own timestamps, read this session **[M]**:

| artifact | `measured_utc` | frames |
|---|---|---|
| `EST_DRIFT-C1-lattice.json` | 2026-08-27T00:47:48Z | 60 |
| `EST_DRIFT-A1..A8.json` | 00:50:48 → 01:12:03Z | 480 each |
| `EST_DRIFT-C2-t{20,40,80}.json` | 01:15:05 → 01:21:07Z | 480 each |
| `EST_DRIFT-C3-wrongseed.json` | 01:24:03Z | 480 |
| `ARM_DIVERGENCE.json` (Arm B) | ~02:45Z | 16 846 |
| `POOLED.json` / `POOLED-V19.json` | 02:45 / 02:50Z | — |

Capture headers record `captured_utc` 2026-08-26T23:59:46Z → 2026-08-27T00:13:08Z for A1..A8 **[M]**.

**Measured wall-time, from those stamps [M]:**

* MuJoCo capture, 480 frames, 480×640: **~1 min 50 s each** (8 captures in 13 min 22 s).
* `measure … --arm both`, 480 frames: **3 min 02 s ± 2 s each** (A1→A8 in 24 min 15 s).
* Arm A end to end (8 captures + 8 both-arm measurements): **≈ 40 min on one RTX 5090.**
* Arm B, 40 whole episodes / 16 846 frames × 2 arms: **≈ 81 min.**

### 2.2 It runs on this workstation *today* — verified, not assumed

I ran the whole chain end to end in this session on a fresh 12-frame capture **[M]**:

* MuJoCo capture: `12 frames in 3.488 s wall` (§7, command C2). MuJoCo 3.10.0, CPU, headless,
  `configs/sim/g1_scene.xml`, `MUJOCO_GL=egl`.
* Both-arm measure: `10.406 s wall` including model load (§7, command C3). GroundingDINO + SAM 2 +
  Depth-Anything all resolved from local cache; nothing was downloaded; the GPU was free
  (31 992 MiB of 32 607 MiB, no compute apps).

**So: yes. The final `EST_DRIFT` measurement can be produced on this workstation, and the cost is
about forty minutes of one RTX 5090 for the full V17 Arm A grid, or three minutes for one capture.
There is no cluster job here.**

### 2.3 What *is* missing is not compute

The artifact I just produced carries **[M]**:

```
"gate_qualified": false,
"gate_disqualified_reasons": [
  "estimator_not_gate_qualified",
  "geom_tol_does_not_record_gate_qualified",
  "geom_tol_is_not_gate_qualified"
]
```

— byte-identical reasons to every one of the eight A-captures **[M]**. Their causes:

| reason | source | what clears it |
|---|---|---|
| `estimator_not_gate_qualified` | `apple_sam2.py:938` `GATE_QUALIFIED = False` | **the owner's signature** on residue (i) + the 5-frame gap |
| `geom_tol_does_not_record_gate_qualified` | `configs/transfer25/pr08_geom_tol.json` has no `gate_qualified` key **[M]** | a `GEOM_TOL` *measurement* landing in that file |
| `geom_tol_is_not_gate_qualified` | `measure_est_drift.py:1588` `if not doc.get("gate_qualified", False)` | same |

Neither is a compute problem for `EST_DRIFT`. The `GEOM_TOL` half genuinely wants the cluster
(§4), the `EST_DRIFT` half does not.

**Confirmed by running the carry [M]** (§7, command C4):

```
FATAL: …/EST_DRIFT-smoke.json records gate_qualified = False, so its p95 MUST NOT be subtracted from
       GEOM_TOL. measure_est_drift's own reasons: estimator_not_gate_qualified;
       geom_tol_does_not_record_gate_qualified; geom_tol_is_not_gate_qualified
       Nothing was written.
```

---

## 3. `--carry-est-drift`, traced end to end

Producer: `scripts/measure_est_drift.py measure` → `configs/transfer25/pr08_est_drift.json`
(`DEFAULT_OUT_REL`, line 232).
Carrier: `scripts/measure_geom_tol.py --carry-est-drift <est_drift.json> [--out <geom_tol.json>]`.
Consumer: `scripts/run_g0_gates.py` `gate_budget()` + `_ca_mask_method_name()`.

### 3.1 The join key

`est_drift_estimator_name` (`measure_geom_tol.py:386` `EST_DRIFT_NAME_FIELD`) is a **measurement
slot**, listed in `CONTRACT_MEASUREMENT_FIELDS` (line 379-383), not a courtesy field. It exists
because the two halves of `GEOM_TOL − EST_DRIFT_P95` are produced by two scripts into two files, and
the merge is the moment the join can be lost. All three spellings of the one string are
`apple_sam2.ESTIMATOR_NAME` = `grounding-dino+sam2+depth-anything-v2` **[M, read from the artifact]**:

| where | field | writer |
|---|---|---|
| `pr08_est_drift.json` | `estimators.name` | `measure_est_drift` |
| `pr08_geom_tol.json` (measured) | `mask_method.name` | `measure_geom_tol` |
| `pr08_geom_tol.json` (contract) | `segmenter.method_name` | committed before measurement |
| `pr08_geom_tol.json` (carried) | `est_drift_estimator_name` | `--carry-est-drift` |

### 3.2 Every refusal, in execution order

**A. `_check_mode_flags` (line 3135-3144).**
1. `--carry-est-drift` together with `--merge` or `--shard`/`--num-shards` → FATAL. *"It MEASURES
   NOTHING … Run it on its own, after the measurement it carries."*

**B. `est_drift_measurement(path)` (line 3436-3494) — reading the drift half.**
2. `path` is not a file → FATAL.
3. not readable JSON → FATAL.
4. `schema != "wam.est_drift/1"` → FATAL. **Verified [M]**: pointing it at
   `runs/pr08-est-drift/v17/POOLED-V19.json` (schema `wam.est_drift_pooled/1`) refuses (§7, C5).
5. **`gate_qualified` falsy → FATAL**, printing the producer's own reasons. This is the refusal that
   fires today **[M]**.
6. `est_drift_p95_px` not a number → FATAL (*"There is no number to carry"*).
7. **`estimators.name` missing or blank → FATAL** — *"PR-08 §4 step 2's 'the SAME segmenter' is the
   join between the two halves … and there is nothing here to join on."*

**C. `carry_est_drift_main` (line 3497-3608) — writing into the committed document.**
8. `--out` does not exist → FATAL. *"it does not create one, because a document with an
   EST_DRIFT_P95 and no committed segmenter contract is exactly what PR-08 §4 step 2 cannot be
   checked against."*
9. `refuse_default_out_without_contract(out)` → refuses writing the tracked default path when no
   contract sits in it.
10. `--out` not readable JSON / not a JSON object → FATAL.
11. `committed_segmenter_contract(doc)` returns `None` (no `segmenter` block) → FATAL. *"Restore the
    committed contract from git."*
12. **Pixel-grid mismatch** — `est.resolution_hw` vs `document_pixel_grid(doc)` (`resolution_hw` →
    `frame_hw` → `segmenter.pixel_grid_hw`) → FATAL. *"PR-08 §6 subtracts them and that is arithmetic
    on one grid only."*
13. **`contract_disagreements(est.segmenter_contract, theirs)` non-empty → FATAL**, printing every
    field with both values. Sixteen fields are compared, and **absence counts as a disagreement**;
    they include the two checkpoint pins with revisions, `object_text_prompt`, both thresholds, the
    retry pair, `box_selection`, all three `mask_validity_*`, `propagation`, `upstream_propagation`
    and `pixel_grid_hw`.
14. `refuse_unnamed_est_drift(record, out)` (line 1982-2032) — **the number and the name land
    together or neither lands**: a missing/blank `est_drift_estimator_name` → FATAL; a name that
    disagrees with the document's own `mask_method.name` → FATAL (*"would name two segmenters"*).

**D. What it then writes** — and nothing else in the document is touched:
`est_drift_p95_px`, `est_drift_estimator_name`, `est_drift_source`
(`<rel-path> measured_utc=… sha256=… estimator='…' is_lower_bound=…`), the union of
`measurement_fields`, `est_drift_p95_blocked_by = None`, and
`gate_margin_px = geom_tol_px − est_drift_p95_px` (or `None` when `geom_tol_px` is null).

**E. Exit status.** `EXIT_NOT_GATE_QUALIFIED` (3) if `geom_tol_px` is null (*"This carry is half of
PR-08 §8 item 4"*) **or** if `margin <= 0` (*"NON-POSITIVE MARGIN … the move is a better estimator,
never a wider gate"*). `0` otherwise.

**F. The consumer, `run_g0_gates.py`.**
* `_first_present` (line 778-806) — `GEOM_TOL_KEYS` / `EST_DRIFT_KEYS` each have two spellings; if
  **both** are present and **disagree**, it refuses rather than picking by key order. Present-and-null
  and absent are deliberately distinct outcomes with distinct messages.
* `gate_budget` (line 809-897) — refuses: no tolerance key; `tol = null`; no drift key;
  `drift = null` (printing `est_drift_source` / `est_drift_p95_blocked_by` as `blocked_by`);
  non-numeric either side; a stated `gate_margin_px` disagreeing with its own subtraction by
  > 1e-9; and `margin <= 0`.
* `_ca_mask_method_name` (line 967-1019) — reads `est_drift_estimator_name`, then
  `est_drift_estimator`, falling back to `estimators.name`. A document that states a drift number
  and **names no segmenter** is a hard `GateRefusal`, not a soft "could not check"; a name that
  differs from the `GEOM_TOL` half's is a hard refusal.

### 3.3 What must be true for the carry to succeed

Exit 0 requires **all** of:

1. `configs/transfer25/pr08_est_drift.json` exists, `schema == "wam.est_drift/1"`, and
   `gate_qualified == true` — which itself requires `apple_sam2.GATE_QUALIFIED is True`, the
   committed `pr08_geom_tol.json` recording `gate_qualified: true`, matching `resolution_hw`,
   matching method name, a field-for-field matching segmenter contract, `coverage ≥ 0.90`, and no
   `--limit`.
2. `est_drift_p95_px` is a number and `estimators.name` is a non-empty string.
3. `configs/transfer25/pr08_geom_tol.json` exists, carries its `segmenter` block, agrees on
   `[480, 640]`, and names the same segmenter.
4. Run alone, without `--merge`/`--shard`.
5. For exit 0 rather than 3: `geom_tol_px` non-null **and** `geom_tol_px − est_drift_p95_px > 0`.

---

## 4. Ordering — does the carry depend on `GEOM_TOL` already being measured?

**Two different questions, two different answers, and conflating them is how the order gets planned
backwards.**

**(a) The carry *step* does not need `GEOM_TOL`'s number.** `carry_est_drift_main` writes the drift
half against a document whose `geom_tol_px` is still `null`, sets `gate_margin_px = None`, prints
*"GEOM_TOL is still null in this document … measure GEOM_TOL for the other half"* and returns exit 3
(line 3600-3604) **[A]**. Symmetrically, a **later** `measure_geom_tol` run preserves a previously
carried drift: lines 2144-2149 — *"The EST_DRIFT_P95 pair is NOT this script's to measure, and it is
not this script's to erase either … Whatever the committed document holds is kept"* — and
`gate_margin_px` is re-derived, never carried forward (line 2150-2153) **[A]**. So at the *document*
level the two halves may land in either order.

**(b) The `EST_DRIFT` *measurement* cannot be gate-qualified before `GEOM_TOL` is.**
`cross_check_geom_tol` (`measure_est_drift.py:1578-1589`) appends
`geom_tol_does_not_record_gate_qualified` when the committed document carries no `gate_qualified`
key, **and** `geom_tol_is_not_gate_qualified` when it is falsy — and `gate_qualified` is only written
by a `measure_geom_tol` *measurement*, never by the committed contract. **Confirmed [M]**: the
committed file today has no `gate_qualified` key at all, and both reasons appear in every artifact
including the one I produced this session.

**Therefore the true dependency order is strict, and `EST_DRIFT` cannot be staged first in a
committable form:**

```
1. apple_sam2.GATE_QUALIFIED = False -> True        OWNER SIGNATURE. Precondition: residue (i)
                                                    (the 92 frames, V18 outcome C) and the open
                                                    5-frame shard-7-vs-census gap.
                                                    No session may make this edit.
        |  (gate_qualified is baked into every shard AT MEASUREMENT TIME)
        v
2. GEOM_TOL re-measured at HEAD, 16 shards + merge  CLUSTER. ~9.115 GPU-h (N=16, from the sbatch's
   -> configs/transfer25/pr08_geom_tol.json          own table). Must be after 1 or it produces
      with gate_qualified: true                      another unusable artifact at the same cost.
        |  (measure_est_drift cross-checks that document)
        v
3. EST_DRIFT measured -> pr08_est_drift.json         LOCAL. ~3 min per 480-frame capture on the
      with gate_qualified: true                      RTX 5090. NO CLUSTER. The captures already
                                                     exist on disk.
        |
        v
4. measure_geom_tol.py --carry-est-drift             LOCAL, seconds.
```

Steps 2 and 3 are the strict part. Step 4 is order-free relative to step 2's *number* but not
relative to step 2's *document*.

**One consequence worth stating plainly [I]:** re-running steps 2 and 3 in the wrong order costs the
9 GPU-h twice, and re-running step 3 before step 2 costs nothing but produces another artifact the
carry will refuse — which is exactly what the thirteen artifacts already on disk are.

---

## 5. V19 — what the replacement control needs, and whether it runs locally

**It needs nothing further. It already ran, here, and it fired.**

`T40_RULE_V19` §3 replaces V17 §5's C1 (which could not fire — a lattice control cannot produce a
run longer than the lattice's own period of 5) with **C3: the propagation arm held on the wrong
object over a coherent trajectory.** What it requires:

| requirement | state |
|---|---|
| Capture `runs/pr08-est-drift/v17/A1`, 480 frames, `turns=1, yaw_turns=1, arm_cycles=2` | **on disk** [M] |
| Ground-truth `seg_ids.npy` carrying the cube distractor on frame 0 | **on disk**, resolved by label [M] |
| Module `estimators.apple_sam2_video_wrongseed`, reached only via `--propagation-module` | **exists**, `scripts/estimators/apple_sam2_video_wrongseed.py` [M] |
| Env `WAM_PR08_CONTROL_SEED_FROM_CAPTURE`, `WAM_PR08_CONTROL_SEED_LABEL=cube` | wired, `apple_sam2_video_wrongseed.py:65-66` [A] |
| One GPU, ~3 min | free; measured 31 992 MiB of 32 607 MiB idle [M] |

**Measured result, `runs/pr08-est-drift/v17/EST_DRIFT-C3-wrongseed.json` [M]:**

| | per-frame | propagation (held on the cube) |
|---|---|---|
| `est_drift_p95_px` | 0.29077062684224225 | 215.44310481569116 *(the cube-to-apple distance; means nothing, and V19 §3 says so)* |
| `n_runs` | 1 | **1** |
| `longest_run` | 1 | **480** |
| `n_frames_in_runs` | 1 | **480** |

Fire condition, carried across from V17 §5 unchanged — `n_runs ≥ 1` **and** `longest_run ≥ 10` —
**fires.** V19 §4's blind prediction (*"`longest_run` on the order of the clip length"*) held
exactly: 480 of 480.

**And V19 §5's requirement that both pools be kept is satisfied [M]:**

```
POOLED.json      outcome V   control EST_DRIFT-C1-lattice.json    fired False  longest 5
POOLED-V19.json  outcome N   control EST_DRIFT-C3-wrongseed.json  fired True   longest 480
```

I re-derived `POOLED-V19` from the artifacts into scratch and it reproduces exactly **[M]** (§7, C1):

```
outcome N: NOT OBSERVED, RATE BOUNDED. …
    per_frame: pooled p95 0.3120786214328541 px over 3827 measured frames of 3840, 3 runs, longest 13
  propagation: pooled p95 0.4486097454155794 px over 3840 measured frames of 3840, 0 runs, longest 0
  control fired: True
```

**One divergence from V19's registered text, and it is in the safe direction [M].** V19 §3 says the
seed is read *"from the renderer's `seg_ids.npy` at geom id 107"*. The module resolves the geom by
**label** (`"cube"`), not by the hard-coded id, and refuses with the list of labels it did find if
the label is absent (`apple_sam2_video_wrongseed.py:104-118`). That is strictly more robust than the
registered wording — an id would silently seed on the wrong geom if the scene were recompiled — but
it *is* a difference between the document and the code, and it belongs in the record rather than in
a docstring.

**One reading hazard that is not a defect.** V17 §4's outcomes say *"either arm"* / *"neither arm"*,
and the word "arm" is overloaded in this protocol: V17 §2/§3 use it for **Arm A** (simulated grid)
and **Arm B** (corpus), while the artifacts use it for **per-frame** vs **propagation**. Outcome N is
reached even though the per-frame side shows a 13-frame run (A3, f294-f306), because outcome D
defines *"such a run"* as **propagation-side**, and "either arm" means Arm A or Arm B.
`pool_est_drift_arms.py:262-271` implements exactly that reading and is correct. A reader resolving
the other way would conclude V17 reached D rather than N — i.e. would read the propagation blocker as
still open. Worth a sentence in any future V-document; **not** a code defect.

---

## 6. Defects

Ordered by what they would cost.

### D1 — `est_drift_p95_px` is hard-wired to the per-frame arm, and the carry reads only that field. **[M+A] — would corrupt the gate, not merely waste a run.**

`scripts/measure_est_drift.py:2333-2337`:

```python
if not run_per_frame:
    # est_drift_p95_px is the per-frame adapter's number by definition — PR-08 §6 subtracts
    # THAT from GEOM_TOL. A run that never measured it must not leave a propagation p95 in
    # the field, and must say why the field is null.
    disqualified.append("per_frame_arm_not_measured")
```

and line 2514 `"est_drift_p95_px": p95` where `p95` is the percentile over `pairs`, the per-frame
arm's displacement list. `--arm both` changes nothing about that field; the propagation p95 lives
only inside `arm_comparison.propagation.est_drift_p95_px`, and
`measure_geom_tol.est_drift_measurement()` reads **only** `doc["est_drift_p95_px"]` (line 3467).

**Consequence:** the first successful `--carry-est-drift` writes **0.31208** into
`configs/transfer25/pr08_geom_tol.json` and G0b gets a per-clip budget of **0.16650 px** instead of
**0.02997 px** — a tolerance **5.56× wider** than the authoritative arm supports, and 28.5 % of
`GEOM_TOL` handed to the generator by a field name. Nothing in `run_g0_gates` can catch it: both
arms record the **same** `SEGMENTER_CONTRACT`, so `contract_disagreements` sees no difference, and
the contract's own `propagation` field says `"per_frame"` on both sides.

**This is not currently mis-gating anything** — `geom_tol_px`, `est_drift_p95_px` and
`gate_margin_px` are all `null` **[M]**, so `gate_budget()` refuses before subtracting. It fires the
moment the carry first succeeds.

**Proposed fix (a diff, not applied).** Do **not** silently swap the field — that would move a gate
number without a rule. Make the artifact state which topology its headline is, and make the carry
refuse a topology the committed contract does not name:

```diff
--- a/scripts/measure_est_drift.py
+++ b/scripts/measure_est_drift.py
@@ artifact assembly (~line 2510)
         "est_drift_p95_px": p95,
+        # WHICH TOPOLOGY THE HEADLINE IS. PR-08 §4 step 2's "the same segmenter" is the
+        # GENERATOR's (measure_geom_tol.py:479; apple_sam2.py:853), and the generator
+        # propagates. Recording the topology beside the number is what lets the carry refuse a
+        # budget measured on the other one instead of subtracting it silently.
+        "est_drift_p95_topology": PROPAGATION_OF_HEADLINE,   # "per_frame" today
+        "est_drift_p95_px_by_arm": {
+            "per_frame": per_frame_block.get("est_drift_p95_px"),
+            "propagation": propagation_block.get("est_drift_p95_px"),
+        },
         "headline_valid": headline_valid,
```

```diff
--- a/scripts/measure_geom_tol.py
+++ b/scripts/measure_geom_tol.py
@@ est_drift_measurement (~line 3480)
+    topology = doc.get("est_drift_p95_topology")
+    committed_topology = (committed_segmenter_contract(...)[0] or {}).get("propagation")
+    if topology is not None and committed_topology is not None and topology != committed_topology:
+        raise MethodUnavailable(
+            f"FATAL: {path}'s est_drift_p95_px was measured on the {topology!r} topology while the "
+            f"committed segmenter contract names {committed_topology!r}. PR-08 §4 step 2's 'the "
+            "same segmenter' includes how it is driven, and the two topologies differ by 0.1365 px "
+            "at p95 on the V17 pool. Nothing was written."
+        )
```

**And the decision the fix cannot make**: which value `SEGMENTER_CONTRACT["propagation"]` should
hold. It is `"per_frame"` today (`apple_sam2.py:525`, and in the committed
`configs/transfer25/pr08_geom_tol.json` **[M]**). Changing it is a **pre-registered contract
change**, i.e. a new V-document and a `GEOM_TOL` re-measure, because the committed contract was
fixed before the number and must not be edited after seeing it. **Owner's call.**

### D2 — the pooled `EST_DRIFT_P95` has no carry path at all. **[M] — would waste the operator's time at the last step.**

`pool_est_drift_arms.py` writes `schema: "wam.est_drift_pooled/1"`;
`measure_geom_tol.est_drift_measurement` requires `"wam.est_drift/1"`. **Verified [M]** (§7, C5):

```
FATAL: runs/pr08-est-drift/v17/POOLED-V19.json does not carry schema 'wam.est_drift/1'
       (got 'wam.est_drift_pooled/1' ).
```

`POOLED*.json` also carries no `gate_qualified`, no `estimators.name` and no `resolution_hw`, so
three further refusals stand behind the schema one. **The number V17 spent eight captures producing
cannot reach the gate document by any committed tool.** Whatever is carried will be a *single*
capture's p95 — and V17 §4 explicitly left open *"whether the pooled number or the single-capture
number is the one G0b subtracts"*, so this is an unresolved rule question with no plumbing on either
side of it.

### D3 — `run_v17_arms.sh` skips any artifact that already exists. **[A] — would silently produce nothing after the flip.**

`scripts/run_v17_arms.sh`, `measure()`:

```bash
if [[ -f "${out}" ]]; then echo "SKIP ${stem} (measured)"; return 0; fi
```

After `GATE_QUALIFIED` flips and `GEOM_TOL` is committed, re-running the script to obtain
gate-qualified artifacts **skips all thirteen** and leaves the stale `gate_qualified: false` files
in place; the carry then refuses with reasons that no longer describe reality. The operator must
point `V17` at a fresh directory or move the old artifacts aside first. There is no warning in the
script and no check that the skipped artifact's `gate_qualified` matches the current flag.

### D4 — the pool docstring promises a device check that does not exist, and no artifact records a device. **[M]**

`pool_est_drift_arms.py:26-30`: *"A capture that disagrees about the instrument is refused, not
averaged in. Different segmenter contract, different pixel grid, **different device**, a schedule
that is not `trajectory` … each refuses the whole pool by name."*

`_instrument_key` (lines 84-95) contains **no device component** — only the segmenter contract, the
resolution, the object class, the propagator spec and the IoU threshold. And `measure_est_drift.py`
writes **no device field anywhere**: I walked every key of `EST_DRIFT-A1.json` for a device-shaped
name and found none **[M]**.

**Consequence:** `PR-08-RESULT-2026-08-27-…-the-per-frame-arm-is-the-one-that-broke.md` §8 asserts
*"device | one RTX 5090, one process, every measurement in this document"* — an assertion no
artifact can support and no code can check. This is the exact provenance class V17 §7 registered a
rule against (*"so a reader six months later is not asked to infer them"*). The fix is one field in
the artifact and one element in `_instrument_key`.

### D5 — V5 §5's floor is registered as a floor and enforced as a comment. **[M] — a 12-frame capture produces a `headline_valid` budget.**

`T40_RULE_V17` §0's "unchanged" table lists *"V5 §5's floor — ≥ 20 distinct scene states, ≥ 200
measured frames | `measure_est_drift.py:878`"* as if it constrained the run. It does not:
`independent_sample_block`'s own docstring (line 907-909) says *"**Nothing here is subtracted from
anything and no disqualification reason depends on it.**"*

**Demonstrated [M]:** my 12-frame / 12-scene-state smoke capture produced

```
est_drift_p95_px 0.5604066865220801   headline_valid True   n_frames 12   n_measured 12
reasons ['estimator_not_gate_qualified', 'geom_tol_does_not_record_gate_qualified',
         'geom_tol_is_not_gate_qualified']
```

— i.e. **the only things stopping a twelve-frame capture from becoming G0b's committed budget are
the two flags that are about to be flipped.** Once they are, `--carry-est-drift` would accept it.
Given that the whole of V17 exists to establish that one 480-frame capture is not enough, a floor
that does not refuse 12 frames is a hole in the same argument. The fix is three lines:

```diff
+    # T40_RULE_V5 §5, enforced rather than only recorded. V17 §0 lists this as a standing floor.
+    if independent["recorded"]:
+        if independent["n_scene_states_with_a_measured_frame"] < 20:
+            disqualified.append("below_v5_scene_state_floor")
+        if independent["n_measured_frames"] < 200:
+            disqualified.append("below_v5_measured_frame_floor")
```

### D6 — the repo contains one comment that reads §4 step 2 the other way, off a false premise. **[A]**

`scripts/restyle_transfer25.py:339-343`: omitting `control_path` *"makes Transfer2.5 estimate the map
itself, with its OWN depth/segmentation models — **a different estimator from the isaac_binding.py
annotators GEOM_TOL was measured against**"*, and line 352 refuses a named-but-missing map because
that *"would swap the estimator GEOM_TOL was measured against for a different one."*

`GEOM_TOL` was measured with `apple_sam2` (GroundingDINO + SAM 2) over the **real corpus**
(`runs/pr08-geom-tol/pr08_geom_tol.json`, `mask_method.name` =
`grounding-dino+sam2+depth-anything-v2` **[M]**), never against the `isaac_binding.py` annotators.
The comment is wrong on the fact and, resting on it, takes the opposite position from
`measure_geom_tol.py:479` and `apple_sam2.py:853` on which segmenter §4 step 2 names. This is the
comment a future reader will find at the moment they decide whether to supply the maps — i.e. the
moment §1.4's fork is resolved — so it is worth correcting before then.

### D7 — `is_lower_bound` is stamped, argued over at length, and read by nobody. **[M]**

PR-08 §4: *"`EST_DRIFT_P95` is a **lower bound** on the real error … and a G0b margin that only
clears under a lower bound is not a pass."* `measure_est_drift.GROUND_TRUTH_BINDINGS` computes
`is_lower_bound` / `error_direction` / `error_direction_measured` per route and the carry embeds
`is_lower_bound=…` in the free-text `est_drift_source` string. **`scripts/run_g0_gates.py` contains
zero occurrences of `is_lower_bound` or `error_direction` [M].** So §4's sentence is enforced by no
code path. Mitigated in practice — the MuJoCo route stamps `is_lower_bound: false` with an *argued,
explicitly unmeasured* direction — but the enforcement gap is real and belongs in the record.

---

## 7. Provenance — exactly what I ran

Interpreter `/home/humanoid/develop/wam/.venv/bin/python`. Working directory
`/home/humanoid/develop/wam`. Scratch
`docs/investigations/2026-08-27-pr08-fronts`
(`$S` below). GPU: RTX 5090, 32 607 MiB, 31 992 MiB free, no compute apps at start **[M]**.

**C1 — re-derive the V19 pool (read-only over `runs/`, writes to scratch).**
```bash
.venv/bin/python scripts/pool_est_drift_arms.py \
  --artifact runs/pr08-est-drift/v17/EST_DRIFT-A[1-8].json \
  --control  runs/pr08-est-drift/v17/EST_DRIFT-C3-wrongseed.json \
  --divergence runs/pr08-est-drift/v17/ARM_DIVERGENCE.json \
  --out "$S/POOLED-V19-rederived.json"
```
Result: `outcome N`, per-frame `0.3120786214328541`, propagation `0.4486097454155794`, control fired.

**C2 — a MuJoCo capture, on this workstation, timed.** `real 0m3.488s` for 12 frames.
```bash
MUJOCO_GL=egl .venv/bin/python scripts/measure_est_drift.py capture \
  --backend mujoco --schedule trajectory --frames 12 --render-hw 480 640 --out "$S/cap-smoke"
```

**C3 — both arms, end to end, timed.** `real 0m10.406s` including model load.
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python scripts/measure_est_drift.py measure \
  --capture "$S/cap-smoke" --estimators estimators.apple_sam2 --arm both --out "$S/EST_DRIFT-smoke.json"
```

**C4 — the carry, refused at `gate_qualified`.**
```bash
cp configs/transfer25/pr08_geom_tol.json "$S/pr08_geom_tol-copy.json"
.venv/bin/python scripts/measure_geom_tol.py \
  --carry-est-drift "$S/EST_DRIFT-smoke.json" --out "$S/pr08_geom_tol-copy.json"
```

**C5 — the pooled artifact, refused at the schema.**
```bash
.venv/bin/python scripts/measure_geom_tol.py \
  --carry-est-drift runs/pr08-est-drift/v17/POOLED-V19.json --out "$S/pr08_geom_tol-copy.json"
```

Plus read-only inspection of `configs/transfer25/pr08_geom_tol.json`,
`runs/pr08-geom-tol/pr08_geom_tol.json`, all thirteen `runs/pr08-est-drift/v17/EST_DRIFT-*.json`,
both `POOLED*.json`, `ARM_DIVERGENCE.json`, and the eight `A{1..8}/capture.json` headers.

---

## 8. What this document does NOT do

* **It licenses no clip and no training.** `T40_RULE_V1` §1 binds in full; §8 items 3 and 4 are open.
* **It does not flip `apple_sam2.GATE_QUALIFIED`** and does not argue that it should be flipped. That
  flag has two preconditions, one of them still carries the open five-frame shard-7-vs-census gap,
  and the flip is the owner's signature.
* **It does not shorten `GATE_QUALIFICATION_BLOCKERS`** — the tuple is already `()` at HEAD, and
  emptiness is not permission.
* **It edits no file in the repository.** Every proposed change in §6 is a diff in this document and
  is not applied.
* **It does not answer whether the pooled or the single-capture p95 is G0b's budget.** V17 §4 left
  that open; §6 D2 records that neither has plumbing.
* **It does not decide `SEGMENTER_CONTRACT["propagation"]`.** §1 establishes which arm §4 step 2
  names; changing the committed contract to match is a new V-document and a `GEOM_TOL` re-measure,
  and it is the owner's.
* **It says nothing about GR00T.** PR-07 §6 forbids it.

---

## Adversarial re-read

Second Claude Code session, same workstation, 2026-08-27. Task: **refute**, not agree. Nothing under
`/home/humanoid/develop/wam` was modified; the only command I ran that writes anything wrote
`…/scratchpad/sprint/adv-POOLED-V19-recheck.json`. No cluster, no git-write.

Labels as in the parent document: **[M]** measured by me this session, **[A]** committed artifact /
tracked file with path+line, **[I]** my inference.

**Verdict: the deliverable does NOT survive.** Its defect catalogue (D1–D7) is almost entirely
sound and I reproduce its arithmetic to the last digit. Its **headline is not.** The single claim
the whole document is organised around — *"the authoritative arm is propagation"*, asserted in §0.3
and §1.2 as a thing *"the project already resolved … in writing, twice"* — is an inference resting
on two truncated quotations, and the repository says the opposite in **four** places, one of which
is the committed contract itself.

### R1 — LOAD-BEARING. "The project already resolved that referent twice, in writing" is false; the repo records it as explicitly UNSETTLED

The parent's §1.2 concludes **"Answer: the propagation arm is authoritative"** with no `[I]` label,
and the structured summary promotes it to a *blocking fact*. Against that:

**(a) `apple_sam2.py:701-710` — the retired propagation blocker, the parent's own third citation,
one sentence past where the parent stops quoting [A]:**

> The bias is **TWO-SIDED**, which is why this cannot be waved through as conservative: (a)
> independent re-detection jitters frame to frame where propagation is temporally smooth, so our
> tail … is INFLATED relative to the generator's … (safe); (b) propagation's own characteristic
> failure … is invisible to a per-frame estimator … (unsafe). … with (a) and (b) together **this
> number is neither a lower nor an upper bound on the generator's mask error**.

**(b) `apple_sam2.py:723-728` — the discharge text, after the measurement [A]:**

> Both readings are recorded because **reporting only the headline would make this file's own
> argument look settled when the sign depends on which percentile the gate uses.**

That is the repo pre-emptively naming the parent document's exact move. The two distributions
**cross between p95 and p99** (per-frame p99 1.0431 / p100 67.633 against propagation 0.5631 /
19.399, `apple_sam2.py:690-692` [A]) — so "propagation is the worse arm" is true *at p95 only*, and
the file says so.

**(c) `apple_sam2.py:519-524`, the comment on the committed constant [A]:**

> See :data:`GATE_QUALIFICATION_BLOCKERS` for the argument about which way that biases the budget —
> **it is not one way.**

**(d) The committed contract itself [M]:** `configs/transfer25/pr08_geom_tol.json`
`segmenter.propagation == "per_frame"`, and `runs/pr08-geom-tol/pr08_geom_tol.json`
`segmenter.propagation == "per_frame"`. The pre-registered contract, fixed *before* the number,
names per-frame. The parent acknowledges this only in §6 D1's last paragraph, after having built
§0–§1 on the opposite premise.

**And the registered discharge condition was never "subtract the propagation number."** It was
(`apple_sam2.py:711-713` [A]) *"measuring the same Isaac capture BOTH ways … and **recording the two
p95s**, so the direction and size of the difference are a measurement rather than the argument
above."* Recording both is what V17 did. Choosing one is what nobody has done.

The parent's own `needs_owner_signature` list concedes this — *"DECIDE WHICH ARM G0b SUBTRACTS, ON
THE RECORD"*. **A document cannot list a decision as unmade and simultaneously report its outcome as
a blocking fact.** The headline is refuted; the underlying open question is real and correctly
identified.

### R2 — LOAD-BEARING. §1.4's inference is reversed by the next sentence of the file it cites

§1.4 cites `build_pr08_source.py:30-35` for *"depth and segmentation are absent … a decision rather
than an oversight"* and infers **[I]** that *"as the pipeline stands today … the propagation arm is
the one that measures the error the run will actually commit."*

The docstring continues, at **lines 33-35** [M]:

> `PR-08 §4's estimated conditioning is blocked on §8 items 4 and 5; **until those land**, the
> honest manifest is one that claims no maps at all. **Adding the keys later is an edit to this
> script and a regenerated manifest, which is visible.**`

The omission is declared **temporary and pending §8 item 4 — the very item this front is about.**
And `restyle_transfer25.py:342-343` [M] states the intended end state directly: *"Where the manifest
carries a map, it is passed, **so the run uses the estimator the geometry budget characterises.**"*

So the code's design intent is that the supplied-map (per-frame `apple_sam2`) branch is the target
state, and the estimate-it-yourself branch is the interim. The parent read the interim as the
settled architecture and inverted the intent. Quote-truncation that reverses the conclusion.

*(This does not rescue the per-frame arm either — no producer for those maps exists, as the parent
correctly notes. It establishes that the fork is open, not that propagation wins it.)*

### R3 — LOAD-BEARING. `measure_geom_tol.py:479-480` is quoted mid-comment, and the clause that follows disclaims it

The parent quotes three lines and stops. The full comment, `measure_geom_tol.py:478-482` [M]:

> `#: Checkpoints Cosmos-Transfer2.5 itself names for this pipeline`
> `#: (…sam2_model.py), so the generator's own segmenter is the one the tolerance is measured with.`
> `#: **Used ONLY to make the failure message concrete: the artifact quotes checkpoints the ADAPTER**`
> `#: **declares and never these, because what Cosmos names upstream is not evidence about what this**`
> `#: **adapter loaded.**`

This is a comment on `COSMOS_SAM2_CHECKPOINT_HINTS`, a constant used at exactly one site
(`measure_geom_tol.py:867`, formatting a failure message [M]). It resolves *which checkpoints*, and
its own next clause says upstream's naming **is not evidence about this adapter**. Presenting it as
one of two written resolutions of §4 step 2's topology referent is out of the context that
undercuts it.

### R4 — LOAD-BEARING. The headline "already measured to the registered standard" is contradicted by the parent's own §8 and D2

Headline §0.1: *"`EST_DRIFT_P95` is already measured to the standard the protocol registered."*
§8: *"It does not answer whether the pooled or the single-capture p95 is G0b's budget."*
§6 D2: neither candidate has a carry path.

`T40_RULE_V17` §4, lines 200-202 [M]:

> whether the pooled number or the single-capture number is the one G0b subtracts is **a separate
> question this document does not answer**.

A quantity for which no registered rule names the admissible estimate, and whose two candidate
estimates differ by 0.1365 px on a 0.4786 px tolerance (28.5 % of the budget), is **not** "measured
to the registered standard." V17 measured a *drift rate* to its registered standard and reached
outcome N. That is a different quantity from the *budget* §6 subtracts. The headline conflates them.

### R5 — The headline finding is already in the sprint document, uncited, and the sprint declines to name the arm

`docs/SPRINT-2026-08-27-cosmos3-dataset-generation.md` §3.4, lines 155-163 [M] — a table the
deliverable never cites, containing its own §1.3 table verbatim:

> | **V17 Arm A pooled, per-frame** | 0.31208 | 0.16650 | 34.79 |
> | **V17 Arm A pooled, propagation** | 0.44861 | **0.02997** | **6.26** |
>
> The propagation row is the one to watch: 6.26 % of the budget is not room, and **which arm is
> authoritative is settled by §4 step 2, not by picking the friendlier number.**

The sprint states the tension and **deliberately does not resolve it**. The deliverable presents the
same three numbers as a re-derivation `[M]` (they are correct — see below) and adds a resolution the
sprint withheld. Its contribution on this axis is the resolution, and the resolution is R1.

### R6 — The structured summary's `runnable_now` list contains repo-mutating and post-signature commands

The deliverable's §7 lists **five** commands (C1–C5), all scratch-only, and I accept all five. The
**structured summary** lists **eight**, and the extra three were never run and are not runnable now:

* **summary C4** — `scripts/run_v17_arms.sh --min-free-mib 10000`. `run_v17_arms.sh:37` hardcodes
  `V17=runs/pr08-est-drift/v17` and `:178` writes `runs/pr08-operating-point/EPISODE_094_CENSUS.json`
  [M] — **both inside `/home/humanoid/develop/wam`**, which this session is forbidden to modify.
  ~3 h of RTX 5090 by the summary's own estimate.
* **summary C7** — `--out configs/transfer25/pr08_est_drift.json`: writes a tracked config path.
* **summary C8** — `measure_geom_tol.py --carry-est-drift` with no `--out`: default target is
  `configs/transfer25/pr08_geom_tol.json`, **the committed pre-registered contract**.

C7/C8 are correctly gated in prose ("once GATE_QUALIFIED is True"), but they sit under the key
`runnable_now`, which is the field a driver script reads. Summary C2 and C3 are also mislabelled:
the parent measured a **12**-frame capture and a smoke measure, never a 480-frame capture nor a
measure over `runs/pr08-est-drift/v17/A1`.

### R7 — D3's stated remediation is not possible

D3 (correct as a defect) advises *"point V17 at a fresh dir or move the old artifacts aside."*
`run_v17_arms.sh:37` is `V17=runs/pr08-est-drift/v17` — a bare assignment, not `${V17:-…}`, and the
flag parser at `:29-34` accepts only `--min-free-mib` and `--wait-minutes` [M]. **There is no way to
point it at a fresh directory without editing the script.** Only the second half of the advice works.

### R8 — D5's proposed fix does not bind on the route PR-08 §4 step 1 names, and is a rule change dressed as a code diff

Two problems.

1. **It is bypassed exactly where it matters.** The fix guards on `if independent["recorded"]:`.
   `scene_state_per_frame` (`measure_est_drift.py:876-892`) returns `(None, reason)` — hence
   `recorded: false` — when the header carries no `steps_per_state`, and its own docstring says
   *"**Isaac captures declare no schedule**"* [M]. PR-08 §4 step 1 is *"Render N **Isaac** episodes"*.
   So the proposed floor would constrain the MuJoCo stand-in and silently exempt the route the
   pre-registration actually names.
2. **It is not a session's to make.** `independent_sample_block`'s docstring states the current
   behaviour as a decision, not an omission: *"**Nothing here is subtracted from anything and no
   disqualification reason depends on it.**"* Turning an advisory block into two new
   `gate_disqualified_reasons` changes what `gate_qualified` means, after the artifacts it would
   judge are on disk. `docs/handoff.md:165` [M]: *"Rules are versioned, never edited in place. **A
   gate rewritten after seeing its output is not a gate.**"* The correct instrument is a V-document,
   which the parent applies to `SEGMENTER_CONTRACT["propagation"]` (§6 D1, correctly) and not here.

*(The underlying observation — a 12-frame capture reaches `headline_valid: true` — I confirm as a
real hole. It is the fix that fails, not the finding.)*

### R9 — D1's proposed diff does not compile at the site it names, and would not prevent the harm it describes

D1's *finding* is verified (see below). Its *fix* is not a diff.

* Hunk 2 inserts into `est_drift_measurement`. That function is `def est_drift_measurement(path:
  Path) -> dict[str, Any]` (`measure_geom_tol.py:3436` [M]) — it takes **only the drift artifact's
  path** and has no reference to the committed `GEOM_TOL` document, so
  `committed_segmenter_contract(...)` — with the argument literally elided as `...` — cannot be
  resolved there. The check belongs in `carry_est_drift_main`, which does hold both.
* Hunk 1 references `PROPAGATION_OF_HEADLINE`, `per_frame_block` and `propagation_block`; none of
  the three exists in `measure_est_drift.py` [M].
* **Most importantly, the semantics are backwards for the stated purpose.** The refusal fires when
  `topology != committed_topology`. The committed contract says `"per_frame"` [M]. So the guard
  would **refuse a propagation-topology measurement and wave the per-frame one straight through** —
  i.e. it hard-codes precisely the outcome D1 calls a 5.56× over-wide gate. The parent's §6 D1
  closing paragraph concedes the contract decision is the owner's, but the summary presents the diff
  as *"Fix diff in the deliverable §6 D1"*, which it is not.

### R10 — two wall-clock figures do not check out

* §2.1: *"`measure … --arm both`, 480 frames: 3 min 02 s ± 2 s each (**A1→A8 in 24 min 15 s**)."*
  Measured `measured_utc` stamps [M]: A1 `00:50:48`, A8 `01:12:03` → **21 min 15 s**, 7 intervals,
  182.1 s each. **24 min 15 s is `C1-lattice` (00:47:48) → A8** — a 60-frame control folded into a
  span labelled A1→A8. The per-item 3 m 02 s is right; the span is not.
* §2.1: *"Arm B … ≈ **81 min** **[M]**."* There is no Arm B duration in any artifact. The figure is
  the gap between two *write* timestamps (`C3` 01:24:03 → `ARM_DIVERGENCE` 02:45:41 = 81 m 38 s)
  [M], which is an upper bound including any idle between two script steps. That is **[I]**, not
  **[M]** — the distinction the parent's own §0 preamble insists on.

### What survives, re-verified independently

Everything below I re-checked from the artifacts and the source, and it holds.

| claim | status |
|---|---|
| `GEOM_TOL = 0.47857992441961017`, 402 ep / 171 625 frames, `mask_method.name = grounding-dino+sam2+depth-anything-v2` | **confirmed [M]** `runs/pr08-geom-tol/pr08_geom_tol.json` |
| pooled per-frame `0.3120786214328541` / propagation `0.4486097454155794`; margins `0.16650130298675608` (34.790699419470585 %) / `0.02997017900403076` (6.262314291677947 %); ratio **5.5556×**; delta 0.13653 px = 28.53 % of `GEOM_TOL` | **confirmed [M]**, exact to the last digit |
| C1 reproduces `POOLED-V19` | **confirmed [M]** — I re-ran the pool into `adv-POOLED-V19-recheck.json`: `outcome N`, both pooled p95s identical, control fired |
| **D1 mechanism** | **confirmed [M]**. `values, dropped = paired_displacements(pairs)` (`measure_est_drift.py:2501`) where `pairs` is the per-frame list; `"est_drift_p95_px": p95` at `:2514`; `measure_geom_tol.est_drift_measurement` reads `doc.get("est_drift_p95_px")` at `:3467` and nothing else. One `SEGMENTER_CONTRACT` per artifact, so `contract_disagreements` cannot see the arm. **The defect is real; only the proposed fix (R9) and the "propagation is authoritative" premise (R1) fail.** |
| **D2** | **confirmed [M]** — `POOLED-V19.json` `schema = wam.est_drift_pooled/1`; no `gate_qualified`, no `estimators.name`, no `resolution_hw`; headline number is `pooled_est_drift_p95_px`, not the field the carry reads |
| **D4** | **confirmed [M]** — `pool_est_drift_arms.py:27` promises *"different device … refuses the whole pool by name"*; `_instrument_key` (`:84-95`) has 5 components, none of them a device; zero device/gpu/cuda/host-shaped keys anywhere in `EST_DRIFT-A1.json`; `PR-08-RESULT-2026-08-27-…md:231` asserts `device \| one RTX 5090, one process` |
| **D6** | **confirmed [M]** — `restyle_transfer25.py:341` says *"a different estimator from the isaac_binding.py annotators GEOM_TOL was measured against"*; `GEOM_TOL` was measured by `measure_geom_tol` with `apple_sam2` over the real corpus. Factually wrong. *(Note: this comment is also a fourth repo citation reading §4 step 2 against the parent's headline — see R1/R2.)* |
| **D7** | **confirmed [M]** — `grep -c "is_lower_bound\|error_direction" scripts/run_g0_gates.py` → **0** |
| every refusal in §3.2, and the exit-3 semantics | **confirmed [M]** at the cited lines |
| dependency order (flip → cluster `GEOM_TOL` → local `EST_DRIFT` → carry) | **confirmed [A]** — matches sprint §3.2's own *"because `gate_qualified` is baked into every shard at measurement time, only after `GATE_QUALIFIED` flips"* |
| `GATE_QUALIFIED = False` at `apple_sam2.py:938`; `GATE_QUALIFICATION_BLOCKERS == ()`; committed config has no `gate_qualified` key; all three measurement slots null | **confirmed [M]** |
| `GEOM_TOL` uncommittable: merged artifact's `segmenter` block has 15 fields, missing `mask_validity_reference_max_frame_fraction`; adapter's `SEGMENTER_CONTRACT` has 16 | **confirmed [M]** |
| V19 C3 fired (`n_runs 1`, `longest_run 480`, `required_longest_run 10`); both pools kept (`POOLED.json` outcome V, `POOLED-V19.json` outcome N) | **confirmed [M]** — stored under key `positive_control`, not `control` |
| the "either arm" overloading note, and the label-vs-geom-id divergence in the C3 seed | **confirmed [M]** — `apple_sam2_video_wrongseed.py:104-118` resolves by label and refuses with the label list |
| local feasibility of the `EST_DRIFT` compute | **confirmed [A/I]** — the artifact stamps show the whole Arm A grid ran here in ~38 min. The *compute* has no cluster dependency. What is refuted is the stronger headline (R4), not this. |

### Bottom line for the driver

Take **D1–D7 as findings** and **discard the resolutions**. Specifically:

1. Do **not** record "propagation is authoritative" as established. Record it as the parent's
   `needs_owner_signature` item #2 already does — an open owner decision worth 0.1365 px — and carry
   R1's four counter-citations with it so the owner sees the repo's own two-sided argument.
2. Keep D1 as *"the headline field silently commits to one arm and nothing downstream can see it"*.
   Drop the diff (R9).
3. Keep D5's observation, drop its diff (R8) — it needs a V-document, and as written it exempts the
   Isaac route.
4. Strike summary commands C4, C7, C8 from `runnable_now`; only §7's C1–C5 were run and only they are
   runnable under this session's constraints (R6).
5. Fix D3's remediation to "move the old artifacts aside" — `V17` is not overridable (R7).
6. The §3.3 **"cluster"** label: the parent is right that the *compute* is local, and that is worth
   propagating. It is **not** right that no further work is needed (R4) — the arm question and the
   pooled-vs-single question are both open, and both are the owner's.
