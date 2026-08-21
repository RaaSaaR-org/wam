---
id: T-040
aliases:
- T-040
- T-40
title: Cosmos-Transfer2.5 photoreal augmentation — pre-register before generating anything
slug: cosmos-transfer-photoreal-augmentation
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- post-mvp
- data
- sim
- backbone
- prereg
sprint: ''
depends_on:
- "[[T-39]]"
due_date: ''
created: 2026-08-06
updated: 2026-08-06
status_note: "Promoted out of backlog 2026-08-06: the deliverable is a document, the document costs
  no GPU, and PR-07 §7's freeze does not name Transfer2.5. `docs/preregistration/PR-08-photoreal-augmentation.md`
  is written — rule `T40_RULE_V1`, four arms, three VOID gates — and closes 9 of 13 acceptance
  criteria. The four left open need the cluster or a written contract, not a decision: throughput on
  an H200, the chunked sbatch, the vla-training consumer contract, and the two measured constants
  GEOM_TOL / EST_DRIFT_P95. GENERATION STILL WAITS ON T-39 — PR-08 §1 binds itself to the reason
  behind the freeze even though its letter does not reach Transfer2.5."
---

# Cosmos-Transfer2.5 photoreal augmentation — pre-register before generating anything

## Description

Write the pre-registration that would let generated video enter a training corpus, and run
`screen_corpus.py` (T-34) on the generated corpus before any of it is trained on. **This task
does not generate data. It produces the document and the screen that would license generating
it.**

The pipeline is the one thing in the 2026-08 material that is concrete
(`docs/backbone-eval.md` §5): **Cosmos-Transfer2.5** consumes depth + segmentation + Canny and
emits photorealistic video, and the Isaac backend (`src/wam/robot/isaac_transport.py`) ~~already
emits exactly those three~~ **[corrected 2026-08-06 — it does not; see the Notes]**. The consumer
is `emai/vla-training` — a 28-dim G1+Dex3 LeRobot v3
pipeline that trains GR00T N1.7 / π0.5 / SmolVLA and evaluates in MuJoCo and Isaac.

**Why Transfer and not Predict, Nano or Super.** Transfer restyles frames of an episode that
*already has actions* — the labels come from the teleop or the sim, not from the generator. A
Predict-family dream has no action labels, and WAM's own action decoder cannot supply them
without circularity: it is the negative result, not a labeller.

### The concrete experiment

Take one dataset — `nvidia/GR00T-N1.7-AppleToPlate`, 402 real teleop demos — and emit N restyled
copies: different apple, tablecloth, wall, lighting. **The robot stays the same G1 + Dex3, and the
joint states and actions are carried over from the original recording unchanged.** Then fine-tune
the VLA on real-only vs. real + restyled and compare. This is **visual domain randomization**, not
new behaviour.

**Appearance may vary; geometry may not.** The actions are the recorded trajectory, so anything
that moves an object desynchronizes pixels from labels and the arm grasps empty air. Allowed:
apple texture/colour/variety, table material, background, lighting. Forbidden: apple or plate
*position*, table height, robot pose, object count. Depth + segmentation conditioning is what
enforces this — the constraint is natural to the tool, which is exactly why it must be gated
rather than assumed.

**This is not T-32.** Ten restyles of 402 demos is still 402 trajectories. It cannot answer the
data-volume question and must not be reported as if it could.

**What it is aimed at.** NVIDIA trained ApplePnP deliberately brittle to appearance —
`real-record.html`: *"Keep the table, object appearance, robot pelvis position, and head pose
consistent"*; the 68 % real success is caveated *"depends on how closely your environment matches
the environment used during data collection"*; `EXPLAINABILITY.md` names lighting and scene layout
as the known risk. That brittleness is the target. **It is *not* the fix for the Isaac 0/10** —
`vla-training/docs/apple-pnp-ursachen.md` ranks that as U1, saturated raw radians against a Dex3
drive at its 5 N·m limit (contact 10/10, lift identically 0.000 mm in 10/10), an actuation defect
with the visual hypotheses refuted 3-to-0. Claiming this task addresses 0/10 would be wrong.

### The conditioning signals do not exist yet — pick a path and price it

Transfer2.5 consumes **depth + segmentation + Canny**. Our real corpus ships none of the first two:
`datasets/gr00t-apple-full/*/manifest.json` declares a single camera `ego` (RGB, 120×160, 30 fps),
and NVIDIA's source is one head RealSense D435 colour topic. Only Canny is computable from it.
Two paths follow, and they are **not** the same experiment:

- **Isaac path.** `isaac_transport.py` ~~renders depth and segmentation for free~~ **[corrected —
  it renders neither today; two annotators have to be attached first, see the Notes]**, so the
  conditioning is exact *once wired*. But then the frames being restyled are *sim* frames, which
  collides with T-25 directly
  rather than at one remove, and the actions come from sim teleop rather than the 402 real demos.
- **Real-teleop path.** Depth and segmentation must be **estimated** (monocular depth, SAM-class
  masks) from RGB. Estimation error lands as geometry drift — precisely what the
  geometry-invariance gate forbids — so the estimator's error has to be measured before it is
  trusted, not after.

**There is now a way to measure that estimator error against ground truth.** `USC-GVL/humanoid-everyday`
(see [[T-041]]) records the **same camera** — one egocentric RealSense D435, 640×480, 30 Hz — and
ships **measured depth** (uint16, 480×640) alongside the RGB on the same G1 + Dex3 embodiment,
with published colour/depth intrinsics and depth-to-colour extrinsics that AppleToPlate does not
have. So the monocular estimator can be calibrated where depth is real and its error budget
carried over, rather than assumed. **Depth lives only in the mixed repo**
`USC-PSI-Lab/humanoid-everyday` (`robot_type: "mixed"`, 8 949 episodes, 3 436 171 frames, float32
480×640 depth) — the 3.84 GB `Humanoid-Everyday-G1` set is RGB-only. So this needs the large
download and a filter to the G1 episodes, unlike T-041 which does not. This does not remove the
gap on AppleToPlate, which stays RGB-only — it makes the gap quantifiable.

Also: the converted corpus is **120×160**. A photoreal restyle at that resolution is worthless to a
VLA that trains on 640×480, so this path re-derives from the HF source at full resolution and does
not reuse `datasets/gr00t-apple-full/`.

### Where it runs, and what it costs

Discoverer+ is the only machine with the capacity, and the cost is the thing that decides whether
this is a project or a paragraph:

- **~172 000 frames per restyle variant** (402 episodes × ~427 frames mean, range 249–749) —
  about 95 minutes of 30 fps video. Per variant. N variants multiply it.
- **Throughput is unmeasured.** No number here until one episode is timed on an H200 at the target
  resolution. A budget line without that measurement is a guess and must not be written as if it
  were not.
- 4 h `MaxWall` and `MaxJobsPU=4` → generation must be **chunked and resumable**, the shape
  `submit_chain.sh` and `MAX_RESTARTS` already use. Billing counts RAM
  (`GPUs×1.0 + MemGB×0.25 + Threads×0.036` per minute), so `--mem` is not free.
- **Isaac is not installed on the cluster.** Under the Isaac path, depth/segmentation are produced
  on dz-226 and shipped; only the restyle runs on Discoverer+.
- The login node is off limits for anything that computes — `sbatch`, `squeue`, file management
  only.

### What this collides with, and must argue against explicitly

Three standing results, none of which this task may quietly step over:

- **"Sim frames are NOT training data"** — `docs/sim.md`, T-25. A standing decision. Overturning
  it is the point of the pre-registration, and it needs an argument, not a paragraph in a
  backbone doc.
- **T-36 / PR-06 already priced generated video as supervision and it lost.** The anchored dream
  scored **16.656** from the truth where holding the conditioning frame scored **12.020** — 39 %
  *worse than standing still*. Transfer2.5 is a different claim (restyle a true frame vs. predict
  an unseen one), and the pre-registration must say why that difference is load-bearing rather
  than assume it.
- **The embodiment defect.** Both priors probed here render a generic manipulator where the G1's
  arm should be (`docs/hf_jobs.md`, `runs/backbone_eval/video/embodiment_grid.png`). A pixel
  metric cannot see it, and it is the defect that would poison a VLA fastest. Any gate here needs
  an embodiment check that is not a distance.
- **PR-07 §7 freeze:** *"Frozen until T-39 reports: T-32, any Cosmos3-Super generation, any
  Cosmos3-Edge work."* Transfer2.5 is not named in that freeze, but the reason behind it applies
  — until T-39 says whether any method clears the bar on this corpus, "the data is wrong" and
  "the method is wrong" are not separable, and generating more data is a bet on the first.

## Acceptance Criteria

- [x] `docs/preregistration/PR-08-*.md` exists: the hypothesis, the arms, the gate, the verdict
      table, and what each verdict forbids — same shape as PR-06/PR-07, written before any clip
      is generated. → `PR-08-photoreal-augmentation.md`, 2026-08-06, rule `T40_RULE_V1`.
- [x] The gate is **borrowed, not coined** — reuse an existing margin (as PR-07 borrows
      `MATERIAL_FLOOR_PP` from `I8_RULE_V3`) so the choice of threshold cannot become the finding.
      → the same constant from the same place; ladder is WAM-Bench's L1/L2.
- [x] The gate includes an **embodiment check** that is not a pixel distance, because
      `video_fidelity` provably cannot see the generic-manipulator defect. Concretely: the G1's own
      pixels must survive the restyle — verify against the robot segmentation mask, and composite
      the real robot back over the generated frame if the generator repaints it.
      → G0c, and stronger than asked: the composite is **unconditional**, so the defect cannot
      enter and no IoU threshold has to be coined. IoU is kept as a generator diagnostic.
- [x] A **geometry-invariance check**: object and plate positions in the restyled clip agree with
      the source within tolerance, so the carried-over actions still describe what is on screen.
      → G0b. `GEOM_TOL` is *derived* — the median per-step object-centroid displacement in the
      source — so it is not a coined number either. The value is measured under §8 item 4.
- [x] The control arm is named and is not trivial: the same policy trained on the un-restyled
      episodes. "Augmented beats nothing" is not a result. → arm A, plus arm **C
      `real+identity`**, which PR-08 adds: same generator, same frame count, zero added diversity.
      Without it a gain from B cannot be attributed to diversity rather than to the generator.
- [x] The **eval set is visually shifted** from the training domain. Evaluated in the same domain,
      augmentation can only be neutral or harmful — the experiment would be unable to produce a
      signal, and a null would say nothing. → disjoint `TRAIN_STYLES` / `EVAL_STYLES`, partition
      committed before generation. **PR-08 §7 records what this still cannot buy:** the shifted
      eval is *generated*, so a P licenses recording a real shifted eval and nothing else.
- [x] The pre-registration states which **`--tune-visual` recipe** it runs under (`vla-training`
      §7, Recipe A vs. B). Varied pixels into a frozen vision tower is a strictly weaker version
      of the experiment; unstated, the result is not readable. → **Recipe B**, lr 5e-5,
      `submit_chain.sh visual`, fixed in PR-08 §8 item 1.
- [x] `screen_corpus.py` (T-34) is specified to run **on the generated corpus** and its output is
      a release gate, not a report. → G0a, **with the AC's own defect corrected**: a restyle
      changes no action, so M1/M2/M3 are identical to the source by construction and the gate as
      written would pass vacuously. Restated as an *identity* check — it must reproduce the source
      within the script's `EXPECT_TOL`, and a deviation means the pipeline corrupted the labels.
- [x] The consumer contract with `emai/vla-training` is written down: LeRobot v3.0, 28-dim
      arms+hands, right hand index-before-middle, and where the action labels come from.
      → `docs/contracts/vla-training-consumer.md`, written 2026-08-15. **This AC and PR-08 §8
      item 2 both name the wrong corpus**, and the contract's §0 says so rather than quietly
      complying: those three fields describe `unitreerobotics/G1_Dex3_*` (v3.0, 28-dim), while
      PR-08 §3 restyles `nvidia/GR00T-N1.7-AppleToPlate`, which is **v2.1 and 43-dim in seven
      groups** — measured from its own `meta/info.json`. The contract covers both corpora and §7
      states what a superseding `T40_RULE_V2` would have to say; `T40_RULE_V1` is registered and
      is not edited (`docs/handoff.md` §3). Its one open item, §3.2, is corpus B's intra-hand
      order, which does not block PR-08.
- [x] **One path chosen and justified** — Isaac (exact conditioning, sim frames) or real teleop
      (real frames, estimated conditioning). Under the real path, the depth/segmentation
      estimator's error is measured *before* generation, against Humanoid Everyday's ground-truth
      depth on the same camera, and that error enters the geometry gate as a budget.
      → **real teleop**, PR-08 §3, because the Isaac path changes the trajectories and so is not an
      augmentation of AppleToPlate at all. **One substitution, deliberate:** the estimator is
      calibrated against **Isaac's** rendered ground truth, not HE's measured depth, because HE is
      unlicensed data and must not sit on the critical path. PR-08 §4 records the cost — synthetic
      renders make `EST_DRIFT_P95` a *lower* bound, so a G0b margin that only clears under it is
      not a pass. HE becomes the confirmatory measurement if the licence resolves.
- [ ] **A measured throughput number** from one timed episode on an H200 at target resolution,
      and a GPU-h ceiling derived from it — enforced in the sbatch, as `MAX_RESTARTS` enforces
      T-39's. ~172 k frames per variant is the multiplier. → **open, needs the cluster.** The
      enforcement half exists (97, above); the measurement half is `TIMING=1`, and its two inputs
      landed 2026-08-18/19: `98_build_transfer25_env.sbatch` builds the framework at a pinned
      commit and `99_stage_transfer25_weights.sbatch` stages the weights at a pinned revision.
      **The licence gate that blocked 99 was accepted by the account holder on 2026-08-19** — the
      probe against the pinned revision returns 200 where it returned 403 the day before.
      A third input was missing and nothing had noticed: 97 reads `${SOURCE}/manifest.json` and its
      own header said *"Nothing writes it today"*, so the timing run had no corpus to time.
      `100_fetch_pr08_source.sbatch` + `scripts/build_pr08_source.py` write it, and the build was
      exercised end to end on the workstation copy 2026-08-19 — **402 episodes, 171 625 frames,
      640×480 av1, every episode's video length equal to its declared label length**, and the
      resulting manifest drives `restyle_transfer25.py` to a clean run under `--backend null`.
      A fourth blocker was found the same way and fixed: 97 passed the driver no `--control`, which
      the driver requires with no default, so **both** the timing and generation invocations would
      have died at argparse with an H200 already allocated.

      **2026-08-20 — the chain ran on the cluster, and three more defects surfaced by running it.**
      `100` and `98` are **done**; `99` is queued (est. 02:08) with the timing run held behind it on
      `--dependency=afterok`. Each defect below cost a queue cycle and would have cost a GPU:
      - **`98` could not build at all.** The checkout ships `.python-version` = 3.13, set by upstream
        ce13887 *"Add Python 3.13 support (cu130+torch29 via **v1.5.0** index)"*, but the pinned sha
        still resolves `cu128` from the **v1.2.0** index, whose flash-attn is cp310/cp312 only. uv
        obeys the file, so the default path was the broken one and the error named a wheel, not a
        Python. Pinned `TRANSFER_PYTHON=3.10`. `98` then completed in 2:46 — torch 2.7.0+cu128,
        CUDA major matched, `cosmos_transfer_env.sh` written.
      - **`100` ran anonymous and was rate-limited**, dying on `429 Too Many Requests` at 80 % of
        813 files, six minutes in. The corpus is public, so no token was thought to be needed; the
        token was never about permission, only about the anonymous rate limit. Now required up front.
      - **`100` then failed with the download SUCCEEDED.** hf 1.28 decorates stdout (`✓ Downloaded`,
        then `  path: …`), so `tail -1` captured the label and the directory guard correctly
        rejected it. `--quiet` is documented as "one ID per line". 99 is unaffected — it passes
        `--local-dir` and never reads stdout.
      **The corpus now exists on the cluster and was verified against the snapshot's own metadata:**
      `${PROJ}/data/pr08-apple-640x480`, 888 MB, **402 episodes / 171 625 frames** matching
      `info.json` exactly, 640×480 av1, 402 materialised files (no symlinks), **0 conditioning maps
      claimed** — the honest state until items 4/5 land, and the manifest Transfer2.5 will estimate
      depth/seg from.

      **2026-08-22 — the timing run reported a number, and the number is not a measurement.**
      `99` completed (job 189135) and the timing run went through as job **189142**: `COMPLETED`,
      exit `0:0`, 2 min on one H200, and it wrote `THROUGHPUT.json` claiming **0.2 s/frame → 9.56
      GPU-h per variant**. It generated **nothing**. Its own log says `[1/1] … error ValidationError`
      and `=== done: 0 success, 1 error`; the 118 s it timed is the time to import torch, build the
      checkpointer and die. **That figure is the input PR-08 §8 item 3 derives the whole-partition
      ceiling from, so the ceiling would have been derived from a crash.** Three separate defects
      had to line up, and all three are now closed:
      - **`model` is required and looks optional.** `SetupArguments` declares it with a default
        (`config.py:305`), but `validate_model` is a `mode="before"` validator (`:263-270`) — it
        sees the raw dict and raises `"model is required"` for the key pydantic would have filled
        in a moment later. The driver never passed it. It now does, derived from the hint keys.
      - **The driver's resumability was load-bearing in the wrong place.** It returns 0 on a dead
        unit *by design*, so a partial chunk can be re-driven — correct for generation, fatal for
        timing, where the sbatch measures a wall clock around it and calls the result throughput.
        `--require-success` now exists and the TIMING path alone passes it, so a dead unit exits 1
        and `THROUGHPUT.json` is never written.
      - **On the committed control set upstream ignores our checkpoint entirely.**
        `CONTROL=depth:0.5,seg:0.5` gives `Control2WorldInference` two hint keys, which takes the
        multi-branch branch (`inference.py:64-72`). That branch never reads `args.checkpoint_path`,
        loads **all four** of `CONTROL_KEYS = ["edge","vis","depth","seg"]` ("even if some have
        control weight = 0"), and resolves each through `download_checkpoint()` at a revision
        **hardcoded per checkpoint** in `checkpoints_transfer2.py` — four *different* commits, none
        of them `ce8440327…`. So `99`'s 29 GB staged tree is unused on this path, `general/blur`
        (which `99` deliberately skipped, and which is what `ModelVariant.VIS` is backed by) is
        required, and the hub cache holds none of it. A cold run downloads ~22 GB **inside the
        measured window**. `99b_stage_transfer25_multibranch.sbatch` now warms exactly those four
        through upstream's own resolver and records them in `MULTIBRANCH_STAGED.json`; the driver
        additionally writes `checkpoints_loaded` and `checkpoint_path_honoured` into each unit's
        record, because the sbatch's `generator … (FROZEN)` log line describes bytes that this path
        does not load.
      **The throughput AC stays OPEN and is now openly worse than it looked:** the one number we
      had has been withdrawn. `99b` → `97 TIMING=1` is resubmitted; the AC closes on the number that
      run produces, not on this one.
- [x] Generation is **chunked and resumable** under 4 h `MaxWall` / `MaxJobsPU=4`.
      → `cluster/discoverer/97_transfer25_restyle.sbatch` (2026-08-16). `CHUNK_INDEX`/`CHUNK_TOTAL`
      are required with no default, `--requeue` plus `--signal=B:USR1@300` hands the run five
      minutes to file what is finished before the wall, and a per-chunk stamp refuses to resume
      into a directory another invocation owns. `PARTITION_CEILING_GPU_H` / `CEILING_GPU_H` are
      likewise required with no default, so the ceiling cannot be a number nobody measured — which
      is why this AC is closed while the one below it is not.
- [ ] Nothing is generated, trained or submitted under this task until the above is reviewed.
      → **standing.** Also gated on T-39 reporting, per PR-08 §1.

## Notes

**Correction, 2026-08-06 — the Isaac conditioning signals do not exist yet.** Two statements above
claimed `isaac_transport.py` already emits depth and segmentation. Checked against the code on
`main`: `isaac_binding.py` makes **exactly one** `AnnotatorRegistry.get_annotator` call and it is
`"rgb"`. `distance_to_camera`, `distance_to_image_plane`, `semantic_segmentation` and
`instance_segmentation` appear **nowhere** in `src/` or `tests/`. The drift is traceable —
`docs/backbone-eval.md:221` says Isaac **"can emit"** those, which is true and is a statement about
Replicator's capability; this task hardened "can" into "already", which is a statement about our
code and is false. Recorded rather than silently patched, because the same hardening would have
made the PR-08 calibration rig look free when it is a code change with tests. It is now PR-08 §8
item 5, and it blocks the estimator error budget entirely.

Rules are versioned, never edited in place — if the gate written here turns out wrong, the fix is
a `V2` alongside it, not an edit. See `docs/handoff.md` §3.

**Scope boundary.** This task uses Transfer2.5 as a **frozen** tool. Fine-tuning a generator on G1
data is [[T-041]] — a better fix for the embodiment defect, a much larger project, and frozen by
PR-07 §7 by name.

%% mc-links: [[T-39]] [[T-34]] [[T-36]] [[T-37]] [[T-041]] %%
