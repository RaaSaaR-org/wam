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
      **Measured the same day (job 189401, 2:31, 22.11 GB, free CPU QoS).** Diffing the four
      against `99`'s `STAGED.json` by sha256: `edge`, `depth` and `seg` are **byte-identical** to
      what `99` staged at `ce8440327…`; only `vis`/`general/blur` (`82ede02539a4b141`) is new. The
      four revisions are therefore commit labels over unchanged content, and the `FROZEN` claim
      holds as a statement about bytes — the only checkable form of it, per api §9. It also settles
      `99`'s `"variant_selected": null`: upstream selects the 5.53 GB member of each pair.
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

**2026-08-22 — PR-08 §4's calibration rig exists, and the one thing still missing is a segmenter.**
§4 step 0 (the annotators) landed 2026-08-21 as `5ef3535`, which unblocked step 1 for the first
time. `scripts/measure_est_drift.py` is steps 1–4, split into two subcommands on purpose:
- `capture` drives an `IsaacBinding` and writes rgb + ground-truth depth + segmentation ids (with
  their `idToLabels`) per frame. It runs today against `FakeIsaacBinding` and stamps
  `is_simulated_binding` into the header, so a laptop capture can never be read as ground truth.
- `measure` is a pure function of that directory: estimated-vs-true object-centroid distance per
  frame, p95 → `EST_DRIFT_P95`, plus absolute depth error over the object with the `inf` background
  excluded and counted. It reuses `centroid_of_mask` and `distribution` from `measure_geom_tol.py`
  rather than reimplementing them, because §6 SUBTRACTS the two numbers and two implementations of
  "centroid" is two different quantities.
It refuses in the same shapes `measure_geom_tol` does — no estimator, ungated estimator, partial
run, coverage below the floor, mixed geometry — and it additionally cross-checks the committed
`GEOM_TOL` artifact's pixel grid, because nothing else in the pipeline checks that the subtraction
is arithmetic. `is_lower_bound: true` is unconditional and not a flag (§4's stated weakness).
22 tests in `tests/test_measure_est_drift.py`, none needing Isaac or a GPU.

**The blocker is now precisely one thing, and it is shared.** Both halves of §8 item 4 need an
object segmenter, and §4 step 2 requires it to be *the same one*. Neither has it. What was found on
the cluster while looking: **`sam2` 1.1.0 is already installed** in the Transfer2.5 venv, with its
hiera configs, and Cosmos drives it at
`cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py` naming
`SAM2_MODEL_CHECKPOINT = "facebook/sam2-hiera-large"`. **No checkpoint is staged** — nothing under
`checkpoints/` or the hub cache matches. So wiring the *generator's own* segmenter is a ~900 MB
fetch plus an adapter, and it closes `GEOM_TOL` and `EST_DRIFT_P95` together. Which segmenter to
commit to is a registered choice (it sets both gate numbers), so it is recorded here rather than
taken.

**2026-08-22, later — the `model` fix worked, and the timing run reached a FOURTH checkpoint nobody
had counted.** Job **189402** ran (after `scontrol update TimeLimit=01:30:00`; at 4 h it sat
`PENDING (Priority)` behind 50 jobs on a 2-node partition at FairShare 0.048). It got past
`SetupArguments` — the defect that killed 189142 — and three layers deeper, into
`text2world_model_rectified_flow.__init__` instantiating the tokenizer. There it died:

    CalledProcessError: uvx hf download nvidia/Cosmos-Predict2.5-2B --revision f176dc95… tokenizer.pth

**`Cosmos-Predict2.5-2B` is a DIFFERENT REPO from Transfer2.5.** `99b` staged the four control
branches Transfer2.5's multi-branch loader resolves; the base model's tokenizer is not among them
and is not in that repo at all. Two things were then measured rather than assumed, using the
workstation's own token and never echoing it:
- `nvidia/Cosmos-Predict2.5-2B` is **`gated: auto`**, and with a VALID token returns **403 on
  `tokenizer.pth` at every revision tried** — the framework's pinned `f176dc95…`, the current
  `85f8ae7b…`, and `main`.
- The control, `nvidia/Cosmos-Transfer2.5-2B` (accepted by the account holder 2026-08-19), returns
  **206** for the same token and method. So the token works and the probe is sound: **this is a
  licence gate, not a revision problem, not a rate limit.**
- `97_transfer25_restyle.sbatch` also carries **no HF token at all** — zero references, where `99`
  and `99b` both read one. Anonymous against a gated repo is a 401 even once the licence is
  accepted, so both halves have to be fixed.

**BLOCKED ON A HUMAN ACTION:** the account holder must accept the licence at
`https://huggingface.co/nvidia/Cosmos-Predict2.5-2B`, exactly as on 2026-08-19 for Transfer2.5.
No session can do this and no retry will clear it.

**What this run DID prove, and it is the point of the previous commit:** `--require-success` fired.
`=== done: 0 success, 1 error` → the driver exited 1 → the sbatch refused → **`THROUGHPUT.json` was
not written.** Under the old code this run would have completed `0:0` and recorded a second
fabricated GPU-h figure. The gate now fails closed.

**2026-08-22, evening — the licence opened, the segmenter staged, and GEOM_TOL turned out to be
blocked on a codec rather than on GPU-hours.**

- **The `Cosmos-Predict2.5-2B` licence is accepted.** Re-probed with the same token and method that
  produced the 403s above: **206 at `f176dc95…`, `85f8ae7b…` and `main`**. The cluster's token is
  byte-identical to the workstation's (compared by sha256, never echoed), so the acceptance applies
  there too. The timing job **189584** is queued at `--time=01:30:00`.
- **The estimator weights are staged and verified.** Job **189583** exited `0:0` after the two
  defects in `102` were fixed. `PR08_ESTIMATORS_STAGED.json`: 5.0 GB, `facebook/sam2-hiera-large`,
  `IDEA-Research/grounding-dino-base` and `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf`,
  every id **and revision** agreeing with `scripts/estimators/apple_sam2.py`. The six `unverified`
  rows are the honest ones: they compare against `pr08_geom_tol.json` and `pr08_est_drift.json`,
  which do not exist yet, so **102 must be re-run once they do**.
- **`103_measure_geom_tol.sbatch` exists, with a PILOT mode that sizes the full run before buying
  it.** Two passes at two frame budgets over the same episodes, so the slope separates per-frame
  cost from the ~3.7 GB of weights loaded first; denominators read out of each pass's own
  `n_frames`. Both passes exit 3 by construction and the pilot path treats that as success.
- **THE PILOT'S FIRST RUN FOUND THE CODEC, NOT THE COST.** Job **189585** died in **7 seconds**:

      [av1 @ ...] Missing Sequence Header.
      FATAL: .../episode_000000.mp4 opened but decoded no frames — the container parses and the
             codec does not.

  **This is the second time this project has hit this exact failure.** Job 186357 captioned 372
  clips and wrote zero captions on a sibling corpus — nothing crashed, `ffprobe` was happy, and
  vLLM's OpenCV backend read every container header and failed every `grab()`.
  `scripts/verify_clip_decode.py` was built *because of that job* and states the lesson in one line:
  **a corpus is only readable by the decoder that will actually read it.** The PR-08 corpus is a
  **copy** of the AV1 source (`manifest.json`: `codecs: ["av1"]`, `materialized: "copy"`), so the
  trap was still armed and nothing ran the gate that exists for it.
- **Job 189586 measured which decoders work here**, in the generator's own venv rather than in the
  abstract (`runs/pr08-geom-tol/CLIP_DECODE_PROBE.json`): **cv2 4.11.0 cannot** (FFMPEG YES, avcodec
  59.37.100, no AV1); **pyav 16.0.1 can, via `libdav1d`**; imageio and torchvision can; **decord
  fails**; and there is **no `ffmpeg` on PATH** to transcode with. Upstream itself is mixed — 13
  files reach for decord, 11 for imageio, and `auxiliary/sam2/sam2_utils.py` uses `VideoCapture`,
  which means **upstream's own SAM2 helper would fail on this corpus too**.
- **The fix is a reader, not a re-encode, and that is a decision about evidence.** Transfer2.5 is
  handed the *path* and decodes it itself (`restyle_transfer25.py:290`), so transcoding would put a
  lossy re-encode between the tolerance and the pixels the generator sees — at a scale of a fraction
  of a pixel, which is the unit `GEOM_TOL` is denominated in. `measure_geom_tol.py` now carries a
  decoder seam: `--decoder auto` **probes** each decoder against a clip of the corpus and takes the
  first that actually returns a frame, every decoder yields **BGR** (`sam2_mask_via` flips once to
  the adapter's `segment(rgb)`; a reader returning RGB would ground "apple" in a world where red is
  blue), and the choice plus the whole probe trail is written into the artifact beside `mask_method`.
  It is deliberately **not** in the `est_drift` cross-check — that side's frames come from a
  renderer, so a decoder field there would be a field about nothing.

**Still blocked on a human:** Isaac Sim is **not on the cluster** and `EST_DRIFT_P95` has nowhere
else to run. Installing it on the workstation's RTX 5090 (32 GB, driver 595.84) is ~10–20 GB and is
the project owner's call. PR-08 §1's generation gate is untouched and remains the owner's call.

**2026-08-22, afternoon — the throughput measurement ran at last, and the corpus codec killed it.**

Job **189584** (`TIMING=1 STYLE_SET=train`) was submitted at `--mem=192G --time=01:30:00` and Slurm
planned it for 2026-08-23T11:45. Cutting it to `--mem=96G --time=01:00:00` via
`scontrol update` halved its billing weight (49 -> 25) and it **backfilled 22 hours early**, starting
2026-08-22T13:51. Then it FAILED in 3:36 with exit 1, inside Cosmos-Transfer2.5's own SAM 2 helper:

    Processing video: .../episode_000000.mp4 ... Number of frames: 590
    Converting video.. Done extracting frames. 0 frames extracted
    [1/1] episode_000000__train-01-oak-tungsten__r00 error RuntimeError: no images found in <tmpdir>

preceded by hundreds of `[av1] Missing Sequence Header` and `Your platform doesn't suppport hardware
accelerated AV1 decoding`. **The job refused to write `THROUGHPUT.json`**, on its own stated grounds
that "a wall clock around a failed unit is not a throughput number, and PR-08 §8 item 3 derives the
budget from this file". That refusal is the file behaving correctly; §8 item 3 is still OPEN.

**This is the THIRD job lost to one defect,** and the first two were ours rather than upstream's:
186357 (372 clips captioned, 0 captions), 189585 (the GEOM_TOL pilot, dead at 7 s), now 189584.
Job 189586's `CLIP_DECODE_PROBE.json` pins it exactly, in the generation venv:

  * `cv2` 4.11.0 — `FFMPEG: YES`, `avcodec 59.37.100` — **cannot decode this AV1 corpus**;
  * `av` (PyAV) 16.0.1 via **libdav1d** — decodes it; `imageio` 2.37.0 — decodes it;
  * `ffmpeg_av1_decoders: null` — **there is no ffmpeg CLI on PATH in that environment at all**;
  * upstream reads clips with `cv2.VideoCapture`, in
    `cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_utils.py`.

So this one cannot be fixed the way ours was. Our own `measure_geom_tol.py` got a decoder seam and
picks PyAV; **upstream's reader is cv2 and cv2 bundles its own libav**, so no PATH change reaches it.

**The transcode objection is withdrawn, by measurement.** This file previously argued against
re-encoding the corpus because "a lossy transcode would sit between the tolerance and the pixels the
generator sees", at the fraction-of-a-pixel scale GEOM_TOL is denominated in. Measured 2026-08-22
with PyAV 18.0.0 on `episode_000000`:

  * AV1 -> `rgb24` -> `VideoFrame.from_ndarray(..., 'rgb24')` -> `yuv420p` @ `crf=0` is **NOT**
    bit-exact: max abs channel delta **7-10/255** over 24 frames. That is the RGB<->YUV chroma
    round-trip, not the codec. This is the naive route and it must not be used.
  * The decoded `av.VideoFrame` passed **straight to the encoder in its native `yuv420p` planes**,
    with `pix_fmt` carried from the input and `crf=0`, is **BIT-EXACT — max abs channel delta 0** on
    every one of the 24 frames compared after decoding both sides back to `rgb24`.

There is no loss, so there is nothing to sit between the tolerance and the pixels, and the objection
does not survive its own measurement. Cost: 3 076 473 B -> 9 336 753 B for one clip (`preset
veryfast`; `slow` gives 9 121 825 B, 2.4 % better), i.e. the corpus's 929 424 549 B of video becomes
roughly **2.8 GB**. `/valhalla` has 4.5 PB free, so the size is not a consideration.

Caveat recorded for whoever implements it: setting `frame.pts = None` before encoding works for a
short prefix and then dies on a full clip with `av.error.ArgumentError ... returned 22` out of
`mux`. Timestamps have to be handled, not nulled.

**Resubmission path once a transcoded tree exists.** `97_transfer25_restyle.sbatch` takes
`SOURCE=${SOURCE:-${PROJ}/data/pr08-apple-640x480}`, and its only path refusal (line 659) is against
the converted 120x160 corpus, so a sibling tree is accepted as long as its manifest declares
640x480.

**2026-08-22, evening — the codec blocker is closed by measurement, and the GEOM_TOL pilot moved
the shape of the remaining work.**

*The transcode.* `scripts/transcode_corpus_lossless.py` (new) re-encoded the whole corpus AV1 ->
H.264 and **proved** rather than assumed the result: every clip is compared back against its source
in **both** `yuv420p` and `rgb24`, and a clip that cannot be proven loses the tree its
`manifest.json`, which is precisely what makes `97_transfer25_restyle.sbatch` reject it. Result:

    OK: 402/402 clips PROVEN bit-exact (171625 frames, stride 1)
        929,424,549 B in -> 6,534,455,777 B out (7.03x), 693.2 s wall, 0 pts repaired

The mux error I could not get past was **not** the transcode being impossible. `frame.pts = None`
makes libx264 invent timestamps the mp4 muxer rejects with errno 22; the fix is to copy the
**container's** `time_base` (`1/15360` here — deriving it from the 30 fps frame rate silently
rescales every timestamp) and carry the source pts through. `pts_repaired` is **0** on this corpus:
the timestamps were never missing, only discarded. `threads=1` is pinned so an output digest does
not depend on `--jobs`; x264 frame threading changes the bitstream without changing the pixels.

Two things I had told the task were wrong and are corrected here: the tree is **6.5 GB, not 2.8 GB**
(the smaller figures came from the truncated files the crashing runs left behind), and on a full
590-frame clip `medium` beats `veryfast` by **4.9 %**, not 2.4 %, with `medium` <= `slow`.

*The clearance, which is the part that counts.* Bit-exactness is a claim about this workstation's
encoder. The question the three lost jobs asked is whether the **generation venv's** cv2 4.11.0 /
avcodec 59.37.100 can read the result, and only that venv's own interpreter can answer it. Job
**189605** ran `verify_clip_decode.py` over the tree on `/valhalla`:

    cv2 4.11.0 from .../third_party/cosmos-transfer2.5/.venv/...
    checking 402 clips, 3 frame(s) each
    OK: 402/402 clips decoded (640x480 @ 30.0 fps)   -> exit 0

`decord`, which threw `DECORDError` on the AV1 original and which upstream reaches for in 13 files,
now returns all 590 frames. All four readers upstream actually uses work.

**`104_probe_clip_decode.sbatch` gained `LIMIT`/`FRAMES` overrides and a codec-aware verdict,
because a FAILURE is categorical and a PASS is not.** Its default `--limit 4` was correct for
diagnosing AV1 — four clips failing identically settles 402 — and wrong for clearing a candidate
tree, where one unreadable file is job 186357 again (372 clips captioned, zero captions, nothing
crashed). The artifact now records `verify_clip_decode_limit` and
`verify_clip_decode_covers_whole_corpus`, so a 4-clip run cannot later be cited as clearing the
corpus, and the verdict branch distinguishes a diagnosis run from a clearance run instead of
printing "this probe is wrong about its own premise" at a corpus that was never AV1.

*What the GEOM_TOL pilot found.* Job **189588** (`PILOT=1`) measured the cost instead of guessing
it: **0.0833 s/frame** plus **116 s** fixed load (interpreter, torch, ~3.7 GB of GroundingDINO +
SAM 2 weights), so the full 402-episode measurement is **4.005 GPU-h** and wants `05:30:00` against
a **4 h `MaxWall`**. It ended `single_job_feasible: false`. Both passes carry `gate_qualified: false`
and exit 3 — `--limit` disqualifies a sample from being the gate by construction, so **nothing in
`GEOM_TOL_PILOT.json` is GEOM_TOL**. The answer is the shard/merge path now in
`measure_geom_tol.py` (`--shard I`, `--merge`) and `103_measure_geom_tol.sbatch` (8-way array at
`%4`, merge as a separate **no-GPU** CPU job under `2cpu-single-host`). GEOM_TOL is a **median**, so
a merge that averages shard medians returns a plausible wrong number rather than an error — that
path is under adversarial verification before it is given four GPU-hours.

*Arm C's identity prompt is measured, and it fails.* Calibration attempt 2 passed all five floors,
including the abstention probes at **10/10** where attempt 1 scored 2/10 — confirming attempt 1's
own diagnosis, since the only change was replacing manufactured occlusion with real corpus
occlusion. The census bounding that number is recorded rather than glossed: of 154 447 frames
searched, exactly **24** are genuinely undecidable and all 24 are one ~1.4 s occlusion in
`episode_000094`, so ten correct answers are ten correlated successes. Run blind over the 40-episode
sheet, the committed prompt scored:

    verdicts    match 0 | mismatch 40 | unsure 0   (coverage 1.000)
    axes        table:40, background:29

All 40 say the same two things — the cloth is **dark grey, not black**, and a pale strip of the
surface behind it runs along the top edge. `apple`, `lighting`, `plate` and `other` are **0**. So
`T40-TODO-01-identity-prompt-provenance` was right on the facts: a machine caption of
`episode_000135_clip000` applied to all 402 describes none of the 40 sampled. The correction is a
minimal one to two clauses, re-measured blind with fresh judges; `configs/transfer25/styles.toml`
is **not** edited here and flipping that todo's status stays a person's step per the judge doc §7.6.

*§8 status after today.* Item 5 is **closed** — `distance_to_camera` and `semantic_segmentation`
are wired in `src/wam/robot/isaac_binding.py` with tests. Item 3 is running as job **189609**
(`TIMING=1`, `--time=02:00:00 --mem=98304`, against the transcoded tree). Item 4 is half-blocked:
GEOM_TOL is now schedulable, `EST_DRIFT_P95` still has nowhere to run. **Item 7 is untouched and
not an engineering question.**

**Still blocked on a human, unchanged:** Isaac Sim for `EST_DRIFT_P95`; PR-08 §1's generation gate;
and whether the corrected identity prompt is pasted into the committed style source.


**2026-08-22, late — somebody finally looked at a mask, and blocker 1 was right.**

`GATE_QUALIFICATION_BLOCKERS[0]` has said since 2026-08-21 that *"coverage 1.0 says a box was
returned on every frame, not that it was the APPLE's box"*, and that this adapter's failure mode is
*"a plausible mask on the wrong object (the plate, the hand, the whole tabletop) which produces a
centroid, a displacement and a p95 that all look like measurements."* `scripts/audit_apple_masks.py`
(new) drives the adapter **unmodified** over a stratified sample and writes overlays a person can
judge. A local CPU run over 12 of 402 episodes, **169 frames, 92 overlays opened one at a time**:

* **156 of 169 masks are correct** — tight on the fruit, score 0.73–0.87, IoU 0.91–0.98 against an
  independent colour heuristic. Including every case the blocker names: the grasp with the Dex3
  fingers closing, fruit held over the plate with the plate untinted, fruit resting on the plate,
  and the apple clipped by the frame edge.
* **9 frames are a confident, well-formed mask of THE PLATE.** All in `episode_000094`, all frames
  where the hand has hidden the fruit almost entirely (54–448 px of apple visible): ~31 000 px,
  plate overlap **0.985–0.992**, IoU **0.00** with the colour heuristic.

Three things make that worse than "a few bad frames", and all three are measurements rather than
arguments:

1. **Not one of them is a no-detection.** `n_frames_without_detection = 0` and
   `n_frames_with_empty_mask = 0` over the whole sample — **coverage 1.000 while nine frames measure
   crockery.** That is blocker 1's claim, confirmed on this corpus.
2. **The plate is stationary**, so consecutive plate masks give adjacent steps of 0.006–0.45 px.
   Those land in GEOM_TOL's displacement pool as near-zero and pull a **median** down, tightening
   G0b; the onset frame contributes a single 245 px step, which distorts a p95 instead.
3. **THE RETRY NEVER FIRED — not once in 169 frames.** `n_frames_retry_fired = 0`. So blocker 2's
   hazard is **real and mislocated**: the wrong-object masks were bought by the primary
   `BOX_THRESHOLD = 0.15`, not by the `(0.10, 0.10)` retry the blocker singles out. `ep000094 f130`
   scored **0.155** — one thousandth over the threshold. At the adapter's previous 0.35 it would
   have returned an honest all-False mask.

The score distribution is cleanly bimodal (p25 **0.758** for correct masks; **0.155–0.264** for all
13 flagged frames), so a gate near 0.30 would separate them — **and adding one is forbidden**, because
it would make this a different segmenter from the generator's, which is exactly what §4 step 2 rules
out. The finding is not "tune the threshold"; it is that the generator's own operating point
mis-segments occluded frames while reporting full coverage.

**Without the census this would have been invisible** — all nine sit in the one episode
`build_identity_calibration probe-scan` forced into the sample.

**`GATE_QUALIFIED` stays `False`, and the audit script cannot change that**: a test asserts the file
contains no assignment to `GATE_QUALIFIED` / `GATE_QUALIFICATION_BLOCKERS` /
`GATE_QUALIFICATION_DISCHARGED`, and the artifact copies all three blockers in verbatim with
`blockers_discharged_by_this_run: []`. The question in front of the owner is no longer *"has anyone
looked"* — it is what to do about ~5 % of hard frames returning the plate at coverage 1.0.

**Three things remain open and are not rounded off.** The **human half** of blockers 1 and 2 is not
discharged: every observer here is a model checking masks produced by a pipeline a model wired up,
which is a correlated observer, and `human_review.looked_at` is `false`. **Blocker 3 is untouched** —
but note that the plate-lock is a run of **~35 consecutive frames**, which is precisely the
propagation-style failure blocker 3 says a per-frame estimator cannot see, appearing *in* a per-frame
estimator. And blocker 2's *"from a full pass"* has **nowhere to land**: neither
`measure_geom_tol.py` nor `measure_est_drift.py` reads `apple_sam2.stats()` into its artifact, so a
full GEOM_TOL run records none of the retry or no-detection counts. That is a small additive change
to both harnesses and it is recorded in the audit artifact as `full_pass_gap`.

`cluster/discoverer/105_audit_apple_masks.sbatch` (new) runs the same audit on the cluster for a
larger, independently-produced sample and the contact sheets a person needs. Sized from the
measurement rather than from 103's inherited request: `--mem=24G --cpus-per-task=8` is
**7.3 billing-min/min against 103's 49.9**, and still 5x job 189588's measured 4.75 GiB peak.


**2026-08-22, night — the cluster audit reproduced the plate masks, and the sheets were looked at.**

Job **189637** ran `audit_apple_masks.py` on the cluster over a different sample from the local CPU
run: **382 frames, 24 episodes, 36 contact sheets, 16 flagged** (`plate_overlap` 12, `low_score` 8,
`centroid_jump` 3, `mask_area_below_band` 3). It cost 6 minutes at `--mem=24G --cpus-per-task=8`,
**7.3 billing-min/min against 103's 49.9**. Summary numbers:

    n_frames_retry_fired  0   n_frames_retry_recovered  0
    no-detection          0
    detection score       min 0.167  p05 0.362  median 0.829  max 0.917  (n=382)
    mask area px          min 471  median 6185  p95 8194  max 31151

**The retry fired zero times on a sample four times larger than the local one.** That closes the
question blocker 2 opened: the wrong-object masks are bought by the **primary** `BOX_THRESHOLD`
= 0.15, not by the `(0.10, 0.10)` retry the blocker names. The blocker's hazard is real and its
attribution is wrong, and both runs agree.

**The contact sheets were pulled back and read** (`runs/pr08-mask-audit/sheets/`, 22 MB, gitignored;
`sync.sh --pull` takes only `.json/.jsonl/.md/.txt`, so the overlays need `scp`). Two things the
JSON did not show:

* **The segmenter oscillates within one episode.** `episode_000094`: f00149 plate → f00150 apple
  (471 px) → f00151 apple (670 px) → f00152 plate, and the panel records the consequence — a
  **245.9 px step**. So the corruption is not only near-zero displacements from a stationary plate
  pulling a median down; it is **large spurious jumps at every switch**, which land in a p95.
* **f00152 shows the apple plainly visible** — 877 warm px, outlined by the colour heuristic — with
  the mask on the plate at score 0.257. **This was never only an occlusion problem.**

The healthy frames are not marginal: the grasp sheets are tight masks on the fruit, IoU 0.97–0.98,
scores 0.77–0.85, plate overlap 0.00, adjacent steps 0.1–2.3 px. This is a good estimator with one
specific failure, which is why the fix is a validity filter and not a different segmenter.

**WHO LOOKED, AND WHAT THAT IS WORTH.** These observations are a model's. `human_review.looked_at`
in `MASK_AUDIT.json` is **`false` and stays false** — a model checking masks produced by a pipeline
a model wired up is a correlated observer, capable of reproducing the same misreading on both sides.
The sheets exist precisely so a person can spend five minutes and settle it; `flagged-00.png` and
`flagged-01.png` are the two to read first.

**Blocker 3 gained evidence against its own premise.** It argues that propagation's characteristic
failure — drifting off the object and staying off for a run of frames — is invisible to a per-frame
estimator "that recovers on the next frame". The plate lock is a run of **~35 consecutive frames**
in a **per-frame** estimator. The failure mode it attributes to propagation occurs without it.


**2026-08-22, late night — the robot masker fails on a third of frames, and the GEOM_TOL array
died proving the pilot's cost model was measured against a segmenter that no longer exists.**

*The finding that stops the next 9.5 GPU-h.* `106_measure_robot_mask_area.sbatch` PILOT (job
**189707**, 9 min 35 s) measured the thing the G0c bound would have been derived from:

    empty_mask.fraction   0.352   (565 frames)      seconds_per_frame  0.1981
    full_run_seconds      34099  ->  9.47 GPU-h     single_job_feasible  false

**`robot_composite.check_mask` refuses a clip on an empty robot mask — "zero is zero", no
threshold.** At 35.2 % essentially every clip refuses at generation time, so **G0c cannot run and
PR-08 cannot generate**, whatever value the area bound takes. A local strided pass over all 402
episodes independently measured **37 %**. The full measurement was **not** submitted: a distribution
produced by a detector that fails on a third of its frames is a distribution of the failure, and a
bound read off it would be meaningless. The open question, being diagnosed now, is whether **the
robot is genuinely out of shot** in those frames — in which case an empty mask is the CORRECT answer
and `check_mask`'s rule is what is wrong, since G0c composites real robot pixels and there are none
to composite — or whether **detection is failing**, in which case the rule is right and the detector
is not. Opposite fixes; the artifact cannot tell them apart. Whether the empty frames cluster in
time within an episode or spread uniformly probably decides it.

*The array, and a mistake worth not repeating.* Job **189658** (`--array=0-3%4 --time=02:00:00`)
produced **no artifact at all** at a cost of roughly six GPU-hours: shard 0 TIMEOUT at 2:00:21,
shard 2 cancelled once it was arithmetically certain to overrun, shards 1 and 3 into the wall
behind them. Shard 0 is exactly **42 673 frames** and did not finish in 7 200 s, so

    p > (7200 - 116) / 42673 = 0.1660 s/frame   against the pilot's 0.0833  ->  at least 1.99x

**The cause is not contention and not the corpus: the GEOM_TOL pilot measured a different
segmenter.** `GEOM_TOL_PILOT.json` was produced at `box_threshold` 0.35 with no retry branch, and
the adapter has since moved to Cosmos-Transfer2.5's own operating point (0.15, single (0.10, 0.10)
retry) because §4 step 2 requires our detection point to **be** the generator's. A lower threshold
means more boxes survive and every surviving box costs a SAM 2 pass. **Two places in this repository
said so before the fact** — `GATE_QUALIFICATION_BLOCKERS[0]`'s *"evidence about a configuration THIS
FILE HAS SINCE REPLACED"*, and 103's own header, *"a 2x error in that extrapolation is entirely
plausible"* — and the array was sized off the stale number anyway. 103's header now carries the
measured correction, why `N = 4` was chosen (`MaxSubmitJobsPU=8` rejected `--array=0-7` outright),
and the conclusion that **N = 4 is not an option**: at 0.18 s/frame its heaviest shard is 148 min
against a 4 h wall. `%j` also does not disambiguate array tasks — all four wrote to one log file —
now `%A_%a` in both array-capable jobs.

*Independent cross-check on the cost.* The robot pilot's **0.1981 s/frame** and the GEOM_TOL
overrun's **> 0.166** are the same detector stack at the same operating point, measured by two
unrelated routes on two different days. **So both remaining §8 measurements are ~9.5 and ~8.6
GPU-h — about 18 GPU-hours before a single clip is generated**, against the ~4 GPU-h the original
pilot implied for GEOM_TOL alone.


**2026-08-22, later — CORRECTION. The robot masker does not fail on a third of frames. The robot is
not there. And the real defect is the opposite one: it masks the APPLE.**

The entry above, and commit `7fdd466`'s subject line, said "the robot masker fails on a third of
frames". **That is wrong and is withdrawn.** `scripts/diagnose_robot_mask_empty.py` measured it.

*The robot is genuinely absent, and the pattern is unmistakable.* A non-learned reference predicate
— dark against the frame's own modal luminance, near-neutral, and different from the episode's
per-pixel temporal median, **with no GroundingDINO anywhere in it** — over 101 episodes / 21 639
frames: **56.5 % robot-present, 36.2 % robot-absent**. Absent fraction **by corpus decile**:

    d0 65.6 %  d1 23.1 %  d2 1.3 %  d3 0.23 %  d4 0.19 %
    d5  2.1 %  d6 20.5 %  d7 64.3 %  d8 90.3 %  d9 94.8 %

That is approach and retreat, not a detector misfiring at random. 98/101 episodes end inside an
absent run, 76/101 begin inside one, and on the pilot's three episodes deciles 2–6 contain **zero**
absent frames and every episode has exactly two runs — one from frame 0, one to the last frame.

*Detection is not failing.* Contingency on the pilot's episodes: **present_empty = 0, present_nonempty
= 917.** Zero failures on 917 robot-present frames. The corpus sample showed 19 apparent failures;
**all 19 were rendered and looked at, and none contains a robot** — the reference predicate had
scored the arm's off-frame shadow, the plate rim or the top-band edge. **Measured detector failures
on robot-present frames: 0.** Every empty mask is `no_boxes_above_threshold`; SAM 2 never segmented
a box to nothing.

*A decision this vindicates.* **91.7 % of empty frames still score above 0.10**, so upstream's
`(0.10, 0.10)` retry — which `robot_composite` deliberately does **not** run — would have
manufactured a box on nearly all of them. That refusal was argued from principle and is now measured.

*So `check_mask`'s "zero is zero" is refusing CORRECT masks.* Its error text offers a dichotomy —
"if the robot is genuinely absent the SOURCE corpus is not what PR-08 §3 describes" — but §3 says
only "the 402 real demos" and never claims per-frame robot visibility. The corpus is what §3
describes. With ~152 robot-absent frames per episode, **every clip refuses and G0c as written cannot
produce one composited clip.**

**THE FINDING NOBODY WENT LOOKING FOR, AND IT IS THE SERIOUS ONE.** On robot-absent frames the
masker returns a **non-empty** mask 14 % of the time on the pilot's episodes and **41 % corpus-wide**
— and the mask is **the apple** (~6–7 k px), the plate (~40 k), or the whole tablecloth (> 0.9 of
frame). `sheets_corpus/nonempty_but_reference_absent.png` shows it plainly: twelve panels, no robot
in any, and the green "robot mask" sitting on the apple in ten of them at detector scores 0.15–0.28.

**Under G0c that composites the SOURCE APPLE back over the generated frame.** The object the entire
task is about silently stops being restyled — and **no gate can see it**: G0b measures geometry and
a pixel-identical apple has moved zero, G0a measures labels, and robot-mask IoU is recorded as
"diagnostic, never a gate" by §6 itself. Arms B, C and D would quietly become arm A **for the
apple**. Scores 0.150–0.414 on these overlap true robot detections 0.165–0.623, so **no score
threshold separates them**, and an apple-sized mask is 0.02 of frame — far below any plausible area
bound. This is exactly `GATE_QUALIFICATION_BLOCKERS[0]`'s "a plausible mask on the wrong object",
now measured on the **robot** prompt rather than the apple one.

*And the bound's own premise is broken.* 106's header says a bound must sit above the observed
maximum. Re-measuring the pilot's own 1 603 frames with the same pins and tree reproduced min,
median, p95 and p99 to three decimals — but **max came back 0.9722 against the pilot's 0.3618**,
because 11 frames ground the whole tablecloth and **10 of the 11 contain no robot**. On this corpus
**the observed maximum IS the defect the bound exists to catch**, so "above the maximum" cannot
define it.

**Three decisions, none of them a script's.** (1) What an empty mask means on a robot-free frame —
there is a threshold-free option that stays inside G0c's own logic: run the committed masker on the
**generated** frame too, and accept when both are empty, refuse when the generated frame grounds a
robot the source never had. (2) Whether Cosmos-Transfer2.5 can hallucinate a manipulator into a
robot-free frame at all — the unmeasured premise under `check_mask`'s error text, and cheap to
settle. (3) The false positives, which the area bound cannot see and which only a person may address,
since the prompt and thresholds are committed constants.


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

**2026-08-23 — COSMOS-TRANSFER2.5 GENERATED FRAMES, AND FOUR THINGS BROKE ON THE WAY.**

Job **189926** (V8 hallucination probe, `runs/pr08-hallucination-probe-QUARANTINE-NOT-A-CORPUS-v2/`)
is the **first successful generation in this project**. Four units, ~55 s per 96-frame chunk on one
H200, `PROBE.json` + contact sheets + quarantined clips all landed.

**Verdict `H`, 3 of 4 units** — `candidate_invention` 16 / 19 / 21 / 0 of 96 frames. `human_review.
looked_at` is **false**, and V8 said in advance the count is an UPPER BOUND. Reading the sheets, the
flagged masks sit at ~7 020–7 400 px and hold that value across a dozen consecutive frames — stable
area over time is an object, and it is apple-scale at 640×480 with the outline on the fruit. So most
of `H` is very likely the masker defect below rather than an invented manipulator. **That is a
reading, not the finding.** Unexplained and not investigated: a persistent cyan vertical smear in
`episode_000001`'s generated frames that is absent from source.

**The green apple is NOT a defect.** `train-01-oak-tungsten`'s committed prompt says verbatim *"A
bright green Granny Smith apple"*; the `apple` style axis explicitly permits colour and variety.
Nearly filed as an unrequested identity change — checked the prompt actually sent first.

**(1) The robot masker grounded the apple, and G0c composited it back — a silent PASS.** `fe72031`,
`PR-08-V9` (unsigned). 2 845 detections re-scored: apple IoU ∈ [0.9364, 0.9847], everything else
≤ 0.5131. `ROBOT_MASK_OBJECT_MAX_IOU = 0.70` sits in a gap where every cut partitions identically.
Also corrects d739a87's "41 % corpus-wide" to "40.8 % of robot-absent frames in the 40-episode
sample" — `stratified_plan` samples equal quotas per bucket and its own docstring says the rates are
not corpus rates. **The corpus rate of the composite defect is not known.**

**(2) NEW AND NOT PREVIOUSLY NAMED — the `plate.` pass refuses 100 % of frames, on the SOURCE
corpus.** `run_g0_gates`'s own documented invocation is `WAM_PR08_OBJECT_PROMPT="plate." …
--source-clips`, which reaches `measure_geom_tol:945` → `module.segment(rgb)`. Executed on 20 source
frames: `n_frames_mask_refused = 20`, validity IoUs 0.0000–0.0036, non-empty on **0 of 20**.
`apple_sam2.object_color_reference` is the APPLE predicate unconditionally and never reads
`OBJECT_TEXT_PROMPT`. **§6's plate half cannot be measured at all**, and it presents as
`coverage: 0.0` — a fact about the corpus — rather than as a filter undefined for the label. This is
not restyle-dependent. **OPEN, not fixed.**

**(3) V6's reference is a warm-colour predicate and five committed styles are not warm** (green
Granny, pale-green waxy, Golden Delicious, russet, Pink Lady). V6 §5.3 anticipated this and says it
fails closed — but on `train-01-oak-tungsten` the reference fires on **34 632 px of warm oak table**
and ~1 000 px of green apple, so `n_frames_mask_refused_no_reference` stays 0. That is the exact
counter §5.3 relies on to separate "the segmenter is wrong here" from "the reference does not fit
here", and it reports the wrong one. **G0c is unaffected** — `composite_clip` masks `src` and never
`gen`, pinned by a test. **OPEN, not fixed.** The proposed direction is to stop asking a colour
predicate to identify the object on frames whose colours the pre-registration deliberately varies:
the colour-free reference is the paired source frame's own mask (G0b already carries
`--restyled-source-map`), and if the source mask is not a valid reference for the generated frame,
**that is the G0b finding**.

**(4) No GEOM_TOL shard could ever land.** `4102e2e`. `GATE_QUALIFIED = False` makes every CORRECT
shard exit 3; the sbatch treated any non-zero RC as fatal and the resume validator refused a
`gate_qualified: false` artifact, so every resubmission re-measured the whole partition and an array
of any width could never converge. Sharding was bought for resumability and handed it straight back.
Re-run submitted as **189935**, N=16 in four waves, `--time=01:30:00`, `--mem=32G` (192G → 32G saves
365 billing-h, >5 GPU-h of runway, on a measurement costing nine). N chosen on **break-even p =
0.3728 s/frame, 2.25× the measured floor** — 189658 died because the truth was 2.2× the plan.

**On the gate-qualification blockers.** Blockers 1 and 2 discharge on a human looking at overlaid
masks spanning the corpus **plus** the detection-score distribution and retry counts from a full
pass. The sheets exist (`runs/pr08-mask-audit/sheets/`): the grasp is clean at IoU 0.97–0.98, and
the predicted failure is visible — on occluded and apple-out-of-frame frames the segmenter returns
the **entire plate**, ~31 000 px, score 0.16–0.29, IoU **0.00**, 98 % plate overlap. The two
populations do not touch, and V6's committed 0.10 splits them with margin. **189935 is producing the
second half of that evidence, so it is not premature spend against a disqualified flag.** Blocker 3
(per-frame re-detection vs upstream's propagation) is untouched and would change the number, not
just the flag. **The discharge is the owner's signature, not a session's.**

**2026-08-23 (later) — THE PARTITION IS 27 % OF THE ALLOCATION, AND STAGE 1 IS 8.8 %.**
`PR-08-V11-staged-partition.md`, **UNSIGNED**.

Job 189926 gives the first and only measurement of what a Transfer2.5 frame costs: 96 frames in
~111 s of H200 time, **1.16 s/frame** (`Average time per chunk: 55.47`, two chunks). The committed
partition is 25 style-instances × 402 episodes = **10 050 clips / 4 290 625 frames ≈ 1 380 GPU-h**
against a 5 000 GPU-h allocation. **That is ~27 % of everything, spent on a generator whose
geometric fidelity has never once been measured** — G0a, G0b and G0c have not returned a verdict
between them. (Billing is not the constraint: ~4 % at `--mem=32G`. GPU-hours are.)

**THE 1.16 s/frame FIGURE IS NOT §8 ITEM 3's MEASUREMENT AND MAY NOT BE A BUDGET LINE.** It sized a
decision; it is one diagnostic clip and likely optimistic (96-frame clips against episodes averaging
~427 frames). A `TIMING=1` run still owes the ceiling.

**The coupling that decides the cost, and it is not obvious.** The 25 instances are 10 `train_styles`
+ 5 `eval_styles` + identity `repeats = 10`. `T40_RULE_V2` requires arm C to MATCH arm B's frame
count, and §5 says arm C "is not optional: … without C a gain from B is uninterpretable". So **one
extra train style costs 804 clips, not 402.** Arm A is the 402 real episodes and generates nothing;
arm D reuses B's clips.

**V11 stages it.** Stage 1 = `train_styles[0:4]` + 4 matched identity repeats = 8 of 25 instances,
3 216 clips, **~442 GPU-h (8.8 %)**. The four are a PREFIX of the committed order, not a selection —
any "four most distinct" rule is a choice made with knowledge the document must not have. Eval is
**DEFERRED, NOT CUT**: no arm trains on it, it is consumed once at evaluation time, and all five
styles stay committed in `styles.toml` unmodified. `styles.toml` is untouched and its hash does not
change.

**Recorded dependency, not a reason to reorder:** three of the four prefix apples are non-warm
(green Granny, Golden Delicious, russet), which puts stage 1 inside the defect `T40_RULE_V10`
addresses — `object_color_reference` fires on ~34 632 px of oak table under `train-01-oak-tungsten`.
V10 must land before stage 1's output is measured. Reordering the prefix to make a measurement
easier would be choosing the experiment to fit the instrument.

**Stage 2 is decided in advance** (V11 §3): B beats A *and* C → scale; B beats C but not A, or the
reverse → stop and read; **B does not beat C → the gain is the generator's fingerprint and scaling k
does not fix it.** The accepted threat: a null at k=4 is weaker than at k=10 and could be a false
negative. Accepted because the sbatch is chunked and resumable, so k=4 → 10 costs the difference and
not a restart. k=2 was rejected — two domains is not domain randomisation.

**CODE CHANGE OWED, NOT YET MADE.** `97_transfer25_restyle.sbatch` prices the whole partition and
hard-codes the ceiling shares 10/25, 5/25, 10/25. Under V11 stage 1 draws at most **8/25** of
`PARTITION_CEILING_GPU_H`, which itself stays the whole-partition number so a sequence of small
stages cannot exceed what nobody measured.

**V11 licenses nothing.** V1 §1 binds, every §6 gate is undischarged, §8 items 3 and 4 are open, and
the determination block is empty — signing is the owner's.


### 2026-08-23 — four defects closed in the gate path, and one V11 claim withdrawn

**`T40_RULE_V10` (unsigned) — the mask-validity reference was answering for objects it has never
seen.** Two measured defects, both of which made a broken instrument read as a fact about the
corpus. (1) The `plate.` pass refused **100 % of SOURCE frames**: 20 segment calls, 20 refusals, all
IoUs 0.0000, while the detector scored 0.7524–0.7773 — because a correct plate mask contains no warm
fruit and `object_color_reference` is the apple predicate unconditionally. §6's plate half could not
be measured at all, and presented as `coverage: 0.0`. (2) On a restyle the reference does not go
quiet, **it moves to the table**: on `train-01-oak-tungsten` it returns 40.5–56.4 % of the frame, all
warm oak, so `MASK_REFUSED_NO_REFERENCE_FRAMES` — the counter V6 §5.3 relies on to separate "the
segmenter is wrong here" from "the reference does not fit here" — stays 0 and reports the wrong one.
A *correct* mask was refused: same box in both styles, kept at IoU 0.46–0.49 on `train-02` and
refused at 0.025–0.030 on `train-01`. Fix refuses instead of pretending to decide: an unknown label
refuses the RUN, and an over-scale reference refuses the FRAME into its own counter. The 0.10 bound
is read off a gap (source max 3.00 % over 154 447 frames vs 20.9–29.1 % mis-firing), not coined.

**G0b compared two sides without recording how either became pixels.** The docstring claimed the two
sides are verified to be one instrument; the **decoder** was neither recorded nor compared, while
G0b's arithmetic is source frame *i* minus restyled frame *i* and `resolve_decoder` probes each side
independently — one command line could resolve two. Now recorded per side and compared. **The rows
do not refuse**: the two sides are not the same codec by construction (av1 source, job 189585 is the
record of cv2 decoding ZERO frames of it), so a hard refusal would be a gate the real corpus cannot
satisfy — the defect this file was already repaired for once. It costs gate qualification instead.

**EST_DRIFT_P95 over 240 frames is a p95 over 20 configurations.** V5 §5's floor (≥ 20 states AND
≥ 200 frames) can be satisfied by near-duplicate frames: measured, the spread *inside* a state is
0.05–0.28 px and *between* states 0.06–39.9 px. Both numbers are now recorded side by side —
additive, read-only, never a gate. Also: `mujoco.FatalError` is not a `RuntimeError`, so the
headless GL crash escaped `main()`'s handler and gave a traceback where every other failure gives a
named refusal.

**106's documented N=3 partition cannot finish.** Its pilot ran (job 189707): p = 0.1981 s/frame
measured, **2.59× slower** than the scaled workstation estimate it was sized on, +1.92 % for V9's
filter, and the GEOM_TOL cross-check is higher still at 0.295 (used as the planning rate). N=3's
heaviest shard is **310 min against a 4 h MaxWall**. Header now gives N=16 in four waves.
Separately: *"every clip refuses"* is a round number — **128 of 129, not 129**; `episode_000243`
has a non-empty robot mask on all 417 frames. And **68 % of the refusals are on frames the V9 filter
did not empty**, so removing the filter would not make G0c pass.

**V11 §2.3 withdrawn and corrected (§2.4).** Read by execution rather than by comment: **nothing
hard-codes 10/25, 5/25, 10/25** — the shares are operator-supplied and the ceiling machinery already
admits any split, so no change is owed there. What is owed is larger: **there is no way to generate a
prefix of a style set at all** (`chosen = styles[style_set]`, no STAGE, no limit), so stage 1 cannot
be expressed today without editing `styles.toml`, which V11 §0 forbids. And the stage selector must
carry arm C's frame-match with it: the guard compares whole-`styles.toml` instances independent of
`chosen`, so **a stage of 4 train styles against 10 identity repeats would pass every check in the
file** — V11 §0's second bullet currently has no enforcement behind it.

**Cluster.** GEOM_TOL wave 1 (189935) landed shards 0–3 with sha256 sidecars; wave 2 (189971,
shards 4–7) and shards 8–9 (189984) are queued — 8/8 submit slots full, `MaxSubmitJobsPU` counts
every array task. Waves 3–4 go in as slots free, then MERGE on the free CPU QoS.

**Nothing here discharges a gate.** V9, V10 and V11 are all unsigned, `GATE_QUALIFIED` is still
`False`, §8 items 3 and 4 are open, and §6 still says that if `GEOM_TOL − EST_DRIFT_P95 ≤ 0` then
generation does not start.

%% mc-links: [[T-39]] [[T-34]] [[T-36]] [[T-37]] [[T-041]] %%
