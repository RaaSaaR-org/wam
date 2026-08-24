# PR-08 — V12 §3.5's two preconditions, checked and answered

**Investigated 2026-08-25 by a nine-agent workflow whose findings were adversarially verified, then
spot-checked by hand. Read-only: no file under `src/`, `scripts/` or `configs/` was modified.**

`T40_RULE_V12` §3.5 recommends its own §3.2 *"conditional on the §2 measurement and on the camera
geometry existing. In that order: run the (a)/(b) split first, confirm the extrinsics are
recoverable second, adopt third."*

**This document executes both checks and reports what they returned.** It adopts nothing, signs
nothing, and changes no gate. `T40_RULE_V1` §1 binds in full.

---

## 0. Headline

| precondition | result |
|---|---|
| **camera geometry recoverable?** | **NO.** §3.2 is unavailable on this corpus |
| **§2's (a)/(b) split measured?** | **taken, but not of record, and by an instrument that no longer matches the masker G0c runs** |
| consequence for V12 | §3.2 falls away. The choice is §3.3 or a new rule, and it is the owner's |
| consequence for `max_frame_fraction` | **a reproducibility problem that job 106 cannot see.** §3 — read this even if nothing else |
| generation licensed | **no** |
| training licensed | **no** |

---

## 1. The camera geometry does not exist in any form a gate may cite

V12 §3.2 predicted this would be the deciding check: *"It requires camera intrinsics and extrinsics
for AppleToPlate that are **not committed anywhere in this repository as this is written**… That is
the first thing a signer should check, because it decides whether this rule is implementable at
all."*

The prediction was correct, and the situation is slightly worse than the prediction.

### 1.1 The FK half is fine — this is not what blocks

`src/wam/robot/kinematics.py` runs a genuine `mj_kinematics` forward pass over the 15 canonical
joints and resolves every joint **by name** through `mj_name2id` (`kinematics.py:97-107`), raising
rather than assuming a positional mapping. Two independent probes reproduced FK on real corpus
state digit-for-digit. The 43-dim layout is attested twice over. **Nothing about the state column or
the kinematics is missing.**

### 1.2 The camera half is empty

- **`configs/calibration/` contains exactly one file**, `example.yaml`, 1 629 bytes — verified by
  `ls`. It labels itself *"Values below are PLACEHOLDERS"* and *"charuco+hand-eye (example
  placeholder, not a real calibration)"*, and its cameras are named `front`/`wrist`, never
  `ego_view`.
- **The corpus carries nothing.** A grep for `intrinsic|extrinsic|camera_matrix|distortion|calibrat|focal|fovy`
  across the raw corpus tree exits 1 — zero hits. `meta/info.json`'s camera entry gives
  `{480, 640, av1, 30 fps}`: pixel geometry, no optics, no pose. `ffprobe` finds no camera metadata.
- **There is no solver.** `src/wam/data/calibration.py:1-6` says so in its first six lines:
  *"Storage + validation ONLY — there is no calibration solver in here."*
- **The repo's own sim cameras are disclaimed in writing.** `docs/sim.md:434`: *"Camera extrinsics
  are invented."* `configs/sim/g1_scene.xml:42-43`: *"Replace fovy with the calibrated intrinsics
  from OD-03 before any sim-to-real transfer."*

### 1.3 The one candidate that looked like a rescue, and why it is not

A fitted `ego_camera` pose *is* committed in an adjacent repository —
`robot-management-system/.../mjcf/g1_dex3/g1_43dof_fixedbase_realism.xml:260-261`,
`pos="0.14 0.06 0.44444" fovy="44"` — with a comment claiming 34 px reprojection RMS. The first
probe reported it as *"verifiable, not asserted"*. **Its verifier refuted that, and the refutation
holds:**

1. **The solver is gone.** `find /home/humanoid -name solve_camera.py` returns nothing. The number
   survives; the derivation does not.
2. **The claimed corroboration is untracked.** `git log --all -- eval/videos/calib_fit.py` is
   **empty** — the file is in no commit on any branch.
3. **And it does not corroborate — it disagrees.** `calib_sol.npy`'s translation has two of three
   components pinned **exactly at the fitter's own ±0.08 m box bound**, i.e. a saturated, degenerate
   solution; mapped into the torso frame it is **opposite in sign** to the adopted pose in x and y,
   and the adopted pose's x offset lies *outside* that fitter's search box. Two different fits, not
   one confirmed fit.
4. **Intrinsics were never solved at all.** `calib_fit.py:27` hard-codes `fx=fy=615.0`; the adopted
   `fovy=44` implies `fx=fy=594.02`. A ~3.5 % disagreement, unreconciled — and since the adopted
   pose's solver is gone, **nobody knows which intrinsics it was fitted under.** No distortion model
   exists at any source.

Importing that into a WAM gate would fail AC-04 traceability on its face.

### 1.4 Two further defeaters, each independently sufficient

- **The witness disagrees with the phenomenon it must adjudicate.** Running the frustum test with
  the adopted pose and `f=594.02` over ten left-arm/hand link origins measures **46.9 %** of
  `episode_000000`'s frames with no arm point in frame, against job 189707's measured **35.2 %**
  empty-mask fraction — an 11.7 pp gap. A further **5.6 %** of frames have their best point within
  34 px of the border, i.e. **undecidable inside the calibration's own stated RMS.**
- **The predicate is a volume question; the instrument answers a point question.** `mj_kinematics`
  fills body **origins**. *"Any arm link lay within the frustum"* is about link extent — a link's
  surface can be in frame while its origin is not. Substituting origins, or padding with an angular
  margin to absorb the 34 px RMS, **is exactly the "approximate frustum test [that] would
  reintroduce the coined tolerance this option's merit rests on avoiding"** (V12 §3.2). The option
  cannot be rescued by the move that would rescue it.

### 1.5 One narrowing in §3.2's favour, recorded because it is true

The missing quantity is **smaller** than "extrinsics in general". The camera is torso-rigid
(`configs/sim/g1_scene.xml:33-40`, welded to `torso_link`, residual 9e-5 m; the real head D435
likewise per `PR-08:107`) and the arms hang off the same body, so legs, waist and the gantry base
**all cancel** in the arm-relative-to-camera transform. §3.2 needs **one fixed 6-DoF `T_torso_cam`
plus intrinsics**, not two transforms.

That does not rescue it. The URDF *nominal* head mount is available (`d435_joint`,
`xyz (0.0576, 0.0175, 0.42987)`, `rpy (0, 0.8308, 0)`), and the one attempt anyone made to fit this
corpus landed **8.2 cm in x away from it**. Either the nominal mount is badly wrong for this gantry
rig, or the fit is bad. **Both readings forbid using either as an unargued witness.**

---

## 2. V12 §2's measurement was taken, and V12 does not know it

**This corrects a factual claim in a rule document, and it is recorded here rather than by editing
V12** — `docs/handoff.md` §3, rules are versioned, never edited in place.

V12 §2 says the (a)/(b) split *"has not been done"*, and §6 repeats it. **That is stale.**
`runs/pr08-robot-mask-empty/DIAGNOSIS.json` (2026-08-22, 15 163 bytes, verified on disk) carries:

> `verdict: "ABSENT. The empty masks are correct answers on frames the robot is not in."`

It predates V12 by two days and V12 cites it nowhere (grep for
`DIAGNOSIS|pr08-robot-mask-empty|diagnose_robot_mask_empty` across the rule documents: zero hits).

**But it is not evidence a gate may be signed against, for four reasons**, all verified by hand:

1. **It is untracked.** `.gitignore:19` is `runs/`. `git check-ignore -v` names that line for both
   `DIAGNOSIS.json` and `G0C_REFUSAL.json`; `git ls-files runs/` returns 20 files and neither is
   among them. **Every number that would discharge this precondition is local-only state in a tree
   this project's own memory records as shared with concurrent peer sessions.**
2. **Its instrument no longer matches the masker G0c runs.** `diagnose_robot_mask_empty.py` last
   changed 2026-08-22; V9's `ROBOT_MASK_OBJECT_MAX_IOU` entered `Sam2RobotMasker.mask` on
   2026-08-23. The diagnose script contains **zero** references to the filter (grep count 0), and
   its own `--verify` guard (`:545-553`) raises *"this module's mask path disagrees with
   `Sam2RobotMasker.mask`. The diagnosis would be about a different masker than the one G0c runs;
   nothing here is usable until that is fixed."* **That guard would now fire.**
3. **Its own file disclaims reproducibility of the governing statistic** — see §3.
4. **The bound does not cover the blind spot.** The 19 inspected `present_empty` frames are
   conditioned on the predicate saying "present" — the region it is good at. The **190 post-V9
   `absent_empty` frames** were adjudicated by the predicate alone, 12 of them rendered. The script
   states at `:145-152` that the predicate *"scores none of"* the white/silver forearm and therefore
   **"UNDERSTATES robot presence"** — which is precisely the `(robot present ∧ mask empty ∧
   predicate absent)` cell, and nothing bounds it.

> **"Zero confirmed instances of (b)" is true. "(b) is refuted" is not.** And no frame anywhere
> carries a human label assigned *before* seeing the masker's answer: every human inspection on
> record was of frames nominated by a disagreement with the masker.

---

## 3. The finding that job 106 cannot see, and that V13 must confront

**Read this even if nothing else in this document is acted on.**

`DIAGNOSIS.json`'s `secondary_finding_oversized_masks.cluster_pilot_disagreement`, verbatim from
disk:

> *"The cluster PILOT (job 189707) reported max area fraction **0.36182** over exactly these three
> episodes at stride 1. This run, same pins, same H.264 tree, same 1603 frames, measures **0.97220**.
> min/median/p95/p99 agree to three decimals (0.0611 vs 0.0654, 0.3059 vs 0.3048, 0.3518 vs 0.3513)
> and the empty count differs by 13 of 1603, so this is the same masker differing on a handful of
> borderline detections across hardware (RTX 5090 here, H200 there) — **but the MAXIMUM is exactly
> the statistic 106's header says a committed bound must sit above, and it is not reproducible.**"*

**Why this matters right now.** Job 106 is measuring the corpus robot-mask area distribution across
16 shards on H200s, in order that a person may set `max_frame_fraction` under `T40_RULE_V13`. V13
§3.1 requires the bound to sit **strictly inside a measured gap**, with `bound_rationale` naming
**both edges**.

That method assumes the tail is a property of the corpus. This says the tail is, at least in part, a
property of **the hardware**: the bulk statistics agree to three decimals while the maximum moves by
a factor of 2.7. A gap identified between "bulk" and "tail" on H200 output is not guaranteed to be
the same gap on other hardware, and a bound placed inside it could fire on frames that were never
anomalous — or fail to fire on ones that are.

**This is not a reason to stop job 106.** The distribution is still the prerequisite for
`load_area_bound` and therefore for the TIMING measurement and §8 item 3. It **is** a reason that:

- V13's `bound_rationale` must record **which hardware the distribution was measured on**, and that
  the maximum is known to be hardware-sensitive on this masker; and
- **V13 §3.3's refusal branch may be the honest outcome for a reason V13 did not anticipate.** V13
  contemplated refusing because the distribution is *unimodal*. It did not contemplate refusing
  because the distribution's tail is *not reproducible*. A separable gap that exists only on one
  vendor's kernels is not the "measured gap" V13 §3.1 means.

**No bound is proposed here and none may be read out of this document.** V13 is unsigned and stays
unsigned; this adds a consideration to it, and the consideration is the owner's to weigh.

---

## 4. What follows for V12, stated without reaching for a workaround

**§3.2 is unavailable.** Not "hard" — unavailable, on the precondition V12 itself nominated as
decisive.

That leaves §3.3 (change nothing, accept that the compositing route does not work on this corpus and
revisit `T40_RULE_V1` §3's route) or a new rule. **§3.3 is the correct standing default today** and
G0c should not be changed. But §3.3's own text conditions its correctness on §2 — *"if §2's
measurement comes back showing (b) dominates, it is the correct answer"* — and §2's measurement is
not of record. **Signing §3.3 today would discard 402 episodes on the same absence of evidence that
blocks §3.2.**

**There is an obvious workaround and this document declines to present it as anything else.** A
"V12.1" whose witness is direct human adjudication of rendered frames instead of FK+frustum would be
*stronger* on V12's own criterion — human inspection coins nothing, whereas a frustum test with an
angular margin coins a number. But it is still **changing a gate's semantics after seeing that
gate's output**, which is the exact shape `handoff.md` §3 names and V12 §2 warns about at length. It
is a rule change, it belongs to the owner, and a session drafting it is not a disinterested party.

---

## 5. What is actually blocking, in one sentence

**A person looking at a few hundred already-renderable frames.** Both mask-audit artifacts carry
`human_review.looked_at: false` and an explicit correlated-observer warning forbidding a model from
standing in; the `absent_empty` cell where case (b) would hide has had 12 of 190 frames adjudicated,
by a predicate that documents its own blind spot. That hour is not compute-gated, and until it
exists on record — tracked, and produced by an instrument that matches the masker G0c runs — both
signing V12 and signing §3.3 are decisions made on evidence that is not of record, in opposite
directions.

---

## 6. Provenance

| | |
|---|---|
| kind | precondition report. **Registers no rule** |
| date | 2026-08-25 |
| executes | `T40_RULE_V12` §3.5's two checks, in the order §3.5 specifies |
| method | nine-agent workflow, three of four probes **refuted** by adversarial verification; decisive facts re-checked by hand and cited |
| corrects | V12 §2 / §6's *"has not been taken"* — by addendum, **not** by editing V12 |
| raises against V13 | §3 — the observed maximum is not hardware-reproducible |
| files modified | **none** outside `docs/` |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
