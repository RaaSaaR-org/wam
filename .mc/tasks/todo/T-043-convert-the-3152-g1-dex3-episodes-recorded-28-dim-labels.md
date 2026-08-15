---
id: T-043
aliases:
- T-043
- T-43
title: "Convert the 3 152 G1+Dex3 episodes — recorded 28-dim labels, route 1"
slug: convert-the-3152-g1-dex3-episodes-recorded-28-dim-labels
status: todo
priority: 2
owner: ''
projects: []
customers: []
tags:
- data
- m1
- prereg
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-15
updated: 2026-08-15
status_note: "Opened 2026-08-15 out of T-042's step 0, which went looking for unlabelled footage
  and found the opposite: 3 152 real G1 episodes with recorded 28-dim actions. **Measured the same
  day, before any code was written, and it changes the task's premise: these sets are ARM-FIRST
  (`[0:14]` arm, `[14:28]` hand) — the opposite of the hand-first block order five documents
  attribute to them.** That claim was real but belongs to a *different* corpus
  (`USC-PSI-Lab/Humanoid-Everyday-G1`, LeRobot v2.1); it was transplanted onto these
  (`unitreerobotics/G1_Dex3_*`, LeRobot v3.0) and would have produced exactly the silent
  arm/hand transposition it was written to warn about. Two independent lines of evidence, both
  local and free, in §1 below. **The blocker is not code: the action parquets were never
  downloaded** — `92_fetch_g1_corpus.sbatch` fetched `meta/` + `videos/` and skipped `data/`.
  647 MB, Apache-2.0, and an ask before it is fetched."
---

# Convert the 3 152 G1+Dex3 episodes — recorded 28-dim labels, route 1

## Description

Convert the 13 `unitreerobotics/G1_Dex3_*` sets into the WAM episode format, the way
`convert_lerobot_g1.py` already converts `nvidia/GR00T-N1.7-AppleToPlate`. **Route 1 — recorded
teleop labels.** Nothing is inferred, nothing is generated; this is conversion work on labels that
already exist, which is why it is not gated behind T-39's positive control (that gate binds
*training runs*, and this produces a dataset).

Every recorded number in this project comes from **402 episodes of one task**. This is **3 152
episodes of thirteen tasks**, and nobody has to collect them.

## 1. The block order — measured, and the standing claim is wrong for this corpus

**This is the first section because getting it wrong is the failure mode the task inherits.** Arm
and hand transposed produces a model that trains, converges, and is silently useless — every
number finite, plausible and wrong.

**`docs/action-labels.md` §3b, `subprojects/data-factory/README.md`,
`subprojects/edge-wam/README.md`, `E-02` and T-042 all state the trap as `action[0:14]` ↔ hand,
`action[14:28]` ↔ arm — hand-first. For these sets that is backwards.**

The hand-first measurement is genuine and is not being overturned: T-041 established it on
**`USC-PSI-Lab/Humanoid-Everyday-G1`** — a *different* corpus, LeRobot **v2.1**, whose state ships
as separate `arm_joints` (14) / `leg_joints` (15) / `hand_joints` (14) fields. The
`unitreerobotics/G1_Dex3_*` sets are LeRobot **v3.0** with a flat 28-dim `observation.state` and no
leg field. One corpus's layout was carried across to the other's conversion task. Both facts are
true of their own corpus; only one of them is true here.

**Evidence line 1 — mechanical joint limits, from `meta/stats.json` of all 13 sets, no download.**
A finger opens from a hard zero and closes to a fixed mechanical limit, so its range is *one-sided*
and its bound is a round number. An arm joint is bidirectional and bounded by nothing round.

| | dims `[0:14]` | dims `[14:28]` |
|---|---|---|
| one-sided dims (a range endpoint at 0 ± 2e-3) | **0, in all 13 sets** | **4–10, in all 13 sets** |
| clean round limit at the far end | none | **100.0°/100.1°** (7 sets) or **120.0°** (5 sets) |

Zero out of fourteen versus four-to-ten out of fourteen, unanimous across 13 independently recorded
sets, is not a marginal call. The two limit families track the two `robot_type` strings below.

**Evidence line 2 — an explicit modality spec from a pipeline that produced a working model.**
`vla-training/groot/modality_g1_dex3.json`, the GR00T `modality.json` for this same 28-dim
`Unitree_G1_Dex3` convention:

```json
"state":  {"arms": {"start": 0, "end": 14}, "hands": {"start": 14, "end": 28}},
"action": {"arms": {"start": 0, "end": 14}, "hands": {"start": 14, "end": 28}}
```

**Arm-first, `[0:14]` arm and `[14:28]` hand.** Written before this task existed, by a pipeline
whose output (`GR00T-N1.7-ApplePnP-V1`) trains and runs.

**Still unverified and not to be assumed:** left/right order *within* each 14-dim block, and the
intra-hand joint order. T-041's lesson on Humanoid-Everyday applies verbatim — a source that is
wrong about the block order earns no trust about the finger order. Both need the same treatment
against the parquet: correlate, do not read the card. **Three mutually inconsistent intra-hand
orderings are already on record** (the corpus card's thumb-first symmetric, Arena's index-first,
and NVIDIA's asymmetric left `[4,5,6,2,3,0,1]` / right `[4,5,6,0,1,2,3]`), so this is a live
question with a known-wrong default, not a formality.

## 2. What the corpus actually is — measured 2026-08-15 from local metadata

All 13 sets are already on the workstation at `~/wam-t041/raw/` (meta + videos, no `data/`).

| | |
|---|---|
| scale | **3 152 episodes · 2 587 515 frames · 23.96 h @ 30 fps** — vs AppleToPlate's 402 ep / 1.6 h |
| format | LeRobot **v3.0** — episodes concatenated, boundaries in `meta/episodes/*/*.parquet` |
| state / action | `float32[28]` both, flat, **no waist and no leg columns** |
| video | **`cam_left_high` at `[480, 640, 3]`, present in all 13 sets**, 30 fps, AV1 |
| licence | Apache-2.0, all 13 |
| two variants | 6 sets `robot_type: Unitree_G1`, 4 cameras, hand limit **120°** · 7 sets `robot_type: Unitree_G1_Dex3`, 2 cameras, hand limit **100°** |

**The resolution is a free hit and should be stated plainly:** `cam_left_high` is **640×480**,
which is exactly the GR00T N1.7 `ego_view` contract fixed in
`subprojects/data-factory/README.md` on 2026-08-15. Unlike `datasets/gr00t-apple-full/` (120×160,
a 4× shortfall), this corpus is natively at the consumer's input resolution and needs no
re-derivation to feed it.

**The two variants are not obviously one corpus.** Two `robot_type` strings and two different hand
mechanical limits (100° vs 120°) is a hand-hardware difference, not a formatting one. Whether they
may be pooled into one dataset is a question this task answers, not one it assumes — pooling two
hands under one normalization is the gripper-mapping mistake of T-31 in a new costume.

## 3. The structural gap the converter has to resolve

`configs/robot/g1.yaml` canonical space is **15 joints + 2 grippers**:
`waist_yaw + left_arm(7) + right_arm(7)`. This corpus is **14 arm + 14 hand and carries no waist
column at all**.

`convert_lerobot_g1.py` builds `q` as `[waist_yaw = state[12], left_arm = state[15:22],
right_arm = state[22:29]]` from AppleToPlate's 43-dim vector. **There is no `state[12]` equivalent
here.** So the converter must either emit a canonical `q` with `waist_yaw` pinned (and say so, in
the manifest, as a recorded absence rather than a measured zero), or the canonical space has to
grow an "upper-body, no waist" variant. **This is a schema decision touching `src/wam/interfaces/`,
which CLAUDE.md marks change-with-care, and it should be settled before code is written.**

Second gap: **the gripper.** AppleToPlate's 7 Dex3-class joints collapse to a 2-dim grasp synergy
through a mapping that took T-31 to get right, and whose `legacy` version is retained only to
reproduce a pinned dataset. That entire apparatus — `--gripper-mapping`, the dataset-level affine,
the refusal-on-clip — has to be re-derived here, against a **different hand with a different
range**, on **two hand variants**. The affine is not transferable and must not be reused.

Third: **v3.0 vs v2.1.** `convert_lerobot_g1.py` is a v2.1 reader. `prepare_cosmos_corpus.py`
already reads both layouts, including the trap that **cameras roll over to new files
independently** (at episode 50 of BlockStacking `cam_left_high` is in `file-001` while the other
cameras are still in `file-000`), so the file must be resolved per (episode, camera). That reader
is the thing to reuse; do not write a second one.

Fourth: **AV1.** Every source is AV1. Whatever decodes here must be checked with the interpreter
that will actually read the frames — `ffprobe` called the AV1 corpus valid throughout while the
consumer got zero pixels and exited 0 (T-041, job 186357).

## 4. The blocker, and it is not code

**The action parquets were never downloaded.** `cluster/discoverer/92_fetch_g1_corpus.sbatch`
passes `--include 'meta/*' --include 'videos/**'`, and `workstation/10_fetch_corpus.sh` narrows
further. The labels this whole task is about are **415 files, 647 MB**, in the same Apache-2.0
repos we already hold 26 GB of video from.

**Nothing is fetched without asking.** Until then §1's intra-block questions cannot be answered and
the converter cannot be tested against real data.

## Acceptance criteria

- [ ] **Block order verified against the parquet, not against §1's inference.** §1 rests on joint
      limits and a third-party modality file; both point the same way, and neither is the data.
      Correlate each block against the recorded state as T-041 did, and record the numbers.
- [ ] **Left/right and intra-hand order established the same way**, against the three conflicting
      orderings on record, with the result written down as measurement.
- [ ] **The waist decision made and recorded** — pinned canonical `waist_yaw` with a manifest note,
      or a schema variant. Not left implicit in code.
- [ ] **The gripper mapping re-derived for this hand**, not inherited. `scripts/audit_gripper.py`
      PASSES the converted output, and the affine is recorded in the manifest's `normalization`
      provenance slot. The two hand variants (100°/120°) are handled explicitly — pooled with
      evidence, or kept separate.
- [ ] **Mutant tests.** The converter's tests kill a deliberately transposed arm/hand block, an
      off-by-one action relabel (`q[t+1]-q[t]` vs `q[t]-q[t-1]`), and a swapped left/right — the
      three defects that are invisible in aggregate metrics. Same standard as T-39's twelve.
- [ ] **Every episode traceable** — dataset snapshot manifest with per-source repo id, revision
      sha, licence and camera key, per AC-04.
- [ ] **`screen_corpus.py` (T-34) run on the result** before it enters any training corpus.

## Notes

**Why this is not a free win**, restated because the size of the number invites it: 28-dim Dex3 is
not 43-dim AppleToPlate, the canonical target is 15 joints + 2 grippers, the hand differs across
the two variants, and the conversion touches the schema. This is a converter plus its mutant tests
plus a schema decision — real work, on data that is genuinely already labelled.

**What it bears on.** PR-07 §1's standing explanation for fourteen negatives is "402 success-only
episodes of one task is not enough", and PR-07 §6's **N** verdict points at *the kind* of data.
Thirteen further G1 tasks with recorded actions bear on both. **That does not make this a
substitute for T-39** — a bigger corpus under a recipe never shown to work is the same experiment
at greater cost. T-39 says whether the scorer can see a signal that is definitely there; this says
what we would feed it afterwards.

%% mc-links: [[T-042]] [[T-041]] [[T-040]] [[T-39]] [[T-34]] [[T-32]] [[E-02]] %%
