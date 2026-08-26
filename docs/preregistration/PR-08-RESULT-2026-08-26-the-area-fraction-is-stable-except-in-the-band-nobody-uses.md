# PR-08 §6 — the robot-mask area fraction across two GPUs: stable at both ends, a coin flip in the middle

**Measured 2026-08-26 on the workstation RTX 5090 against the cluster's own pooled measurement,
driving `scripts/robot_composite.Sam2RobotMasker` unmodified at its committed operating point. No
gate, config or blocker was touched, and no bound is written or implied.**

This document exists because a control run turned up a disagreement large enough to threaten
`T40_RULE_V13`'s whole method, and then a second control contained it. **Both halves are recorded,
in that order, because the first half alone would justify abandoning the bound and it should not.**

---

## 1. What was measured

`runs/pr08-robot-mask-area/POOLED.json` is the corpus area distribution, measured on the cluster
(job 106, 16 shards, `git_commit 8b710d0119b6…`). Three bands of it were re-rendered here, frame for
frame, through the same decode function and the same masker:

| band | frames | why this band |
|---|---:|---|
| tail, `f >= 0.680277` | 48 | the sample a person judged for V13 §3.2 |
| middle, `0.36 <= f <= 0.601546` | **44 — the complete population**, not a sample | the frames immediately below the measured gap |
| bulk, `0.001 <= f <= 0.36` | 48 | 98.7 % of the corpus lives here and it was unchecked |

## 2. The result

| band | mismatch > 0.01 | median abs delta | crosses the candidate midpoint 0.640911 |
|---|---:|---:|---:|
| tail | 8 / 48 | 0.000133 | 0 leave the tail |
| **middle** | **37 / 44** | **0.2138** | **5 move up: 4 above the tail edge, 1 into the gap** |
| bulk | 19 / 48 | 0.000879 | **0** |

The five that move, by name:

```
episode_000004:223   0.3615 -> 0.8266
episode_000004:224   0.3631 -> 0.8182
episode_000234:267   0.5812 -> 0.7656
episode_000388:143   0.4700 -> 0.8800
episode_000222:83    0.6015 -> 0.6152   (into the gap the bound sits in)
```

A shift of 0.47 is not numerical noise. **It is a different mask.**

## 3. The two controls, which are what make this readable

### 3.1 The workstation is deterministic

The 44-frame middle band was rendered **twice on this machine**, same command, separate runs:
**0 of 44 frames differ, bit-identically.** So the masker does not wander between runs, and the
disagreement is between machines rather than within one.

### 3.2 The pins are identical

Both runs record the same estimator string, character for character:

```
det=IDEA-Research/grounding-dino-base@12bdfa3120f3e7ec7b434d90674b3396eccf88eb;
seg=facebook/sam2-hiera-large@e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251;
prompt='robot arm. robotic hand. robotic gripper.';box_thr=0.15;text_thr=0.25;retry=none
```

Same weights, same revisions, same prompt, same thresholds. The difference is the GPU and its stack
(here: RTX 5090, torch 2.13.0+cu130).

## 4. The mechanism this is consistent with

`PR-08-RESULT-2026-08-25-detector-noise-floor.md` established that Grounding-DINO's best candidate on
this scene sits at **0.12–0.149** whether or not a robot is present, against a `BOX_THRESHOLD` of
**0.15** — a worst-case margin of **0.0009**. `Sam2RobotMasker` then **unions every box above
threshold**. So one borderline box, worth a thousandth of a score, decides whether a
tablecloth-sized region joins the mask, and the area jumps between two attractors: a small mask at
~0.1–0.2 or a grounded one at ~0.85.

**That predicts exactly the pattern measured here.** Frames deep in either attractor are stable —
both ends of the distribution agree across machines. Frames the cluster placed in the middle are the
frames where the borderline box was marginal, and those are the frames that flip.

**Recorded as consistent with the observation, not as a proven cause.** Proving it would mean
capturing the per-box scores on both machines for these same frames, which has not been done.

## 5. What follows for the bound, stated carefully

**It does not refute the gap.** The bimodality is present in both machines' measurements; the region
between the attractors is nearly empty in both. One of 92 re-rendered non-tail frames landed inside
it, at 0.6152, from a frame recorded at the gap's own lower edge.

**It does not open a path from the bulk into the tail.** 0 of 48 bulk frames cross the candidate
midpoint. The bulk moves DOWN if it moves at all — recorded median 0.109 against a re-rendered
0.0032, with many masks vanishing entirely.

**What it does establish is that per-frame classification near the bound is hardware-dependent for
one narrow band, and that band is 44 frames in a corpus of 171 625 — 0.026 %.**

**And it establishes a condition that belongs in any `bound_rationale`:** `check_mask` refuses a
whole clip when a single frame exceeds the bound (`scripts/robot_composite.py:1391`), so the set of
refused clips is a property of the machine that runs the composite, not only of the corpus. The
bound proposed under V13 is measured on the cluster and would be applied by the cluster. **That is a
condition, not a fix**, and a reader who moves generation to different hardware inherits an
unmeasured question.

## 6. What this does NOT establish

- **Not a bound.** No `max_frame_fraction` is written, proposed or implied here. V13 stays unsigned.
- **Not that the cluster is right and this machine is wrong**, or the reverse. Two machines
  disagree; nothing here adjudicates which mask is the better mask, and neither was checked against
  ground truth.
- **Not a corpus rate for the middle band beyond its own 44 frames** — though that band IS the
  complete population, so for once the sample is the thing.
- **Not a bulk rate.** 48 frames of 112 361 is a sample, and a larger one could find a crossing.
- **Not a discharge of anything.** No blocker moved.

---

## 7. Provenance

| | |
|---|---|
| kind | measurement report with two controls. **Registers no rule** |
| date | 2026-08-26 |
| hardware | workstation **RTX 5090**, torch 2.13.0+cu130, against cluster job 106's pooled artifact |
| corpus | `pr08-apple-640x480-h264-lossless`, `source_manifest_sha256 a988dd60db6ba8ab…` |
| masker | `scripts/robot_composite.Sam2RobotMasker`, unmodified, committed operating point |
| driven by | `scripts/render_area_tail_sheet.py` with `--max-fraction` |
| artifacts | `runs/pr08-area-tail-look/`, `runs/pr08-below-bound-look/`, `runs/pr08-below-bound-repeat/`, `runs/pr08-bulk-stability/` |
| determinism control | two identical runs over the 44-frame band, 0 differences |
| blockers discharged | **none** |
| `GATE_QUALIFIED` | still `False`, untouched |
| generation licensed | **no** |
| training licensed | **no** |
