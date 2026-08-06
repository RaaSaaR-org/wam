---
id: T-31
aliases:
- T-31
title: "The gripper was never flat — we flattened it"
slug: the-gripper-was-never-flat-we-flattened-it
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- eval
sprint: ''
depends_on: []
due_date: ''
created: 2026-08-01
updated: 2026-08-02
---

# The gripper was never flat — we flattened it

## Description

The gripper was never flat — we flattened it (I-9, `docs/improvements.md`;
`scripts/convert_lerobot_g1.py`, `src/wam/evaluation/gripper.py`, `scripts/audit_gripper.py`) —
*T-27 recorded that `datasets/gr00t-apple-full`'s demonstrated gripper has peak-to-peak range 0.120
sitting on the 0.5 binarization threshold, concluded the benchmark is structurally blind to
grasping, and prescribed acquiring a public LeRobot dataset with real transitions. **The
prescription was wrong.** Measured on `data/raw/gr00t_apple`, parquet only, no decode: the left
hand's joints span up to **0.826 rad** and the commanded `action[29:36]` spans the full range; the
right hand is frozen at 0.0007 rad across all 402 episodes. `hand_synergy`'s `clip((mean + 1) / 2,
0, 1)` assumes source joints in [-1, 1] — a range the Dex3 hand never uses — and squashes 0.826 →
0.157; `relabel_chunks` then averages **both** hands into `gripper_target`, and the dead right hand
halves it again to 0.0785. That is the 0.120. Rescaled, **30/30 episodes show exactly two debounced
open/close transitions** — one grasp, one release. The grasp was always on disk; we destroyed it in
the converter and read the flat channel as a property of the task. **What shipped:**
`--gripper-mapping active-hand` fits one **dataset-level** affine over the raw synergy of the hand
that moves and takes `gripper_target` from that hand alone (dataset-level, not per-episode — a
per-episode min-max makes the same physical aperture a different number in every episode, which is
unlearnable). Because the fit depends on which episodes are in the conversion (30 eps →
−0.39980/0.41004, 402 eps → −0.43865/0.46675), `--gripper-affine OFFSET SPAN` pins it so two
conversions are comparable, and a pinned affine that would clip **any** sample is refused — clipping
is silent in the output and moves every admissibility clause in the *passing* direction. The legacy
mapping is now held to the same bar (`legacy_clipped_frac`); until 2026-08-01 the one mapping that
**assumes** a scale was the only one allowed to be silently wrong about it.
`scripts/audit_gripper.py` is the gate: dynamic range, debounced transitions per episode, fraction
of episodes with a transition, and fraction with a full close-**and**-reopen **cycle** — the cycle
clause exists because a monotone ramp from 0 to 1 clears a transition count and is not a grasp.
Saturation is reported against `expected_saturated_frac`, since a fitted affine rails exactly the
two extremal samples it was fitted on and anything far above that is "clipped, not measured".
`bench.json` withholds `gripper_accuracy` with a reason on an inadmissible channel, and `e1.md` now
does the same — the two artifacts of one run were disagreeing, because T-27 put the diagnostics next
to the number instead of in front of it.* **Re-converted 2026-08-02** — `--source
data/raw/gr00t_apple --out datasets/gr00t-apple-grip --episodes 402 --gripper-mapping active-hand`,
one CPU pass, ~7 min, 80 MB out, into a **new** root: `datasets/gr00t-apple-full` is byte-untouched
and still verifies its per-episode checksums, so every archived `dataset_snapshot_ref` keeps
resolving (AC-04). The affine fitted over all 402 episodes is exactly the one the converter
docstring predicts — `active=left offset=-0.438654 span=0.466748`, mean per-episode p2p left 0.3214
/ right 0.0008 — and it travels with the data in each manifest's `normalization` +
`mapping.gripper_synergy`. `scripts/audit_gripper.py` **PASSES all four clauses** on the result and
still **FAILS** the legacy set, on the same code and the same 402 episodes: debounced
transitions/episode **2.01 vs 0.00**, episodes with a full close-and-reopen cycle **99.0 % vs 0.0
%**, per-episode p2p **0.6885 vs 0.0804**, zero saturation on either rail of the scored channel.
That is `PR-01-FOLLOWUP`'s predicted 2.015/ep reproduced end to end. One non-gated finding:
`state.gripper[1]` (the frozen right hand) rails on 0.1685 of samples, because the left hand's
affine is applied to both — expected, and it does not touch the scored `action.gripper_target`.
**Still not done:** re-scoring the ladder on the new root (needs a GPU pass) and the 40 → 150
holdout widening `PR-01-GRIPPER.md` §2 asks for; a rescaled gripper still changes nothing about what
`skill_vs_repeat_pct` measured on arm trajectories

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
