---
id: T-25b
aliases:
- T-25b
title: "Adversarial-review repairs on the MuJoCo track"
slug: adversarial-review-repairs-on-the-mujoco-track
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m4
- interfaces
- sim
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# Adversarial-review repairs on the MuJoCo track

## Description

Adversarial-review repairs on the MuJoCo track (2026-07-28). Four reproduced blocking defects fixed
at the root: (1) `MujocoG1Transport.write_gripper_cmd` accepted non-finite input while its sibling
rejected it — `np.clip(nan, 0, 1)` is `nan`, and one NaN in `data.ctrl` makes MuJoCo zero **all 43**
controls, slamming the robot from `ready` to its zero pose in 0.2 s with no exception, finite
`read_low_state` output and a normally advancing tick; now rejected in both write paths. (2)
`emergency_damp()` wrapped its body in `except BaseException` and recorded the failure on
`last_damp_error`, which **no caller reads** — a totally failed safe-stop (|dq| unchanged at 3.67
rad/s) reported success, contradicting `G1Adapter.estop()`'s documented "the exception still
propagates"; now catches `Exception` only and re-raises after recording. (3) `estop()` from a second
thread **SIGSEGVed the interpreter 3/3**, violating `RobotAdapter.estop()`'s "safe … from any
thread"; every `MjModel`/`MjData` access is now serialised behind a public re-entrant
`MujocoG1Transport.lock`, which `render_frames` and the sim-sleep also hold. (4) sim gains were
tuned against a fixed-absolute-target protocol the adapter never issues; replaced with the vendor
model's own design point (`SIM_KP`=500, `SIM_KD`= per-joint critical damping re-derivable from the
scene), which more than doubles per-control-period execution (mean 0.14 → 0.39) and drops the CLI
rollout's safety interventions to 0. Plus: the un-fetched-model hint moved into
`MujocoG1Transport.__init__` so both entry points give it; `G1Adapter.forget_tick()` replaces the
private `_last_tick_ns` poke; `SIM_DQ_MAX` makes the no-config path match the versioned yaml;
`configs/robot/mujoco_g1.yaml` now covered by `tests/test_versioning.py`; two inaccurate limit
claims in that yaml corrected (<5 mrad, not <1; ranges rounded *inward*). **8 new tests** in
`tests/test_mujoco_g1.py` (604 → 617 passed), each a regression guard for a reproduced defect.

---

Migrated from `TASKS.md` (milestone M4) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
