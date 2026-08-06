---
id: T-25c
aliases:
- T-25c
title: "Bounded feed-forward in `G1Adapter.execute()`"
slug: bounded-feed-forward-in-g1adapter-execute
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m4
- interfaces
- sim
- hardware
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-08-02
---

# Bounded feed-forward in `G1Adapter.execute()`

## Description

*(follow-up, design change — deliberately NOT done in T-25b)* Bounded feed-forward in
`G1Adapter.execute()`. The adapter re-based its target on the **measured** `q` at every call, so the
position loop's lag was discarded rather than caught up: a joint executed ~0.39 of a
one-control-period step and ~0.95 of a 25-step chunk, i.e. the executed magnitude depended on
`prefix_steps`, a rollout knob. **Not tunable away** — even flat kp=4000 with critical damping
reaches only mean 0.86, because ~full execution inside 20 ms needs a closed-loop bandwidth no arm
has. Consequence: sim action labels were not commanded actions, safety-intervention rates were not
calibrated, and a zero-delta chunk was not a hold (the arm ratcheted 0.73 rad in 30 s under
gravity). — *✅ **done 2026-08-02, 1 091 tests green** (`G1Config.q_track_window`,
`G1Adapter._carry_in`/`forget_command`/`tracking_error`/`tracking_clamp_count`,
`mujoco_g1.SIM_Q_TRACK_WINDOW`, 13 new tests). The carry is the previous **commanded** target,
clamped to within `q_track_window` of the measured `q`; integration, the per-step `dq_max·dt` clip
and the `[q_min, q_max]` clip are untouched, and the SafetyLayer is not in this path at all (FR-07
unchanged). **Measured, not asserted** — `left_shoulder_yaw`, 0.400 rad commanded, prefix 1/5/25:
`0.306 / 0.666 / 0.947` at window 0 → **`0.987 / 0.987 / 0.987`** at 0.05, i.e. the dependence on
`prefix_steps` is gone, not merely reduced. Zero-delta ratchet over 10 s: `0.0803 → 0.1338 → 0.2047
→ 0.2698 → 0.3294` rad (still growing) → a **flat 0.0091**, which is exactly the bounded
fixed-target droop `mujoco_g1.yaml` always quoted and the runtime protocol could never reach. **The
window is the smallest value that works, and that is the safety argument:** an uncorrected carry is
an integrator with no anti-wind-up, so a blocked joint stores commanded error and releases it as one
swing when freed; the clamp bounds the stored error at `q_track_window` per joint (measured bound
`window + steps·per_step`, **settled** — 50 chunks against a frozen joint leave the lead unchanged
to 1e-9, where an unclamped carry would climb 0.01 rad per chunk). Sizing is a measurement, not a
taste: the window must exceed the steady-state tracking error at `dq_max` (worst joint **0.0299
rad** at 1.5 rad/s) or the clamp bites in normal fast motion — **0.02 was tried and rejected**,
clamping 58 of 60 steps at `dq_max` and dropping execution back to 0.846; 0.05 clamps **zero** times
from 0.2 to 2.0 rad/s. The carry is dropped on `connect`/`hold`/`estop`/`forget_command` and on
`MujocoG1Robot.reset()` — after any of those the robot may be somewhere this adapter did not put it,
and `estop` matters most, since damping lets the arm settle and the first act after `clear_estop()`
must not be a lunge back to the pre-estop target. A partially executed chunk carries only what was
actually written, and a non-finite carry falls back to measurement (`np.clip` with a nan bound
returns nan, and nan reaches the motors). **Shipped OFF on hardware:** `configs/robot/g1.yaml` keeps
the `0.0` default, because the window is a property of `kp`/`kd` and `g1.yaml`'s are OD-08
placeholders whose ~0.17 rad sag is ~6× this window — a copied value would clamp every step and
throttle the feed-forward to nothing. `tracking_error` is reported **regardless** of the window, so
the number needed to size a real one is free to collect during bring-up. The archived
under-execution bands are not deleted but kept live as
`test_the_disabled_window_reproduces_the_archived_under_execution`, which is both the A/B arm and
the pin on the hardware configuration's actual behaviour. Docs corrected at every site that asserted
the defect as current: `docs/sim.md` §1–2, `docs/ROADMAP.md`, `src/wam/robot/README.md`, and the
`mujoco_g1`/`mujoco_transport` module docstrings.* **Adversarial review 2026-08-02 found two real
defects in the above, both fixed (8 confirmed of 21 raised; 13 refuted):** (1) **the safety property
the commit message claimed was false.** `execute()` tested the e-stop latch **once, before the
loop**, and wrote the carry on **every step**, so an `estop()` arriving mid-chunk had its carry-drop
immediately undone by the still-running loop — and `hold()` cannot repair it because it
early-returns while estopped, and `clear_estop()` deliberately does not forget. The pre-estop
setpoint therefore survived `clear_estop()`, and the first motion after an emergency stop was an
unrequested lunge back toward it, on a chunk as innocuous as the SafetyLayer's own zero-delta hold.
Reproduced two ways: `FakeG1Transport` with two real threads (3/3), and on the MuJoCo rig where a
zero-delta chunk after `clear_estop` moved `left_shoulder_pitch` **−0.0452 rad** against **+0.0057**
(gravity only) with the carry properly dropped. The reproduction also exposed a **pre-existing**
defect it depends on: the loop kept writing full-gain position targets for all 25 steps after the
latch was set, i.e. commanding an arm the vendor had just damped. Both fixed — the loop now breaks
on the latch, and a re-check after the loop closes every ordering including an e-stop landing after
the final write. Note the old test could not see this: it used `tracking_error > 0.0` as its proxy,
and `estop()` leaves that at 0.0 while the carry is armed. (2) **`scripts/view_sim.py` silently
ignored `control.q_track_window`**, so the viewer ran the *pre-T-25c* control law while a rollout
from the same config file used the new one — 0.44 vs 0.96 of a commanded travel at `prefix_steps=1`,
and **3.3× the `accel_limit` interventions** (16 vs 53), with `docs/sim.md` quoting the 16 as what
the operator should expect from "the same chain". Fixed, and pinned by a **generic** test asserting
`view_sim` honours every robot-config field `G1Config` accepts, so the next forgotten field fails
too. **Still not done:** the window for real hardware (needs OD-08 gains + a measured tracking error
on the robot), and `hold()` deliberately still re-bases on measurement.

---

Migrated from `TASKS.md` (milestone M4) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
