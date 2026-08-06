---
id: T-25
aliases:
- T-25
title: "Real MuJoCo simulation of the G1 behind the `G1Transport` seam"
slug: real-mujoco-simulation-of-the-g1-behind-the-g1transport-seam
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m4
- safety
- interfaces
- sim
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-29
updated: 2026-07-29
---

# Real MuJoCo simulation of the G1 behind the `G1Transport` seam

## Description

Real MuJoCo simulation of the G1 behind the existing `G1Transport` seam, so the closed loop runs on
contact physics and rendered pixels before hardware: `configs/sim/g1_scene.xml` (Menagerie
`unitree_g1` + Dex3 hands, welded base, 14 locked joints, table + cube, cameras
`head`/`wrist_left`), `MujocoG1Transport` as the **third** implementation of the seam next to
Fake/Dds, `MujocoG1Robot` composing the *real* `G1Adapter`, `scripts/fetch_g1_model.py` (pinned,
gitignored `assets/`), `configs/robot/mujoco_g1.yaml`, `scripts/view_sim.py` + `scripts/view_sim.sh`
(interactive viewer driving the *real* closed loop — same chain as `rollout.py`, safety layer never
bypassed, logs to `runs/view/`; the wrapper exists because `mjpython` loads the interpreter with
`dlopen`, so a uv-managed CPython's `libpython` is off every `@rpath` and the launch dies before
Python starts), `mujoco` as optional extra `wam[sim]` — *✅ verified 2026-07-27 by ad-hoc scripts:
scene settles (pelvis drift 0.0042 mm, worst locked joint 6.6e-05 rad, cube z jitter 0.000 µm),
renders carry real structure (256×256 std 86.8 / 77.9), motor slots resolve by name and cross-check
against `G1_JOINT_MAP`, `tick_ns` advances only on motor writes (deltas exactly 2e7 ns) so the
stale-sample path degrades validity as on hardware, gripper round-trip strictly monotonic (max err
0.024 at full close), `emergency_damp()` 2.549 → 0.198 rad/s, two rollouts bit-identical, 28.5×
realtime physics-only / 0.40× with 2 renders per step, full `ClosedLoopExecutor` rollout 20/20
cycles at 1.19× realtime. Sim gains (`SIM_KP`=500 = the vendor Menagerie class stiffness, `SIM_KD` =
per-joint critical damping) kept in a **separate** config — `g1.yaml`'s kp=20 hardware placeholders
leave 0.17 rad of sag (OD-08 untouched). Guide: `docs/sim.md`. Pinned by `tests/test_mujoco_g1.py`
(29 tests in the normal run, 617 passed; whole module skips with an actionable reason when `mujoco`
or the fetched model is absent) — protocols, registry, motor-slot names, low-state contract, tick
semantics, stale-tick validity through the **unmodified** adapter, clipping, e-stop latch, gripper
monotonicity + mirroring, bit-identical determinism, render shape, base/locked-joint drift.*
**Honestly pending:** the tests pin contracts, not physics — image assertions check variance and
shape only (renders are not bit-portable across GL backends), and the reach distances, damping
curves and throughput numbers above come from ad-hoc scripts that nothing re-runs; no stable grasp
(fingers contact the cube, it slips after ~6 mm); `available_robots()` deliberately does not list it
(see `optional_robots()`). **Sim frames are NOT training data** (see `docs/sim.md`)

---

Migrated from `TASKS.md` (milestone M4) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
