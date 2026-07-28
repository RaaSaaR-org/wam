# MuJoCo simulation of the G1 (E2)

**TL;DR** — `configs/sim/g1_scene.xml` puts the Unitree G1 with Dex3 hands in front of a table
with a cube, on two cameras. `MujocoG1Transport` implements the existing `G1Transport` protocol,
so the **real** `G1Adapter` — joint mapping, unit conversion, defense-in-depth clipping, pacing,
latched e-stop — runs unchanged on contact physics and real rendered pixels. Nothing about the
robot code path is simulated; only the robot is.

```bash
uv pip install mujoco                            # extra: wam[sim]
.venv/bin/python scripts/fetch_g1_model.py       # ~38 MB into assets/ (gitignored)
```

## What this is, and what it is not

It **is** a second implementation of the hardware seam. `G1Transport` (`g1_transport.py`) now has
three implementations — `FakeG1Transport` (a first-order lag, for unit tests), `DdsG1Transport`
(real vendor DDS), `MujocoG1Transport` (physics). Everything above the seam is byte-identical in
all three cases. That is the whole design: the sim buys coverage of the *production* adapter, not
of a sim-specific one. There is deliberately no `MujocoAdapter`.

It is **not** a training environment, not a digital twin, and not a source of training data. The
scene is hand-built, the contact parameters are MuJoCo defaults, and the renderings look nothing
like a RealSense D435i. See "What this does not prove" at the bottom — read it before quoting any
number from here.

## Setup

```bash
cd /Users/heussers/develop/emai/wam
uv pip install mujoco                          # or: uv pip install -e '.[sim]'
.venv/bin/python scripts/fetch_g1_model.py     # idempotent; --check verifies, --force re-fetches
```

`mujoco` is an **optional** dependency. `wam.robot`, `wam.robot.registry`, `wam.robot.mujoco_g1`
and the full test suite all import and run without it; only *constructing* the robot raises
`RuntimeError: MuJoCo simulation support requires the 'mujoco' package …`. Verified with mujoco
blocked at the import hook.

The model is **fetched, not vendored**: `scripts/fetch_g1_model.py` does a shallow + blobless +
sparse clone of MuJoCo Menagerie pinned to commit `71f066ad0be9` — the revision the scene was
authored and measured against — into `assets/mujoco/unitree_g1/` (`assets/` is in `.gitignore`).
Staging is a same-filesystem rename, so an interrupted fetch cannot leave a half-written tree; a
`--force` re-fetch produced a byte-identical tree (`diff -rq` clean, 39.6 MB). Attribution:
Menagerie ships `unitree_g1` under Unitree's BSD-3-Clause license, and the `LICENSE` file travels
inside the fetched folder (the script checks for it).

## Running it

```python
from wam.robot.registry import get_robot

robot = get_robot("mujoco_g1")          # constructed already connected, no SDK, no socket
state = robot.read_state()              # canonical 15-joint RobotState + validity flags
images = robot.render_frames(1)         # {"head": (1,256,256,3) uint8, "wrist_left": ...}
robot.execute(safe_chunk, prefix_steps=5)
robot.reset()                           # episode reset; a latched e-stop survives it
```

`get_robot("mujoco_g1")` takes ~0.27 s. `MujocoG1Robot` satisfies `RobotAdapter` structurally
(composition, not subclassing — a subclass would inherit `G1Adapter.connect()`'s `DdsG1Transport`
fallback, i.e. a sim robot able to open a DDS socket). It adds `render_frames`, `reset`,
`clear_estop`, `sim_time_ns`, `close`, and read-only `adapter` / `transport` / `cameras` /
`image_hw` / `scene_path` accessors.

**`scripts/rollout.py --robot mujoco_g1` works** — the flag is wired and runs the full
`ClosedLoopExecutor` (DummyPolicy + SafetyLayer + Watchdog + JSONL) on MuJoCo:

```bash
.venv/bin/python scripts/rollout.py --robot mujoco_g1 --policy dummy --rollouts 1
```

Measured on this machine (12 cycles, `prefix_steps=5`, both cameras rendering): E2 static checks
PASS on 16 probes, 0 e-stops, 0 watchdog timeouts, 0 deadline misses, 0 safety interventions,
`min_rate 14.5 Hz`. `--fault-injection` exercises the NaN-reject and deadline-miss paths.

Two configuration knobs that matter:

- **Pacing runs on sim time.** `G1Adapter.execute()` paces per-step commands so step `i` is sent
  no earlier than `t0 + i·dt_s` — that pacing is what makes its `dq_max·dt` clip a real velocity
  limit. `MujocoG1Robot` injects the transport's sim tick as clock and a sleep that *steps
  physics* instead of blocking, so the velocity reasoning is identical to hardware while a
  10-step chunk costs 8 ms of wall time instead of 180 ms. Pass `clock=time.monotonic,
  sleep=time.sleep` for realtime pacing (measured 0.186 s for the same chunk).
- **Keep `chunk.dt_s == control_dt_s` (0.02 s).** A larger chunk `dt_s` makes the executor step
  extra hold-physics between commands (correct, slower); a smaller one advances more sim time per
  step than the chunk claims, and the trajectory plays back in slow motion.

### Watching it live

```bash
scripts/view_sim.sh --amplitude-rad 0.2      # wall-clock paced, close the window to stop
scripts/view_sim.sh --fast                   # sim-time paced, as fast as the machine allows
```

`scripts/view_sim.py` runs the **same** chain as `scripts/rollout.py --robot mujoco_g1`
(`ClosedLoopExecutor` → `SafetyLayer` + `Watchdog` → `G1Adapter` → `MujocoG1Transport`) and only
adds a viewer window. It never bypasses the safety layer and never writes `data.ctrl` itself, so
what you watch is the real control loop, not a playback. It writes the standard rollout log
(`run_metadata` + `control_cycle` + `rollout_summary`) to `runs/view/` (AC-04), and pacing is
wall-clock by default so the motion runs at the rate the robot would.

The executor runs on a worker thread while the main thread syncs the window, so every `MjData`
read holds `MujocoG1Transport.lock` — the same lock that makes a cross-thread `estop()` safe.
Without it this script segfaults reliably.

**Always launch through the `.sh` wrapper, not `mjpython` directly.** Two macOS constraints stack
up. The interactive viewer needs `mjpython` (the main thread must run the native event loop), and
`mjpython` loads the interpreter with `dlopen` — so the `@rpath` lookup uses *mjpython's* rpaths,
not the Python binary's. A uv-managed CPython keeps `libpython3.x.dylib` in the uv toolchain
directory, which is on none of those paths, and the launch dies before Python starts:

```
failed to dlopen path '.../.venv/bin/python3': Library not loaded: @rpath/libpython3.12.dylib
```

Nothing inside `view_sim.py` can fix that, so the wrapper derives the directory via `sysconfig`
(not hard-coded) and exports `DYLD_FALLBACK_LIBRARY_PATH` before exec'ing `mjpython`.

At `--amplitude-rad 0.2` a run reports ~16 `accel_limit` interventions over 25 cycles; at the
calm default (0.05) it reports none. That is limitation 1 below, not a viewer bug.

## The scene

`configs/sim/g1_scene.xml` (nq=71, nv=67, nu=43, ncam=2, neq=17, timestep 2 ms) `<include>`s the
vendor `g1_with_hands.xml` verbatim and adds table, cube, lighting and cameras.

| | |
|---|---|
| cameras | `head` (fovy 70), `wrist_left` — become the `Observation.images` keys |
| keyframe | `ready` (id 0; the vendor `stand` is id 1) |
| cube | body `cube`, joint `cube_free`, geom `cube` |
| table | body `table`, geoms `table_top` + 4 legs |
| constraints | weld `pelvis_to_world`, 14 joint locks (legs + waist roll/pitch), 2 camera welds |

Two ordering constraints in the MJCF are load-bearing and easy to break: `meshdir` resolves
against the **top-level** file's directory, so the scene's `<compiler>` must come *after* the
include and override the vendor one; and MuJoCo merges keyframes in document order, so `ready`
must come *before* the include or the vendor `stand` becomes key 0 and — zero-padded — drops the
cube at the world origin.

Measured over 2 s of settling: pelvis drift **0.0042 mm** / 0.0021°, worst locked joint
**6.6e-05 rad** (`waist_pitch_joint`), cube sink **+0.0039 mm** with 0.000 µm peak-to-peak z
jitter over the last second. Renders at 256×256: `head` mean 110.3 / std 86.8 / 2100 unique
colours, `wrist_left` mean 110.0 / std 77.9 / 1381 unique colours — i.e. real structure, not a
flat frame. Two identical rollouts are bit-identical (max diff 0.0).

Reach, honestly: at `ready` the left fingertips sit 103–133 mm from the cube centre with the hand
37 mm above the table and `ncon=0`. Driven to an IK'd pre-grasp the open hand straddles the cube
without touching it (cube moves 0.00 mm); closing the fingers gives **2 sustained finger↔cube
contacts** (`thumb_2`, `middle_0`, 0.2 mm penetration) and nudges the cube 14.7 mm. **It is not a
stable grasp** — the cube rises ~6 mm and slips. Reach and contact are the scene's job; holding is
the policy's. The right hand reaches the mirrored spot and contacts `table_top`, so the table is
genuinely two-handed.

Three scene compromises, stated rather than hidden:

1. **Cameras ride on welded, near-massless free bodies.** MJCF cannot add a child to a body that
   arrived via `<include>` (re-declaring `torso_link` is a duplicate-name compile error), so the
   camera mounts are welded to `torso_link` / `left_wrist_yaw_link`. Cost: 2×6 extra DoF in
   `qpos`/`qvel` (index by name, never by raw index) and residual motion under a 0.8 Hz shoulder
   sweep of 8.8e-05 m (head) / 4.7e-04 m (wrist).
2. **Equality stiffness is a modelling device, not physics.** At MuJoCo's stock softness the
   welded pelvis sags ~25 mm and `waist_pitch` creeps ~4e-3 rad; `class="wam_rigid"` tightens
   `solref` to 0.5 ms. Direct-stiffness `solref` values (`-1e5 -1e3` and beyond) blow up.
3. **A vendor-model quirk we did not paper over:** `wrist_pitch` declares
   `actuatorfrcrange ±5 Nm` while its position actuator inherits kp=500. Holding the hand out it
   asks for ~55 Nm, gets 5, and sits **0.109 rad (6°)** below command; every other arm joint
   tracks to <0.015 rad. Under table contact at kp=300 the error reaches 0.54 rad (163 Nm
   demanded). Fixing it means overriding the vendor `g1` default class for the whole robot.

One measurement gotcha for anyone extending the scene: `mujoco.mj_geomDistance` is **unreliable on
the mirrored hand meshes** — it returned 0.00 mm for `left_hand_middle_1_link` where the
geometrically identical right-hand mesh correctly returned 37.5 mm. All reach conclusions above
come from the contact list (`d.ncon`) and kinematic body distances instead.

## The transport

`MujocoG1Transport` implements `read_low_state` / `write_motor_cmd` / `write_gripper_cmd` /
`emergency_damp` and passes `isinstance(t, G1Transport)`. All 43 scene actuators are *position*
actuators, so `write_motor_cmd` overrides `actuator_gainprm[:,0]=kp`, `biasprm[:,1]=-kp`,
`biasprm[:,2]=-kd` on the 29 body actuators, writes `ctrl`, then steps 10 × 2 ms = one 20 ms
control period. `dq_target` becomes a `kd·dq_target` feed-forward on `qfrc_applied`, because a
position actuator's affine bias cannot express a nonzero velocity target.

All 29 motors are resolved **by name** via `mj_name2id` and cross-checked against `G1_JOINT_MAP`;
the constructor raises if the Menagerie joint order ever stops matching. `write_gripper_cmd`
writes finger `ctrl` and deliberately **does not step** — `G1Adapter.execute()` issues one motor
plus one gripper command per step, and stepping in both would double the simulated period behind
the adapter's back.

`tick_ns` follows the protocol exactly: bare `read_low_state()` calls do **not** advance it
(`0, 0, 0`), five motor commands give `0, 2e7, 4e7, 6e7, 8e7, 1e8` ns (deltas exactly
`control_dt_s`), bare gripper commands leave it unchanged, `reset()` returns it to 0. That is what
makes the adapter's stale-sample detection testable here: an immediate re-read after a write
degrades every validity flag, exactly as on hardware.

`emergency_damp()` (defaults `damp_kd=20`, `damp_duration_s=0.2`) took a moving arm from
|dq|max 2.549 → 0.198 rad/s in one call. Repeated calls plateau at 0.148 rad/s — that floor is
real, not a bug: with kp=0 the arm creeps down at the gravity terminal rate `tau/damp_kd`
(measured 1.53 / 0.61 / 0.198 / 0.082 / 0.041 rad/s at damp_kd = 2 / 5 / 20 / 50 / 100), which is
how vendor damp mode behaves too. It never raised — not on a fresh transport, not with
`damp_duration_s=0`, not with `qpos` set to NaN. Damping gains persist until the next
`write_motor_cmd` restores tracking.

### Dex3 grasp synergy and why the canonical gripper stays scalar

OD-01 fixed the canonical gripper at **2 scalars** `[left, right]` in `[0, 1]`; per-finger control
is post-MVP. The transport realises that as a synergy derived from the model's own `jnt_range`:
open = 0 rad on all 7 finger joints (= the `ready` pose), closed = the range endpoint farther from
zero for the 6 curling joints. Mirroring falls out of the ranges — left closed is
`[0, +1.0472, +1.7453, -1.5708, -1.7453, -1.5708, -1.7453]` and right is its exact negation.
`thumb_0` is held at 0 in both: measured at full curl, the thumb tip sits 37.2 mm from *both* the
middle and index tips at `thumb_0=0` versus 44.6–69.1 mm at ±1.047, so zero is the opposing pose.

Round-trip (`SIM_KP`/`SIM_KD`, 1 s settle) is strictly monotonic and near-exact: commands 0.00 / 0.25 /
0.50 / 0.75 read back identically; 1.00 reads back 0.9756 (left) / 0.9760 (right). The 0.024
residual is joint-limit constraint softness — the closed pose sits *on* the limits. Asymmetric
commands work (L=1.0 / R=0.0 → `[0.9756, 0.0000]`).

Caveat: `write_gripper_cmd` clips to `[0, 1]`, which assumes `G1Config.gripper_vendor_min/max`
stay at their 0/1 defaults. A non-unit vendor range would need the clip bounds passed in.

Note the asymmetry with the DDS track: MuJoCo's synergy is *measured against the real Menagerie
kinematics*, while `DdsG1Transport`'s gripper mapping (command all 7 joints to the same angle,
read back their mean) is an explicit placeholder pending hardware (OD-08).

## Gains: sim numbers are not hardware numbers

`configs/robot/g1.yaml` keeps `kp=20 / kd=0.5` — conservative **hardware placeholders** pending
OD-08. Those gains do not hold this arm up: with no gravity compensation, a −0.5 rad elbow step
leaves 0.171 rad of steady-state error and never settles.

There are **two** ways to measure a gain set here, and they disagree badly. Quote both.

**(A) Fixed absolute target, held 1.5 s** — a protocol the runtime never uses. Steady-state
|error| after a −0.5 rad `left_elbow` step in free space:

| kp / kd | elbow err | mean (15 joints) | max (15) |
|---|---|---|---|
| 20 / 0.5 (`g1.yaml` default) | 0.1711 | 0.0468 | 0.1874 |
| 100 / 5 | 0.0323 | 0.0144 | 0.0436 |
| 200 / 10 | 0.0160 | 0.0073 | 0.0223 |
| 300 / 15 (flat — the *previous* sim default) | 0.0107 | 0.0049 | 0.0150 |
| **500 / per-joint critical damping (`SIM_KP`/`SIM_KD`)** | **0.0064** | **0.0030** | **0.0091** |

**(B) Through `G1Adapter.execute()`** — the only protocol the runtime ever uses. The adapter
re-bases its target on the **measured** `q` at every call, so whatever the position loop has not
caught up with by the end of a chunk is discarded. Metric: the fraction of one control period's
commanded step actually executed inside that period (`prefix_steps=1`, one joint at a time,
free space, 0.004 rad/step):

| gain set | min | mean | max |
|---|---|---|---|
| 300 / 15 flat | −0.20 | 0.14 | 0.30 |
| **`SIM_KP`/`SIM_KD`** | −0.11 | **0.39** | 0.97 |

and the same 0.400 rad of commanded travel executed with different `prefix_steps`
(`left_shoulder_yaw`, identical 2 s of sim in every row):

| `prefix_steps` | 1 | 2 | 5 | 10 | 25 | 100 |
|---|---|---|---|---|---|---|
| 300 / 15 flat | 0.162 | 0.269 | 0.477 | 0.675 | 0.895 | 0.975 |
| **`SIM_KP`/`SIM_KD`** | 0.309 | 0.439 | 0.666 | 0.851 | 0.947 | 0.987 |

**`SIM_KP` = 500** is the vendor Menagerie `g1` actuator-class stiffness — the value the model
itself was authored for. **`SIM_KD`** is per-joint *critical* damping, `2·√(kp·m_eff)`, which is
exactly what the vendor's `dampratio="1"` compiles to (`mujoco_g1.scene_critical_damping()`
recovers it from the scene; a test re-derives `SIM_KD` from the model so code and config cannot
drift apart). A **flat** kd is the wrong shape: the wrist roll's effective inertia is 0.011 kg·m²
against the waist's 0.398, so one number is either heavily overdamped at the wrist or underdamped
at the waist.

These gains live in a **separate** `configs/robot/mujoco_g1.yaml` and must not be copied into
`g1.yaml` — they describe MuJoCo's actuator model, not the robot. The transport never clamps or
substitutes the caller's gains; `MujocoG1Robot` merely defaults to them when no config is passed
(along with `SIM_DQ_MAX`, which mirrors the yaml's `dq_max` so the no-config path is not looser
than the versioned file).

`configs/robot/mujoco_g1.yaml` also takes `q_min`/`q_max` from the scene's own `jnt_range`,
rounded *inward* to 4 decimals (6 of 15 joints differ from the raw range by 2–3e-5 rad, always
narrower — a wider limit would make the adapter's clipping lie). They agree with `g1.yaml`'s
datasheet placeholders to **<5 mrad** (worst: elbow and wrist pitch/yaw, 4.40 mrad; `g1.yaml` is
rounded to 2 decimals and on `shoulder_roll` it is *wider* than the scene, so it is not uniformly
the conservative side), and are much wider than `G1Config`'s ±1.5708 default, which would clip
0.52 rad off elbow flexion. `dq_max`/`ddq_max` are **identical** to `g1.yaml` on purpose: they are
MVP policy caps (PRD §11.2), not hardware facts, and a sim-validated policy should carry the same
velocity envelope onto the robot.

## Measured performance

| | |
|---|---|
| physics only (write motor + write gripper + read state) | **1424 control steps/s = 28.5× realtime** |
| with 2 × 256×256 offscreen renders per step | **19.9 steps/s = 0.40× realtime** |
| `render_frames(1)`, both cameras | 44–48 ms |
| `get_robot("mujoco_g1")` | 0.27 s |
| closed-loop rollout (executor + safety + watchdog + 2 renders/cycle) | 1.19× realtime, 20/20 cycles |

Rendering is the bottleneck, not physics, and 19.9 steps/s still leaves ~10× headroom over the
FR-05 2 Hz policy-rate floor. Offscreen rendering via `mujoco.Renderer` works natively and
headless on this macOS/arm64 machine.

Determinism holds end to end: two fresh transports running 200 steps of a sin/cos arm sweep plus a
sawtooth gripper differ by max abs 0.0; `reset()` is bit-identical across resets and across
instances. No Python-level RNG is used anywhere in the sim path.

## The Docker DDS track (separate, and it passes)

Independent of the sim, `docker/dds/` runs the real `DdsG1Transport` against a fake G1 on a real
CycloneDDS bus inside a `linux/arm64` container — the same architecture as the G1 EDU4's onboard
Jetson Orin, so the vendor SDK, the CycloneDDS build and the wire format are the real ones.

```bash
docker/dds/run.sh          # build + run; exit 0 = no FAIL
```

Status 2026-07-27: **11 PASS / 0 FAIL / 0 SKIP**, reproduced across 4 consecutive runs. Checks:
`sdk_import`, `dds_init_loopback`, `fake_peer_process`, `transport_connect`, `lowstate_roundtrip`,
`lowcmd_roundtrip_crc`, `lowcmd_crc_detects_corruption`, `emergency_damp`,
`dex3_gripper_roundtrip`, `adapter_closed_loop`, `stale_tick_degrades_validity`. The peer is a
**separate process** with its own DomainParticipant, so discovery and RTPS exchange are real; a
negative CRC check proves the positive one is not vacuous.

Consequence for the repo: `DdsG1Transport` is **no longer stubbed** — its five methods are
implemented. On the host, where `unitree_sdk2py` is absent, they still raise
`RuntimeError("G1 hardware support requires unitree_sdk2py")`, and `tests/test_g1.py` /
`tests/test_robot.py` are untouched.

Two deliberate design calls worth knowing: `emergency_damp()` is a **wire** command (`kp=0`,
`kd=damp_kd`, `q=dq=tau=0` on all 29 motors, 3×, valid CRC) rather than `LocoClient().Damp()`,
because it must work when the vendor service is dead — exactly when an e-stop matters; the service
call is documented as a bring-up escalation. And `cmd_topic=rt/arm_sdk` + `arm_sdk_weight` is
supported (legs stay with the vendor controller) as the safer first-contact path — the constructor
rejects that topic without a weight, since a missing weight silently voids every command.

Full detail, the 8-point "does not prove" list, the Docker-Desktop multicast caveat and 9 ordered
hardware bring-up steps: `docker/dds/README.md`.

## What this does **not** prove

Read this before quoting any number above.

1. **The pixels are wrong, and that is the important one.** MuJoCo renderings are not RealSense
   images — no sensor noise, no rolling shutter, no motion blur, no real materials or lighting,
   no scene clutter. The video backbone (Wan2.2-TI2V-5B) has never seen a MuJoCo rendering, and
   the T-15/T-24 probes already showed both Wan and Cosmos3 hallucinate embodiment they have not
   observed. **Sim frames are not a substitute for real D2 teleop data** and must not be fed into
   the LoRA fine-tune as if they were. What the sim gives the vision path is a *plumbing* test:
   frames of the right shape, dtype and cadence, changing when the robot moves (mean |pixel delta|
   across one `execute()`: 11.9 head / 34.7 wrist).
2. **The dynamics are Menagerie's, not the robot's.** Masses, inertias, friction, damping and
   contact softness are whatever the vendor MJCF and MuJoCo defaults say. Nothing here has been
   validated against a physical G1. `SIM_KP`/`SIM_KD` are the vendor model's own design point, tuned to *this model*.
3. **No stable grasp exists yet.** Fingers make contact, the cube slips. Any "pick-and-place in
   sim" claim would be false today.
4. **The base is welded and 14 joints are locked.** No balance, no walking, no whole-body
   dynamics, no falling. This is a fixed-base manipulation rig wearing a humanoid.
5. **Camera extrinsics are invented.** `head` uses `fovy=70` (≈ the D435i's 69° *horizontal* RGB
   FoV) on a square crop, while the real sensor is 69×42°; `wrist_left` has no hardware
   counterpart at all. Neither is calibrated against anything.
6. **The `wrist_pitch` torque quirk is unmodelled behaviour, not verified behaviour.** 0.109 rad
   of steady-state error in free space, up to 0.54 rad in contact. Whether the real joint behaves
   this way is unknown.
7. **The DDS track is self-consistency, not vendor conformance.** Both sides use the same
   `unitree_sdk2py` IDL classes and the same CRC routine. No physics, no vendor RPC services, no
   timing/load characterisation, and the Dex3 topic names and RIS mode byte are documentation, not
   measurements. All 8 caveats are enumerated in `docker/dds/README.md`.
8. **The regression tests are contract tests, not physics validation.** `tests/test_mujoco_g1.py`
   (29 tests, see below) pins the seam — protocols, tick semantics, clipping, e-stop, determinism,
   frame shape — but image assertions are on **variance and shape only, never pixel values**
   (renders are not bit-portable across GL backends), and the detailed numbers in this document
   (gain sweeps, reach distances, damping curves, throughput) come from ad-hoc verification
   scripts, not from the suite. Nothing automatically re-measures them.

## Test coverage

`tests/test_mujoco_g1.py` — 29 tests, part of the normal `pytest tests/ -q` run (617 passed on
this machine). The **whole module skips with an actionable reason** when `mujoco` is not installed
(`pytest.importorskip`) or when `configs/sim/g1_scene.xml` / the fetched vendor model is missing,
so a bare install is unaffected. One scene + one renderer are built per module (loading the MJCF
~0.2 s, the offscreen GL context ~0.4 s); the per-test fixture rewinds to `ready` and clears the
e-stop.

What it pins: `MujocoG1Transport` satisfies `G1Transport` and `MujocoG1Robot` satisfies
`RobotAdapter`; the registry lists it under `optional_robots()` and constructs it; default gains
are `SIM_KP`/`SIM_KD` and the limits come from the scene's own ranges; all 29 motor slots resolve
to the `G1_JOINT_MAP` names *in order*, and a missing joint fails at construction rather than
mid-rollout; the low-state dict matches the documented contract and returns copies, not live
views; `tick_ns` advances on a motor write and never without one, and a stale tick degrades
validity **through the unmodified adapter**; `execute()` moves the commanded joint only, clips an
over-large delta to the configured limits, and rejects `EE_DELTA` and a wrong width; `hold()`
advances sim time by one control period; the e-stop latches and `clear_estop()` releases it while
`reset()` does not; gripper closure is monotonic and mirrored between hands; two freshly built
sims and two resets are bit-identical; renders have the right shape/dtype and are non-degenerate;
the welded base and the 14 locked joints do not drift over a rollout.

Added after the adversarial review, all of them regression guards for a defect that was actually
reproduced: `SIM_KD` is re-derived from the scene's own critical damping (code and config cannot
drift apart); a non-unit `gripper_vendor_min/max` is rejected at construction; **non-finite input
is rejected by both write paths** (a NaN reaching `data.ctrl` makes MuJoCo zero all 43 controls and
slam the robot to its zero pose with no exception and an advancing tick); a **failed
`emergency_damp` propagates** instead of reporting success; `estop()` is exercised **concurrently
from a second thread** (this segfaulted the interpreter 3/3 before the transport lock); the
`prefix_steps`-dependent under-execution and the zero-delta ratchet are pinned to measured bands.

## Known limitations — read before quoting any number out of this sim

### 1. Every commanded joint delta is under-executed, by a `prefix_steps`-dependent factor

`G1Adapter.execute()` integrates chunk deltas onto the **measured** `q`, re-read at the start of
every call. Within a chunk the targets accumulate correctly; **between** chunks the position
loop's lag is discarded rather than caught up. A joint therefore executes ~0.39 of a
one-control-period step within that period and ~0.95 of a 25-step chunk — see the `prefix_steps`
table above. `MockRobot` (a kinematic integrator) shows 1.000 everywhere, which is why this never
surfaced before there was contact physics behind the seam.

**This is not tunable away.** Executing ~all of a 20 ms position step inside 20 ms needs a
closed-loop bandwidth of several hundred rad/s; the scene's own effective inertias put the waist
at 35 rad/s for the vendor kp=500 and would need kp ≈ 1.6e4 N·m/rad to reach 200 rad/s. Measured:
even flat kp=4000 with critical damping only reaches mean 0.86 / min 0.56 per period. The cause is
architectural — no feed-forward and no integral action anywhere in the chain — not MuJoCo.

**What this invalidates, concretely:**

- A recorded `(state, action)` pair from this sim is **not** a commanded/achieved pair. Do not
  train an action head on sim labels and expect the magnitudes to transfer.
- Safety-intervention rates and `accel_limit` counts are **not calibrated**: `SafetyLayer` seeds
  `v_prev` from the *measured* `dq`, so on this robot the gate partly measures tracking error.
  (Substituting the last *commanded* velocity drives interventions to 0 — that would **mask** this
  limitation, not fix it. Do not do it.)
- Any velocity-envelope or "the loop closes at X Hz of real motion" claim moves when
  `prefix_steps` moves.

`tests/test_mujoco_g1.py::test_execute_under_executes_every_delta_by_a_prefix_dependent_factor`
pins the measured bands so this cannot silently drift in either direction.

**Follow-up (design change, deliberately out of scope here):** bounded feed-forward in
`G1Adapter.execute()` — carry the previous commanded target forward, clamped to a tracking window
around the measured `q` so a blocked joint cannot wind up and slam when freed. That touches the
safety-critical hardware path and needs its own review. Tracked as T-25c in `TASKS.md`.

### 2. A zero-delta chunk (and `hold()`) is not a position hold

Same root cause. `execute()`/`hold()` re-read `q`, so each cycle forgives the previous cycle's
gravity droop and the arm creeps monotonically. Max |q − keyframe| through zero-delta chunks:
**0.080 rad @ 2 s, 0.329 @ 10 s, 0.730 @ 30 s, 0.971 @ 60 s** — still growing. The **bounded**
0.0091 rad figure in the gains section is the *fixed-target* droop and describes a protocol the
runtime never uses; never quote it as this sim's hold accuracy. On hardware the vendor's gravity
compensation masks this; in sim it does not.

### 3. An e-stop is not a freeze, and it moves the sim clock

`emergency_damp()` steps `damp_duration_s` (0.2 s) of physics, so `tick_ns` jumps by 10 control
periods and the arm creeps ~0.04 rad under gravity *during* the call. Anything that timestamps on
sim time sees a discontinuity at every e-stop. `Δq` measured *after* `estop()` returns is
legitimately 0 — that is the latch, not the damping.

## Known gaps and follow-ups

- **`available_robots()` does not list `"mujoco_g1"`.** `tests/test_robot.py:246` asserts
  `available_robots() == ("g1", "mock")` and `:254-256` *constructs* every listed entry, so
  listing an optional-dependency robot there would fail on any machine without `mujoco` or the
  38 MB model. The registry therefore splits the tiers: `available_robots()` = constructible in
  any install, `optional_robots()` = `('mujoco_g1',)`. Both are named in `get_robot`'s
  unknown-name error, and `get_robot("mujoco_g1")` works either way. To merge them: extend the
  expected tuple at `:246` and make `:254` skip names whose dependency is absent.
- `MujocoG1Robot` is reachable via `get_robot` or a direct import, but is **not** exported from
  `wam/robot/__init__.py` — an eager export would defeat the lazy optional import.
- `mujoco_transport.py` imports the private `_as_motor_array` from `g1_transport.py` so that
  shape-validation error messages stay byte-identical across the three transports. Promoting it to
  a public name would be cleaner.
- `hold()` advances sim time by one control period, unlike `MockRobot.hold()`. Holding a real arm
  is physics, so this is intended, but it differs from the mock.
