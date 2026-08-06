# `wam.robot` — hardware abstraction layer

**TL;DR** — Translates between the canonical schema and one concrete robot API. Joint mapping,
units, calibration and vendor limits live **only** here (FR-06). Everything upstream stays
robot-agnostic. Torch-free, numpy + pydantic.

## Files

| File | Contains |
|------|----------|
| `registry.py` | `get_robot(name, **cfg)` — the single construction entry point |
| `mock.py` | `MockRobot` — deterministic kinematic integrator, no hardware |
| `g1.py` | `G1Adapter` — Unitree G1 upper body: mapping, units, limits, e-stop |
| `g1_transport.py` | `G1Transport` seam + `FakeG1Transport` (tests) and `DdsG1Transport` (real DDS) |
| `mujoco_transport.py` | `MujocoG1Transport` — the same seam, backed by MuJoCo physics (optional dep) |
| `mujoco_g1.py` | `MujocoG1Robot` — `G1Adapter` + MuJoCo transport + cameras, episode reset, sim clock |
| `isaac_binding.py` | `IsaacBinding` protocol + `IsaacSimBinding` (real) and `FakeIsaacBinding` (CPU) |
| `isaac_transport.py` | `IsaacG1Transport` — the same seam, backed by Isaac Sim / PhysX (optional dep) |
| `isaac_g1.py` | `IsaacG1Robot` — `G1Adapter` + Isaac transport + USD cameras, episode reset, sim clock |

## The `RobotAdapter` contract

Four methods (see `wam.interfaces.protocols`): `read_state()`, `execute(chunk, prefix_steps)`,
`hold()`, `estop()`, plus a `limits` dict in canonical order (`q_min`, `q_max`, `dq_max`,
optionally gripper bounds). `execute` runs only the **first `prefix_steps`** of a chunk — that is
what makes the receding horizon work (FR-05). The chunk must already have passed the safety layer.

Adding a robot means adding a factory in `registry.py`, never branching on names elsewhere.
Construction never touches hardware; the G1 connects lazily via `connect()`. Robots whose module
needs an optional dependency go into `_LAZY_FACTORIES` (dotted path, imported on first use) and
are listed by `optional_robots()` rather than `available_robots()` — that keeps the registry
importable and every listed robot constructible in a bare install.

## `mock.py`

An in-memory robot for pipeline tests. `q` integrates the chunk deltas with hard clipping to
`[q_min, q_max]`; `dq` is the finite difference over `dt_s`, so clipping is visible in the
velocity. **No real time passes** — per-step latency is simulated on an internal monotonic clock
(`timestamp_ns += dt_s + step_latency_s` per step), it never sleeps. Optional Gaussian noise
perturbs *reads only*; the internal state stays clean. Deterministic under a fixed seed.

`render_frames(n)` gives a synthetic camera so vision pipelines have something to chew on: flat
per-camera background plus a bright dot whose **column encodes `q[0]`** within its limits.

## `g1.py`

`G1_JOINT_MAP` is the one place canonical order meets G1 motor indices — explicit data, not code:

```
waist_yaw -> 12    left arm  -> 15..21    right arm -> 22..28      (29-DoF G1JointIndex)
```

15 canonical joints (upper body: waist + 2×7 arm), `gripper_dims=2` for `[left, right]`, each in
`[0, 1]`. Only WaistYaw is used, so the subset is also valid on the 23-DoF variant.

Behaviour worth knowing:

- **Imports without the vendor SDK.** All hardware I/O goes through the `G1Transport` seam. The
  default `DdsG1Transport` is built lazily in `connect()` and raises
  `RuntimeError("G1 hardware support requires unitree_sdk2py")` if the SDK is absent. With an
  injected transport the whole adapter runs SDK-free.
- **Stale samples degrade validity.** If the vendor tick did not advance since the previous
  `read_state()`, every validity flag is cleared — the upstream safety layer then rejects the
  state and the watchdog is not fed fresh data.
- **Defense in depth in `execute()`.** Deltas are integrated onto the *current* `q`, clipped to
  `dq_max * dt` per step and to `[q_min, q_max]` before sending. The upstream `SafetyLayer`
  stays authoritative (FR-07); this is a second wall, not the first.
- **Wall-clock pacing.** Successive step commands are sent no earlier than `t0 + i * dt_s`.
  The `dq_max * dt` clip *is* a velocity limit only if targets arrive `dt_s` apart — streaming
  them back-to-back would collapse the chunk into one large position jump at the motors.
  `clock` / `sleep` are injectable for deterministic tests.
- **Unmapped motors hold.** Legs and waist roll/pitch are commanded to their current position
  with zero gains, so the vendor controller keeps authority there.
- **E-stop latches** even without a transport attached, and latches *even if damping raises* —
  a failed damp must never leave the adapter willing to command motion again.
- `JOINT_DELTA` only. `EE_DELTA` needs an IK layer that does not exist yet (OD-02).

Limit and gain defaults in `G1Config` are **conservative placeholders pending OD-08**. Override
from a versioned config file before any real-robot use.

## `g1_transport.py` — the seam, and its three implementations

The narrow hardware boundary. Everything above it is pure, testable logic. **This is the central
design idea of the package:** `G1Adapter` is written once and exercised three ways, so unit tests,
physics and real hardware all run the *same* mapping, clipping, pacing and e-stop code.

Low-state dict contract: `q` [29] rad, `dq` [29] rad/s, `imu` (`quat_wxyz`, `gyro`, `acc`),
optional `gripper` [2] in vendor units, and `tick_ns` (a tick that does not advance = stale).

| Implementation | Backed by | Needs | Used for |
|---|---|---|---|
| `FakeG1Transport` | first-order lag | — | unit tests |
| `MujocoG1Transport` | MuJoCo physics + renderer | `mujoco` + fetched model | E2 sim, closed-loop rollouts |
| `DdsG1Transport` | CycloneDDS / vendor SDK | `unitree_sdk2py` | real robot |

- `FakeG1Transport` — first-order lag toward `q_target`, records every command, and can
  `freeze_tick` to simulate a stalled vendor controller for watchdog tests.
- `DdsG1Transport` — **implemented** (LowState/LowCmd IDL mapping, CRC, `rt/lowcmd` or
  `rt/arm_sdk`, Dex3 hand topics). The constructor stores config only and never imports the SDK;
  without `unitree_sdk2py` every method raises `RuntimeError`. Verified against a fake G1 on a
  real CycloneDDS bus in an arm64 container — 11/11 checks, `docker/dds/README.md`. Not yet run
  against hardware: the Dex3 mapping and the vendor limits/gains stay placeholders (OD-08).
- `MujocoG1Transport` — position actuators driven from `kp`/`kd` (gains written into
  `actuator_gainprm`/`biasprm`), 10 × 2 ms substeps per 20 ms control period, `tick_ns` advancing
  only on motor writes. Motors resolved **by name** and cross-checked against `G1_JOINT_MAP`.
  `mujoco` is imported lazily, so this module imports fine without it. **Non-finite input is
  rejected by both write paths** (a NaN reaching `data.ctrl` makes MuJoCo zero all 43 controls and
  slam the robot to its zero pose, undetectably). **`emergency_damp()` propagates a failed damp**
  after recording it on `last_damp_error` — matching the other two transports and
  `G1Adapter.estop()`'s contract. **Thread-safe:** `MjModel`/`MjData` are not, so every access is
  serialised behind the public re-entrant `MujocoG1Transport.lock`, which is what makes
  `RobotAdapter.estop()`'s "safe from any thread" true here; anything outside the class that
  touches `.model`/`.data` must hold it too.

## `mujoco_g1.py`

`MujocoG1Robot` **composes** `G1Adapter` with a `MujocoG1Transport` and forwards the four protocol
methods verbatim — deliberately not a subclass, which would inherit `connect()`'s `DdsG1Transport`
fallback and could override the safety code this class exists to exercise unchanged. On top it
adds `render_frames(n)` (cameras `head` / `wrist_left`, one lazily-built reused `mujoco.Renderer`;
rendering never steps physics), `reset()` (episode reset — a latched e-stop survives it),
`clear_estop()`, `sim_time_ns` and `close()`.

Three defaults differ from `G1Config`, on purpose: gains are `SIM_KP` (500, the vendor Menagerie
class stiffness) with `SIM_KD` (per-joint *critical* damping, re-derivable from the scene — a flat
kd is the wrong shape when the wrist's effective inertia is 1/36 of the waist's), `dq_max` is
`SIM_DQ_MAX` (mirroring `configs/robot/mujoco_g1.yaml`, so the no-config path is not looser than
the versioned file), and `q_min`/`q_max` come from the scene's own `jnt_range`. `G1Config`'s
`gripper_vendor_min/max` must stay `(0.0, 1.0)` — the Dex3 synergy fraction *is* the vendor unit
here — and `__init__` rejects anything else. Pacing runs on the **sim clock**, so `execute()`'s
`dq_max·dt` velocity clip means what it means on hardware while the rollout runs faster than
realtime.

**Formerly a known limitation, fixed by T-25c:** `G1Adapter.execute()` re-based on the measured
`q`, so every commanded delta was under-executed by a `prefix_steps`-dependent factor (mean 0.39
of a one-control-period step). The bounded feed-forward (`q_track_window`, 0.05 rad in
`configs/robot/mujoco_g1.yaml`) carries the previous commanded target forward and reaches 0.987 at
every prefix. It is **off by default and off on the hardware config** — the window has to exceed
the tracking error of the gains in use, and `g1.yaml`'s are OD-08 placeholders. See "Known
limitations" in `docs/sim.md`.

`tests/test_mujoco_g1.py` runs the seam contract one layer below `tests/test_g1.py` — same
adapter, real physics instead of the kinematic lag. The whole module skips with an actionable
reason when `mujoco` or the fetched model is absent.

Setup, scene layout, measured numbers and the honest limits: `docs/sim.md`.

## `isaac_binding.py`, `isaac_transport.py`, `isaac_g1.py`

The fourth `G1Transport` (after `FakeG1Transport`, `DdsG1Transport` and `MujocoG1Transport`) and
the same composition shape as `mujoco_g1.py`: `IsaacG1Robot` **holds** a `G1Adapter`, forwards the
four protocol methods, and adds `render_frames`, `reset()`, `clear_estop()`, `sim_time_ns`,
`close()`.

One extra layer that MuJoCo does not need. `IsaacBinding` is a 17-method protocol between the
transport and Omniverse, with two implementations: `IsaacSimBinding` (the real one — **written
against NVIDIA's documentation and never executed**, because Isaac Sim runs on Linux + an NVIDIA
GPU) and `FakeIsaacBinding` (a caricature PD integrator that runs anywhere). Everything above the
binding — the transport, the robot, and the unmodified `G1Adapter` driving them — is therefore
testable on a Mac with no GPU, which is what `tests/test_isaac_g1.py` and
`tests/test_isaac_binding.py` do. What the fake cannot test, `scripts/preflight_isaac.py` tests on
the box: it is the gate that turns each documented assumption into pass/fail before a rollout.

**Three things about this backend are not like the others, and none of them are papered over:**

1. **The e-stop is not at parity with hardware.** The Omniverse API is main-thread-only, so
   `emergency_damp()` from a watchdog thread only *latches*, in pure Python; the damping is
   applied by a `PHYSICS_PRE_STEP` callback at the next physics step. The latch is also what
   stops the control loop stepping — so in the ordinary case that step never comes and the gains
   are never lowered at all. Safe in sim (a stopped clock is a stopped arm) and exactly what an
   e-stop must not do on a robot. `damp_applied_count` / `is_damping` / `last_damp_error` report
   the truth; read them.
2. **It runs in a different python.** `isaacsim-core` 6.0.1 pins torch 2.11.0 and this repo's
   lock resolves 2.13.0, so the Isaac side is deliberately torch-free at module scope (asserted
   in a subprocess) and the supported topology is `rollout.py --robot isaac_g1 --policy remote`
   in the Isaac venv against `scripts/serve_policy.py` in this one.
3. **The gains in `configs/robot/isaac_g1.yaml` are unmeasured.** They are Isaac Lab's published
   G1 magnitudes and disagree with the MuJoCo rig's measured ones by an order of magnitude. An
   Isaac number and a MuJoCo number are not comparable until someone measures on the box.

Setup, the preflight's 31 checks, and the full list of ways this differs from the MuJoCo backend:
`docs/isaac.md`.
