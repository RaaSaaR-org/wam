# Architecture — how WAM fits together

**TL;DR** — One vertical path turns pixels, proprioception and a sentence into joint deltas, and a
deterministic gate sits between the learned part and the robot. Everything interesting about this
codebase is at the **seams**: five structural protocols that let the backbone, the robot and the
hardware transport be swapped without a single call site changing. The other half of the story is
the **control cycle**, which has four possible endings — and only one of them moves the robot.

Read this after `README.md`, before any module `README`. Module detail:
[`src/wam/*/README.md`](../src/wam) · Sim: [`sim.md`](sim.md) · Order of work:
[`ROADMAP.md`](ROADMAP.md).

---

## 1. The runtime path

![WAM runtime architecture: camera frames, robot state and a language instruction enter a policy in which a state encoder, a swappable video backbone and an action head produce a canonical action chunk; a deterministic safety layer filters every chunk before the G1 adapter executes only its first steps; and one G1Transport seam carries that same code onto a fake robot, MuJoCo physics or the real Unitree G1 over DDS.](img/wam-architecture.png)

| Stage | Type | What it owns |
|-------|------|--------------|
| `Observation` | `interfaces.protocols` | `{camera: HxWxC array}`, a canonical `RobotState`, the instruction string |
| `StateMLP` | `encoders` | `concat(q, dq, imu, gripper)` → embedding; owns the **missing-sensor** case |
| `WanFlowBackbone` | `backbones` | Wan2.2-TI2V-5B DiT + VAE + umT5, LoRA on top; emits **intermediate** features |
| `ActionHead` | `decoders` | features → `ActionChunk`; tanh/sigmoid-bounded, mean-pools token dims |
| `SafetyLayer` | `safety` | limits, projection, rejection, watchdog. No ML, no randomness |
| `G1Adapter` | `robot` | canonical ↔ vendor units, joint mapping, calibration, latched e-stop |
| `G1Transport` | `robot` | the one hardware seam: fake / MuJoCo / DDS |

Three properties of that path are load-bearing and easy to miss:

**Units are physical, everywhere.** `ActionHead` is tanh-bounded to `(-1, 1)` and those numbers
*are* per-step radians — the MVP pipeline is identity-normalized end to end, there is no
denormalization step anywhere. `SafetyConfig` limits are in the same units, so the safety layer
compares directly. The cost: training data must keep per-step `|targets| < 1`, which
`EpisodeDataset` enforces rather than letting it fail silently. `NormalizationSpec` exists in the
schema but is parked; a non-identity spec makes the dataset refuse the episode.

**A missing sensor is not a crash.** `RobotState` carries a `ValidityMask` per field group, and
`StateMLP` substitutes a learned "missing" embedding via `torch.where` — never a multiply, so
`NaN`/`Inf` inside a masked-out group cannot poison the gradient.

**The action decoder reads intermediate activations, not pixels.** The video branch and the action
branch share a forward pass; `features()` returns `[B, S, feature_dim]` with video tokens first,
and `num_video_tokens()` tells the action branch where its own tokens start.

### The two small trainable modules

Everything on that path is either frozen, deterministic, or vendor code — except two MLPs. They are
small enough to describe in one picture each, and both encode a decision worth knowing.

![The WAM state encoder takes four proprioception groups — joint positions, joint velocities, IMU and gripper — each carrying its own validity flag; any group flagged invalid is swapped for a learned missing embedding by torch.where before the groups are concatenated and run through a small MLP, so a dead sensor produces an embedding instead of a crash and its NaN values reach neither the output nor the gradient.](img/wam-state-mlp.png)

`StateMLP` (`encoders/state_mlp.py`) is a 2–4 layer MLP over `concat(q, dq, imu, gripper)`. The
interesting part is the front door: each group owns a learned "missing" parameter of that group's
raw width, and `torch.where` picks it whenever the validity flag says the group is untrustworthy.
`where` rather than a multiply is the whole point — a mask-and-multiply would still touch the
invalid values, and `0 · NaN` is `NaN`. The inference path (`encode`) never even reads a group
flagged invalid, so garbage or `None` sitting in that field is tolerated.

![The WAM action head takes one pooled backbone feature vector, runs it through a small MLP trunk and splits into two bounded heads: a tanh target head emitting 8 to 32 per-step joint deltas and a sigmoid gripper head emitting a 0 to 1 command per step. Together they form one action chunk, of which the closed loop executes only the first few steps. The bounded tanh output is already in radians — no denormalization step exists anywhere.](img/wam-action-head.png)

`ActionHead` (`decoders/action_head.py`) is a trunk plus two bounded heads: `tanh` for the targets,
`sigmoid` for the gripper. One forward pass emits the **entire** chunk — `num_steps × target_dim`
values, 8–32 steps per PRD 9.10 — not one step at a time. `decode` mean-pools any leading dimensions
(backbone tokens) into a single feature vector and reduces the per-step gripper across
`gripper_dims` to the one scalar the canonical chunk carries.

Its bounds are **soft**. `(-1, 1)` is a shape constraint on the output distribution, not a safety
guarantee: hard limits live downstream in the safety layer, and the chunk is only useful because
those bounded numbers are read directly as radians (see the unit contract above).

---

## 2. The seams

Every swappable module is a `@runtime_checkable` `Protocol` in `wam.interfaces.protocols`
(`INTERFACES_VERSION = "0.3.0"`) — structural, so conformance is `isinstance`, never inheritance.
The protocol module is torch-free: tensors are typed `Any` with a documented contract ("last
dimension is the feature dim"), and anything crossing the robot/safety boundary is numpy.

| Seam | Implementations | Constructed by |
|------|-----------------|----------------|
| `BackboneAdapter` / `FlowBackbone` | `tiny` · `wan_i2v` · `flux3` | `get_backbone(name, **cfg)` |
| `SafetyFilter` | `SafetyLayer` — exactly one, on purpose | direct |
| `RobotAdapter` | `mock` · `g1` · `mujoco_g1` | `get_robot(name, **cfg)` |
| `G1Transport` | `FakeG1Transport` · `MujocoG1Transport` · `DdsG1Transport` | injected into `G1Adapter` |
| `Policy` | `CheckpointPolicy` (local) · `RemotePolicy` (WebSocket) | direct |

Two of these carry a design decision worth stating out loud.

### One hardware seam, three implementations

`G1Transport` is the *only* line where robot-specific I/O lives. Above it — joint mapping, unit
conversion, defense-in-depth clipping, pacing, the latched e-stop — the code is byte-identical
whether it is talking to a first-order-lag fake, to MuJoCo contact physics, or to a real vendor DDS
bus. There is deliberately **no** `MujocoAdapter`: the simulator buys coverage of the *production*
adapter, not of a sim-only twin. Same for the registry — adding a robot means adding a factory in
`robot/registry.py`, never branching on a name somewhere else. Optional-dependency robots
(`mujoco_g1`) are registered by dotted path and imported on first use, so a bare install still
imports and constructs everything `available_robots()` reports.

### The module boundary is drawn around what trains

`JointWorldActionModel` depends on the `FlowBackbone` protocol, never on a concrete backbone. And
`WanFlowBackbone` registers exactly two things as parameters: the state→text-context projection,
and an `nn.ParameterDict` **aliasing** the LoRA parameters peft injected into the DiT. The 5B DiT,
the VAE and the text tower are held as plain attributes, so they never enter `state_dict()`,
`parameters()` or `.to()`.

That is not a micro-optimization. It makes "checkpoint the adapter, not the 5B model" a *structural*
property instead of a flag someone has to remember: `state_dict()` stays small, the training
monitor's parameter snapshots stay cheap, and the authoritative artifact remains peft's own adapter
directory — which loads into a stock diffusers Wan pipeline, so a checkpoint can be watched as video
without any of WAM. The price is one override (`_apply` forwards **device** moves only, never dtype,
then re-points the aliases) — documented at the top of `backbones/wan_flow.py`.

The flow convention is likewise WAM's, not any backbone's: `x_t = (1-t)·x0 + t·x1`, `v = x1 - x0`,
`t = 1` is clean. Wan counts denoising steps *downwards* from 1000, so its adapter flips both the
timestep and the velocity sign internally. Nothing outside the adapter may learn about that.

---

## 3. One control cycle

![One WAM control cycle has four possible endings: a stale loop trips the watchdog into hold or e-stop, a late prediction is discarded and the robot holds, an unusable state produces a zero-delta hold chunk that deliberately does not feed the watchdog, and only a clean cycle executes the first few steps of the safety-filtered chunk before the unexecuted remainder is thrown away.](img/wam-control-cycle.png)

`ClosedLoopExecutor.run_rollout` walks three gates in a fixed order. The gates are cheap; what makes
them subtle is **who feeds the watchdog**.

| Ending | What the robot gets | Watchdog fed? |
|--------|---------------------|---------------|
| Watchdog already expired | `hold()` or `estop()` per its configured action | re-armed, but **only after** the safe state was commanded |
| Prediction later than `policy_deadline_ms` (default 500 ms) | `hold()`; the fresh chunk is discarded unexecuted | **no** — so chronic lateness eventually trips it |
| Safety filter rejects (`nan_reject`, `schema_reject`, `state_reject`) | the layer's zero-delta HOLD chunk | **no** — stale robot data must not keep it armed |
| Clean | `execute(safe_chunk, prefix_steps)` | yes |

A late prediction is never executed late. That is the whole point of the deadline: a stale action is
worse than no action, so the chunk is thrown away and the robot holds.

**The two clocks.** The watchdog arms and expires on **robot time** (`state.timestamp_ns` —
simulated for the mock, wall-clock-ish on hardware). Policy latency and the reject streak are timed
on the **host** clock (`time.monotonic`). This split exists because of one failure mode: a stalled
robot freezes its own timestamps, so a watchdog running purely on robot time would never expire
while the robot serves the same frozen state forever. The executor therefore measures uninterrupted
safety rejections on the host clock and escalates to HOLD or STOP once the streak passes the
watchdog timeout.

**Why only a prefix.** Every cycle predicts a fresh chunk from a fresh observation, executes its
first `prefix_steps`, and **discards the remainder** (FR-05). The next observation is more
informative than the tail of a chunk planned one cycle ago, so that tail is never worth running. The
robot only ever sees prefixes; no stale chunk tail survives a re-plan.

![The WAM closed loop, one cycle: a fresh observation of camera frames and joint state reaches the policy, one forward pass fills a whole horizon of sixteen action steps, the deterministic safety layer clips the three steps that exceed the limits, only the first steps of the chunk are executed, and the untouched tail is discarded when the loop goes back for a fresh observation.](anim/receding-horizon-poster.png)

That rhythm is easier to watch than to read. `anim/receding-horizon.html` plays the same figure as a
ten-second loop — open it locally; GitHub renders the poster above instead. Both are generated; see
`anim/README.md`.

Every cycle writes one `control_cycle` JSONL record and every rollout one `rollout_summary`, both
stamped with `run_id` + `config_hash` — that is how a rollout stays traceable to checkpoint,
dataset snapshot and config (AC-04). The MVP floor is 2 Hz; below that, `RolloutResult` carries
`below_min_policy_rate`.

---

## 4. What the safety layer actually does

Not a clamp on the final numbers — a **step-wise, per-joint projection** in a fixed order, so the
limits hold simultaneously rather than fighting each other:

```
per step t, per joint j:
  v      = delta / dt
  v_acc  = v_prev + clip(v - v_prev, ±ddq_max·dt)     accel_limit
  v_lim  = clip(v_acc, ±dq_max)                       velocity_limit
  q_next = clip(q + v_lim·dt, q_min, q_max)           joint_limit
  step   = clip(q_next - q, ±dq_max·dt)               joint_limit_recovery
```

The last line is the non-obvious one. If the robot starts **outside** its position limits —
overtravel, miscalibration, someone moved the arm by hand — the position clamp alone would snap it
back to the boundary in a single arbitrarily fast step. Re-clipping to the velocity limit turns that
into a ramp back at legal speed over several steps. It is a no-op whenever the running position is
inside the limits. EE mode has the same shape (`workspace` → `workspace_recovery`), plus one honest
gap: without a forward-kinematics callable, only per-step translation magnitude is bounded and the
chunk carries a `workspace_skipped` intervention. Rotation deltas pass through unmodified in v0.

Unusable input never gets projected, only replaced: `NaN`/`Inf`, a wrong shape, or a state that
cannot be integrated produce a **single-step zero-delta HOLD chunk**. And every modification —
each kind, each step, each affected joint — is emitted as a `SafetyIntervention` and logged. `filter`
is pure with respect to its inputs; the only internal mutation is an intervention counter.

---

## 5. Where the code lives

| Package | Torch? | Contents |
|---------|--------|----------|
| `interfaces/` | no | `schema.py` (canonical types), `protocols.py` (the seams), `versioning.py` (config hashes, provenance) |
| `backbones/` | lazily | `registry.py`, `tiny.py`, `wan_i2v.py`, `wan_flow.py`, `flux3.py` |
| `encoders/` · `decoders/` | yes | `StateMLP`, `ActionChunkEncoder` (training only) · `ActionHead` |
| `safety/` | **no** | `config.py`, `layer.py`, `watchdog.py` |
| `robot/` | no | `registry.py`, `g1.py`, `g1_transport.py`, `mujoco_transport.py`, `mujoco_g1.py`, `mock.py` |
| `data/` | no | episode format, `SyncRecorder`, replay, the validation gates |
| `training/` | yes | `losses.py`, `datasets.py`, `action_only.py` (M2 baseline), `joint.py` (T-16), `monitor.py` |
| `runtime/` | only `policies.py` | `executor.py`, `server.py`, `client.py`, `wire.py`, `mock_loop.py` |
| `evaluation/` | no | E1 offline replay, E2 checks, AC-07 ablation, the acceptance harness |

The torch column is a real constraint, not trivia: the safety layer, the executor, the robot HAL and
the whole evaluation ladder import and run in an environment with no torch at all.

---

## 6. What this design buys — and what it does not

It buys **substitution without edits**: the same executor drives a dummy policy, a local checkpoint
and a remote WebSocket policy; the same `G1Adapter` drives a fake, MuJoCo and a real DDS bus; the
same trainer accepts the self-contained `tiny` backbone and a 5B DiT. That is why M0–M4 could be finished
and tested before a robot existed.

It does not buy correctness of the *idea*. The open claim is AC-07 — that predicting video helps the
action head at all — and the LoRA fine-tune it rested on has now run and come back **negative**:
WAM-Bench L0, `skill_vs_repeat_pct` −32.4 % **tiled** — measured on one frame repeated nine times —
i.e. losing to repeat-last-action. One of the three confounds under that verdict has since been
measured: T-29 re-scored the same checkpoint with a real frame window and the gap narrows to
**−21.80 %**, +10.65 pp. The confound was real and worth about a third of the gap, and it did not
come close to closing it: the verdict survives, the published figure does not. Two confounds remain
open — T-30 (flow readout) is running now, T-32 (data scaling) is unrun — and only
`t16-lora-seed0` has been re-scored, so the baselines it is held against are still tiled-only
numbers and AC-07 is back to *undetermined* rather than answered. The honest reading is now
"negative in distribution, and still through a readout training never deployed". See
`docs/benchmark.md` and the "Not proven" table in `README.md` before quoting anything from here as
evidence.
