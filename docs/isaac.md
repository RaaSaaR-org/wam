# Isaac Sim on the 5090 — the G1 robot backend (E2)

For the same box as `docs/local_gpu.md`, doing the one thing that page cannot: driving the closed
loop against **NVIDIA Isaac Sim 6.0.1** instead of MuJoCo. Read `docs/local_gpu.md` §0 first — the
WAM venv, the torch build and `scripts/preflight_gpu.py` are set up there and not repeated here.
This page only covers what is different, which is: **a second interpreter**.

> **Nothing on this page has ever run.** `src/wam/robot/isaac_binding.py`,
> `src/wam/robot/isaac_transport.py`, `src/wam/robot/isaac_g1.py` and
> `configs/robot/isaac_g1.yaml` were written against NVIDIA's *documentation*, on a Mac with no
> Isaac Sim and no GPU. Every module path, symbol name, argument name and return shape in the
> Isaac half is an assumption. `scripts/preflight_isaac.py` is the thing that converts each
> assumption into a pass or a fail on your box, and it is §2 for a reason. §6 lists exactly which
> command breaks first for each assumption that turns out to be wrong.

What *is* executed and tested, on a laptop with no GPU: everything driven by `FakeIsaacBinding` —
the physics-step tick, the 43-DoF name resolution, the gain round-trip including `kp=0`, the camera
warmup, the wedged main thread (`tests/test_isaac_binding.py`), and the transport and robot layers
on top of it — the tick-equality staleness path, the e-stop latch and its `PHYSICS_PRE_STEP` drain,
the grasp synergy and its readback (`tests/test_isaac_g1.py`). That fake exists so that everything
above the seam is genuinely testable even though the vendor half cannot be. Run it first, anywhere:

```bash
.venv/bin/python -m pytest tests/ -k isaac -q     # the one thing on this page you can check
```

---

## 0. Why two venvs — the pin, in full

**One venv cannot hold both.** This is not a preference, it is a version conflict with no overlap:

| | pins torch to | how well that is established |
|---|---|---|
| `isaacsim-core==6.0.1.0` | **2.11.0** | **not confirmed from an NVIDIA-hosted page.** It is the researched pin this backend was designed around, recorded in `src/wam/robot/isaac_binding.py`'s module docstring and in `ISAAC_MISSING_MSG`. NVIDIA's own Isaac Lab pip page still documents Isaac Sim **5.1.0** with torch **2.7.0** as of 2026-08-05, and the 6.0 figure came from a third-party doc mirror. Nobody here has read the wheel metadata |
| this repo's `uv.lock` | **2.13.0**, from PyPI | verified by reading the file — `uv.lock:1969-1971`, `name = "torch"` / `version = "2.13.0"` / `source = { registry = "https://pypi.org/simple" }` |

**Confirm the left-hand row on the box** — one line after the install in §1:
`.venv-isaac/bin/pip show torch`. The conclusion survives being wrong about the exact number: the
two venvs are needed unless Isaac's pin turns out to be *exactly* 2.13.0. If it is, say so and this
whole page collapses into one venv.

Installing one over the other gives you a broken half either way: Isaac's tensor backend against a
torch it was not built for, or the Wan backbone against a torch two minor versions back. There is
also a second, independent conflict, and this one *is* confirmed on NVIDIA's own docs — **Isaac Sim
6.x requires Python 3.12** (5.1 wanted 3.11), and `preflight_isaac.py` check A fails a non-3.12
interpreter for that reason.

So the box runs **two interpreters**, and the split is drawn at the seam that already exists and is
already tested (T-20, `docs/local_gpu.md` §4):

```
.venv          (WAM venv,   torch 2.13)      scripts/serve_policy.py    -> ws://127.0.0.1:8765
.venv-isaac    (Isaac venv, Isaac's torch)   scripts/rollout.py --policy remote --server-uri ...
```

The consequence for the code, and it is enforced: **`isaac_binding.py`, `isaac_transport.py` and
`isaac_g1.py` import numpy and nothing heavier.** No torch, directly or transitively. Proved in a
subprocess rather than by trusting an import list, by two tests covering different amounts of the
claim: `tests/test_isaac_binding.py` imports the binding alone, `tests/test_isaac_g1.py` imports all
three and drives a whole `IsaacG1Robot` through read / hold / render / estop.
Measured here on 2026-08-05, on the Mac, with `torch` blocked at the import hook:
`wam.robot.isaac_binding`, `wam.robot.isaac_transport`, `wam.robot.isaac_g1`, `wam.runtime.client`
(`RemotePolicy`) and `scripts/rollout.py` all import clean and `torch` never lands in `sys.modules`.
That is what makes the Isaac venv able to run the rollout at all.

---

## 1. Install Isaac Sim 6.0.1 into its own venv

NVIDIA's stated requirements for Isaac Sim (read from their requirements page on 2026-08-05,
x86_64 table): **16 GB VRAM minimum** ("GPUs with less than 16GB VRAM may be insufficient to run a
complex scene"), 32 GB RAM minimum / 64 GB recommended, 50 GB SSD minimum / 500 GB recommended,
Ubuntu 22.04 or 24.04, Linux driver **595.58.03**. A 5090 is 32 GiB, which clears the VRAM minimum
on its own — but not while `serve_policy.py` is also resident. That is §3, and it is the constraint
that actually bites.

```bash
# a SECOND venv, on python 3.12 exactly — not the one docs/local_gpu.md §0 built
python3.12 -m venv .venv-isaac
. .venv-isaac/bin/activate
pip install --upgrade pip                        # NVIDIA's docs ask for this first

# Isaac Sim itself — NVIDIA's documented install line for 6.0.1, verbatim
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com

# what torch did that pull? THIS is the answer to §0's open question — record it.
pip show torch

# and WAM itself. The base dependency set is numpy + pydantic + pyyaml + typing-extensions —
# no torch — and `serve` adds websockets, which --policy remote needs.
pip install -e '.[serve]'
```

**Let Isaac resolve its own torch; do not pre-install one.** The version above is exactly the thing
§0 could not confirm, so pinning it by hand would be guessing where the resolver knows. If the
wheel Isaac pulls has no sm_120 kernels you will find out at the first kernel launch with
`no kernel image is available for execution on the device` — the fix is then to reinstall **the same
version** from the cu128 index (`pip install torch==<what pip show said> --index-url
https://download.pytorch.org/whl/cu128`), which is the same reasoning `docs/local_gpu.md` §0a
applies to the WAM venv.

**Do not run `uv sync` in this checkout, in either venv.** `docs/local_gpu.md` §0a explains why for
the WAM venv (it reconciles the environment to `uv.lock` and will remove or replace a hand-installed
torch); in the Isaac venv the same mechanism would replace Isaac's torch with the locked 2.13.0 and
break Isaac instead.

**On `isaac-python`.** `scripts/preflight_isaac.py` and `isaac_binding.ISAAC_MISSING_MSG` both say
to run things with "Isaac Sim's own interpreter (`isaac-python`)". With the pip install above, that
interpreter *is* `.venv-isaac/bin/python` — there is no separate binary to find. `isaac-python` /
`./python.sh` is what the archive and container installs ship; if you installed that way, substitute
it everywhere this page says `.venv-isaac/bin/python`. Either way the rule is the same one those
messages are protecting: **never run the Isaac side out of `.venv`.**

---

## 2. `scripts/preflight_isaac.py` — run this FIRST, and read the JSON

```bash
.venv-isaac/bin/python scripts/preflight_isaac.py --out runs/preflight/isaac.json
```

Exit code is 0 iff every check passed. It is the Isaac equivalent of `preflight_gpu.py`, with one
difference in kind: `preflight_gpu.py` checks *your box*, this one checks *NVIDIA's API against
what we assumed it was*. It deliberately does **not** import `wam.robot.isaac_transport` — that
would conflate "is the vendor API what we think" with "is our code correct against it", and only
the first question can be answered here. Flags: `--asset`, `--hz` (default 500), `--device`
(default `cuda:0`), `--camera-hw H W` (default 256 256), `--warmup-frames` (default 20), `--out`,
`--gui` (default is headless).

Every result is printed as `[PASS]`/`[FAIL]` with `flush=True` as it happens, and the same list is
written to `--out` as `{"checks": [{name, ok, detail}], "info": {...}}`. **The report is written
even when checks fail** — but only for checks that *return* false. A vendor call that *raises*
(rather than returning something wrong) propagates out of `main()` and takes the report with it;
two spots are hardened against that on purpose (the `isaacsim` import itself, and the callback
registration), the rest are not. If you get a traceback and no JSON, the last printed line is your
result.

What it checks, in the order it runs them:

| # | check names | what a failure means |
|---|---|---|
| A | `python_is_3_12`, `platform_is_linux` | wrong interpreter — stop, nothing below is meaningful |
| M | `main_thread_guard_discriminates`, `preflight_runs_on_main_thread` | the identity test the binding uses to *refuse* an off-main-thread call. Runs before Isaac loads, so it is free |
| — | `isaacsim_importable`, `simulation_app_started` | you are in the wrong venv, or the install is broken. Reported as a check, not a traceback, because it is the most likely state of a fresh box |
| D | `api_surface_complete`, `physics_pre_step_event` | a symbol moved between Isaac releases. Named one by one so the message points at the symbol, not at a bare `ImportError`. `physics_pre_step_event` is the e-stop drain point (§5) |
| E | `hz_is_exact`, `physics_dt_round_trips` | `PhysxScene.set_dt` does `steps_per_second = int(1.0 / dt)` and a float round-trip can land on 499 when you asked for 500, silently. Also the only safe moment to set rates: changing them after `play()` is a known fatal crash in 6.0 |
| F | `asset_root_resolves`, `num_dofs_is_43` | asset streaming is unreachable, or you loaded a different G1. 43 = 29 body motors + 2 × 7 Dex3-1 finger joints |
| G | `body_joints_resolve`, `left_fingers_resolve`, `right_fingers_resolve` | **this is the discovery check — see below** |
| H | `tick_is_integer`, `tick_advances_exactly_7`, `tick_frozen_on_zero_steps` | `G1Adapter.read_state()` decides staleness by *equality* against the previous tick. A float clock makes that comparison meaningless and the watchdog fires at random. `numbers.Integral`, not `int`: a pybind counter may surface as `numpy.int64`, which is exact under `==`; a float must fail, and does |
| L | `pre_step_callback_registers`, `pre_step_callback_fires_once_per_step`, `pre_step_callback_runs_on_the_main_thread` | **the e-stop would be a silent no-op** (§5). Nothing else in the system would notice, which is why it is a preflight check and not a unit test |
| I | `render_advances_no_physics` | the adapter owns the clock. A render that steps behind its back corrupts staleness detection and stops the `dq_max * dt` clip from being a velocity limit |
| J | `gains_round_trip`, `zero_kp_accepted` | the caller owns the gains. `kp=0` is the e-stop damping mode and must not be clamped to a floor. This is the check that says the backend can stay raw Isaac Sim rather than Isaac Lab |
| G′ | `same_process_determinism` | recorded as a measured max-abs delta in rad, not just a boolean. Same-process only — NVIDIA makes no cross-machine guarantee, and the rollout manifest should carry the weaker claim |
| K | `camera_returns_a_frame`, `camera_dtype_uint8`, `camera_shape`, `camera_frame_is_not_blank` | the first frames come back as `None`, not black — up to 20 of them in NVIDIA's own test. Code that does not gate on `is not None` records **black frames into a dataset**, which is worse than a crash: they pass the T-11 data-quality gates and poison training silently |

### The part that is discovery, not verification: `info.dof_names`

Two things are genuinely unknown until this script runs, and the script *records* them rather than
asserting them:

- **`info.dof_names`** — the full 43-entry list, in PhysX's own order. PhysX walks joints
  breadth-first from the base link, which is neither URDF order nor
  `wam.robot.g1_transport.G1_MOTOR_JOINT_NAMES` order. `G1Adapter` gathers by hard-coded index, so
  a positional guess produces a plausible-looking robot **moving the wrong arm**, undetectable
  without hardware. This is why `resolve_g1_dof_indices()` resolves every one of the 43 by name and
  why nothing else in the backend is allowed to re-implement it.
- **`info.body_joint_pattern` / `info.left_finger_pattern` / `info.right_finger_pattern`** — which
  naming convention the shipped USD actually uses. The candidates tried are `{name}_joint` then
  `{name}` for the body, and `{side}_hand_{finger}_joint` → `{side}_hand_{finger}` →
  `{side}_{finger}_joint` for the fingers. **A mismatch here is a finding about the asset, not a
  bug in the logic**, which is why check G reports the near-miss list instead of raising — but the
  *repair* is an edit to two Python files, not to a YAML: append the observed convention to
  `BODY_NAME_CANDIDATES` / `FINGER_NAME_CANDIDATES` in **both** `src/wam/robot/isaac_binding.py`
  and `scripts/preflight_isaac.py` (nothing under `configs/` has a knob for this). Never fall back
  to positional indexing, which is what `resolve_g1_dof_indices` refuses to do for you.

The candidate lists are duplicated **on purpose** between `scripts/preflight_isaac.py` and
`src/wam/robot/isaac_binding.py`, with a test asserting the two copies are identical. The preflight
must not depend on what the library believes — testing the vendor API is its entire job. Keep that
property if you edit either list.

Read `info.dof_names` out of the report even on a full pass. It is the only record of what the
asset actually contains, and diffing it after an asset upgrade is how a silent re-ordering gets
caught.

---

## 3. The two-process topology, and what the GPU has to hold

Same box, same GPU, two processes:

```bash
# terminal 1 — WAM venv. Holds the weights. --offload-text is not optional here (see below).
. .venv/bin/activate
python scripts/serve_policy.py --joint --offload-text \
    --checkpoint runs/t16-lora-seed0/checkpoints/step-020000/model.safetensors \
    --backbone-source /path/to/Wan2.2-TI2V-5B --device cuda --policy-camera persp
# prints: serving ws://127.0.0.1:8765

# terminal 2 — Isaac venv. Holds Isaac Sim. No weights, no torch on the WAM side of anything.
. .venv-isaac/bin/activate
python scripts/rollout.py --robot isaac_g1 --policy remote --server-uri ws://127.0.0.1:8765 \
    --instruction "move the apple to the plate" --rollouts 1
```

`serve_policy.py --host` defaults to `127.0.0.1`, which is right for two processes on one box —
you do **not** want `0.0.0.0` here.

**The camera name is chosen on the server side.** `rollout.py --policy-camera` only applies to
`--policy checkpoint|joint`; on the remote path the rollout ships the whole `Observation.images`
dict and the served policy indexes it by *its* camera key. So `serve_policy.py --policy-camera
persp` must name one of `configs/robot/isaac_g1.yaml`'s `sim.cameras` — which, with the bare
`g1.usd`, is `persp` and nothing else (`sim.camera_prims` is where that name is bound to
`/OmniverseKit_Persp`). There is no Isaac equivalent of `configs/sim/g1_scene.xml` yet: no table,
no cube, no head/wrist cameras, and the viewport camera is a safe default precisely because it is a
poor observation. Render size comes from `sim.image_hw`; `rollout.py --image-hw` is the CLI
override and its help text names `--robot mujoco_g1` only, so check it applies before relying on it.

### The VRAM arithmetic — and why `--offload-text` is mandatory on this topology

Decimal GB, against a 32 GiB card. `docs/local_gpu.md` §0c is the source for the WAM-side rows and
for the two correction terms (allocator slack **+0.90 measured**, CUDA context **+0.80 estimated**);
the board total as `nvidia-smi` reports it is taken there as **34.19**, itself unverified.

| resident | GB | basis |
|---|---|---|
| `serve_policy.py`, umT5 resident | ~26.0 occupied | ~24.28 allocated (**estimated** — inferred from the smoke job's H200 peak; no artifact in `runs/` carries a `peak_vram_gb` for this entry point) + 1.70 correction |
| `serve_policy.py --offload-text` | ~14.6 occupied | 24.28 − 11.36 (the umT5 tower) + 1.70 |
| Isaac Sim | ≥16.0 | **the weakest row here.** It is NVIDIA's stated system-requirement *minimum for the card*, not a measured footprint of this process on a bare G1 stage — nobody has measured that, and it is plausibly lower |

**Both processes at once, with the tower resident: ~42 GB against a ~34.19 GB board. It does not
fit.** With `--offload-text`: ~30.6 GB, about 3.6 GB spare — and that margin is smaller than the
error bar on all three terms, so run headless (`sim.headless: true`, which is the config default)
and do not also have a desktop compositor on this card (0.3–1.5 GB, per `local_gpu.md` §0c). Watch
it rather than trusting it: `nvidia-smi --query-gpu=memory.used --format=csv -l 1` in a third
shell, and write down what Isaac actually takes — that is the number this table is missing.

The recommendation does not depend on the Isaac row being right. `--offload-text` costs one CPU
umT5 forward per distinct instruction and buys 11.36 GB; there is no configuration of this box
where that is a bad trade.

`--offload-text` is nearly free on the serving side: `condition_text` memoizes per prompt string,
so a served policy pays one CPU umT5 forward per *distinct* instruction and reads the cache for the
rest of the session (`docs/local_gpu.md` §0b). None of the above is measured on a 5090 — no number
on this page is.

---

## 4. Before the policy server: the one-process smoke run

`--policy dummy` drives the loop with a deterministic sinusoid. No weights, no policy server, no
torch on the policy path, one process. It is the cheapest way to find out whether the Isaac backend
works at all, and it is what to run immediately after the preflight passes:

```bash
.venv-isaac/bin/python scripts/rollout.py --robot isaac_g1 --policy dummy --rollouts 1
```

Everything above the `G1Transport` seam is byte-identical to the MuJoCo and DDS backends — joint
mapping, unit conversion, defense-in-depth clipping, sim-clock pacing, the latched e-stop. That is
the design (`src/wam/robot/README.md`, `docs/sim.md`): the sim buys coverage of the *production*
adapter, not of a sim-specific one. There is deliberately no `IsaacAdapter`.

What an Isaac rollout measures: latency against the deadline, the safety and watchdog paths under
real model timing, whether predicted chunks survive the filter. What it does **not** measure: task
competence. Raytraced Isaac frames are not the frames any checkpoint here trained on. Task success
is E3 and needs the robot.

---

## 5. The e-stop is **not** at parity with the DDS path

**Read this before running an Isaac rollout next to a physical robot, or before quoting an Isaac
safe-stop latency at anybody.**

The Omniverse API is main-thread-only. An e-stop arriving on a watchdog thread therefore cannot
touch Isaac at all. The design: `emergency_damp()` latches a pending-damp flag in **pure Python**
from any thread, and a `PHYSICS_PRE_STEP` callback drains it on the main thread. Compare
`DdsG1Transport.emergency_damp()`, which publishes a damping `LowCmd` (kp=0, kd=`damp_kd`) on the
wire immediately and synchronously, independent of the control loop's health.

Two real differences, neither of which is papered over:

1. **A latency floor of one physics step.** The damp takes effect at the next `PHYSICS_PRE_STEP` —
   up to one `physics_dt` (2 ms at the config's 500 Hz) later, *plus* however long the current
   `step(steps=N)` batch has left to run, because the callback fires per step but the Python caller
   does not regain control until the batch ends. On hardware there is no such floor.
2. **Usually no e-stop at all, not just when something is wrong.** If the main thread is not
   calling `step()`, the flag is never drained and **nothing happens in the simulator**. Read that
   twice: a latched e-stop is precisely what makes `execute()` and `hold()` stop stepping, so
   unless the request lands inside a batch that was already in flight, the ordinary
   watchdog-e-stop path ends with `damp_count == 1`, `damp_applied_count == 0`, `is_damping`
   False, and the body gains untouched. `clear_estop()` then discards the pending flag, so an
   operator resume closes the window for good. A wedged main thread is the same outcome for a more
   exotic reason. On hardware the DDS write does not depend on the control loop being alive — that
   is the entire point of an e-stop. Both cases are tests rather than footnotes
   (`FakeIsaacBinding.wedge_main_thread()` for the exotic one).

   Safe here only because a stopped clock is a stopped arm. Nothing moves in a simulator nobody is
   stepping, which is why this is a fidelity gap rather than a hazard — in Isaac.

What the Isaac path *does* still guarantee, and what `G1Adapter.estop()` actually depends on:
**no further motor command reaches the sim after `estop()` returns.** That is enforced in pure
Python by the latch, on whatever thread it is called from, with no Isaac involvement — the same
property `FakeG1Transport` and `MujocoG1Transport` provide.

**The consequence, spelled out.** An Isaac rollout is *not* evidence for AC-06 (real-robot
safe-stop) and must never be quoted as one. The acceptance harness detects simulated evidence
**solely** via the `sim:` task prefix (`--task`, default `sim:reach`; `--fault-injection` forces
`sim:fault_injection`), and then reports AC-06 as `pending_hardware` instead of safe-stop evidence —
so keeping that prefix is the operator's job, not something the harness can enforce for you. And if
you are running this simulator on a box
that is *also* connected to a G1: **the physical robot's e-stop is not this one.** A latched Isaac
e-stop stops the simulator. Keep the hardware e-stop within reach, keep the DDS path as the thing
that stops the arm, and do not let a green Isaac safe-stop number stand in for an untested one on
the robot.

---

## 6. What is verified, what is not, and what breaks first

**Verified** (executed, on a Mac, no GPU): `FakeIsaacBinding` and the binding-level contracts —
the physics-step tick, `resolve_g1_dof_indices` on all 43 names, the gain round-trip including
`kp=0`, the camera warmup returning `None`, the wedged main thread; torch-freeness of the
Isaac-side import graph, checked in a subprocess; and the constant parity between the preflight and
the binding — the naming candidates, the expected DoF count, the effort-getter candidates and the
asset subpath (`tests/test_isaac_binding.py`). On top of that, `tests/test_isaac_g1.py` drives
`IsaacG1Transport` and `IsaacG1Robot` against the fake: the tick-equality staleness path, the e-stop
latch and its `PHYSICS_PRE_STEP` drain (including the wedged-main-thread case, asserted as the
documented behaviour), the refusal of motor *and* gripper writes after the latch, the grasp synergy
and its projection readback, and the two-latch `clear_estop`. `configs/robot/isaac_g1.yaml` is now
covered by `tests/test_versioning.py`'s `TestShippedConfigs`, which discovers `configs/robot/*.yaml`
by glob instead of naming them — it loads, builds a `G1Config` and declares a canonical space equal
to `G1_SPEC` — and a test there also asserts that every key under its `sim:` block reaches
`rollout.py`'s `_build_isaac_g1` and is a real `IsaacG1Robot.__init__` parameter.

Added after the adversarial review of this backend, because each of these could have been wrong
with no symptom: the gripper readback does not transpose the two hands (every other gripper test
commands the same value to both, so a transposed projection passed all of them); commanding the
gripper open leaves the fingers at the asset's open pose, read off the DOFs rather than through the
projection that would hide the error; a command's own physics steps execute that command's target
rather than its predecessor's; `clear_estop()` discards a damp that never drained; `last_damp_error`
describes the last attempt rather than the worst one ever seen; and `reset()` drops the carried
commanded target — inert at the shipped `q_track_window` of 0, which is exactly why the test turns
the window on.

`pytest tests/ -k isaac -q` is the whole of it — and every one of those tests runs against
`FakeIsaacBinding`. **Passing them says our code is correct against the binding we wrote; it says
nothing about whether that binding matches Isaac Sim.** Only §2 answers that.

**Not verified — essentially everything on the Isaac side.** No `omni`/`isaacsim`/`pxr` call in
this repo has ever executed. Specifically unverified: every module path and symbol name in
`check_api_surface`; the DoF names and the joint naming convention; that `g1.usd` has 43 DoFs; that
`get_num_physics_steps()` is an integer counter; that `render()` does not advance physics; that
`set_dof_gains` round-trips; the effort-getter name (a guess with fallbacks, and diagnostic only —
no rollout needs it); every gain value in `configs/robot/isaac_g1.yaml`; the joint limits in that
file, which are the MuJoCo Menagerie MJCF's ranges reused on lineage, never read from the USD, and
which **the preflight does not check** — the safety layer clips to them regardless, so a mismatch
makes the clip conservative or loose, never absent.

**Which command fails first, per assumption:**

| assumption | first thing that breaks |
|---|---|
| the venv split, python version | `preflight_isaac.py` check A — before Isaac is even imported |
| Isaac's torch pin is not 2.13.0 (§0) | `pip show torch` in the Isaac venv, right after the install. Nothing later depends on it, but the whole two-venv premise does |
| Isaac is installed, in *this* venv | `preflight_isaac.py` → `isaacsim_importable` FAIL with the repair line. Nothing below it ran |
| module paths / symbol names | `preflight_isaac.py` → `api_surface_complete`, listing the missing names. This is the check that says whether the transport will even import |
| 500 Hz survives `int(1.0/dt)` | `preflight_isaac.py` → `hz_is_exact` / `physics_dt_round_trips`. Silent otherwise — every velocity limit would be quietly off |
| the asset is the 43-DoF G1 | `preflight_isaac.py` → `num_dofs_is_43`; then the binding's own constructor assertion |
| the joint naming convention | `preflight_isaac.py` → `body_joints_resolve` / `*_fingers_resolve`, with the near-miss list. Then `resolve_g1_dof_indices` raises naming the joint. **The fix is the candidate lists in `isaac_binding.py` *and* `preflight_isaac.py`, not a YAML** (§2) |
| the tick is an exact integer | `preflight_isaac.py` → `tick_is_integer`. If this were wrong and unchecked, the watchdog would fire at random |
| `PHYSICS_PRE_STEP` fires | `preflight_isaac.py` → `pre_step_callback_*`. **Nothing else would ever notice** — the latch still blocks motor commands, so the failure is invisible from above (§5) |
| gains are ours to set | `preflight_isaac.py` → `gains_round_trip` / `zero_kp_accepted` |
| a camera produces pixels | `preflight_isaac.py` → `camera_*`. Raise `--warmup-frames` or add a dome light before assuming it is broken |
| the backend imports and constructs | `rollout.py --robot isaac_g1 --policy dummy --rollouts 1` (§4) — the first command that touches our code rather than NVIDIA's |
| both processes fit on one card | the `--policy remote` run in §3, as an OOM in whichever process loses. Watch `nvidia-smi` |
| the gains actually track | nothing fails. It just tracks badly, and you find out by measuring — see the note in `configs/robot/isaac_g1.yaml`'s `gains` block for the two protocols to repeat |

If a preflight check contradicts what the binding believes, **fix the binding, not the preflight**.
The preflight is the record of the assumption; the binding is the assumption.

---

## See also

- `docs/local_gpu.md` — the WAM venv, the torch build, `preflight_gpu.py`, the VRAM budget (§0c), `--offload-text` (§0b), the closed loop (§4)
- `scripts/preflight_isaac.py` — the gate in §2; its module docstring is the authoritative record of every Isaac API assumption
- `src/wam/robot/isaac_binding.py` — the one module allowed to import `omni`/`isaacsim`/`pxr`, plus `FakeIsaacBinding`
- `configs/robot/isaac_g1.yaml` — limits, gains and Isaac wiring, each with its provenance
- `docs/sim.md` — the MuJoCo backend: what a sim rollout proves and what it does not
- `src/wam/robot/README.md` — the `G1Transport` seam and its implementations
