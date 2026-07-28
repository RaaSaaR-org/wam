# Roadmap to real usage

Status 2026-07-27: M0–M4 code-complete (604 tests, acceptance report in `runs/acceptance/`), and
the closed loop now also runs on MuJoCo contact physics with rendered pixels (`docs/sim.md`) and
over a real DDS bus in an arm64 container (`docker/dds/README.md`). Everything below is decisions,
hardware, and real data — in order.
Details per task: `TASKS.md`. PRD §16 (internal document, outside this repository).

## 1. Decisions — resolved 2026-07-26

| Decision | Outcome | Unblocks |
|----------|---------|----------|
| OD-01 | **Unitree G1 EDU4 + Dex3-1 hands**; scalar gripper channel → grasp synergy in the adapter (per-finger post-MVP) | robot bring-up (order the robot) |
| OD-03 | **Standard G1 EDU4 sensor set** — head RealSense D435i, RGB stream for WAM; depth/LiDAR unused in MVP | perception, data recording |
| OD-07 | **VR teleop** (Unitree `xr_teleoperate`; headset model picked at purchase) | data recording |
| OD-04 | Wan2.1/2.2 are Apache 2.0; start on `Wan2.2-TI2V-5B` (`docs/hf_jobs.md`) | M3 training |
| OD-05 | **Free tier now** (Mac MPS + ZeroGPU); for T-16: EuroHPC **Discoverer+** (NVIDIA H200) — access verified 2026-07-27, 5 000 GPU-hours, 4 h walltime cap, `docs/discoverer.md` | M3 training (**unblocked**) |
| OD-02 | joint-delta primary, EE-delta supported in schema/safety | — |

## 2. Robot bring-up

**Nothing is hardware-stubbed any more.** `G1Adapter` now runs against three implementations of
the same `G1Transport` seam — `FakeG1Transport` (unit tests), `MujocoG1Transport` (physics +
pixels), `DdsG1Transport` (real vendor DDS, implemented and wire-verified in a container). The
adapter code that will drive the robot is the *same* code in all three cases.

Already testable without the robot (2026-07-27):

- **Physics + pixels (T-25, `docs/sim.md`).** `uv pip install mujoco` +
  `scripts/fetch_g1_model.py`, then `get_robot("mujoco_g1")` gives the real `G1Adapter` on a
  MuJoCo G1 with Dex3 hands, a table, a cube and two cameras. Full `ClosedLoopExecutor` rollouts
  run at 1.19× realtime including two 256×256 renders per cycle (28.5× realtime physics-only).
  Sim gains kp=300/kd=15 live in `configs/robot/mujoco_g1.yaml`; `g1.yaml` keeps its conservative
  hardware placeholders untouched. Caveats that matter: no stable grasp yet, welded base with 14
  locked joints, invented camera extrinsics. Pinned by `tests/test_mujoco_g1.py` (21 tests,
  skipped without the `mujoco` extra) — but those are contract tests; the measured physics
  numbers come from ad-hoc scripts nothing re-runs.
- **DDS wire layer (T-25a, `docker/dds/README.md`).** `docker/dds/run.sh` runs `DdsG1Transport`
  against a fake G1 on a real CycloneDDS bus in a `linux/arm64` container — the EDU4's onboard
  Jetson architecture — **11 PASS / 0 FAIL / 0 SKIP**: IDL mapping, CRC (plus a corruption
  negative check), Dex3 topics, `emergency_damp`, the whole `G1Adapter` closed loop and the
  stale-tick validity path, with the peer in a separate process so DDS discovery is real.

Still genuinely needs the physical robot:

- **Vendor conformance.** The container test is self-consistent (same `unitree_sdk2py` IDL and CRC
  on both sides); that the vendor's Python struct layout matches the robot's C++ `LowCmd`
  byte-for-byte is asserted, not verified. Confirm `rt/lowstate` at ~500 Hz first.
- **Vendor RPC services.** `MotionSwitcherClient().ReleaseMode()` (the moment the robot stops
  holding itself up) and `LocoClient().Damp()` as an escalation on top of the implemented wire
  damping. Decide deliberately whether release belongs in `open()` or in an operator script.
- **The E-stop chain, physically.** With the robot suspended: `G1Adapter.estop()` → joints go
  limp-damped. Nothing autonomous runs before this passes.
- **First contact on `rt/arm_sdk`, not `rt/lowcmd`** (`cmd_topic=G1_ARM_SDK_TOPIC,
  arm_sdk_weight=1.0` — legs stay with the vendor controller). Implemented, untested.
- **Machine variant**: reported `mode_machine`, 23- vs. 29-DoF waist, `mode_pr` 0 (series) vs. 1.
- **Real limits and gains** (OD-08) → replace the placeholders in `configs/robot/g1.yaml` from the
  datasheet; re-confirm `damp_kd`. Do **not** copy the sim gains over.
- **The real Dex3 gripper mapping**: verify topic names and the RIS mode byte, determine the
  per-joint vendor range, set `gripper_vendor_min/max`, and replace the mean-of-7 placeholder. The
  MuJoCo synergy (measured from the Menagerie joint ranges) is the reference to aim at.
- **Timing under load**: `execute()` paces on the clock, so the `dq_max·dt` clip is a velocity
  limit only if commands really arrive `dt_s` apart. Measure jitter at the chosen `control_dt_s`.

Ordered bring-up checklist: `docker/dds/README.md` § "Remaining steps for real-hardware bring-up".

## 3. Perception + teleop

- Real `FrameSource` implementation (`src/wam/data/capture.py` protocol) for the G1's head
  RealSense D435i (RGB; pyrealsense2).
- Calibrate: intrinsics/extrinsics → `CalibrationSet` (`configs/calibration/`).
- Verify timestamp sync against the 20 ms gate (`SyncRecorder`).
- VR teleop rig (Unitree `xr_teleoperate` → canonical actions), wire into `SyncRecorder` —
  workflow: `docs/teleop.md`.

## 4. Real data (M1 gates)

- **D0** systems test (30–60 min): all validation gates green (`wam.data.validation`).
- **D1** overfit set (1–3 h, one pick-and-place task): rerun `scripts/overfit_d1.py`
  on it — the real M2 go/no-go gate.

## 5. Training (M3)

- `WanI2VAdapter` (`src/wam/backbones/wan_i2v.py`) is **verified on real Wan2.2-TI2V-5B
  weights** (ZeroGPU, 13/13 checks, `docs/hf_jobs.md`); rerun with
  `scripts/deploy_wan_space.py` (free) or `scripts/launch_wan_smoke_job.py` (HF Jobs).
- Readout blocks are **label-validated** (2026-07-26, ridge probes on real GR00T action
  chunks, ZeroGPU, 8/8 checks): early blocks (2, 10) beat both the label-free ablation pick
  (20, 29) and the (15, 22) heuristic — `configs/model/wan22_ti2v_5b.yaml`, details in
  `docs/hf_jobs.md`. Caveat: no frozen video features beat a state-only ridge yet, so the
  action value must come from fine-tuning.
  Next: LoRA fine-tune — ZeroGPU cannot hold a training run. **Compute and code are both
  solved.** Discoverer+ (H200) is verified and the job chain is checked in at
  `cluster/discoverer/` (env, weight staging, requeue probe, smoke test, readout probe — steps
  1–3 cost no GPU hours). **T-16a done 2026-07-27**: `FlowBackbone` protocol, backbone-agnostic
  joint model, LoRA on the Wan DiT via `WanFlowBackbone`, and `scripts/train_t16_lora.py` with
  SIGUSR1-checkpoint/resume for the 4 h cap. 583 tests green, all runnable on CPU without Wan
  weights. What remains is *running* it: `configs/training/joint_wan_gr00t.yaml` + `sbatch
  50_train_t16.sbatch`.
  Note for interpreting the result: the video flow loss now lives in **VAE latent space**, so
  its magnitude is not comparable to the tiny-backbone pixel loss (the ~0.72 plateau in T-18).
- **Backbone bake-off (T-24) — decided 2026-07-26:** probed `nvidia/Cosmos3-Nano` (16B MoT,
  robotics-pretrained with actions; its VAE is the Wan2.2 VAE) with the *same*
  frozen-feature ridge suite (`scripts/hf_job_cosmos3_probe.py`, ZeroGPU, 9/9 checks).
  Frozen Cosmos3 features do **not** beat the state-only ridge either (joints test R²
  0.359 vs. 0.456; gripper 0.708 vs. 0.881) → **stay on Wan** for the T-16 LoRA.
  Details: `docs/hf_jobs.md`, run: `runs/cosmos3_probe/`.
- **D2** MVP set (10–30 h) → joint video/action training (`wam.training.joint`).
- Ablation world-action vs. action-only on real data → **first AC-07 verdict recorded**
  (2026-07-26, `scripts/run_ablation.py`, tiny backbone, 402 GR00T episodes, identical
  362/40 split + config as the baseline): the video branch **hurts** offline — holdout MSE
  2.09e-05 vs. 1.10e-05, gripper acc −2%. The tiny video loss plateaus (~0.72), so the
  shared trunk pays a multi-task tax; a measurable AC-07 advantage must come from the
  pretrained prior. Rerun after the Wan LoRA fine-tune (`runs/t18-real-ablation-seed0`).

## 6. Real rollouts (M4 / E3)

- Rehearse in MuJoCo first (`docs/sim.md`): the same `G1Adapter`, safety layer, watchdog and
  executor, on contact physics — `scripts/rollout.py --robot mujoco_g1 --policy dummy` runs the
  whole loop, E2 gates included. Open work before it is a proper E2 *rehearsal rig*: every
  commanded joint delta is under-executed by a `prefix_steps`-dependent factor (the adapter
  re-bases on the measured `q`, so loop lag is discarded — mean 0.39 of a one-control-period step,
  0.95 of a 25-step chunk), which means sim action labels, safety-intervention rates and velocity-
  envelope numbers are **not calibrated**. See "Known limitations" in `docs/sim.md`; the design
  fix (bounded feed-forward in `G1Adapter.execute()`) is T-25c.
- GPU box: `scripts/serve_policy.py --checkpoint …`; robot side:
  `scripts/rollout.py --robot g1 --policy remote --server-uri ws://…`.
- Verify policy rate ≥ 2 Hz end-to-end over the wire.
- E3 controlled rollouts with operator + physical E-stop.
- 100-rollout acceptance run on hardware → real AC-01/02/03/06
  (`scripts/run_acceptance.py`; AC-03/04/05 already pass in sim).

## Parallelizable start

Decisions are made — order the G1 EDU4 (+ VR headset) and run robot bring-up (step 2) as
soon as it arrives. The code-touching part of bring-up is **done**: `DdsG1Transport` is
implemented and wire-verified, and the MuJoCo sim exercises the same adapter on physics. What is
left on arrival is verification, not implementation. Until then, real-data validation of
encoders/decoders runs on public LeRobot datasets (video + joint states) — no robot required.

One thing the sim explicitly does **not** shorten: **D2**. MuJoCo renderings are not RealSense
images, and the video backbone (Wan) has never seen one — the T-15/T-24 probes already showed both
Wan and Cosmos3 hallucinate embodiment they have not observed. Sim frames test the vision
*plumbing* (shape, dtype, cadence, pixels that change when the robot moves); they are not training
data. Real teleop episodes remain the gate for T-16.

**Done (2026-07-26):** `scripts/convert_lerobot_g1.py` converts LeRobot-v2.1 G1 episodes
(nvidia/GR00T-N1.7-AppleToPlate, CC-BY-4.0) into the WAM episode format: 43-dim G1 state →
15-joint canonical space, Dex3 7-DoF hands → grasp-synergy gripper scalar, BC relabeling
of executed states into bounded joint-delta chunks, IMU flagged invalid (exercises the
FR-02 missing-group path on real data). 10 converted episodes pass all T-11 validation
gates, and the T-13 overfit gate passes on them (`scripts/overfit_d1.py --dataset
datasets/gr00t-apple`): action loss → 0.01 % of initial, E1 holdout mse 1.8e-5.
The state encoder + action head demonstrably learn real robot data end-to-end.

**Generalization (2026-07-26, `runs/d1-full-gen-seed0`):** trained on the full converted
dataset (402 episodes, 362 train / 40 holdout, tiny backbone, 6000 steps on MPS). On the
40 unseen episodes: E1 mse 1.10e-5 vs. 1.63e-5 for the zero-delta baseline (−32 %),
cosine(predicted, true motion) 0.60; on moving steps (71 % of data) the error is −39 %
vs. baseline, while still phases pick up small jitter (~0.002 rad/step RMS). The
action-only policy learns the task, not just the episodes — next quality jump expected
from the real Wan backbone (LoRA, 5090/H200).
