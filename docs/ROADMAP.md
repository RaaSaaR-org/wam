# Roadmap to real usage

Status 2026-07-26: M0–M4 code-complete and verified on mock/sim (465 tests, acceptance report
in `runs/acceptance/`). Everything below is decisions, hardware, and real data — in order.
Details per task: `TASKS.md`. PRD §16 (internal document, outside this repository).

## 1. Decisions — resolved 2026-07-26

| Decision | Outcome | Unblocks |
|----------|---------|----------|
| OD-01 | **Unitree G1 EDU4 + Dex3-1 hands**; scalar gripper channel → grasp synergy in the adapter (per-finger post-MVP) | robot bring-up (order the robot) |
| OD-03 | **Standard G1 EDU4 sensor set** — head RealSense D435i, RGB stream for WAM; depth/LiDAR unused in MVP | perception, data recording |
| OD-07 | **VR teleop** (Unitree `xr_teleoperate`; headset model picked at purchase) | data recording |
| OD-04 | Wan2.1/2.2 are Apache 2.0; start on `Wan2.2-TI2V-5B` (`docs/hf_jobs.md`) | M3 training |
| OD-05 | **Free tier now** (Mac MPS + ZeroGPU); own RTX 5090 + H200 cluster later for T-16 | M3 training |
| OD-02 | joint-delta primary, EE-delta supported in schema/safety | — |

## 2. Robot bring-up

- Install `unitree_sdk2py`; implement the real methods in
  `src/wam/robot/g1_transport.py::DdsG1Transport` (LowState/LowCmd topics, CRC, damping
  service — hook points documented in the file). This is the **only** hardware-stubbed file;
  everything above it is tested against `FakeG1Transport`.
- Replace placeholder limits in `configs/robot/g1.yaml` with datasheet values.
- Clarify which safety functions the vendor controller covers (OD-08); verify the E-stop
  chain physically (`G1Adapter.estop()` → damping).

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
  Next: LoRA fine-tune — that one needs Jobs, ZeroGPU cannot hold a training run.
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

- GPU box: `scripts/serve_policy.py --checkpoint …`; robot side:
  `scripts/rollout.py --robot g1 --policy remote --server-uri ws://…`.
- Verify policy rate ≥ 2 Hz end-to-end over the wire.
- E3 controlled rollouts with operator + physical E-stop.
- 100-rollout acceptance run on hardware → real AC-01/02/03/06
  (`scripts/run_acceptance.py`; AC-03/04/05 already pass in sim).

## Parallelizable start

Decisions are made — order the G1 EDU4 (+ VR headset) and run robot bring-up (step 2) as
soon as it arrives. First code-touching step: `DdsG1Transport`. Until hardware arrives,
real-data validation of encoders/decoders can run on public LeRobot datasets
(video + joint states) — no robot required.

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
