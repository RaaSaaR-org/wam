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
- Readout blocks are **measured** (2026-07-26, `--ablate` on ZeroGPU, 18/18 checks): (20, 29)
  beat the (15, 22) heuristic on motion/instruction/state sensitivity —
  `configs/model/wan22_ti2v_5b.yaml`, details in `docs/hf_jobs.md`.
  Next: LoRA fine-tune — that one needs Jobs, ZeroGPU cannot hold a training run.
- **D2** MVP set (10–30 h) → joint video/action training (`wam.training.joint`).
- Ablation world-action vs. action-only on real data → AC-07 verdict.

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
