# Roadmap to real usage

Status 2026-07-26: M0–M4 code-complete and verified on mock/sim (465 tests, acceptance report
in `runs/acceptance/`). Everything below is decisions, hardware, and real data — in order.
Details per task: `TASKS.md`. PRD §16 (internal document, outside this repository).

## 1. Decisions (blockers)

| Decision | What to decide | Unblocks |
|----------|----------------|----------|
| OD-01 | Platform + gripper: confirm G1, which hand/gripper | robot bring-up |
| OD-03 | Cameras: which, resolution, framerate | perception, data recording |
| OD-07 | Teleop system: leader-follower vs. VR | data recording |
| OD-04 | *Answered:* Wan2.1/2.2 are Apache 2.0; start on `Wan2.2-TI2V-5B` (`docs/hf_jobs.md`) | M3 training |
| OD-05 | Training GPU + budget — **ZeroGPU** is free on PRO for the T-15 check, **HF Jobs** ($0.80–5/h, pre-paid credits) for T-16 training | M3 training |
| OD-02 | *De-facto done:* joint-delta primary, EE-delta supported in schema/safety | — |

## 2. Robot bring-up

- Install `unitree_sdk2py`; implement the real methods in
  `src/wam/robot/g1_transport.py::DdsG1Transport` (LowState/LowCmd topics, CRC, damping
  service — hook points documented in the file). This is the **only** hardware-stubbed file;
  everything above it is tested against `FakeG1Transport`.
- Replace placeholder limits in `configs/robot/g1.yaml` with datasheet values.
- Clarify which safety functions the vendor controller covers (OD-08); verify the E-stop
  chain physically (`G1Adapter.estop()` → damping).

## 3. Perception + teleop

- Real `FrameSource` implementations (`src/wam/data/capture.py` protocol) for the chosen cameras.
- Calibrate: intrinsics/extrinsics → `CalibrationSet` (`configs/calibration/`).
- Verify timestamp sync against the 20 ms gate (`SyncRecorder`).
- Build the teleop rig, wire into `SyncRecorder` — workflow: `docs/teleop.md`.

## 4. Real data (M1 gates)

- **D0** systems test (30–60 min): all validation gates green (`wam.data.validation`).
- **D1** overfit set (1–3 h, one pick-and-place task): rerun `scripts/overfit_d1.py`
  on it — the real M2 go/no-go gate.

## 5. Training (M3)

- `WanI2VAdapter` (`src/wam/backbones/wan_i2v.py`) is **verified on real Wan2.2-TI2V-5B
  weights** (ZeroGPU, 13/13 checks, `docs/hf_jobs.md`); rerun with
  `scripts/deploy_wan_space.py` (free) or `scripts/launch_wan_smoke_job.py` (HF Jobs).
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

Decide OD-01/03/07, order teleop hardware, and run robot bring-up (step 2) in parallel.
First code-touching step: `DdsG1Transport`.
