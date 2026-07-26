# WAM — MVP Tasks

Derived from PRD roadmap (M0–M4 = MVP; M5/M6 post-MVP). Order is strict: don't start a milestone before the previous exit criterion is met.

> **Build status (2026-07-26):** M0–M4 code-complete and tested (465 tests green, mock/sim only).
> Everything marked *hw* still needs real hardware, real teleop data (D1/D2), or an open decision.
> Ordered path to real usage: `docs/ROADMAP.md`.

## M0 · Architecture & Safety Baseline (2–4 weeks)

- [x] **T-01** Canonical robot state/action schema (PRD Anhang A) as versioned code: `q, dq, imu, gripper, validity_mask` / `relative_joint_target | ee_delta, gripper_target, duration` + normalization spec
- [x] **T-02** Core interfaces in `wam.interfaces`: `BackboneAdapter`, `StateEncoder`, `ActionDecoder`, `SafetyFilter`, `RobotAdapter` (typed protocols, versioned)
- [x] **T-03** Mock robot adapter + dummy policy → end-to-end loop runs without hardware (`scripts/run_mock_loop.py`)
- [x] **T-04** Safety layer v0: joint/velocity/accel/workspace limits, NaN/Inf rejection, timeout → hold/stop, every intervention logged (FR-07, §11.2)
- [x] **T-05** Config + experiment versioning: config hash, software commit, adapter version in every log record (FR-10)
- [ ] **T-06** Resolve open decisions OD-01/02/03/07 (platform+gripper, action space, cameras, teleop) — *user/business decision; code supports joint-delta (primary) + EE-delta*

**Exit:** interfaces + canonical schema approved, dummy policy loop with E-stop + logging works.

## M1 · Data Factory (3–6 weeks)

- [x] **T-07** Episode format writer/reader per Anhang A (mp4 + parquet + manifest with checksums)
- [x] **T-08** Synchronized capture: cameras + robot state + commanded/executed actions; timestamp tolerance check (FR-01) — *hw: verified with mock sources only*
- [x] **T-09** Replay + visualization: episode report from stored data (FR-08)
- [x] **T-10** Teleop workflow + camera/kinematics calibration, versioned — *hw: storage/spec + `docs/teleop.md`; real teleop rig pending OD-07*
- [x] **T-11** Automatic dataset validation gates; record D0 + D1 — *hw: gates done; D0/D1 recorded synthetically (`scripts/record_mock_dataset.py`), real teleop sets pending*

**Exit:** reproducible synchronized recording + replay — met with mock episodes; real teleop episodes pending hardware.

## M2 · Action-Only Baseline (3–5 weeks)

- [x] **T-12** State encoder (2–4 layer MLP, handles missing sensor fields) + action decoder (chunked, 8–32 steps) (FR-02, FR-04)
- [x] **T-13** Action-only policy on open backbone; overfit D1 successfully — *gate passed on synthetic D1: loss → 0.09 % of initial (`scripts/overfit_d1.py`)*
- [x] **T-14** Offline eval E1: action prediction on holdout episodes, metrics dashboard

**Exit:** overfit proof — pipeline and action space are learnable. ✅ (synthetic D1)

## M3 · World-Action Prototype (6–10 weeks)

- [x] **T-15** Backbone adapters behind one interface: `tiny` (functional), `wan_i2v` (real diffusers integration: VAE + umT5 + DiT residual-stream hooks, Wan2.1-I2V-14B and Wan2.2-TI2V-5B layouts), `flux3` stub (OD-06) — *gpu: verified against stubs offline; real-weights check runs on a free ZeroGPU Space (`scripts/deploy_wan_space.py`) or HF Jobs (`scripts/launch_wan_smoke_job.py`) — `docs/hf_jobs.md`*
- [x] **T-16** Action encoder + joint video/action flow-matching training; frozen parts registry, selective blocks (FR-03, §10.3) — *trained only on synthetic data so far*
- [x] **T-17** Loss monitoring (video/action/alignment/smoothness/limit penalty) + gradient checks, divergence detection (R-07)
- [x] **T-18** Ablation harness: world-action vs. action-only (AC-07) — *hw: verdict needs real D2 data*

**Exit:** ablation machinery ready; "video branch helps" verdict pending real data.

## M4 · Real-Robot MVP (6–10 weeks)

- [x] **T-19** Closed-loop runtime: receding horizon, replanning replaces unexecuted chunk rest, deadline + watchdog (FR-05, ≥2 Hz policy rate)
- [x] **T-20** Inference server (WebSocket, versioned wire protocol) separated from robot-side controller; safety stays robot-side
- [x] **T-21** G1 robot adapter: joint mapping, units, limits, E-stop integration behind swappable transport (FR-06, OD-08) — *hw: real DDS transport needs `unitree_sdk2py` + robot*
- [x] **T-22** E2 kinematic/sim checks — *passed with trained checkpoint; E3 controlled rollouts with operator pending hardware*
- [x] **T-23** Acceptance harness + 100-rollout sim run: AC-03/04/05 PASS, AC-06 10/10 sim faults handled (pending real-robot fault injection), sim proxy `sim:reach` 100/100, AC-01/02/07 pending real data (`runs/acceptance/`)

**Exit:** MVP acceptance criteria evaluated — 3 PASS / 0 FAIL / 4 pending hardware+data.

## Post-MVP

- M5: FLUX 3 Dev integration (backbone swap, license check, benchmark vs. fallback — AC-05)
- M6: generalization, video-only data, cross-embodiment, multiple future hypotheses (FR-11/12)

## Open decisions (PRD §16)

| ID | Decision | Blocks |
|----|----------|--------|
| OD-01 | Platform + gripper/hand | M1 |
| OD-02 | Action space: joint delta vs. EE delta | M2 |
| OD-03 | Camera setup, resolution, framerate | data recording |
| OD-04 | Open fallback backbone + license | M3 |
| OD-05 | Training hardware + budget | after M2 overfit gate |
| OD-06 | FLUX 3 access, weights, fine-tuning rights | M5 |
| OD-07 | Teleoperation system | M1 |
| OD-08 | Which safety functions the vendor controller covers | real-robot tests |
