# WAM — MVP Tasks

Derived from PRD roadmap (M0–M4 = MVP; M5/M6 post-MVP). Order is strict: don't start a milestone before the previous exit criterion is met.

> **Build status (2026-07-26):** M0–M4 code-complete and tested (470 tests green, mock/sim only).
> Everything marked *hw* still needs real hardware, real teleop data (D1/D2), or an open decision.
> Ordered path to real usage: `docs/ROADMAP.md`.

## M0 · Architecture & Safety Baseline (2–4 weeks)

- [x] **T-01** Canonical robot state/action schema (PRD Anhang A) as versioned code: `q, dq, imu, gripper, validity_mask` / `relative_joint_target | ee_delta, gripper_target, duration` + normalization spec
- [x] **T-02** Core interfaces in `wam.interfaces`: `BackboneAdapter`, `StateEncoder`, `ActionDecoder`, `SafetyFilter`, `RobotAdapter` (typed protocols, versioned)
- [x] **T-03** Mock robot adapter + dummy policy → end-to-end loop runs without hardware (`scripts/run_mock_loop.py`)
- [x] **T-04** Safety layer v0: joint/velocity/accel/workspace limits, NaN/Inf rejection, timeout → hold/stop, every intervention logged (FR-07, §11.2)
- [x] **T-05** Config + experiment versioning: config hash, software commit, adapter version in every log record (FR-10)
- [x] **T-06** Resolve open decisions OD-01/02/03/07 — *decided 2026-07-26: Unitree G1 EDU4 + Dex3-1 hands; joint-delta; standard EDU4 sensor set; VR teleop (see table below)*

**Exit:** interfaces + canonical schema approved, dummy policy loop with E-stop + logging works.

## M1 · Data Factory (3–6 weeks)

- [x] **T-07** Episode format writer/reader per Anhang A (mp4 + parquet + manifest with checksums)
- [x] **T-08** Synchronized capture: cameras + robot state + commanded/executed actions; timestamp tolerance check (FR-01) — *hw: verified with mock sources only*
- [x] **T-09** Replay + visualization: episode report from stored data (FR-08)
- [x] **T-10** Teleop workflow + camera/kinematics calibration, versioned — *hw: storage/spec + `docs/teleop.md`; real teleop rig pending OD-07*
- [x] **T-11** Automatic dataset validation gates; record D0 + D1 — *hw: gates done; D0/D1 recorded synthetically (`scripts/record_mock_dataset.py`); all gates also PASS on real G1 data converted from nvidia/GR00T-N1.7-AppleToPlate (`scripts/convert_lerobot_g1.py`); real teleop sets pending*

**Exit:** reproducible synchronized recording + replay — met with mock episodes; real teleop episodes pending hardware.

## M2 · Action-Only Baseline (3–5 weeks)

- [x] **T-12** State encoder (2–4 layer MLP, handles missing sensor fields) + action decoder (chunked, 8–32 steps) (FR-02, FR-04)
- [x] **T-13** Action-only policy on open backbone; overfit D1 successfully — *gate passed on synthetic D1: loss → 0.09 % of initial (`scripts/overfit_d1.py`); passed again on real D1-scale data (10 GR00T G1 apple-to-plate episodes, `--dataset datasets/gr00t-apple`): loss → 0.01 % of initial, E1 holdout mse 1.8e-5 / mae 0.0028*
- [x] **T-14** Offline eval E1: action prediction on holdout episodes, metrics dashboard — *first generalization result (2026-07-26): trained on 362 real GR00T episodes, evaluated on 40 unseen: E1 mse 1.10e-5 vs zero-delta baseline 1.63e-5 (−32 %), cosine(pred, truth) 0.60, moving-step error −39 % (`runs/d1-full-gen-seed0`)*

**Exit:** overfit proof — pipeline and action space are learnable. ✅ (synthetic D1 + real G1 data; generalizes to unseen episodes)

## M3 · World-Action Prototype (6–10 weeks)

- [x] **T-15** Backbone adapters behind one interface: `tiny` (functional), `wan_i2v` (real diffusers integration: VAE + umT5 + DiT residual-stream hooks, Wan2.1-I2V-14B and Wan2.2-TI2V-5B layouts), `flux3` stub (OD-06) — *✅ verified on real weights: Wan2.2-TI2V-5B on a ZeroGPU RTX PRO 6000, 13/13 checks, features `[1, 224, 3072]`, 24.3 GB peak VRAM (`docs/hf_jobs.md`); rerun with `scripts/deploy_wan_space.py`. Readout blocks measured via `--ablate` (18/18), then **label-validated** with ridge probes on real GR00T action chunks (8/8, `scripts/hf_job_wan_probe.py`): early blocks (2, 10) overturn the label-free pick (20, 29) → `configs/model/wan22_ti2v_5b.yaml`; no frozen features beat state-only yet — LoRA (T-16) carries the burden*
- [x] **T-16** Action encoder + joint video/action flow-matching training; frozen parts registry, selective blocks (FR-03, §10.3) — *trained only on synthetic data so far*
- [x] **T-17** Loss monitoring (video/action/alignment/smoothness/limit penalty) + gradient checks, divergence detection (R-07)
- [x] **T-24** Cosmos3-Nano frozen-feature probe (backbone bake-off vs. Wan, OD-04): `scripts/hf_job_cosmos3_probe.py` runs the *identical* T-15 experiment — same GR00T windows, labels, episode split and ridge code (imported from the Wan probe) — against Cosmos3-Nano's generator tower (diffusers `Cosmos3OmniTransformer`, 36 MoT layers; its VAE **is** the Wan2.2 VAE). Free on ZeroGPU: `scripts/deploy_cosmos3_space.py`. Decision rule: frozen Cosmos3 features beat the state-only ridge (which Wan's could not) → Cosmos3 becomes the primary backbone candidate for the T-16 LoRA; otherwise stay on Wan — *✅ ran 2026-07-26 (ZeroGPU, 9/9 checks, `runs/cosmos3_probe/`): best block pair joints test R² 0.359 / gripper 0.708 vs. state-only 0.456 / 0.881 → **frozen Cosmos3 does not beat state-only either; stay on Wan** for T-16. Robotics pretraining shows up only in the gripper channel (best single block 0.822 vs. Wan's 0.698), not in joints. Details: `docs/hf_jobs.md`*
- [x] **T-18** Ablation harness: world-action vs. action-only (AC-07) — *first real-data verdict (2026-07-26, `scripts/run_ablation.py`, 402 GR00T episodes, identical 362/40 split + config as the action-only baseline): **hurts** at tiny scale — holdout MSE 2.09e-05 vs. 1.10e-05 (−89.5%), gripper acc 0.853 vs. 0.871. The tiny backbone's video loss plateaus (~0.72), so the shared trunk pays a multi-task tax. Consistent with the T-15 probe: the AC-07 advantage must come from the pretrained prior (Wan LoRA). Run: `runs/t18-real-ablation-seed0`*

**Exit:** ablation machinery ready; first real AC-07 verdict recorded — at tiny scale the video branch hurts, so "video helps" now rests on the pretrained prior (T-16 LoRA, GPU pending).

## M4 · Real-Robot MVP (6–10 weeks)

- [x] **T-19** Closed-loop runtime: receding horizon, replanning replaces unexecuted chunk rest, deadline + watchdog (FR-05, ≥2 Hz policy rate)
- [x] **T-20** Inference server (WebSocket, versioned wire protocol) separated from robot-side controller; safety stays robot-side
- [x] **T-21** G1 robot adapter: joint mapping, units, limits, E-stop integration behind swappable transport (FR-06, OD-08) — *hw: real DDS transport needs `unitree_sdk2py` + robot*
- [x] **T-22** E2 kinematic/sim checks — *passed with trained checkpoint; E3 controlled rollouts with operator pending hardware*
- [x] **T-23** Acceptance harness + 100-rollout sim run: AC-03/04/05 PASS, AC-06 10/10 sim faults handled (pending real-robot fault injection), sim proxy `sim:reach` 100/100, AC-01/02 pending real data, AC-07 first verdict via T-18 (`runs/acceptance/`)

**Exit:** MVP acceptance criteria evaluated — 3 PASS / 0 FAIL / 4 pending hardware+data.

## Post-MVP

- M5: FLUX 3 Dev integration (backbone swap, license check, benchmark vs. fallback — AC-05)
- M6: generalization, video-only data, cross-embodiment, multiple future hypotheses (FR-11/12)

## Open decisions (PRD §16) — resolved 2026-07-26

| ID | Decision | Status |
|----|----------|--------|
| OD-01 | Platform + gripper/hand | ✅ **Unitree G1 EDU4 + Dex3-1 three-finger hands.** MVP maps the canonical scalar gripper channel `[left, right]` to a grasp synergy in the G1 adapter; per-finger control is post-MVP |
| OD-02 | Action space | ✅ joint-delta primary, EE-delta supported in schema/safety |
| OD-03 | Cameras | ✅ standard G1 EDU4 sensor set (head RealSense D435i, RGB for WAM; depth/LiDAR unused in MVP) |
| OD-04 | Open fallback backbone + license | ✅ Wan2.2-TI2V-5B (Apache 2.0), verified on real weights (`docs/hf_jobs.md`) |
| OD-05 | Training hardware + budget | ✅ free tier now (Mac MPS + ZeroGPU); own RTX 5090 + H200 cluster available later for T-16 real-data training |
| OD-06 | FLUX 3 access, weights, fine-tuning rights | ⏳ deferred to M5 (post-MVP), nothing blocks on it |
| OD-07 | Teleoperation system | ✅ VR teleop (Unitree `xr_teleoperate` path; headset model — Vision Pro vs. Quest 3 — still to pick at purchase time) |
| OD-08 | Vendor controller safety coverage | ⏳ verify during G1 bring-up (which functions Unitree's controller covers vs. WAM safety layer) |
