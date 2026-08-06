---
id: T-24
aliases:
- T-24
title: "Cosmos3-Nano frozen-feature probe — the backbone bake-off vs. Wan"
slug: cosmos3-nano-frozen-feature-probe-the-backbone-bake-off-vs-w
status: done
priority: 3
owner: ''
projects: []
customers: []
tags:
- m3
- data
- backbone
- eval
- training
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-26
updated: 2026-07-26
---

# Cosmos3-Nano frozen-feature probe — the backbone bake-off vs. Wan

## Description

Cosmos3-Nano frozen-feature probe (backbone bake-off vs. Wan, OD-04):
`scripts/hf_job_cosmos3_probe.py` runs the *identical* T-15 experiment — same GR00T windows, labels,
episode split and ridge code (imported from the Wan probe) — against Cosmos3-Nano's generator tower
(diffusers `Cosmos3OmniTransformer`, 36 MoT layers; its VAE **is** the Wan2.2 VAE). Free on ZeroGPU:
`scripts/deploy_cosmos3_space.py`. Decision rule: frozen Cosmos3 features beat the state-only ridge
(which Wan's could not) → Cosmos3 becomes the primary backbone candidate for the T-16 LoRA;
otherwise stay on Wan — *✅ ran 2026-07-26 (ZeroGPU, 9/9 checks, `runs/cosmos3_probe/`): best block
pair joints test R² 0.359 / gripper 0.708 vs. state-only 0.456 / 0.881 → **frozen Cosmos3 does not
beat state-only either; stay on Wan** for T-16. Robotics pretraining shows up only in the gripper
channel (best single block 0.822 vs. Wan's 0.698), not in joints. Details: `docs/hf_jobs.md`*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
