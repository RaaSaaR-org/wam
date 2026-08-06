---
id: T-15
aliases:
- T-15
title: "Backbone adapters behind one interface — tiny, wan_i2v, flux3 stub"
slug: backbone-adapters-behind-one-interface-tiny-wan-i2v-flux3-st
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
sprint: ''
depends_on: []
due_date: ''
created: 2026-07-26
updated: 2026-07-29
---

# Backbone adapters behind one interface — tiny, wan_i2v, flux3 stub

## Description

Backbone adapters behind one interface: `tiny` (functional), `wan_i2v` (real diffusers integration:
VAE + umT5 + DiT residual-stream hooks, Wan2.1-I2V-14B and Wan2.2-TI2V-5B layouts), `flux3` stub
(OD-06) — *✅ verified on real weights: Wan2.2-TI2V-5B on a ZeroGPU RTX PRO 6000, 13/13 checks,
features `[1, 224, 3072]`, 24.3 GB peak VRAM (`docs/hf_jobs.md`); rerun with
`scripts/deploy_wan_space.py`. Readout blocks measured via `--ablate` (18/18), then
**label-validated** with ridge probes on real GR00T action chunks (8/8,
`scripts/hf_job_wan_probe.py`): early blocks (2, 10) overturn the label-free pick (20, 29) →
`configs/model/wan22_ti2v_5b.yaml`; no frozen features beat state-only yet — LoRA (T-16) carries the
burden. **T-26 re-tested this without the mean-pool and it held**, so the claim is now measured
rather than an artefact of pooling*

---

Migrated from `TASKS.md` (milestone M3) on 2026-08-06. The Description above is that entry verbatim,
re-wrapped only.
